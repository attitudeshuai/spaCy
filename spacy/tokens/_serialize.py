import os
import struct
import time
import zlib
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set, Union

import numpy
import srsly
from numpy import ndarray
from thinc.api import NumpyOps

from ..attrs import IDS, ORTH, SPACY, intify_attr
from ..compat import copy_reg
from ..errors import Errors
from ..util import SimpleFrozenList, ensure_path
from ..vocab import Vocab
from ._dict_proxies import SpanGroups
from .doc import DOCBIN_ALL_ATTRS as ALL_ATTRS, Doc

# Version of the DocBin payload (the msgpack structure describing the docs).
DOCBIN_VERSION = "0.1"
SUPPORTED_DOCBIN_VERSIONS = ("0.1",)
# Version of the segmented container framing around the payloads.
CONTAINER_VERSION = 1
# The container magic starts with 0x53 ("S"), so it can never be confused with
# a zlib stream (monolithic DocBin), which starts with 0x78.
CONTAINER_MAGIC = b"SPDOCBIN"
SEG_HEADER_MAGIC = b"SEGH"
SEG_FOOTER_MAGIC = b"SEGF"
EOF_MAGIC = b"EOF1"
EOF_TAIL = b"SPEND\r\n"

# File header: magic, container version, segment size (docs per segment), flags
FILE_HEADER_STRUCT = struct.Struct(">8sIIB")
# Segment header: magic, index, num docs, raw length, compressed length, crc
SEG_HEADER_STRUCT = struct.Struct(">4sQIQQI")
# Segment footer: magic, index, crc of the compressed payload
SEG_FOOTER_STRUCT = struct.Struct(">4sQI")
# Completion record: magic, num segments, num docs, crc of frame crc list
EOF_STRUCT = struct.Struct(">4sQQI")

FILE_HEADER_SIZE = FILE_HEADER_STRUCT.size
SEG_HEADER_SIZE = SEG_HEADER_STRUCT.size
SEG_FOOTER_SIZE = SEG_FOOTER_STRUCT.size
EOF_SIZE = EOF_STRUCT.size + len(EOF_TAIL)

_FLAG_STORE_USER_DATA = 1
_READ_CHUNK = 1024 * 1024


def _read_exact(file_, size: int) -> bytes:
    """Read exactly size bytes in bounded chunks. Returns fewer bytes at EOF."""
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = file_.read(min(remaining, _READ_CHUNK))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _scan_for_record(file_, start_pos: int):
    """Scan forward from start_pos for a segment header or completion record.
    Seeks to the record and returns its magic, or returns None at end of file.
    """
    file_.seek(start_pos)
    pos = start_pos
    tail = b""
    while True:
        chunk = file_.read(_READ_CHUNK)
        if not chunk:
            return None
        data = tail + chunk
        data_base = pos - len(tail)
        hits = [data.find(magic) for magic in (SEG_HEADER_MAGIC, EOF_MAGIC)]
        hits = [hit for hit in hits if hit >= 0]
        if hits:
            offset = min(hits)
            abs_pos = data_base + offset
            file_.seek(abs_pos)
            return data[offset : offset + 4]
        tail = data[-3:]
        pos += len(chunk)


class DocBinStats:
    """Statistics of a (possibly partial) DocBin container read.

    All counts are cumulative for the read that produced them, so they can be
    used to reconcile a read that skipped damaged segments.
    """

    def __init__(self) -> None:
        self.complete: bool = False
        self.segments_seen: int = 0  # structurally valid frames encountered
        self.segments_read: int = 0  # frames whose contents were returned
        self.segments_skipped: int = 0  # damaged frames that were skipped
        self.segments_filtered: int = 0  # valid frames excluded by selection
        self.docs_seen: int = 0  # docs in structurally valid frames
        self.docs_read: int = 0  # docs returned
        self.docs_skipped: int = 0  # docs in skipped frames (count known)
        self.expected_segments: Optional[int] = None  # from completion record
        self.expected_docs: Optional[int] = None
        self.messages: List[str] = []

    @property
    def missing_segments(self) -> Optional[int]:
        """Difference between the expected segment count and frames seen."""
        if self.expected_segments is None:
            return None
        return self.expected_segments - self.segments_seen

    @property
    def missing_docs(self) -> Optional[int]:
        """Difference between the expected doc count and docs accounted for."""
        if self.expected_docs is None:
            return None
        return self.expected_docs - self.docs_read - self.docs_skipped

    def summary(self) -> str:
        """Human-readable summary with cumulative counts and discrepancies."""
        lines = [
            f"Container complete: {'yes' if self.complete else 'no'}",
            (
                f"Segments read: {self.segments_read}, skipped: "
                f"{self.segments_skipped}, filtered: {self.segments_filtered}"
            ),
            f"Docs read: {self.docs_read}, skipped: {self.docs_skipped}",
        ]
        if self.expected_segments is not None:
            lines.append(
                f"Segments expected: {self.expected_segments}, difference: "
                f"{self.missing_segments}"
            )
        if self.expected_docs is not None:
            lines.append(
                f"Docs expected: {self.expected_docs}, difference: "
                f"{self.missing_docs}"
            )
        if self.messages:
            lines.append("Notes:")
            lines.extend(f"- {message}" for message in self.messages)
        return "\n".join(lines)


class DocBinReader:
    """Stream documents from a segmented DocBin container.

    Each reader opens its own file handle, so several readers can advance
    through the same container independently (in threads or processes). A
    segment is only ever returned once its complete frame, including its
    commit footer and checksums, is available.

    path (str / Path): The file path.
    on_error (str): Whether to "strict"ly fail or "skip" a damaged segment.
    segments (Optional[Iterable[int]]): Only return these segment indices.
    start_segment (int): Ignore segments before this index (resume reading).
    wait (bool): If the file is not complete yet, wait for more data instead
        of stopping.
    timeout (Optional[float]): Maximum time to wait in seconds.
    poll_interval (float): Polling interval while waiting.
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        on_error: str = "strict",
        segments: Optional[Iterable[int]] = None,
        start_segment: int = 0,
        wait: bool = False,
        timeout: Optional[float] = None,
        poll_interval: float = 0.1,
    ) -> None:
        if on_error not in ("strict", "skip"):
            raise ValueError(Errors.E1062.format(found=on_error))
        self.path = ensure_path(path)
        self.on_error = on_error
        self.segment_filter: Optional[Set[int]] = (
            None if segments is None else set(segments)
        )
        self.start_segment = start_segment
        self.wait = wait
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.store_user_data = False
        self.stats = DocBinStats()

    def get_docs(self, vocab: Vocab) -> Iterator[Doc]:
        """Yield Doc objects in the order they were written.

        vocab (Vocab): The shared vocab.
        YIELDS (Doc): The Doc objects.
        """
        for msg in self.iter_segments():
            yield from DocBin._from_segment_msg(msg).get_docs(vocab)

    def get_bins(self) -> Iterator["DocBin"]:
        """Yield one DocBin per selected segment, in segment order."""
        for msg in self.iter_segments():
            yield DocBin._from_segment_msg(msg)

    def iter_segments(self) -> Iterator[Dict]:
        """Yield the decoded msgpack structure of each selected segment.

        Both the segmented container and the monolithic DocBin format are
        supported: a monolithic file is yielded once as a single segment.
        """
        stats = self.stats
        with self.path.open("rb") as file_:
            header = _read_exact(file_, FILE_HEADER_SIZE)
            if header[:8] != CONTAINER_MAGIC:
                yield from self._iter_monolithic_segment(
                    header, file_, stats
                )
                return
            if not self._validate_file_header(header):
                return
            for index, num_docs, msg in self._iter_container_records(
                file_, stats
            ):
                if index < self.start_segment or (
                    self.segment_filter is not None
                    and index not in self.segment_filter
                ):
                    stats.segments_filtered += 1
                    continue
                stats.segments_read += 1
                stats.docs_read += num_docs
                yield msg

    def _iter_container_records(self, file_, stats) -> Iterator:
        """Walk the frames of a segmented container and yield
        (index, num_docs, msg) for each structurally valid frame."""
        expected_index = 0
        crc_accum = b""
        attrs: Optional[List[int]] = None
        while True:
            frame_pos = file_.tell()
            magic = self._advance_to_record(file_, frame_pos, stats)
            if magic is None:
                return
            if magic == EOF_MAGIC:
                self._read_eof(file_, crc_accum, stats)
                return
            header_rest = _read_exact(file_, SEG_HEADER_SIZE - 4)
            if len(header_rest) < SEG_HEADER_SIZE - 4:
                if self.wait and self._wait_for_data(
                    file_, frame_pos + SEG_HEADER_SIZE
                ):
                    continue
                self._interrupted("segment header is truncated")
                return
            _, index, num_docs, raw_len, comp_len, raw_crc = (
                SEG_HEADER_STRUCT.unpack(magic + header_rest)
            )
            if index != expected_index:
                stats.messages.append(
                    f"segment sequence gap: expected segment "
                    f"{expected_index}, found segment {index}"
                )
            frame_end = (
                frame_pos + SEG_HEADER_SIZE + comp_len + SEG_FOOTER_SIZE
            )
            reason, msg = self._decode_frame(
                file_,
                frame_pos,
                index,
                num_docs,
                raw_len,
                comp_len,
                raw_crc,
                frame_end,
                stats,
            )
            if reason is not None:
                expected_index = index + 1
                continue
            if attrs is None:
                attrs = list(msg["attrs"])
            elif list(msg["attrs"]) != attrs:
                self._corrupt(
                    index,
                    num_docs,
                    "segment attrs do not match earlier segments",
                    frame_end,
                    file_,
                    stats,
                )
                expected_index = index + 1
                continue
            stats.segments_seen += 1
            stats.docs_seen += num_docs
            crc_accum += struct.pack(">QI", index, raw_crc)
            expected_index = index + 1
            yield index, num_docs, msg

    def _advance_to_record(self, file_, frame_pos, stats):
        """Position the file at a valid record start.

        Returns the record magic (SEG_HEADER_MAGIC / EOF_MAGIC), or None to
        signal that reading should stop.
        """
        magic = file_.read(4)
        if not magic:
            if self.wait and self._wait_for_data(file_, frame_pos):
                magic = file_.read(4)
            else:
                self._interrupted(
                    "reached end of file before the completion record"
                )
                return None
        if magic in (SEG_HEADER_MAGIC, EOF_MAGIC):
            return magic
        found = _scan_for_record(file_, frame_pos)
        if found is None:
            self._interrupted("unreadable region followed by end of file")
            return None
        stats.messages.append(
            f"skipped unreadable region from byte {frame_pos} to byte "
            f"{file_.tell()}; number of segments/docs in this region is "
            f"unknown"
        )
        return found

    def _read_checked_payload(self, file_, frame_pos, index, comp_len, stats):
        """Read the compressed payload and commit footer of one frame.

        Returns (payload, None) or (None, reason). Handles waiting for more
        data while the file is being written.
        """
        payload = _read_exact(file_, comp_len)
        if len(payload) < comp_len:
            if self.wait and self._wait_for_data(
                file_, frame_pos + SEG_HEADER_SIZE + len(payload)
            ):
                payload += _read_exact(file_, comp_len - len(payload))
        if len(payload) < comp_len:
            return (
                None,
                f"segment {index} payload is truncated "
                f"({len(payload)} of {comp_len} bytes)",
            )
        footer = _read_exact(file_, SEG_FOOTER_SIZE)
        if len(footer) < SEG_FOOTER_SIZE:
            if self.wait and self._wait_for_data(file_, file_.tell()):
                footer += _read_exact(
                    file_, SEG_FOOTER_SIZE - len(footer)
                )
        if len(footer) < SEG_FOOTER_SIZE:
            return None, f"segment {index} footer is truncated"
        _, footer_index, comp_crc = SEG_FOOTER_STRUCT.unpack(footer)
        if footer_index != index:
            return None, "frame footer does not match the segment index"
        if zlib.crc32(payload) & 0xFFFFFFFF != comp_crc:
            return None, "checksum mismatch for compressed payload"
        return payload, None

    def _decode_frame(
        self,
        file_,
        frame_pos,
        index,
        num_docs,
        raw_len,
        comp_len,
        raw_crc,
        frame_end,
        stats,
    ):
        """Validate and decode one frame. Applies the on-error policy:
        strict raises, skip positions the file at frame_end. Returns
        (reason, None) or (None, msg)."""
        payload, reason = self._read_checked_payload(
            file_, frame_pos, index, comp_len, stats
        )
        if reason is not None:
            self._corrupt(
                index, num_docs, reason, frame_end, file_, stats
            )
            return reason, None
        try:
            raw = zlib.decompress(payload)
        except zlib.error:
            reason = "could not decompress segment"
            self._corrupt(
                index, num_docs, reason, frame_end, file_, stats
            )
            return reason, None
        if len(raw) != raw_len or (
            zlib.crc32(raw) & 0xFFFFFFFF != raw_crc
        ):
            reason = "checksum or length mismatch for segment data"
            self._corrupt(
                index, num_docs, reason, frame_end, file_, stats
            )
            return reason, None
        try:
            msg = srsly.msgpack_loads(raw)
        except Exception:
            reason = "could not decode segment msgpack"
            self._corrupt(
                index, num_docs, reason, frame_end, file_, stats
            )
            return reason, None
        payload_version = msg.get("version", DOCBIN_VERSION)
        if payload_version not in SUPPORTED_DOCBIN_VERSIONS:
            raise ValueError(
                Errors.E1058.format(
                    found=payload_version, supported=DOCBIN_VERSION
                )
            )
        return None, msg

    def _iter_monolithic_segment(self, header, file_, stats):
        """Read a monolithic (gzipped msgpack) file as a single segment."""
        data = header + file_.read()
        try:
            msg = srsly.msgpack_loads(zlib.decompress(data))
        except (zlib.error, ValueError, srsly.MsgpackDeserializationError):
            raise ValueError(Errors.E1014) from None
        version = msg.get("version", DOCBIN_VERSION)
        if version not in SUPPORTED_DOCBIN_VERSIONS:
            raise ValueError(
                Errors.E1058.format(found=version, supported=DOCBIN_VERSION)
            )
        lengths = numpy.frombuffer(msg["lengths"], dtype="int32")
        num_docs = int(lengths.size)
        stats.segments_seen += 1
        stats.docs_seen += num_docs
        if 0 < self.start_segment or (
            self.segment_filter is not None and 0 not in self.segment_filter
        ):
            stats.segments_filtered += 1
            return
        stats.segments_read += 1
        stats.docs_read += num_docs
        stats.expected_segments = 1
        stats.expected_docs = num_docs
        stats.complete = True
        yield msg

    def _validate_file_header(self, header: bytes) -> bool:
        if len(header) < FILE_HEADER_SIZE:
            self._interrupted("file header is truncated")
            return False
        magic, container_version, _, flags = FILE_HEADER_STRUCT.unpack(header)
        if magic != CONTAINER_MAGIC:
            raise ValueError(
                Errors.E1063.format(path=self.path, magic=CONTAINER_MAGIC)
            )
        if container_version != CONTAINER_VERSION:
            raise ValueError(
                Errors.E1059.format(
                    path=self.path,
                    found=container_version,
                    supported=CONTAINER_VERSION,
                )
            )
        self.store_user_data = bool(flags & _FLAG_STORE_USER_DATA)
        return True

    def _corrupt(
        self, index, num_docs, reason, frame_end, file_, stats
    ) -> None:
        stats.segments_skipped += 1
        stats.docs_skipped += num_docs
        stats.messages.append(f"segment {index} skipped: {reason}")
        if self.on_error == "strict":
            raise ValueError(
                Errors.E1060.format(index=index, path=self.path, reason=reason)
            )
        file_.seek(frame_end)

    def _interrupted(self, reason: str) -> None:
        stats = self.stats
        stats.messages.append(f"reading interrupted: {reason}")
        if self.on_error == "strict":
            raise ValueError(
                Errors.E1061.format(path=self.path, reason=reason)
            )

    def _wait_for_data(self, file_, position: int) -> bool:
        deadline = None
        if self.timeout is not None:
            deadline = time.monotonic() + self.timeout
        while True:
            time.sleep(self.poll_interval)
            try:
                size = self.path.stat().st_size
            except OSError:
                size = position
            if size > position:
                file_.seek(position)
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False

    def _read_eof(self, file_, crc_accum, stats) -> None:
        rest = _read_exact(file_, EOF_SIZE - 4)
        if len(rest) < EOF_SIZE - 4:
            self._interrupted("completion record is truncated")
            return
        body = rest[: EOF_STRUCT.size - 4]
        tail = rest[EOF_STRUCT.size - 4 :]
        _, num_segments, num_docs, expected_crc = EOF_STRUCT.unpack(
            EOF_MAGIC + body
        )
        if tail != EOF_TAIL:
            self._interrupted("completion record is corrupted")
            return
        stats.expected_segments = num_segments
        stats.expected_docs = num_docs
        if (zlib.crc32(crc_accum) & 0xFFFFFFFF) != expected_crc:
            stats.messages.append(
                "completion record checksum does not match the frames read"
            )
        if num_segments != stats.segments_seen:
            stats.messages.append(
                f"segment count discrepancy: expected {num_segments}, saw "
                f"{stats.segments_seen}"
            )
        accounted = stats.docs_read + stats.docs_skipped
        if num_docs != accounted:
            stats.messages.append(
                f"doc count discrepancy: expected {num_docs}, accounted for "
                f"{accounted} ({stats.docs_read} read, "
                f"{stats.docs_skipped} skipped)"
            )
        stats.complete = True


class DocBin:
    """Pack Doc objects for binary serialization.

    The DocBin class lets you efficiently serialize the information from a
    collection of Doc objects. You can control which information is serialized
    by passing a list of attribute IDs, and optionally also specify whether the
    user data is serialized. The DocBin is faster and produces smaller data
    sizes than pickle, and allows you to deserialize without executing arbitrary
    Python code.

    The default serialization format is gzipped msgpack, where the msgpack
    object has the following structure:

    {
        "attrs": List[uint64], # e.g. [TAG, HEAD, ENT_IOB, ENT_TYPE]
        "tokens": bytes, # Serialized numpy uint64 array with the token data
        "spans": List[Dict[str, bytes]], # SpanGroups data for each doc
        "spaces": bytes, # Serialized numpy boolean array with spaces data
        "lengths": bytes, # Serialized numpy int32 array with the doc lengths
        "strings": List[str] # List of unique strings in the token data
        "version": str, # DocBin version number
    }

    Strings for the words, tags, labels etc are represented by 64-bit hashes in
    the token data, and every string that occurs at least once is passed via the
    strings object. This means the storage is more efficient if you pack more
    documents together, because you have less duplication in the strings.

    A notable downside to this format is that you can't easily extract just one
    document from the DocBin. To enable segment-wise reading, write to disk
    with a `segment_size` and use a DocBinReader.
    """

    def __init__(
        self,
        attrs: Iterable[str] = ALL_ATTRS,
        store_user_data: bool = False,
        docs: Iterable[Doc] = SimpleFrozenList(),
    ) -> None:
        """Create a DocBin object to hold serialized annotations.

        attrs (Iterable[str]): List of attributes to serialize. 'orth' and
            'spacy' are always serialized, so they're not required.
        store_user_data (bool): Whether to write the `Doc.user_data` to bytes/file.
        docs (Iterable[Doc]): Docs to add.

        DOCS: https://spacy.io/api/docbin#init
        """
        int_attrs = [intify_attr(attr) for attr in attrs]
        if None in int_attrs:
            non_valid = [attr for attr in attrs if intify_attr(attr) is None]
            raise KeyError(
                Errors.E983.format(dict="attrs", key=non_valid, keys=IDS.keys())
            ) from None
        attrs = sorted(int_attrs)
        self.version = DOCBIN_VERSION
        self.attrs = [attr for attr in attrs if attr != ORTH and attr != SPACY]
        self.attrs.insert(0, ORTH)  # Ensure ORTH is always attrs[0]
        self.tokens: List[ndarray] = []
        self.spaces: List[ndarray] = []
        self.cats: List[Dict] = []
        self.span_groups: List[bytes] = []
        self.user_data: List[Optional[bytes]] = []
        self.flags: List[Dict] = []
        self.strings: Set[str] = set()
        # Per-doc strings, used to keep each written segment self-contained.
        self.doc_strings: List[Set[str]] = []
        self.store_user_data = store_user_data
        for doc in docs:
            self.add(doc)

    def __len__(self) -> int:
        """RETURNS: The number of Doc objects added to the DocBin."""
        return len(self.tokens)

    def add(self, doc: Doc) -> None:
        """Add a Doc's annotations to the DocBin for serialization.

        doc (Doc): The Doc object.

        DOCS: https://spacy.io/api/docbin#add
        """
        array = doc.to_array(self.attrs)
        if len(array.shape) == 1:
            array = array.reshape((array.shape[0], 1))
        self.tokens.append(array)
        spaces = doc.to_array(SPACY)
        assert array.shape[0] == spaces.shape[0]  # this should never happen
        spaces = spaces.reshape((spaces.shape[0], 1))
        self.spaces.append(numpy.asarray(spaces, dtype=bool))
        self.flags.append({"has_unknown_spaces": doc.has_unknown_spaces})
        doc_strings = set()
        for token in doc:
            for string in (
                token.text,
                token.tag_,
                token.lemma_,
                token.norm_,
                str(token.morph),
                token.dep_,
                token.ent_type_,
                token.ent_kb_id_,
                token.ent_id_,
            ):
                self.strings.add(string)
                doc_strings.add(string)
        self.cats.append(doc.cats)
        if self.store_user_data:
            self.user_data.append(srsly.msgpack_dumps(doc.user_data))
        self.span_groups.append(doc.spans.to_bytes())
        for _, group in doc.spans.items():
            for span in group:
                doc_strings.add(span.label_)
                self.strings.add(span.label_)
                if span.kb_id in span.doc.vocab.strings:
                    doc_strings.add(span.kb_id_)
                    self.strings.add(span.kb_id_)
                if span.id in span.doc.vocab.strings:
                    doc_strings.add(span.id_)
                    self.strings.add(span.id_)
        self.doc_strings.append(doc_strings)

    def get_docs(self, vocab: Vocab) -> Iterator[Doc]:
        """Recover Doc objects from the annotations, using the given vocab.
        Note that the user data of each doc will be read (if available) and returned,
        regardless of the setting of 'self.store_user_data'.

        vocab (Vocab): The shared vocab.
        YIELDS (Doc): The Doc objects.

        DOCS: https://spacy.io/api/docbin#get_docs
        """
        for string in self.strings:
            vocab[string]
        orth_col = self.attrs.index(ORTH)
        for i in range(len(self.tokens)):
            flags = self.flags[i]
            tokens = self.tokens[i]
            spaces: Optional[ndarray] = self.spaces[i]
            if flags.get("has_unknown_spaces"):
                spaces = None
            doc = Doc(vocab, words=tokens[:, orth_col], spaces=spaces)  # type: ignore
            doc = doc.from_array(self.attrs, tokens)  # type: ignore
            doc.cats = self.cats[i]
            # backwards-compatibility: may be b'' or serialized empty list
            if self.span_groups[i] and self.span_groups[i] != SpanGroups._EMPTY_BYTES:
                doc.spans.from_bytes(self.span_groups[i])
            else:
                doc.spans.clear()
            if i < len(self.user_data) and self.user_data[i] is not None:
                user_data = srsly.msgpack_loads(self.user_data[i], use_list=False)
                doc.user_data.update(user_data)
            yield doc

    def merge(self, other: "DocBin") -> None:
        """Extend the annotations of this DocBin with the annotations from
        another. Will raise an error if the pre-defined attrs of the two
        DocBins don't match, or if they differ in whether or not to store
        user data.

        other (DocBin): The DocBin to merge into the current bin.

        DOCS: https://spacy.io/api/docbin#merge
        """
        if self.attrs != other.attrs:
            raise ValueError(
                Errors.E166.format(param="attrs", current=self.attrs, other=other.attrs)
            )
        if self.store_user_data != other.store_user_data:
            raise ValueError(
                Errors.E166.format(
                    param="store_user_data",
                    current=self.store_user_data,
                    other=other.store_user_data,
                )
            )
        self.tokens.extend(other.tokens)
        self.spaces.extend(other.spaces)
        self.strings.update(other.strings)
        self.cats.extend(other.cats)
        self.span_groups.extend(other.span_groups)
        self.flags.extend(other.flags)
        self.user_data.extend(other.user_data)
        if len(other.doc_strings) == len(other.tokens):
            self.doc_strings.extend(other.doc_strings)
        else:
            self.doc_strings = []

    def _segment_msg(self, start: int, end: int) -> Dict:
        """Build the self-contained msgpack structure for docs [start:end]."""
        seg_tokens = self.tokens[start:end]
        seg_spaces = self.spaces[start:end]
        lengths = [len(tokens) for tokens in seg_tokens]
        tokens = numpy.vstack(seg_tokens) if seg_tokens else numpy.asarray([])
        spaces = numpy.vstack(seg_spaces) if seg_spaces else numpy.asarray([])
        if len(self.doc_strings) == len(self.tokens):
            seg_strings: Set[str] = set()
            for doc_string_set in self.doc_strings[start:end]:
                seg_strings.update(doc_string_set)
        else:
            # Bins loaded via from_bytes don't carry per-doc string sets:
            # include the full string table to keep docs resolvable.
            seg_strings = set(self.strings)
        msg = {
            "version": self.version,
            "attrs": self.attrs,
            "tokens": tokens.tobytes("C"),
            "spaces": spaces.tobytes("C"),
            "lengths": numpy.asarray(lengths, dtype="int32").tobytes("C"),
            "strings": sorted(seg_strings),
            "cats": self.cats[start:end],
            "flags": self.flags[start:end],
            "span_groups": self.span_groups[start:end],
        }
        if self.store_user_data:
            msg["user_data"] = self.user_data[start:end]
        return msg

    def to_bytes(self) -> bytes:
        """Serialize the DocBin's annotations to a bytestring.

        RETURNS (bytes): The serialized DocBin.

        DOCS: https://spacy.io/api/docbin#to_bytes
        """
        for tokens in self.tokens:
            assert len(tokens.shape) == 2, tokens.shape  # this should never happen
        lengths = [len(tokens) for tokens in self.tokens]
        tokens = numpy.vstack(self.tokens) if self.tokens else numpy.asarray([])
        spaces = numpy.vstack(self.spaces) if self.spaces else numpy.asarray([])
        msg = {
            "version": self.version,
            "attrs": self.attrs,
            "tokens": tokens.tobytes("C"),
            "spaces": spaces.tobytes("C"),
            "lengths": numpy.asarray(lengths, dtype="int32").tobytes("C"),
            "strings": sorted(self.strings),
            "cats": self.cats,
            "flags": self.flags,
            "span_groups": self.span_groups,
        }
        if self.store_user_data:
            msg["user_data"] = self.user_data
        return zlib.compress(srsly.msgpack_dumps(msg))

    @classmethod
    def _from_segment_msg(cls, msg: Dict) -> "DocBin":
        """Construct a one-segment DocBin from a decoded segment structure."""
        self = cls.__new__(cls)
        self.version = msg.get("version", DOCBIN_VERSION)
        if self.version not in SUPPORTED_DOCBIN_VERSIONS:
            raise ValueError(
                Errors.E1058.format(
                    found=self.version, supported=DOCBIN_VERSION
                )
            )
        self.attrs = msg["attrs"]
        self.strings = set(msg["strings"])
        lengths = numpy.frombuffer(msg["lengths"], dtype="int32")
        flat_spaces = numpy.frombuffer(msg["spaces"], dtype="bool")
        flat_tokens = numpy.frombuffer(msg["tokens"], dtype="uint64")
        shape = (flat_tokens.size // len(self.attrs), len(self.attrs))
        flat_tokens = flat_tokens.reshape(shape)
        flat_spaces = flat_spaces.reshape((flat_spaces.size, 1))
        self.tokens = NumpyOps().unflatten(flat_tokens, lengths)
        self.spaces = NumpyOps().unflatten(flat_spaces, lengths)
        self.cats = msg["cats"]
        self.span_groups = msg.get(
            "span_groups", [b"" for _ in lengths]
        )
        self.flags = msg.get("flags", [{} for _ in lengths])
        if "user_data" in msg:
            self.user_data = list(msg["user_data"])
            self.store_user_data = True
        else:
            self.user_data = [None] * len(self)
            self.store_user_data = False
        self.doc_strings = []
        for tokens in self.tokens:
            assert len(tokens.shape) == 2, tokens.shape  # this should never happen
        return self

    def from_bytes(self, bytes_data: bytes) -> "DocBin":
        """Deserialize the DocBin's annotations from a bytestring.

        bytes_data (bytes): The data to load from.
        RETURNS (DocBin): The loaded DocBin.

        DOCS: https://spacy.io/api/docbin#from_bytes
        """
        try:
            msg = srsly.msgpack_loads(zlib.decompress(bytes_data))
        except zlib.error:
            raise ValueError(Errors.E1014)
        version = msg.get("version", DOCBIN_VERSION)
        if version not in SUPPORTED_DOCBIN_VERSIONS:
            raise ValueError(
                Errors.E1058.format(found=version, supported=DOCBIN_VERSION)
            )
        self.version = version
        self.attrs = msg["attrs"]
        self.strings = set(msg["strings"])
        lengths = numpy.frombuffer(msg["lengths"], dtype="int32")
        flat_spaces = numpy.frombuffer(msg["spaces"], dtype="bool")
        flat_tokens = numpy.frombuffer(msg["tokens"], dtype="uint64")
        shape = (flat_tokens.size // len(self.attrs), len(self.attrs))
        flat_tokens = flat_tokens.reshape(shape)
        flat_spaces = flat_spaces.reshape((flat_spaces.size, 1))
        self.tokens = NumpyOps().unflatten(flat_tokens, lengths)
        self.spaces = NumpyOps().unflatten(flat_spaces, lengths)
        self.cats = msg["cats"]
        self.span_groups = msg.get("span_groups", [b"" for _ in lengths])
        self.flags = msg.get("flags", [{} for _ in lengths])
        if "user_data" in msg:
            self.user_data = list(msg["user_data"])
        else:
            self.user_data = [None] * len(self)
        self.doc_strings = []
        for tokens in self.tokens:
            assert len(tokens.shape) == 2, tokens.shape  # this should never happen
        return self

    def to_disk(
        self, path: Union[str, Path], *, segment_size: int = 0
    ) -> None:
        """Save the DocBin to a file (typically called .spacy).

        By default the DocBin is written as a single gzipped msgpack object,
        exactly as in previous spaCy versions. If segment_size is a positive
        number, the docs are split into frames of at most segment_size docs:
        each frame carries its own checksums and a frame is only visible to
        readers once its commit footer is written. The file ends with a
        completion record, so an interrupted write can never be mistaken for a
        complete file.

        path (str / Path): The file path.
        segment_size (int): Maximum number of docs per segment, or 0 to write
            the monolithic format.

        DOCS: https://spacy.io/api/docbin#to_disk
        """
        path = ensure_path(path)
        segment_size = int(segment_size)
        if segment_size < 0:
            raise ValueError(Errors.E1064.format(value=segment_size))
        elif segment_size > 0:
            self._to_disk_segmented(path, segment_size)
            return
        with path.open("wb") as file_:
            try:
                file_.write(self.to_bytes())
            except ValueError:
                raise ValueError(Errors.E870)

    def _to_disk_segmented(
        self, path: Path, segment_size: int
    ) -> None:
        flags = _FLAG_STORE_USER_DATA if self.store_user_data else 0
        with path.open("wb") as file_:
            file_.write(
                FILE_HEADER_STRUCT.pack(
                    CONTAINER_MAGIC, CONTAINER_VERSION, segment_size, flags
                )
            )
            file_.flush()
            crc_accum = b""
            total = len(self.tokens)
            index = 0
            for start in range(0, total, segment_size):
                end = min(start + segment_size, total)
                raw = srsly.msgpack_dumps(self._segment_msg(start, end))
                raw_crc = zlib.crc32(raw) & 0xFFFFFFFF
                payload = zlib.compress(raw)
                comp_crc = zlib.crc32(payload) & 0xFFFFFFFF
                file_.write(
                    SEG_HEADER_STRUCT.pack(
                        SEG_HEADER_MAGIC,
                        index,
                        end - start,
                        len(raw),
                        len(payload),
                        raw_crc,
                    )
                )
                file_.write(payload)
                file_.write(
                    SEG_FOOTER_STRUCT.pack(
                        SEG_FOOTER_MAGIC, index, comp_crc
                    )
                )
                file_.flush()
                crc_accum += struct.pack(">QI", index, raw_crc)
                index += 1
            eof_crc = zlib.crc32(crc_accum) & 0xFFFFFFFF
            file_.write(
                EOF_STRUCT.pack(EOF_MAGIC, index, total, eof_crc)
            )
            file_.write(EOF_TAIL)
            file_.flush()
            try:
                os.fsync(file_.fileno())
            except (OSError, ValueError):
                pass

    def from_disk(
        self, path: Union[str, Path], *, on_error: str = "strict"
    ) -> "DocBin":
        """Load the DocBin from a file (typically called .spacy).

        Both the monolithic format and the segmented container format are
        supported. Files written by older spaCy versions are read back
        unchanged.

        path (str / Path): The file path.
        on_error (str): For segmented files, whether to "strict"ly fail or
            "skip" damaged segments.
        RETURNS (DocBin): The loaded DocBin.

        DOCS: https://spacy.io/api/docbin#from_disk
        """
        path = ensure_path(path)
        reader = DocBinReader(path, on_error=on_error)
        return self._load_from_reader(reader)

    def _load_from_reader(self, reader: DocBinReader) -> "DocBin":
        """Load all segments of a container (uses memory proportional to the
        whole file, same as a monolithic load; streaming reads should use the
        DocBinReader directly)."""
        bins = [DocBin._from_segment_msg(msg) for msg in reader.iter_segments()]
        if not bins:
            self.__init__(store_user_data=reader.store_user_data)
        else:
            self.version = bins[0].version
            self.attrs = list(bins[0].attrs)
            self.store_user_data = reader.store_user_data
            self.tokens = []
            self.spaces = []
            self.strings = set()
            self.cats = []
            self.span_groups = []
            self.user_data = []
            self.flags = []
            self.doc_strings = []
            for seg_bin in bins:
                self.tokens.extend(seg_bin.tokens)
                self.spaces.extend(seg_bin.spaces)
                self.strings.update(seg_bin.strings)
                self.cats.extend(seg_bin.cats)
                self.span_groups.extend(seg_bin.span_groups)
                self.user_data.extend(seg_bin.user_data)
                self.flags.extend(seg_bin.flags)
        self.load_stats = reader.stats
        return self


def merge_bins(bins):
    merged = None
    for byte_string in bins:
        if byte_string is not None:
            doc_bin = DocBin(store_user_data=True).from_bytes(byte_string)
            if merged is None:
                merged = doc_bin
            else:
                merged.merge(doc_bin)
    if merged is not None:
        return merged.to_bytes()
    else:
        return b""


def pickle_bin(doc_bin):
    return (unpickle_bin, (doc_bin.to_bytes(),))


def unpickle_bin(byte_string):
    return DocBin().from_bytes(byte_string)


copy_reg.pickle(DocBin, pickle_bin, unpickle_bin)
# Compatibility, as we had named it this previously.
Binder = DocBin

__all__ = ["DocBin", "DocBinReader", "DocBinStats"]
