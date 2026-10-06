import os
import subprocess
import sys
import threading
import time
import zlib

import numpy
import pytest
import srsly

import spacy
from spacy.lang.en import English
from spacy.tokens import Doc, DocBin, DocBinReader
from spacy.tokens import _serialize as seg_mod
from spacy.tokens.underscore import Underscore


@pytest.mark.issue(4367)
def test_issue4367():
    """Test that docbin init goes well"""
    DocBin()
    DocBin(attrs=["LEMMA"])
    DocBin(attrs=["LEMMA", "ENT_IOB", "ENT_TYPE"])


@pytest.mark.issue(4528)
def test_issue4528(en_vocab):
    """Test that user_data is correctly serialized in DocBin."""
    doc = Doc(en_vocab, words=["hello", "world"])
    doc.user_data["foo"] = "bar"
    # This is how extension attribute values are stored in the user data
    doc.user_data[("._.", "foo", None, None)] = "bar"
    doc_bin = DocBin(store_user_data=True)
    doc_bin.add(doc)
    doc_bin_bytes = doc_bin.to_bytes()
    new_doc_bin = DocBin(store_user_data=True).from_bytes(doc_bin_bytes)
    new_doc = list(new_doc_bin.get_docs(en_vocab))[0]
    assert new_doc.user_data["foo"] == "bar"
    assert new_doc.user_data[("._.", "foo", None, None)] == "bar"


@pytest.mark.issue(5141)
def test_issue5141(en_vocab):
    """Ensure an empty DocBin does not crash on serialization"""
    doc_bin = DocBin(attrs=["DEP", "HEAD"])
    assert list(doc_bin.get_docs(en_vocab)) == []
    doc_bin_bytes = doc_bin.to_bytes()
    doc_bin_2 = DocBin().from_bytes(doc_bin_bytes)
    assert list(doc_bin_2.get_docs(en_vocab)) == []


def test_serialize_doc_bin():
    doc_bin = DocBin(
        attrs=["LEMMA", "ENT_IOB", "ENT_TYPE", "NORM", "ENT_ID"], store_user_data=True
    )
    texts = ["Some text", "Lots of texts...", "..."]
    cats = {"A": 0.5}
    nlp = English()
    for doc in nlp.pipe(texts):
        doc.cats = cats
        span = doc[0:2]
        span.label_ = "UNUSUAL_SPAN_LABEL"
        span.id_ = "UNUSUAL_SPAN_ID"
        span.kb_id_ = "UNUSUAL_SPAN_KB_ID"
        doc.spans["start"] = [span]
        doc[0].norm_ = "UNUSUAL_TOKEN_NORM"
        doc[0].ent_id_ = "UNUSUAL_TOKEN_ENT_ID"
        doc_bin.add(doc)
    bytes_data = doc_bin.to_bytes()

    # Deserialize later, e.g. in a new process
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_bytes(bytes_data)
    reloaded_docs = list(doc_bin.get_docs(nlp.vocab))
    for i, doc in enumerate(reloaded_docs):
        assert doc.text == texts[i]
        assert doc.cats == cats
        assert len(doc.spans) == 1
        assert doc.spans["start"][0].label_ == "UNUSUAL_SPAN_LABEL"
        assert doc.spans["start"][0].id_ == "UNUSUAL_SPAN_ID"
        assert doc.spans["start"][0].kb_id_ == "UNUSUAL_SPAN_KB_ID"
        assert doc[0].norm_ == "UNUSUAL_TOKEN_NORM"
        assert doc[0].ent_id_ == "UNUSUAL_TOKEN_ENT_ID"


def test_serialize_doc_bin_unknown_spaces(en_vocab):
    doc1 = Doc(en_vocab, words=["that", "'s"])
    assert doc1.has_unknown_spaces
    assert doc1.text == "that 's "
    doc2 = Doc(en_vocab, words=["that", "'s"], spaces=[False, False])
    assert not doc2.has_unknown_spaces
    assert doc2.text == "that's"

    doc_bin = DocBin().from_bytes(DocBin(docs=[doc1, doc2]).to_bytes())
    re_doc1, re_doc2 = doc_bin.get_docs(en_vocab)
    assert re_doc1.has_unknown_spaces
    assert re_doc1.text == "that 's "
    assert not re_doc2.has_unknown_spaces
    assert re_doc2.text == "that's"


@pytest.mark.parametrize(
    "writer_flag,reader_flag,reader_value",
    [
        (True, True, "bar"),
        (True, False, "bar"),
        (False, True, "nothing"),
        (False, False, "nothing"),
    ],
)
def test_serialize_custom_extension(en_vocab, writer_flag, reader_flag, reader_value):
    """Test that custom extensions are correctly serialized in DocBin."""
    Doc.set_extension("foo", default="nothing")
    doc = Doc(en_vocab, words=["hello", "world"])
    doc._.foo = "bar"
    doc_bin_1 = DocBin(store_user_data=writer_flag)
    doc_bin_1.add(doc)
    doc_bin_bytes = doc_bin_1.to_bytes()
    doc_bin_2 = DocBin(store_user_data=reader_flag).from_bytes(doc_bin_bytes)
    doc_2 = list(doc_bin_2.get_docs(en_vocab))[0]
    assert doc_2._.foo == reader_value
    Underscore.doc_extensions = {}


# ---------------------------------------------------------------------------
# Segmented container
# ---------------------------------------------------------------------------

ATTR_NAMES = ["LEMMA", "ENT_IOB", "ENT_TYPE", "NORM", "ENT_ID"]


def _build_docbin(texts, *, store_user_data=True):
    nlp = English()
    cats = {"A": 0.5, "B": 0.25}
    doc_bin = DocBin(attrs=ATTR_NAMES, store_user_data=store_user_data)
    for i, doc in enumerate(nlp.pipe(texts)):
        doc.cats = cats
        if len(doc) >= 2:
            span = doc[0:2]
            span.label_ = "UNUSUAL_SPAN_LABEL"
            span.id_ = f"SPAN_ID_{i}"
            span.kb_id_ = f"SPAN_KB_{i}"
            doc.spans["start"] = [span]
        doc[0].norm_ = f"NORM_{i}"
        doc.user_data["doc_index"] = i
        doc_bin.add(doc)
    return doc_bin


def _assert_docs_equivalent(docs1, docs2):
    """Compare two doc lists item by item (order, annotations, content)."""
    assert len(docs1) == len(docs2)
    for i in range(len(docs1)):
        d1, d2 = docs1[i], docs2[i]
        assert d1.text == d2.text
        assert d1.cats == d2.cats
        arr1 = d1.to_array(ATTR_NAMES)
        arr2 = d2.to_array(ATTR_NAMES)
        assert numpy.array_equal(arr1, arr2)
        assert set(d1.spans.keys()) == set(d2.spans.keys())
        for key in d1.spans:
            g1, g2 = d1.spans[key], d2.spans[key]
            assert len(g1) == len(g2)
            for j in range(len(g1)):
                s1, s2 = g1[j], g2[j]
                assert (s1.text, s1.label_, s1.id_, s1.kb_id_) == (
                    s2.text,
                    s2.label_,
                    s2.id_,
                    s2.kb_id_,
                )
        assert d1.user_data == d2.user_data


def _frame_positions(raw):
    """Return [(frame start, compressed length), ...] for the frames in raw."""
    positions = []
    pos = seg_mod.FILE_HEADER_SIZE
    while raw[pos : pos + 4] == seg_mod.SEG_HEADER_MAGIC:
        _, _, _, _, comp_len, _ = seg_mod.SEG_HEADER_STRUCT.unpack(
            raw[pos : pos + seg_mod.SEG_HEADER_SIZE]
        )
        positions.append((pos, comp_len))
        pos += seg_mod.SEG_HEADER_SIZE + comp_len + seg_mod.SEG_FOOTER_SIZE
    return positions


def test_monolithic_format_unchanged(tmp_path):
    texts = ["Some text", "Lots of texts...", "..."]
    doc_bin = _build_docbin(texts)
    path = tmp_path / "mono.spacy"
    doc_bin.to_disk(path)
    data = path.read_bytes()
    # Default write produces exactly the old bytes (zlib stream, 0x78).
    assert data == doc_bin.to_bytes()
    assert data[0] == 0x78
    # Old bytes are read back unchanged through from_disk.
    nlp = English()
    reloaded = list(DocBin().from_disk(path).get_docs(nlp.vocab))
    expected = list(doc_bin.get_docs(nlp.vocab))
    _assert_docs_equivalent(reloaded, expected)


@pytest.mark.parametrize("segment_size", [1, 2, 3, 7, 100])
def test_segmented_roundtrip_matches_monolithic(tmp_path, segment_size):
    texts = [f"document number {i} with several words here" for i in range(20)]
    doc_bin = _build_docbin(texts)
    path = tmp_path / "seg.spacy"
    doc_bin.to_disk(path, segment_size=segment_size)
    data = path.read_bytes()
    assert data[:8] == seg_mod.CONTAINER_MAGIC

    reader = DocBinReader(path)
    nlp = English()
    reloaded = list(reader.get_docs(nlp.vocab))
    expected = list(
        DocBin().from_bytes(doc_bin.to_bytes()).get_docs(nlp.vocab)
    )
    _assert_docs_equivalent(reloaded, expected)
    assert reader.stats.complete
    expected_segments = (len(texts) + segment_size - 1) // segment_size
    assert reader.stats.segments_read == min(expected_segments, len(texts))
    assert reader.stats.expected_docs == len(texts)
    assert reader.stats.missing_docs == 0

    # from_disk auto-detects the segmented container.
    reloaded_bin = DocBin().from_disk(path)
    assert len(reloaded_bin) == len(texts)
    assert reloaded_bin.load_stats.complete
    _assert_docs_equivalent(
        list(reloaded_bin.get_docs(nlp.vocab)), expected
    )


def test_segmented_empty_bin(tmp_path):
    path = tmp_path / "empty.spacy"
    DocBin().to_disk(path, segment_size=5)
    nlp = English()
    reader = DocBinReader(path)
    assert list(reader.get_docs(nlp.vocab)) == []
    assert reader.stats.complete
    assert reader.stats.expected_segments == 0
    assert len(DocBin().from_disk(path)) == 0


def test_segmented_corrupt_payload(tmp_path):
    texts = [f"document number {i} with several words here" for i in range(12)]
    doc_bin = _build_docbin(texts)
    path = tmp_path / "seg.spacy"
    doc_bin.to_disk(path, segment_size=3)
    raw = bytearray(path.read_bytes())
    positions = _frame_positions(bytes(raw))
    frame_start, comp_len = positions[2]
    raw[frame_start + seg_mod.SEG_HEADER_SIZE + 10] ^= 0xFF
    corrupt_path = tmp_path / "corrupt.spacy"
    corrupt_path.write_bytes(bytes(raw))

    with pytest.raises(ValueError) as excinfo:
        list(DocBinReader(corrupt_path).get_docs(English().vocab))
    assert "segment 2" in str(excinfo.value)

    reader = DocBinReader(corrupt_path, on_error="skip")
    reloaded = list(reader.get_docs(English().vocab))
    stats = reader.stats
    assert len(reloaded) == 9
    assert stats.segments_read == 3
    assert stats.segments_skipped == 1
    assert stats.docs_read == 9
    assert stats.docs_skipped == 3
    assert stats.expected_docs == 12
    # 9 read + 3 skipped = 12: fully reconciled
    assert stats.missing_docs == 0
    assert stats.missing_segments == 1
    assert any("segment 2 skipped" in msg for msg in stats.messages)
    summary = stats.summary()
    for token in ("9", "3", "12"):
        assert token in summary


def test_segmented_truncated(tmp_path):
    texts = [f"document number {i} with several words here" for i in range(12)]
    doc_bin = _build_docbin(texts)
    path = tmp_path / "seg.spacy"
    doc_bin.to_disk(path, segment_size=3)
    trunc_path = tmp_path / "trunc.spacy"
    trunc_path.write_bytes(path.read_bytes()[:-120])

    with pytest.raises(ValueError):
        list(DocBinReader(trunc_path).get_docs(English().vocab))

    reader = DocBinReader(trunc_path, on_error="skip")
    reloaded = list(reader.get_docs(English().vocab))
    # Only fully committed segments come back.
    assert len(reloaded) == 9
    assert not reader.stats.complete
    assert reader.stats.expected_docs is None
    assert any("interrupted" in msg for msg in reader.stats.messages)


def test_docbin_payload_version_mismatch():
    texts = ["Some text", "Lots of texts"]
    doc_bin = _build_docbin(texts)
    msg = srsly.msgpack_loads(zlib.decompress(doc_bin.to_bytes()))
    version_key = b"version" if b"version" in msg else "version"
    msg[version_key] = "9.9"
    bad_bytes = zlib.compress(srsly.msgpack_dumps(msg))
    with pytest.raises(ValueError) as excinfo:
        DocBin().from_bytes(bad_bytes)
    message = str(excinfo.value)
    assert "9.9" in message and "0.1" in message


def test_container_version_mismatch(tmp_path):
    texts = [f"document number {i} with words" for i in range(6)]
    doc_bin = _build_docbin(texts)
    path = tmp_path / "seg.spacy"
    doc_bin.to_disk(path, segment_size=3)
    raw = bytearray(path.read_bytes())
    raw[8:12] = (99).to_bytes(4, "big")
    bad_path = tmp_path / "badver.spacy"
    bad_path.write_bytes(bytes(raw))
    with pytest.raises(ValueError) as excinfo:
        list(DocBinReader(bad_path).get_docs(English().vocab))
    message = str(excinfo.value)
    assert "99" in message and "1" in message


def test_segment_selection(tmp_path):
    texts = [f"document number {i} with words" for i in range(12)]
    doc_bin = _build_docbin(texts)
    path = tmp_path / "seg.spacy"
    doc_bin.to_disk(path, segment_size=3)

    reader = DocBinReader(path, segments=[0, 3])
    reloaded = list(reader.get_docs(English().vocab))
    assert [d.text for d in reloaded] == texts[0:3] + texts[9:12]
    assert reader.stats.segments_filtered == 2

    reader2 = DocBinReader(path, start_segment=2)
    reloaded2 = list(reader2.get_docs(English().vocab))
    assert [d.text for d in reloaded2] == texts[6:]
    assert reader2.stats.segments_filtered == 2


def test_parallel_readers(tmp_path):
    texts = [f"document number {i} with words" for i in range(12)]
    doc_bin = _build_docbin(texts)
    path = tmp_path / "seg.spacy"
    doc_bin.to_disk(path, segment_size=3)
    results = {}

    def worker(key):
        nlp = English()
        results[key] = [d.text for d in DocBinReader(path).get_docs(nlp.vocab)]

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert all(result == texts for result in results.values())


def _write_slow_segment(file_, doc_bin, index, start, end):
    raw = srsly.msgpack_dumps(doc_bin._segment_msg(start, end))
    raw_crc = zlib.crc32(raw) & 0xFFFFFFFF
    payload = zlib.compress(raw)
    comp_crc = zlib.crc32(payload) & 0xFFFFFFFF
    file_.write(
        seg_mod.SEG_HEADER_STRUCT.pack(
            seg_mod.SEG_HEADER_MAGIC,
            index,
            end - start,
            len(raw),
            len(payload),
            raw_crc,
        )
    )
    file_.write(payload)
    file_.write(
        seg_mod.SEG_FOOTER_STRUCT.pack(
            seg_mod.SEG_FOOTER_MAGIC, index, comp_crc
        )
    )
    file_.flush()


def test_read_while_writing(tmp_path):
    texts = [f"document number {i} with words" for i in range(12)]
    doc_bin = _build_docbin(texts)
    path = tmp_path / "slow.spacy"

    def slow_writer():
        with path.open("wb") as file_:
            file_.write(
                seg_mod.FILE_HEADER_STRUCT.pack(
                    seg_mod.CONTAINER_MAGIC,
                    seg_mod.CONTAINER_VERSION,
                    3,
                    seg_mod._FLAG_STORE_USER_DATA,
                )
            )
            file_.flush()
            index = 0
            for start in range(0, len(doc_bin.tokens), 3):
                end = min(start + 3, len(doc_bin.tokens))
                _write_slow_segment(file_, doc_bin, index, start, end)
                index += 1
                time.sleep(0.25)
            file_.write(
                seg_mod.EOF_STRUCT.pack(
                    seg_mod.EOF_MAGIC, index, len(doc_bin.tokens), 0
                )
            )
            file_.write(seg_mod.EOF_TAIL)
            file_.flush()

    writer_thread = threading.Thread(target=slow_writer)
    writer_thread.start()
    time.sleep(0.35)
    # A skip-mode reader only sees committed segments.
    early = list(
        DocBinReader(path, on_error="skip").get_docs(English().vocab)
    )
    assert 3 <= len(early) < 12
    # A waiting reader follows the writer to the completion record.
    waiting = DocBinReader(path, wait=True, timeout=30)
    reloaded = list(waiting.get_docs(English().vocab))
    assert [d.text for d in reloaded] == texts
    assert waiting.stats.complete
    writer_thread.join()


def test_subprocess_reader(tmp_path):
    texts = [f"document number {i} with words" for i in range(12)]
    doc_bin = _build_docbin(texts)
    path = tmp_path / "seg.spacy"
    doc_bin.to_disk(path, segment_size=3)
    code = (
        "import os, spacy; from spacy.tokens import DocBinReader; "
        "r = DocBinReader(os.environ['SPACY_TEST_PATH']); "
        "print(len(list(r.get_docs(spacy.blank('en').vocab))))"
    )
    env = os.environ.copy()
    env["SPACY_TEST_PATH"] = str(path)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "12"


def test_segmented_after_from_bytes(tmp_path):
    texts = [f"document number {i} with words" for i in range(10)]
    doc_bin = _build_docbin(texts)
    # A bin loaded via from_bytes has no per-doc string sets.
    rebin = DocBin().from_bytes(doc_bin.to_bytes())
    assert rebin.doc_strings == []
    path = tmp_path / "rebin.spacy"
    rebin.to_disk(path, segment_size=4)
    reloaded = list(DocBinReader(path).get_docs(English().vocab))
    assert [d.text for d in reloaded] == texts


def test_corpus_reads_segmented(tmp_path):
    from spacy.training.corpus import Corpus

    texts = [f"document number {i} with words" for i in range(9)]
    doc_bin = _build_docbin(texts, store_user_data=False)
    path = tmp_path / "seg.spacy"
    doc_bin.to_disk(path, segment_size=3)
    nlp = English()
    examples = list(Corpus(path)(nlp))
    assert len(examples) == len(texts)
    assert [eg.reference.text for eg in examples] == texts


def test_invalid_segment_size_and_on_error(tmp_path):
    path = tmp_path / "x.spacy"
    with pytest.raises(ValueError):
        DocBin().to_disk(path, segment_size=-1)
    with pytest.raises(ValueError):
        DocBinReader(path, on_error="bogus")
