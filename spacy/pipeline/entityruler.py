import importlib
import json
import threading
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import srsly

from ..errors import Errors, MatchPatternError, Warnings
from ..language import Language
from ..matcher import Matcher, PhraseMatcher
from ..matcher.levenshtein import levenshtein_compare
from ..scorer import get_ner_prf
from ..tokens import Doc, Span
from ..training import Example
from ..util import SimpleFrozenList, ensure_path, from_disk, to_disk
from .pipe import Pipe

DEFAULT_ENT_ID_SEP = "||"
PatternType = Dict[str, Union[str, List[Dict[str, Any]]]]


def entity_ruler_score(examples, **kwargs):
    return get_ner_prf(examples)


def make_entity_ruler_scorer():
    return entity_ruler_score


class EntityRuler(Pipe):
    """The EntityRuler lets you add spans to the `Doc.ents` using token-based
    rules or exact phrase matches. It can be combined with the statistical
    `EntityRecognizer` to boost accuracy, or used on its own to implement a
    purely rule-based entity recognition system. After initialization, the
    component is typically added to the pipeline using `nlp.add_pipe`.

    DOCS: https://spacy.io/api/entityruler
    USAGE: https://spacy.io/usage/rule-based-matching#entityruler
    """

    def __init__(
        self,
        nlp: Language,
        name: str = "entity_ruler",
        *,
        phrase_matcher_attr: Optional[Union[int, str]] = None,
        matcher_fuzzy_compare: Callable = levenshtein_compare,
        validate: bool = False,
        overwrite_ents: bool = False,
        ent_id_sep: str = DEFAULT_ENT_ID_SEP,
        patterns: Optional[List[PatternType]] = None,
        scorer: Optional[Callable] = entity_ruler_score,
    ) -> None:
        """Initialize the entity ruler. If patterns are supplied here, they
        need to be a list of dictionaries with a `"label"` and `"pattern"`
        key. A pattern can either be a token pattern (list) or a phrase pattern
        (string). For example: `{'label': 'ORG', 'pattern': 'Apple'}`.

        nlp (Language): The shared nlp object to pass the vocab to the matchers
            and process phrase patterns.
        name (str): Instance name of the current pipeline component. Typically
            passed in automatically from the factory when the component is
            added. Used to disable the current entity ruler while creating
            phrase patterns with the nlp object.
        phrase_matcher_attr (int / str): Token attribute to match on, passed
            to the internal PhraseMatcher as `attr`.
        matcher_fuzzy_compare (Callable): The fuzzy comparison method for the
            internal Matcher. Defaults to
            spacy.matcher.levenshtein.levenshtein_compare.
        validate (bool): Whether patterns should be validated, passed to
            Matcher and PhraseMatcher as `validate`
        patterns (iterable): Optional patterns to load in.
        overwrite_ents (bool): If existing entities are present, e.g. entities
            added by the model, overwrite them by matches if necessary.
        ent_id_sep (str): Separator used internally for entity IDs.
        scorer (Optional[Callable]): The scoring method. Defaults to
            spacy.scorer.get_ner_prf.

        DOCS: https://spacy.io/api/entityruler#init
        """
        self.nlp = nlp
        self.name = name
        self.overwrite = overwrite_ents
        # Re-entrant lock guarding the pattern list / matchers / ent-ID map as
        # one consistent generation. Readers and writers take this lock so no
        # caller can ever observe a half-applied update.
        self._lock = threading.RLock()
        self.token_patterns = defaultdict(list)  # type: ignore
        self.phrase_patterns = defaultdict(list)  # type: ignore
        self._validate = validate
        self.matcher_fuzzy_compare = matcher_fuzzy_compare
        self.phrase_matcher_attr = phrase_matcher_attr
        self.ent_id_sep = ent_id_sep
        self._ent_ids = defaultdict(tuple)  # type: ignore
        self.clear()
        if patterns is not None:
            self.add_patterns(patterns)
        self.scorer = scorer

    def __len__(self) -> int:
        """The number of all patterns added to the entity ruler."""
        with self._lock:
            n_token_patterns = sum(len(p) for p in self.token_patterns.values())
            n_phrase_patterns = sum(len(p) for p in self.phrase_patterns.values())
            return n_token_patterns + n_phrase_patterns

    def __contains__(self, label: str) -> bool:
        """Whether a label is present in the patterns."""
        with self._lock:
            return label in self.token_patterns or label in self.phrase_patterns

    def __call__(self, doc: Doc) -> Doc:
        """Find matches in document and add them as entities.

        doc (Doc): The Doc object in the pipeline.
        RETURNS (Doc): The Doc with added entities, if available.

        DOCS: https://spacy.io/api/entityruler#call
        """
        error_handler = self.get_error_handler()
        try:
            # Hold the lock for the full match + annotation step so the
            # matches and the label/ID map always come from the same pattern
            # generation, even if another thread updates the ruler in between.
            with self._lock:
                matches = self.match(doc)
                self.set_annotations(doc, matches)
            return doc
        except Exception as e:
            return error_handler(self.name, self, [doc], e)

    def match(self, doc: Doc):
        with self._lock:
            self._require_patterns()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="\\[W036")
                # Resolve both matchers from the same generation before calling
                # them: an update swaps the pair atomically under this lock.
                matcher = self.matcher
                phrase_matcher = self.phrase_matcher
                matches = list(matcher(doc)) + list(phrase_matcher(doc))

        final_matches = set(
            [(m_id, start, end) for m_id, start, end in matches if start != end]
        )
        get_sort_key = lambda m: (m[2] - m[1], -m[1])
        final_matches = sorted(final_matches, key=get_sort_key, reverse=True)
        return final_matches

    def set_annotations(self, doc, matches):
        """Modify the document in place"""
        # Snapshot the map once so a concurrent update can't interleave with
        # the per-match lookups below.
        ent_ids = self._ent_ids
        entities = list(doc.ents)
        new_entities = []
        seen_tokens = set()
        for match_id, start, end in matches:
            if any(t.ent_type for t in doc[start:end]) and not self.overwrite:
                continue
            # check for end - 1 here because boundaries are inclusive
            if start not in seen_tokens and end - 1 not in seen_tokens:
                if match_id in ent_ids:
                    label, ent_id = ent_ids[match_id]
                    span = Span(doc, start, end, label=label, span_id=ent_id)
                else:
                    span = Span(doc, start, end, label=match_id)
                new_entities.append(span)
                entities = [
                    e for e in entities if not (e.start < end and e.end > start)
                ]
                seen_tokens.update(range(start, end))
        doc.ents = entities + new_entities

    @property
    def labels(self) -> Tuple[str, ...]:
        """All labels present in the match patterns.

        RETURNS (set): The string labels.

        DOCS: https://spacy.io/api/entityruler#labels
        """
        with self._lock:
            keys = set(self.token_patterns.keys())
            keys.update(self.phrase_patterns.keys())
            all_labels = set()

            for l in keys:
                if self.ent_id_sep in l:
                    label, _ = self._split_label(l)
                    all_labels.add(label)
                else:
                    all_labels.add(l)
            return tuple(sorted(all_labels))

    def initialize(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        nlp: Optional[Language] = None,
        patterns: Optional[Sequence[PatternType]] = None,
    ):
        """Initialize the pipe for training.

        get_examples (Callable[[], Iterable[Example]]): Function that
            returns a representative sample of gold-standard Example objects.
        nlp (Language): The current nlp object the component is part of.
        patterns Optional[Iterable[PatternType]]: The list of patterns.

        DOCS: https://spacy.io/api/entityruler#initialize
        """
        self.clear()
        if patterns:
            self.add_patterns(patterns)  # type: ignore[arg-type]

    @property
    def ent_ids(self) -> Tuple[Optional[str], ...]:
        """All entity ids present in the match patterns `id` properties

        RETURNS (set): The string entity ids.

        DOCS: https://spacy.io/api/entityruler#ent_ids
        """
        with self._lock:
            keys = set(self.token_patterns.keys())
            keys.update(self.phrase_patterns.keys())
            all_ent_ids = set()

            for l in keys:
                if self.ent_id_sep in l:
                    _, ent_id = self._split_label(l)
                    all_ent_ids.add(ent_id)
            return tuple(all_ent_ids)

    @property
    def patterns(self) -> List[PatternType]:
        """Get all patterns that were added to the entity ruler.

        RETURNS (list): The original patterns, one dictionary per pattern.

        DOCS: https://spacy.io/api/entityruler#patterns
        """
        with self._lock:
            all_patterns = []
            for label, patterns in self.token_patterns.items():
                for pattern in patterns:
                    ent_label, ent_id = self._split_label(label)
                    p = {"label": ent_label, "pattern": pattern}
                    if ent_id:
                        p["id"] = ent_id
                    all_patterns.append(p)
            for label, patterns in self.phrase_patterns.items():
                for pattern in patterns:
                    ent_label, ent_id = self._split_label(label)
                    p = {"label": ent_label, "pattern": pattern.text}
                    if ent_id:
                        p["id"] = ent_id
                    all_patterns.append(p)
            return all_patterns

    def add_patterns(self, patterns: List[PatternType]) -> None:
        """Add patterns to the entity ruler. A pattern can either be a token
        pattern (list of dicts) or a phrase pattern (string). For example:
        {'label': 'ORG', 'pattern': 'Apple'}
        {'label': 'GPE', 'pattern': [{'lower': 'san'}, {'lower': 'francisco'}]}

        The update is atomic: either all patterns are added or none of them
        are, and the ruler keeps its previous state. If a pattern is invalid,
        a ValueError is raised that reports the index of the offending entry
        and the reason. Duplicate patterns in a batch (or relative to the
        patterns already present) are ignored, so the resulting state does
        not depend on the order of the entries.

        patterns (list): The patterns to add.

        DOCS: https://spacy.io/api/entityruler#add_patterns
        """
        patterns = list(patterns)
        # Phase 1: validate the whole batch structurally, before anything is
        # touched. An invalid entry is reported with its index and the ruler
        # keeps its previous state.
        for index, entry in enumerate(patterns):
            self._validate_pattern_entry(index, entry)

        # Phase 2: turn phrase strings into Docs. This runs through the shared
        # pipeline with subsequent components disabled, but it completes
        # BEFORE any state is touched, so a failing pipeline run cannot leave
        # the ruler in a half-updated state.
        token_entries = []
        phrase_entries = []
        phrase_texts = []
        for index, entry in enumerate(patterns):
            has_id = "id" in entry
            ent_id = entry.get("id")
            if isinstance(entry["pattern"], str):
                phrase_entries.append((index, entry["label"], has_id, ent_id))
                phrase_texts.append(entry["pattern"])
            else:
                token_entries.append(
                    (index, entry["label"], has_id, ent_id, entry["pattern"])
                )

        # disable the nlp components after this one in case they hadn't been initialized / deserialised yet
        try:
            current_index = -1
            for i, (name, pipe) in enumerate(self.nlp.pipeline):
                if self == pipe:
                    current_index = i
                    break
            subsequent_pipes = [pipe for pipe in self.nlp.pipe_names[current_index:]]
        except ValueError:
            subsequent_pipes = []
        with self.nlp.select_pipes(disable=subsequent_pipes):
            phrase_docs = list(self.nlp.pipe(phrase_texts))

        # Phases 3 and 4: build the complete new generation (pattern lists,
        # matchers and label/ID map) off to the side and then publish it in a
        # single swap. Writers are serialized, while readers either see the
        # state before or after this call, never an intermediate one.
        with self._lock:
            staged = self._stage_patterns(
                token_entries,
                [
                    (index, label, has_id, ent_id, doc)
                    for (index, label, has_id, ent_id), doc in zip(
                        phrase_entries, phrase_docs
                    )
                ],
            )
            self._commit(staged)

    def _validate_pattern_entry(self, index: int, entry: PatternType) -> None:
        """Check one raw add_patterns entry and raise an E1058 ValueError that
        names the entry index and the reason if it is malformed."""

        def invalid(reason: str) -> None:
            raise ValueError(
                Errors.E1058.format(
                    index=index, component=self.name, reason=reason
                )
            )

        if not isinstance(entry, dict):
            invalid(
                f"expected a dict with 'label' and 'pattern' keys, but got: "
                f"{type(entry).__name__}"
            )
        if "label" not in entry or not isinstance(entry.get("label"), str):
            invalid("the 'label' key is required and has to be a string")
        if "pattern" not in entry:
            invalid("the 'pattern' key is required")
        pattern = entry["pattern"]
        if isinstance(pattern, str):
            return
        if isinstance(pattern, list):
            if len(pattern) == 0:
                invalid(Errors.E012.format(key=entry["label"]))
            if not all(isinstance(token_spec, dict) for token_spec in pattern):
                invalid("each token pattern has to be a list of dicts")
        else:
            invalid(Errors.E097.format(pattern=pattern))

    @staticmethod
    def _token_pattern_key(pattern: List[Dict[str, Any]]) -> str:
        """Hashable, order-insensitive identity for a token pattern."""
        return json.dumps(pattern, sort_keys=True, ensure_ascii=False)

    def _new_matcher(self) -> Matcher:
        return Matcher(
            self.nlp.vocab,
            validate=self._validate,
            fuzzy_compare=self.matcher_fuzzy_compare,
        )

    def __reduce__(self):
        # The Cython-generated Pipe.__reduce__ serializes the full __dict__
        # and bypasses __getstate__, so the unpicklable lock has to be removed
        # from the state here (e.g. for nlp.pipe(n_process>1)).
        func, args, state = super().__reduce__()
        name, attrs = state
        attrs = {key: value for key, value in attrs.items() if key != "_lock"}
        return func, args, (name, attrs)

    def __setstate__(self, state) -> None:
        super().__setstate__(state)
        self._lock = threading.RLock()

    def _new_phrase_matcher(self) -> PhraseMatcher:
        return PhraseMatcher(
            self.nlp.vocab,
            attr=self.phrase_matcher_attr,
            validate=self._validate,
        )

    def _stage_patterns(self, token_entries, phrase_entries):
        """Build a complete, self-consistent new generation of the pattern
        lists, matchers and label/ID map by carrying over the current state
        and applying the (already structurally validated) new entries. Must be
        called with self._lock held. Returns the staged 5-tuple."""
        token_patterns = defaultdict(
            list,
            {label: list(pats) for label, pats in self.token_patterns.items()},
        )
        phrase_patterns = defaultdict(
            list,
            {label: list(docs) for label, docs in self.phrase_patterns.items()},
        )
        matcher = self._new_matcher()
        phrase_matcher = self._new_phrase_matcher()
        ent_ids = dict(self._ent_ids)

        # Re-populate the fresh matchers with the carried-over patterns.
        for label, pats in token_patterns.items():
            matcher.add(label, list(pats))
        for label, docs in phrase_patterns.items():
            phrase_matcher.add(label, list(docs))

        # Identity set used to make repeated additions idempotent.
        seen = set()
        for label, pats in token_patterns.items():
            for pattern in pats:
                seen.add(("token", label, self._token_pattern_key(pattern)))
        for label, docs in phrase_patterns.items():
            for doc in docs:
                seen.add(("phrase", label, doc.text))

        # Token patterns are applied before phrase patterns to preserve the
        # historical ordering of the pattern lists within one add call.
        def apply_entry(index, raw_label, has_id, ent_id, kind, value):
            created_label = (
                self._create_label(raw_label, ent_id) if has_id else raw_label
            )
            if kind == "token":
                identity = (
                    "token",
                    created_label,
                    self._token_pattern_key(value),
                )
                if identity not in seen:
                    try:
                        matcher.add(created_label, [value])
                    except MatchPatternError:
                        # Keep the dedicated validation error type as-is.
                        raise
                    except Exception as e:
                        raise ValueError(
                            Errors.E1058.format(
                                index=index, component=self.name, reason=str(e)
                            )
                        ) from e
                    seen.add(identity)
                    token_patterns[created_label].append(value)
            else:
                if len(value) == 0:
                    raise ValueError(
                        Errors.E1058.format(
                            index=index,
                            component=self.name,
                            reason="phrase pattern doesn't contain any tokens",
                        )
                    )
                identity = ("phrase", created_label, value.text)
                if identity not in seen:
                    try:
                        phrase_matcher.add(created_label, [value])
                    except Exception as e:
                        raise ValueError(
                            Errors.E1058.format(
                                index=index, component=self.name, reason=str(e)
                            )
                        ) from e
                    seen.add(identity)
                    phrase_patterns[created_label].append(value)
            if has_id:
                ent_ids[matcher._normalize_key(created_label)] = (
                    raw_label,
                    ent_id,
                )

        for index, label, has_id, ent_id, pattern in token_entries:
            apply_entry(index, label, has_id, ent_id, "token", pattern)
        for index, label, has_id, ent_id, doc in phrase_entries:
            apply_entry(index, label, has_id, ent_id, "phrase", doc)

        # Keep the map in exact correspondence with the pattern lists: any key
        # without a matching label (e.g. left behind by older code or by a
        # deletion) is swept away.
        final_labels = set(token_patterns) | set(phrase_patterns)
        final_keys = {matcher._normalize_key(label) for label in final_labels}
        ent_ids = defaultdict(
            tuple, {key: val for key, val in ent_ids.items() if key in final_keys}
        )
        return (
            token_patterns,
            phrase_patterns,
            ent_ids,
            matcher,
            phrase_matcher,
        )

    def _commit(self, staged) -> None:
        """Publish a staged generation. Must be called with self._lock held."""
        (
            self.token_patterns,
            self.phrase_patterns,
            self._ent_ids,
            self.matcher,
            self.phrase_matcher,
        ) = staged

    def clear(self) -> None:
        """Reset all patterns."""
        with self._lock:
            self._commit(
                (
                    defaultdict(list),
                    defaultdict(list),
                    defaultdict(tuple),
                    self._new_matcher(),
                    self._new_phrase_matcher(),
                )
            )

    def remove(
        self,
        ent_id: Optional[str] = None,
        *,
        label: Optional[str] = None,
    ) -> None:
        """Remove patterns by their `id` and/or `label`.

        Passing only an `ent_id` removes all patterns with that ID (across
        labels), passing only `label` removes all patterns with that label
        (with or without IDs), and passing both removes only the pattern(s)
        for the exact label/ID combination. The pattern lists, the matchers
        and the label/ID map are updated together, so no stale entries remain.

        If no pattern matches the given `ent_id`/`label`, a ValueError is
        raised and the state is left unchanged. Removing the same pattern
        twice therefore succeeds once and raises on the second call.

        ent_id (str): ID of the pattern(s) to be removed.
        label (str): Label of the pattern(s) to be removed.
        RETURNS: None
        DOCS: https://spacy.io/api/entityruler#remove
        """
        if ent_id is None and label is None:
            raise TypeError(Errors.E1059.format(component=self.name))
        with self._lock:
            current_labels = set(self.token_patterns) | set(self.phrase_patterns)
            remove_labels = set()
            for created_label in current_labels:
                ent_label, _ = self._split_label(created_label)
                if label is not None and ent_label != label:
                    continue
                if ent_id is not None:
                    # ID-based removal only affects patterns that were added
                    # with this ID (recorded in the label/ID map).
                    mapped = self._ent_ids.get(
                        self.matcher._normalize_key(created_label)
                    )
                    if mapped is None or mapped[1] != ent_id:
                        continue
                remove_labels.add(created_label)
            if not remove_labels:
                if label is None:
                    attr_type, value = "ID", ent_id
                elif ent_id is None:
                    attr_type, value = "label", label
                else:
                    attr_type, value = (
                        "ID/label",
                        f"{label}{self.ent_id_sep}{ent_id}",
                    )
                raise ValueError(
                    Errors.E1024.format(
                        attr_type=attr_type, label=value, component=self.name
                    )
                )
            token_patterns = defaultdict(
                list,
                {
                    key: list(pats)
                    for key, pats in self.token_patterns.items()
                    if key not in remove_labels
                },
            )
            phrase_patterns = defaultdict(
                list,
                {
                    key: list(docs)
                    for key, docs in self.phrase_patterns.items()
                    if key not in remove_labels
                },
            )
            self._commit(self._rebuild_state(token_patterns, phrase_patterns))

    def _rebuild_state(self, token_patterns, phrase_patterns):
        """Create matchers and a label/ID map that exactly correspond to the
        given pattern lists. Must be called with self._lock held."""
        matcher = self._new_matcher()
        phrase_matcher = self._new_phrase_matcher()
        for key, pats in token_patterns.items():
            matcher.add(key, list(pats))
        for key, docs in phrase_patterns.items():
            phrase_matcher.add(key, list(docs))
        final_keys = {
            matcher._normalize_key(key)
            for key in set(token_patterns) | set(phrase_patterns)
        }
        ent_ids = defaultdict(
            tuple,
            {key: val for key, val in self._ent_ids.items() if key in final_keys},
        )
        return (
            token_patterns,
            phrase_patterns,
            ent_ids,
            matcher,
            phrase_matcher,
        )

    def _require_patterns(self) -> None:
        """Raise a warning if this component has no patterns defined."""
        if len(self) == 0:
            warnings.warn(Warnings.W036.format(name=self.name))

    def _split_label(self, label: str) -> Tuple[str, Optional[str]]:
        """Split Entity label into ent_label and ent_id if it contains self.ent_id_sep

        label (str): The value of label in a pattern entry
        RETURNS (tuple): ent_label, ent_id
        """
        if self.ent_id_sep in label:
            ent_label, ent_id = label.rsplit(self.ent_id_sep, 1)
        else:
            ent_label = label
            ent_id = None  # type: ignore
        return ent_label, ent_id

    def _create_label(self, label: Any, ent_id: Any) -> str:
        """Join Entity label with ent_id if the pattern has an `id` attribute
        If ent_id is not a string, the label is returned as is.

        label (str): The label to set for ent.label_
        ent_id (str): The label
        RETURNS (str): The ent_label joined with configured `ent_id_sep`
        """
        if isinstance(ent_id, str):
            label = f"{label}{self.ent_id_sep}{ent_id}"
        return label

    def from_bytes(
        self, patterns_bytes: bytes, *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> "EntityRuler":
        """Load the entity ruler from a bytestring.

        patterns_bytes (bytes): The bytestring to load.
        RETURNS (EntityRuler): The loaded entity ruler.

        DOCS: https://spacy.io/api/entityruler#from_bytes
        """
        cfg = srsly.msgpack_loads(patterns_bytes)
        with self._lock:
            if isinstance(cfg, dict):
                # Apply the configuration first so the matchers created by
                # add_patterns already use the restored settings. This keeps
                # the pattern lists, matchers and label/ID map in sync after
                # the round-trip.
                self.overwrite = cfg.get("overwrite", False)
                self.phrase_matcher_attr = cfg.get("phrase_matcher_attr", None)
                self.ent_id_sep = cfg.get("ent_id_sep", DEFAULT_ENT_ID_SEP)
                patterns = cfg.get("patterns", cfg)
            else:
                patterns = cfg
            self.clear()
            if patterns:
                self.add_patterns(patterns)
        return self

    def to_bytes(self, *, exclude: Iterable[str] = SimpleFrozenList()) -> bytes:
        """Serialize the entity ruler patterns to a bytestring.

        RETURNS (bytes): The serialized patterns.

        DOCS: https://spacy.io/api/entityruler#to_bytes
        """
        serial = {
            "overwrite": self.overwrite,
            "ent_id_sep": self.ent_id_sep,
            "phrase_matcher_attr": self.phrase_matcher_attr,
            "patterns": self.patterns,
        }
        return srsly.msgpack_dumps(serial)

    def from_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> "EntityRuler":
        """Load the entity ruler from a file. Expects a file containing
        newline-delimited JSON (JSONL) with one entry per line.

        path (str / Path): The JSONL file to load.
        RETURNS (EntityRuler): The loaded entity ruler.

        DOCS: https://spacy.io/api/entityruler#from_disk
        """
        path = ensure_path(path)
        depr_patterns_path = path.with_suffix(".jsonl")
        with self._lock:
            if path.suffix == ".jsonl":  # user provides a jsonl
                if path.is_file:
                    patterns = srsly.read_jsonl(path)
                    self.clear()
                    self.add_patterns(patterns)
                else:
                    raise ValueError(Errors.E1023.format(path=path))
            elif depr_patterns_path.is_file():
                patterns = srsly.read_jsonl(depr_patterns_path)
                self.clear()
                self.add_patterns(patterns)
            elif path.is_dir():  # path is a valid directory
                cfg = {}
                deserializers_patterns = {
                    "patterns": lambda p: self.add_patterns(
                        srsly.read_jsonl(p.with_suffix(".jsonl"))
                    )
                }
                deserializers_cfg = {"cfg": lambda p: cfg.update(srsly.read_json(p))}
                from_disk(path, deserializers_cfg, {})
                # Apply the configuration before re-adding the patterns so the
                # matchers created during add_patterns use it directly; this
                # avoids replacing a populated phrase matcher with an empty one
                # after the fact.
                self.overwrite = cfg.get("overwrite", False)
                self.phrase_matcher_attr = cfg.get("phrase_matcher_attr")
                self.ent_id_sep = cfg.get("ent_id_sep", DEFAULT_ENT_ID_SEP)
                self.clear()
                from_disk(path, deserializers_patterns, {})
            else:  # path is not a valid directory or file
                raise ValueError(Errors.E146.format(path=path))
        return self

    def to_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = SimpleFrozenList()
    ) -> None:
        """Save the entity ruler patterns to a directory. The patterns will be
        saved as newline-delimited JSON (JSONL).

        path (str / Path): The JSONL file to save.

        DOCS: https://spacy.io/api/entityruler#to_disk
        """
        path = ensure_path(path)
        cfg = {
            "overwrite": self.overwrite,
            "phrase_matcher_attr": self.phrase_matcher_attr,
            "ent_id_sep": self.ent_id_sep,
        }
        serializers = {
            "patterns": lambda p: srsly.write_jsonl(
                p.with_suffix(".jsonl"), self.patterns
            ),
            "cfg": lambda p: srsly.write_json(p, cfg),
        }
        if path.suffix == ".jsonl":  # user wants to save only JSONL
            srsly.write_jsonl(path, self.patterns)
        else:
            to_disk(path, serializers, {})


# Setup backwards compatibility hook for factories
def __getattr__(name):
    if name == "make_entity_ruler":
        module = importlib.import_module("spacy.pipeline.factories")
        return module.make_entity_ruler
    raise AttributeError(f"module {__name__} has no attribute {name}")
