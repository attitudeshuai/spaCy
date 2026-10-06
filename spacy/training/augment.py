import contextvars
import hashlib
import itertools
import random
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

from ..errors import Errors
from ..util import logger
from .example import Example
from .iob_utils import _doc_to_biluo_tags_with_partial, split_bilu_label

from ..pipeline._parser_internals.nonproj import contains_cycle as _contains_cycle

if TYPE_CHECKING:
    from ..language import Language  # noqa: F401


# Per-call RNG used by the built-in augmenters. The AugmentingScheduler sets
# this to a stream derived from the training seed for each individual input,
# so augmentation results don't depend on the process-global random stream
# (which is also consumed by shuffling etc.). Outside of a scheduler the
# variable is unset and the process-global ``random`` module is used, keeping
# the old behavior for direct calls.
_augment_rng: "contextvars.ContextVar[Optional[random.Random]]" = (
    contextvars.ContextVar("augment_rng", default=None)
)


def get_augmenting_rng() -> Any:
    """Return the RNG active for the current augmentation call. Falls back to
    the process-global ``random`` module when no per-call RNG was set (e.g.
    when an augmenter is invoked directly rather than through a scheduler).
    """
    rng = _augment_rng.get()
    return random if rng is None else rng


def create_combined_augmenter(
    lower_level: float,
    orth_level: float,
    orth_variants: Optional[Dict[str, List[Dict]]],
    whitespace_level: float,
    whitespace_per_token: float,
    whitespace_variants: Optional[List[str]],
) -> Callable[["Language", Example], Iterator[Example]]:
    """Create a data augmentation callback that uses orth-variant replacement.
    The callback can be added to a corpus or other data iterator during training.

    lower_level (float): The percentage of texts that will be lowercased.
    orth_level (float): The percentage of texts that will be augmented.
    orth_variants (Optional[Dict[str, List[Dict]]]): A dictionary containing the
        single and paired orth variants. Typically loaded from a JSON file.
    whitespace_level (float): The percentage of texts that will have whitespace
        tokens inserted.
    whitespace_per_token (float): The number of whitespace tokens to insert in
        the modified doc as a percentage of the doc length.
    whitespace_variants (Optional[List[str]]): The whitespace token texts.
    RETURNS (Callable[[Language, Example], Iterator[Example]]): The augmenter.
    """
    return partial(
        combined_augmenter,
        lower_level=lower_level,
        orth_level=orth_level,
        orth_variants=orth_variants,
        whitespace_level=whitespace_level,
        whitespace_per_token=whitespace_per_token,
        whitespace_variants=whitespace_variants,
    )


def combined_augmenter(
    nlp: "Language",
    example: Example,
    *,
    lower_level: float = 0.0,
    orth_level: float = 0.0,
    orth_variants: Optional[Dict[str, List[Dict]]] = None,
    whitespace_level: float = 0.0,
    whitespace_per_token: float = 0.0,
    whitespace_variants: Optional[List[str]] = None,
) -> Iterator[Example]:
    rng = get_augmenting_rng()
    if rng.random() < lower_level:
        example = make_lowercase_variant(nlp, example)
    if orth_variants and rng.random() < orth_level:
        raw_text = example.text
        orig_dict = example.to_dict()
        orig_dict["doc_annotation"]["entities"] = _doc_to_biluo_tags_with_partial(
            example.reference
        )
        variant_text, variant_token_annot = make_orth_variants(
            nlp,
            raw_text,
            orig_dict["token_annotation"],
            orth_variants,
            lower=False,
            rng=rng,
        )
        orig_dict["token_annotation"] = variant_token_annot
        example = example.from_dict(nlp.make_doc(variant_text), orig_dict)
    if whitespace_variants and rng.random() < whitespace_level:
        for _ in range(int(len(example.reference) * whitespace_per_token)):
            example = make_whitespace_variant(
                nlp,
                example,
                rng.choice(whitespace_variants),
                rng.randrange(0, len(example.reference)),
            )
    yield example


def create_orth_variants_augmenter(
    level: float, lower: float, orth_variants: Dict[str, List[Dict]]
) -> Callable[["Language", Example], Iterator[Example]]:
    """Create a data augmentation callback that uses orth-variant replacement.
    The callback can be added to a corpus or other data iterator during training.

    level (float): The percentage of texts that will be augmented.
    lower (float): The percentage of texts that will be lowercased.
    orth_variants (Dict[str, List[Dict]]): A dictionary containing
        the single and paired orth variants. Typically loaded from a JSON file.
    RETURNS (Callable[[Language, Example], Iterator[Example]]): The augmenter.
    """
    return partial(
        orth_variants_augmenter, orth_variants=orth_variants, level=level, lower=lower
    )


def create_lower_casing_augmenter(
    level: float,
) -> Callable[["Language", Example], Iterator[Example]]:
    """Create a data augmentation callback that converts documents to lowercase.
    The callback can be added to a corpus or other data iterator during training.

    level (float): The percentage of texts that will be augmented.
    RETURNS (Callable[[Language, Example], Iterator[Example]]): The augmenter.
    """
    return partial(lower_casing_augmenter, level=level)


def dont_augment(nlp: "Language", example: Example) -> Iterator[Example]:
    yield example


def lower_casing_augmenter(
    nlp: "Language", example: Example, *, level: float
) -> Iterator[Example]:
    rng = get_augmenting_rng()
    if rng.random() >= level:
        yield example
    else:
        yield make_lowercase_variant(nlp, example)


def make_lowercase_variant(nlp: "Language", example: Example):
    example_dict = example.to_dict()
    example_dict["doc_annotation"]["entities"] = _doc_to_biluo_tags_with_partial(
        example.reference
    )
    doc = nlp.make_doc(example.text.lower())
    example_dict["token_annotation"]["ORTH"] = [t.lower_ for t in example.reference]
    return example.from_dict(doc, example_dict)


def orth_variants_augmenter(
    nlp: "Language",
    example: Example,
    orth_variants: Dict[str, List[Dict]],
    *,
    level: float = 0.0,
    lower: float = 0.0,
) -> Iterator[Example]:
    rng = get_augmenting_rng()
    if rng.random() >= level:
        yield example
    else:
        raw_text = example.text
        orig_dict = example.to_dict()
        orig_dict["doc_annotation"]["entities"] = _doc_to_biluo_tags_with_partial(
            example.reference
        )
        variant_text, variant_token_annot = make_orth_variants(
            nlp,
            raw_text,
            orig_dict["token_annotation"],
            orth_variants,
            lower=raw_text is not None and rng.random() < lower,
            rng=rng,
        )
        orig_dict["token_annotation"] = variant_token_annot
        yield example.from_dict(nlp.make_doc(variant_text), orig_dict)


def make_orth_variants(
    nlp: "Language",
    raw: str,
    token_dict: Dict[str, List[str]],
    orth_variants: Dict[str, List[Dict[str, List[str]]]],
    *,
    lower: bool = False,
    rng: Any = None,
) -> Tuple[str, Dict[str, List[str]]]:
    if rng is None:
        rng = get_augmenting_rng()
    words = token_dict.get("ORTH", [])
    tags = token_dict.get("TAG", [])
    # keep unmodified if words are not defined
    if not words:
        return raw, token_dict
    if lower:
        words = [w.lower() for w in words]
        raw = raw.lower()
    # if no tags, only lowercase
    if not tags:
        token_dict["ORTH"] = words
        return raw, token_dict
    # single variants
    ndsv = orth_variants.get("single", [])
    punct_choices = [rng.choice(x["variants"]) for x in ndsv]
    for word_idx in range(len(words)):
        for punct_idx in range(len(ndsv)):
            if (
                tags[word_idx] in ndsv[punct_idx]["tags"]
                and words[word_idx] in ndsv[punct_idx]["variants"]
            ):
                words[word_idx] = punct_choices[punct_idx]
    # paired variants
    ndpv = orth_variants.get("paired", [])
    punct_choices = [rng.choice(x["variants"]) for x in ndpv]
    for word_idx in range(len(words)):
        for punct_idx in range(len(ndpv)):
            if tags[word_idx] in ndpv[punct_idx]["tags"] and words[
                word_idx
            ] in itertools.chain.from_iterable(ndpv[punct_idx]["variants"]):
                # backup option: random left vs. right from pair
                pair_idx = rng.choice([0, 1])
                # best option: rely on paired POS tags like `` / ''
                if len(ndpv[punct_idx]["tags"]) == 2:
                    pair_idx = ndpv[punct_idx]["tags"].index(tags[word_idx])
                # next best option: rely on position in variants
                # (may not be unambiguous, so order of variants matters)
                else:
                    for pair in ndpv[punct_idx]["variants"]:
                        if words[word_idx] in pair:
                            pair_idx = pair.index(words[word_idx])
                words[word_idx] = punct_choices[punct_idx][pair_idx]
    token_dict["ORTH"] = words
    raw = construct_modified_raw_text(token_dict)
    return raw, token_dict


def make_whitespace_variant(
    nlp: "Language",
    example: Example,
    whitespace: str,
    position: int,
) -> Example:
    """Insert the whitespace token at the specified token offset in the doc.
    This is primarily intended for v2-compatible training data that doesn't
    include links or spans. If the document includes links, spans, or partial
    dependency annotation, it is returned without modifications.

    The augmentation follows the basics of the v2 space attachment policy, but
    without a distinction between "real" and other tokens, so space tokens
    may be attached to space tokens:
    - at the beginning of a sentence attach the space token to the following
      token
    - otherwise attach the space token to the preceding token

    The augmenter does not attempt to consolidate adjacent whitespace in the
    same way that the tokenizer would.

    The following annotation is used for the space token:
    TAG: "_SP"
    MORPH: ""
    POS: "SPACE"
    LEMMA: ORTH
    DEP: "dep"
    SENT_START: False

    The annotation for each attribute is only set for the space token if there
    is already at least partial annotation for that attribute in the original
    example.

    RETURNS (Example): Example with one additional space token.
    """
    example_dict = example.to_dict()
    example_dict["doc_annotation"]["entities"] = _doc_to_biluo_tags_with_partial(
        example.reference
    )
    doc_dict = example_dict.get("doc_annotation", {})
    token_dict = example_dict.get("token_annotation", {})
    # returned unmodified if:
    # - doc is empty
    # - words are not defined
    # - links are defined (only character-based offsets, which is more a quirk
    #   of Example.to_dict than a technical constraint)
    # - spans are defined
    # - there are partial dependencies
    if (
        len(example.reference) == 0
        or "ORTH" not in token_dict
        or len(doc_dict.get("links", [])) > 0
        or len(example.reference.spans) > 0
        or (
            example.reference.has_annotation("DEP")
            and not example.reference.has_annotation("DEP", require_complete=True)
        )
    ):
        return example
    words = token_dict.get("ORTH", [])
    length = len(words)
    assert 0 <= position <= length
    if example.reference.has_annotation("ENT_TYPE"):
        # I-ENTITY if between B/I-ENTITY and I/L-ENTITY otherwise O
        entity = "O"
        if position > 1 and position < length:
            ent_prev = doc_dict["entities"][position - 1]
            ent_next = doc_dict["entities"][position]
            if "-" in ent_prev and "-" in ent_next:
                ent_iob_prev, ent_type_prev = split_bilu_label(ent_prev)
                ent_iob_next, ent_type_next = split_bilu_label(ent_next)
                if (
                    ent_iob_prev in ("B", "I")
                    and ent_iob_next in ("I", "L")
                    and ent_type_prev == ent_type_next
                ):
                    entity = f"I-{ent_type_prev}"
        doc_dict["entities"].insert(position, entity)
    else:
        del doc_dict["entities"]
    token_dict["ORTH"].insert(position, whitespace)
    token_dict["SPACY"].insert(position, False)
    if example.reference.has_annotation("TAG"):
        token_dict["TAG"].insert(position, "_SP")
    else:
        del token_dict["TAG"]
    if example.reference.has_annotation("LEMMA"):
        token_dict["LEMMA"].insert(position, whitespace)
    else:
        del token_dict["LEMMA"]
    if example.reference.has_annotation("POS"):
        token_dict["POS"].insert(position, "SPACE")
    else:
        del token_dict["POS"]
    if example.reference.has_annotation("MORPH"):
        token_dict["MORPH"].insert(position, "")
    else:
        del token_dict["MORPH"]
    if example.reference.has_annotation("DEP", require_complete=True):
        if position == 0:
            token_dict["HEAD"].insert(position, 0)
        else:
            token_dict["HEAD"].insert(position, position - 1)
        for i in range(len(token_dict["HEAD"])):
            if token_dict["HEAD"][i] >= position:
                token_dict["HEAD"][i] += 1
        token_dict["DEP"].insert(position, "dep")
    else:
        del token_dict["HEAD"]
        del token_dict["DEP"]
    if example.reference.has_annotation("SENT_START"):
        token_dict["SENT_START"].insert(position, False)
    else:
        del token_dict["SENT_START"]
    raw = construct_modified_raw_text(token_dict)
    return Example.from_dict(nlp.make_doc(raw), example_dict)


def construct_modified_raw_text(token_dict):
    """Construct modified raw text from words and spaces."""
    raw = ""
    for orth, spacy in zip(token_dict["ORTH"], token_dict["SPACY"]):
        raw += orth
        if spacy:
            raw += " "
    return raw


# ---------------------------------------------------------------------------
# Scheduled augmentation
#
# Unlike the stateless per-example augmenters above, the scheduler keeps the
# state of a single training epoch: how many inputs are still allowed to be
# augmented (quota) and counters for what was emitted or skipped. The random
# streams used for selection and augmentation are derived from the training
# seed and a stable per-input id, so results are reproducible and independent
# of the process-global random stream and of interleaved iterators.
# ---------------------------------------------------------------------------

ALIGN_ENTITIES = "entities"
ALIGN_SPANS = "spans"
ALIGN_DEPS = "deps"
ALIGN_FIELDS = (ALIGN_ENTITIES, ALIGN_SPANS, ALIGN_DEPS)
ON_FAILURE_SKIP = "skip"
ON_FAILURE_PASS = "pass"
ON_FAILURE_VALUES = (ON_FAILURE_SKIP, ON_FAILURE_PASS)

_SEED_NAMESPACE = "spacy.augment.v1"


def derive_seed(*parts) -> int:
    """Derive an independent 64-bit seed from arbitrary hashable parts.

    Two calls with the same parts always produce the same seed, while changing
    any part produces an unrelated stream. Only stdlib hashing is used so the
    result does not consume (or depend on) the process-global RNG.
    """
    payload = b"\x1f".join(
        str(part).encode("utf8", "backslashreplace") for part in parts
    )
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _normalized_span_text(text: str) -> str:
    return text.lower().strip().replace(" ", "")


def validate_variant(
    orig: Example, variant: Example, fields: Iterable[str] = ALIGN_FIELDS
) -> List[str]:
    """Check that an augmented Example keeps its annotations aligned with its
    predicted tokenization.

    Entities, spans (all span groups) and the dependency parse are checked, but
    only for annotation that is present on the original input.

    orig (Example): The unaugmented input example.
    variant (Example): The candidate variant produced by an augmenter.
    fields (Iterable[str]): Annotation levels to check, any of "entities",
        "spans" and "deps".
    RETURNS (List[str]): A list of problem identifiers; the variant is valid
        exactly when the list is empty.
    """
    fields = tuple(fields)
    problems: List[str] = []
    x = variant.predicted
    y = variant.reference
    alignment = variant.alignment

    # Basic coverage: every token on both sides must map to the other side.
    if fields:
        x2y_covered = all(alignment.x2y.lengths > 0)
        y2x_covered = all(alignment.y2x.lengths > 0)
        if not (x2y_covered and y2x_covered):
            problems.append("tokens:unaligned")

    if ALIGN_ENTITIES in fields and orig.reference.has_annotation("ENT_IOB"):
        orig_ents = [
            (ent.label, _normalized_span_text(ent.text)) for ent in orig.reference.ents
        ]
        var_ents = [(ent.label, _normalized_span_text(ent.text)) for ent in y.ents]
        if orig_ents != var_ents:
            problems.append("entities:mismatch")
        else:
            # Every entity in the variant reference must align losslessly to
            # one or more predicted tokens.
            x_ents, _ = variant.get_aligned_ents_and_ner()
            if len(x_ents) != len(var_ents):
                problems.append("entities:unaligned")

    if ALIGN_SPANS in fields:
        for key in orig.reference.spans:
            orig_spans = [
                (span.label, _normalized_span_text(span.text))
                for span in orig.reference.spans[key]
            ]
            var_group = list(y.spans.get(key, []))
            var_spans = [
                (span.label, _normalized_span_text(span.text)) for span in var_group
            ]
            if orig_spans != var_spans:
                problems.append(f"spans:{key}:mismatch")
                continue
            aligned = variant.get_aligned_spans_y2x(var_group)
            if len(aligned) != len(var_spans):
                problems.append(f"spans:{key}:unaligned")

    if ALIGN_DEPS in fields and orig.reference.has_annotation(
        "DEP", require_complete=True
    ):
        if not y.has_annotation("DEP", require_complete=True):
            problems.append("deps:incomplete")
        elif (
            len(x) != len(y)
            or not all(alignment.x2y.lengths == 1)
            or not all(alignment.y2x.lengths == 1)
        ):
            # A dependency parse cannot be projected without loss across
            # split/merged tokens, so require a 1:1 tokenization.
            problems.append("deps:unaligned")
        else:
            heads = [token.head.i for token in y]
            if any(head < 0 or head >= len(y) for head in heads):
                problems.append("deps:head")
            elif _contains_cycle(heads):
                problems.append("deps:cycle")

    return problems


def _as_uid_example_pairs(
    items: Iterable[Union[Example, Tuple[int, Example]]],
) -> Iterator[Tuple[int, Example]]:
    """Normalize a stream of Examples or (uid, Example) pairs."""
    for index, item in enumerate(items):
        if isinstance(item, tuple):
            yield item  # type: ignore[misc]
        else:
            yield index, item


class AugmentingScheduler:
    """Stateful, quota-based data augmentation scheduler.

    The scheduler wraps an existing augmenter callable (the same
    ``(nlp, example) -> Iterator[Example]`` protocol) and drives it once per
    training epoch:

    * Exactly ``quota`` inputs per epoch are augmented when the epoch size is
      known (absolute quota), or ``round(ratio * n_inputs)`` (target ratio).
      Inputs not selected for augmentation pass through unchanged.
    * For streaming epochs of unknown length a fixed quota caps the number of
      augmented inputs, while a ratio is applied as a per-input derived
      Bernoulli draw.
    * Which inputs are selected and which variant is produced is decided by
      ``random.Random`` streams derived from the training seed, the epoch
      number and a stable per-input id (plus an optional shard id). The
      process-global random stream is never consumed, so repeated runs with
      the same config/data are item-by-item identical and interleaved
      iterators do not share random state.
    * A selected input may produce zero, one or multiple variants (the inner
      augmenter controls this, and ``variants`` draws independent variant
      sets). Every variant is checked for entity/span/dependency alignment
      before it is emitted; failed or unalignable variants are skipped and
      counted.
    * When both quota and ratio are 0 the scheduler is disabled: every input
      passes through unchanged and no random numbers are drawn.

    quota (int): Maximum number of inputs augmented per epoch. 0 disables
        augmentation unless ratio is set.
    ratio (float): Target fraction of inputs to augment per epoch when the
        epoch size is known. Mutually exclusive with quota.
    variants (int): Number of independent variant sets to draw per selected
        input (each with its own derived stream).
    seed (Optional[int]): Base seed. When None, the training seed is read
        from the nlp config at the start of the epoch.
    shard (int): Optional shard id mixed into the derived streams, so parallel
        data iterators get independent random streams.
    on_failure (str): "skip" drops an input whose augmentation failed or whose
        variants do not align; "pass" falls back to the unchanged input. Both
        are counted.
    align (Optional[Iterable[str]]): Annotation levels enforced via
        validate_variant. Defaults to ("entities", "spans", "deps").
    """

    def __init__(
        self,
        inner: Optional[Callable[["Language", Example], Iterator[Example]]] = None,
        *,
        quota: Optional[int] = 0,
        ratio: float = 0.0,
        variants: int = 1,
        seed: Optional[int] = None,
        shard: int = 0,
        on_failure: str = ON_FAILURE_SKIP,
        align: Optional[Iterable[str]] = None,
    ) -> None:
        quota = int(quota or 0)
        ratio = float(ratio)
        variants = int(variants)
        if quota < 0:
            raise ValueError(
                Errors.E1032.format(var="quota", forbidden="negative", value=quota)
            )
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(
                Errors.E1032.format(
                    var="ratio", forbidden="outside [0.0, 1.0]", value=ratio
                )
            )
        if quota > 0 and ratio > 0.0:
            raise ValueError(Errors.E1058.format(quota=quota, ratio=ratio))
        if variants < 1:
            raise ValueError(
                Errors.E1032.format(
                    var="variants", forbidden="smaller than 1", value=variants
                )
            )
        if on_failure not in ON_FAILURE_VALUES:
            raise ValueError(
                Errors.E1059.format(
                    name="on_failure",
                    value=on_failure,
                    expected=", ".join(ON_FAILURE_VALUES),
                )
            )
        align_fields = tuple(ALIGN_FIELDS if align is None else align)
        invalid = sorted(set(align_fields) - set(ALIGN_FIELDS))
        if invalid:
            raise ValueError(
                Errors.E1059.format(
                    name="align",
                    value=", ".join(invalid),
                    expected=", ".join(ALIGN_FIELDS),
                )
            )
        self.inner = inner if inner is not None else dont_augment
        self.quota = quota
        self.ratio = ratio
        self.variants = variants
        self._seed = seed
        self.shard = int(shard)
        self.on_failure = on_failure
        self.align_fields = align_fields
        self._state: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, Any]] = []
        self.last_epoch: Optional[Dict[str, Any]] = None
        self._implicit_uid = 0

    @property
    def enabled(self) -> bool:
        return self.quota > 0 or self.ratio > 0.0

    def resolve_seed(self, nlp: "Language") -> int:
        """Get the configured base seed, falling back to the nlp training
        seed and then to 0."""
        if self._seed is not None:
            return int(self._seed)
        try:
            value = nlp.config.interpolate()["training"]["seed"]
        except Exception:
            value = None
        return 0 if value is None else int(value)

    def _new_state(self, seed: int, epoch: int, quota: Optional[int]) -> Dict[str, Any]:
        return {
            "seed": seed,
            "epoch": epoch,
            "quota": quota,
            "inputs": 0,
            "used": 0,
            "passed": 0,
            "variants": 0,
            "skipped_unaligned": 0,
            "skipped_empty": 0,
            "failed": 0,
        }

    def begin_epoch(
        self, nlp: "Language", epoch: int = 0, n_inputs: Optional[int] = None
    ) -> Dict[str, Any]:
        """Start a new epoch and resolve the augmentation quota for it."""
        seed = self.resolve_seed(nlp)
        if self.quota > 0:
            quota: Optional[int] = self.quota
        elif n_inputs is not None:
            quota = int(round(self.ratio * n_inputs))
        else:
            # Streaming epoch with an unknown size: no fixed quota, the ratio
            # is applied as a per-input derived draw instead.
            quota = None
        self._state = self._new_state(seed, epoch, quota)
        self._implicit_uid = 0
        return self._state

    def finish_epoch(self) -> Optional[Dict[str, Any]]:
        """Close the current epoch and return/summarize its counts."""
        state = self._state
        if state is None:
            return None
        quota = state["quota"]
        stats = {
            "epoch": state["epoch"],
            "inputs": state["inputs"],
            "quota": quota,
            "augmented": state["used"],
            "quota_remaining": None if quota is None else max(quota - state["used"], 0),
            "passed": state["passed"],
            "variants": state["variants"],
            "skipped_unaligned": state["skipped_unaligned"],
            "skipped_empty": state["skipped_empty"],
            "failed": state["failed"],
        }
        self.history.append(stats)
        self.last_epoch = stats
        self._state = None
        logger.info(
            "Augmentation epoch %d: %d variants from %d/%d inputs, "
            "%d passed unchanged, %d skipped (%d unaligned, %d empty, %d "
            "failed), quota remaining: %s",
            stats["epoch"],
            stats["variants"],
            stats["augmented"],
            stats["inputs"],
            stats["passed"],
            stats["skipped_unaligned"] + stats["skipped_empty"] + stats["failed"],
            stats["skipped_unaligned"],
            stats["skipped_empty"],
            stats["failed"],
            "n/a" if quota is None else stats["quota_remaining"],
        )
        return stats

    def _select_score(self, seed: int, epoch: int, uid: int) -> float:
        rng = random.Random(
            derive_seed(_SEED_NAMESPACE, "select", seed, self.shard, epoch, uid)
        )
        return rng.random()

    def _select_exact(self, seed: int, epoch: int, uids: List[int], quota: int) -> set:
        """Select exactly quota uids deterministically and independently of
        iteration order (smallest derived selection keys)."""
        if quota <= 0:
            return set()
        scored = [(self._select_score(seed, epoch, uid), uid) for uid in uids]
        # Tie-break on the uid so results stay unambiguous on equal keys.
        scored.sort(key=lambda item: (item[0], item[1]))
        return {uid for _, uid in scored[:quota]}

    def _select_streaming(self, uid: int, state: Dict[str, Any]) -> bool:
        quota = state["quota"]
        if quota is not None:
            # Fixed cap on a stream of unknown length: augment inputs until
            # the quota is used up, then pass everything else through.
            return state["used"] < quota
        return self._select_score(state["seed"], state["epoch"], uid) < self.ratio

    def run_epoch(
        self,
        nlp: "Language",
        items: Iterable[Union[Example, Tuple[int, Example]]],
        epoch: int = 0,
        n_inputs: Optional[int] = None,
    ) -> Iterator[Example]:
        """Run one epoch over inputs (Examples or stable (uid, Example)
        pairs), yielding the augmented epoch stream.
        """
        if not self.enabled:
            for item in items:
                yield item[1] if isinstance(item, tuple) else item
            return
        state = self.begin_epoch(nlp, epoch=epoch, n_inputs=n_inputs)
        if n_inputs is not None:
            # Epoch size known: select an exact, order-independent subset.
            pairs = list(_as_uid_example_pairs(items))
            quota = state["quota"] or 0
            selected = self._select_exact(
                state["seed"], state["epoch"], [uid for uid, _ in pairs], quota
            )
            for uid, example in pairs:
                yield from self._augment_one(nlp, uid, example, uid in selected, state)
        else:
            for uid, example in _as_uid_example_pairs(items):
                yield from self._augment_one(
                    nlp, uid, example, self._select_streaming(uid, state), state
                )

    def _augment_one(
        self,
        nlp: "Language",
        uid: int,
        example: Example,
        selected: bool,
        state: Dict[str, Any],
    ) -> Iterator[Example]:
        state["inputs"] += 1
        if not selected:
            state["passed"] += 1
            yield example
            return
        state["used"] += 1
        emitted = 0
        produced_any = False
        for variant_idx in range(self.variants):
            rng = random.Random(
                derive_seed(
                    _SEED_NAMESPACE,
                    "variant",
                    state["seed"],
                    self.shard,
                    state["epoch"],
                    uid,
                    variant_idx,
                )
            )
            token = _augment_rng.set(rng)
            try:
                variants = list(self.inner(nlp, example))
            except Exception as e:
                state["failed"] += 1
                logger.debug(
                    "Augmentation failed for input uid=%s (epoch=%d): %r",
                    uid,
                    state["epoch"],
                    e,
                )
                continue
            finally:
                _augment_rng.reset(token)
            for variant in variants:
                produced_any = True
                if not isinstance(variant, Example):
                    state["failed"] += 1
                    logger.debug(
                        "Augmenter returned non-Example output for uid=%s: %r",
                        uid,
                        type(variant),
                    )
                    continue
                problems = validate_variant(example, variant, self.align_fields)
                if problems:
                    state["skipped_unaligned"] += 1
                    logger.debug(
                        "Skipping unaligned variant for uid=%s (epoch=%d): %s",
                        uid,
                        state["epoch"],
                        ", ".join(problems),
                    )
                    continue
                state["variants"] += 1
                emitted += 1
                yield variant
        if not produced_any:
            state["skipped_empty"] += 1
        if emitted == 0 and self.on_failure == ON_FAILURE_PASS:
            state["passed"] += 1
            yield example

    def __call__(self, nlp: "Language", example: Example) -> Iterator[Example]:
        """Plain ``(nlp, example)`` augmenter protocol for data iterators that
        are not driven per epoch by the training loop. An implicit streaming
        epoch is opened lazily."""
        if not self.enabled:
            yield example
            return
        state = self._state
        if state is None:
            state = self.begin_epoch(nlp, epoch=0, n_inputs=None)
        uid = self._implicit_uid
        self._implicit_uid += 1
        yield from self._augment_one(
            nlp, uid, example, self._select_streaming(uid, state), state
        )


def create_scheduled_augmenter(
    inner: Optional[Callable[["Language", Example], Iterator[Example]]] = None,
    *,
    quota: Optional[int] = 0,
    ratio: float = 0.0,
    variants: int = 1,
    seed: Optional[int] = None,
    shard: int = 0,
    on_failure: str = ON_FAILURE_SKIP,
    align: Optional[List[str]] = None,
) -> AugmentingScheduler:
    """Create a stateful, quota/ratio-based data augmentation scheduler that
    wraps another augmenter. Can be used as the ``augmenter`` of a
    :class:`~spacy.training.Corpus`.

    inner (Optional[Callable]): The augmenter to invoke for selected inputs.
        Defaults to no augmentation (selected inputs are re-emitted as-is).
    quota (int): Maximum number of inputs augmented per epoch. 0 disables
        augmentation unless ratio is set.
    ratio (float): Target fraction of inputs augmented per epoch. Mutually
        exclusive with quota.
    variants (int): Number of independent variant sets drawn per selected
        input.
    seed (Optional[int]): Base seed for derived streams. Defaults to the
        training seed from the nlp config.
    shard (int): Shard id for independent streams across parallel iterators.
    on_failure (str): "skip" or "pass" failed/unalignable inputs.
    align (Optional[List[str]]): Annotation levels to enforce, any of
        "entities", "spans", "deps".
    RETURNS (AugmentingScheduler): The scheduler.
    """
    return AugmentingScheduler(
        inner,
        quota=quota,
        ratio=ratio,
        variants=variants,
        seed=seed,
        shard=shard,
        on_failure=on_failure,
        align=align,
    )
