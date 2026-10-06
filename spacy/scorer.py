import uuid
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np

from .errors import Errors
from .morphology import Morphology
from .tokens import Doc, Span, Token
from .training import Example
from .util import SimpleFrozenList, ensure_path, get_lang_class

if TYPE_CHECKING:
    # This lets us add type hints for mypy etc. without causing circular imports
    from .language import Language  # noqa: F401


DEFAULT_PIPELINE = ("senter", "tagger", "morphologizer", "parser", "ner", "textcat")
MISSING_VALUES = frozenset([None, 0, ""])

STATE_VERSION = 1


class PRFScore:
    """A precision / recall / F score."""

    def __init__(
        self,
        *,
        tp: int = 0,
        fp: int = 0,
        fn: int = 0,
    ) -> None:
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def __len__(self) -> int:
        return self.tp + self.fp + self.fn

    def __iadd__(self, other):
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    def __add__(self, other):
        return PRFScore(
            tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn
        )

    def score_set(self, cand: set, gold: set) -> None:
        self.tp += len(cand.intersection(gold))
        self.fp += len(cand - gold)
        self.fn += len(gold - cand)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-100)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + 1e-100)

    @property
    def fscore(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * ((p * r) / (p + r + 1e-100))

    def to_dict(self) -> Dict[str, float]:
        return {"p": self.precision, "r": self.recall, "f": self.fscore}


class ROCAUCScore:
    """An AUC ROC score. This is only defined for binary classification.
    Use the method is_binary before calculating the score, otherwise it
    may throw an error."""

    def __init__(self) -> None:
        self.golds: List[Any] = []
        self.cands: List[Any] = []
        self.saved_score = 0.0
        self.saved_score_at_len = 0

    def score_set(self, cand, gold) -> None:
        self.cands.append(cand)
        self.golds.append(gold)

    def is_binary(self):
        return len(np.unique(self.golds)) == 2

    @property
    def score(self):
        if not self.is_binary():
            raise ValueError(Errors.E165.format(label=set(self.golds)))
        if len(self.golds) == self.saved_score_at_len:
            return self.saved_score
        self.saved_score = _roc_auc_score(self.golds, self.cands)
        self.saved_score_at_len = len(self.golds)
        return self.saved_score


# ###########################################################################
# State helpers: the mergeable intermediate state is built from minimum-size
# integer counts (tp/fp/fn) and, for ROC AUC scores which cannot be recovered
# from counts, the per-example (gold, cand) point pairs.
# ###########################################################################


def _prf_counts(prf: PRFScore) -> List[int]:
    return [int(prf.tp), int(prf.fp), int(prf.fn)]


def _prf_from_counts(counts: List[int]) -> PRFScore:
    return PRFScore(tp=int(counts[0]), fp=int(counts[1]), fn=int(counts[2]))


def _add_counts(counts1: List[int], counts2: List[int]) -> List[int]:
    return [a + b for a, b in zip(counts1, counts2)]


def _merge_per_type_counts(d1: Dict, d2: Dict) -> Dict:
    result = {k: list(v) for k, v in d1.items()}
    for k, counts in d2.items():
        if k in result:
            result[k] = _add_counts(result[k], counts)
        else:
            result[k] = list(counts)
    return result


def _auc_points(auc: ROCAUCScore) -> List[List[float]]:
    return [[float(gold), float(cand)] for gold, cand in zip(auc.golds, auc.cands)]


def _auc_from_points(points: Iterable[Iterable[float]]) -> ROCAUCScore:
    auc = ROCAUCScore()
    for gold, cand in points:
        auc.score_set(cand, gold)
    return auc


def _merge_auc_per_type(d1: Dict, d2: Dict) -> Dict:
    result = {k: {"points": list(v["points"])} for k, v in d1.items()}
    for k, value in d2.items():
        if k in result:
            result[k]["points"] = result[k]["points"] + list(value["points"])
        else:
            result[k] = {"points": list(value["points"])}
    return result


def _make_part(part_type: str, cfg: Dict[str, Any], scores: Dict[str, Any]):
    return {"type": part_type, "cfg": cfg, "scores": scores}


def scorer_with_state(fn):
    """Mark a scorer callable as supporting the keyword-only '_state' argument,
    i.e. it returns a component state dict instead of final scores."""
    fn.supports_scorer_state = True  # type: ignore[attr-defined]
    return fn


def _scorer_supports_state(component) -> bool:
    scorer = getattr(component, "scorer", None)
    return bool(getattr(scorer, "supports_scorer_state", False))


def _finalize_component_state(component_state: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for part in component_state["parts"]:
        result.update(FINALIZERS[part["type"]](part))
    return result


def _sorted_values(values: Iterable[Any]) -> List[Any]:
    """Sort a set of potentially mixed-type config values deterministically."""
    return sorted(values, key=repr)


def _merge_part_scores(part_type: str, s1: Dict, s2: Dict) -> Dict:
    if part_type == "tokenization":
        return {
            "acc": _add_counts(s1["acc"], s2["acc"]),
            "prf": _add_counts(s1["prf"], s2["prf"]),
        }
    if part_type == "token_attr":
        return {"score": _add_counts(s1["score"], s2["score"])}
    if part_type == "token_per_feat":
        return {
            "micro": _add_counts(s1["micro"], s2["micro"]),
            "per_feat": _merge_per_type_counts(s1["per_feat"], s2["per_feat"]),
        }
    if part_type == "spans":
        result = {"score": _add_counts(s1["score"], s2["score"])}
        if "per_type" in s1 and "per_type" in s2:
            result["per_type"] = _merge_per_type_counts(
                s1["per_type"], s2["per_type"]
            )
        return result
    if part_type == "cats":
        return {
            "micro": _add_counts(s1["micro"], s2["micro"]),
            "f_per_type": _merge_per_type_counts(s1["f_per_type"], s2["f_per_type"]),
            "auc_per_type": _merge_auc_per_type(
                s1["auc_per_type"], s2["auc_per_type"]
            ),
        }
    if part_type == "links":
        return {
            "micro": _add_counts(s1["micro"], s2["micro"]),
            "f_per_type": _merge_per_type_counts(s1["f_per_type"], s2["f_per_type"]),
        }
    if part_type == "deps":
        return {
            "unlabelled": _add_counts(s1["unlabelled"], s2["unlabelled"]),
            "labelled": _add_counts(s1["labelled"], s2["labelled"]),
            "per_dep": _merge_per_type_counts(s1["per_dep"], s2["per_dep"]),
        }
    if part_type == "ner":
        return {
            "score": _add_counts(s1["score"], s2["score"]),
            "per_type": _merge_per_type_counts(s1["per_type"], s2["per_type"]),
        }
    raise ValueError(Errors.E1063.format(msg=f"unknown part type '{part_type}'"))


# ###########################################################################
# Part finalizers: restore the score objects from a part and produce the same
# final score dictionaries as the original scoring methods. `canonical=True`
# (i.e. the part results from a merge) uses sorted key orders so that merged
# results are independent of the merge order. Non-canonical parts keep the
# original insertion order, so scores without sharding are bit-identical.
# ###########################################################################


def _sorted_per_type(per_type: Dict[str, PRFScore]) -> Dict[str, Dict[str, float]]:
    return {k: per_type[k].to_dict() for k in sorted(per_type)}


def _finalize_tokenization(part, *, canonical: bool = False):
    acc = _prf_from_counts(part["scores"]["acc"])
    if len(acc) == 0:
        return {
            "token_acc": None,
            "token_p": None,
            "token_r": None,
            "token_f": None,
        }
    prf = _prf_from_counts(part["scores"]["prf"])
    return {
        "token_acc": acc.precision,
        "token_p": prf.precision,
        "token_r": prf.recall,
        "token_f": prf.fscore,
    }


def _finalize_token_attr(part, *, canonical: bool = False):
    attr = part["cfg"]["attr"]
    prf = _prf_from_counts(part["scores"]["score"])
    if len(prf) == 0:
        return {f"{attr}_acc": None}
    return {f"{attr}_acc": prf.fscore}


def _finalize_token_per_feat(part, *, canonical: bool = False):
    attr = part["cfg"]["attr"]
    micro = _prf_from_counts(part["scores"]["micro"])
    result: Dict[str, Any] = {}
    if len(micro) == 0:
        result[f"{attr}_micro_p"] = None
        result[f"{attr}_micro_r"] = None
        result[f"{attr}_micro_f"] = None
        result[f"{attr}_per_feat"] = None
        return result
    result[f"{attr}_micro_p"] = micro.precision
    result[f"{attr}_micro_r"] = micro.recall
    result[f"{attr}_micro_f"] = micro.fscore
    per_feat = {k: _prf_from_counts(c) for k, c in part["scores"]["per_feat"].items()}
    if canonical:
        result[f"{attr}_per_feat"] = _sorted_per_type(per_feat)
    else:
        result[f"{attr}_per_feat"] = {k: v.to_dict() for k, v in per_feat.items()}
    return result


def _finalize_spans(part, *, canonical: bool = False):
    attr = part["cfg"]["attr"]
    labeled = part["cfg"]["labeled"]
    final_scores: Dict[str, Any] = {
        f"{attr}_p": None,
        f"{attr}_r": None,
        f"{attr}_f": None,
    }
    if labeled:
        final_scores[f"{attr}_per_type"] = None
    score = _prf_from_counts(part["scores"]["score"])
    if len(score) > 0:
        final_scores[f"{attr}_p"] = score.precision
        final_scores[f"{attr}_r"] = score.recall
        final_scores[f"{attr}_f"] = score.fscore
        if labeled:
            per_type = {
                k: _prf_from_counts(c) for k, c in part["scores"]["per_type"].items()
            }
            if canonical:
                final_scores[f"{attr}_per_type"] = _sorted_per_type(per_type)
            else:
                final_scores[f"{attr}_per_type"] = {
                    k: v.to_dict() for k, v in per_type.items()
                }
    return final_scores


def _finalize_cats(part, *, canonical: bool = False):
    cfg = part["cfg"]
    attr = cfg["attr"]
    multi_label = cfg["multi_label"]
    positive_label = cfg["positive_label"]
    f_per_type = {
        k: _prf_from_counts(c) for k, c in part["scores"]["f_per_type"].items()
    }
    micro_prf = _prf_from_counts(part["scores"]["micro"])
    auc_per_type = {
        k: _auc_from_points(v["points"])
        for k, v in part["scores"]["auc_per_type"].items()
    }
    n_cats = len(f_per_type) + 1e-100
    macro_p = sum(prf.precision for prf in f_per_type.values()) / n_cats
    macro_r = sum(prf.recall for prf in f_per_type.values()) / n_cats
    macro_f = sum(prf.fscore for prf in f_per_type.values()) / n_cats
    # Limit macro_auc to those labels with gold annotations,
    # but still divide by all cats to avoid artificial boosting of datasets with missing labels
    macro_auc = (
        sum(auc.score if auc.is_binary() else 0.0 for auc in auc_per_type.values())
        / n_cats
    )
    results: Dict[str, Any] = {
        f"{attr}_score": None,
        f"{attr}_score_desc": None,
        f"{attr}_micro_p": micro_prf.precision,
        f"{attr}_micro_r": micro_prf.recall,
        f"{attr}_micro_f": micro_prf.fscore,
        f"{attr}_macro_p": macro_p,
        f"{attr}_macro_r": macro_r,
        f"{attr}_macro_f": macro_f,
        f"{attr}_macro_auc": macro_auc,
        f"{attr}_f_per_type": {k: v.to_dict() for k, v in f_per_type.items()},
        f"{attr}_auc_per_type": {
            k: v.score if v.is_binary() else None for k, v in auc_per_type.items()
        },
    }
    if len(cfg["labels"]) == 2 and not multi_label and positive_label:
        positive_label_f = f_per_type[positive_label].fscore
        results[f"{attr}_score"] = positive_label_f
        results[f"{attr}_score_desc"] = f"F ({positive_label})"
    elif not multi_label:
        results[f"{attr}_score"] = results[f"{attr}_macro_f"]
        results[f"{attr}_score_desc"] = "macro F"
    else:
        results[f"{attr}_score"] = results[f"{attr}_macro_auc"]
        results[f"{attr}_score_desc"] = "macro AUC"
    return results


def _finalize_links(part, *, canonical: bool = False):
    f_per_type = {
        k: _prf_from_counts(c) for k, c in part["scores"]["f_per_type"].items()
    }
    micro_prf = _prf_from_counts(part["scores"]["micro"])
    label_order = sorted(f_per_type) if canonical else list(f_per_type)
    n_labels = len(f_per_type) + 1e-100
    macro_p = sum(f_per_type[k].precision for k in label_order) / n_labels
    macro_r = sum(f_per_type[k].recall for k in label_order) / n_labels
    macro_f = sum(f_per_type[k].fscore for k in label_order) / n_labels
    if canonical:
        per_type = _sorted_per_type(f_per_type)
    else:
        per_type = {k: v.to_dict() for k, v in f_per_type.items()}
    results = {
        "nel_score": micro_prf.fscore,
        "nel_score_desc": "micro F",
        "nel_micro_p": micro_prf.precision,
        "nel_micro_r": micro_prf.recall,
        "nel_micro_f": micro_prf.fscore,
        "nel_macro_p": macro_p,
        "nel_macro_r": macro_r,
        "nel_macro_f": macro_f,
        "nel_f_per_type": per_type,
    }
    return results


def _finalize_deps(part, *, canonical: bool = False):
    attr = part["cfg"]["attr"]
    unlabelled = _prf_from_counts(part["scores"]["unlabelled"])
    if len(unlabelled) == 0:
        return {
            f"{attr}_uas": None,
            f"{attr}_las": None,
            f"{attr}_las_per_type": None,
        }
    labelled = _prf_from_counts(part["scores"]["labelled"])
    per_dep = {k: _prf_from_counts(c) for k, c in part["scores"]["per_dep"].items()}
    if canonical:
        per_type = _sorted_per_type(per_dep)
    else:
        per_type = {k: v.to_dict() for k, v in per_dep.items()}
    return {
        f"{attr}_uas": unlabelled.fscore,
        f"{attr}_las": labelled.fscore,
        f"{attr}_las_per_type": per_type,
    }


def _finalize_ner(part, *, canonical: bool = False):
    score = _prf_from_counts(part["scores"]["score"])
    if len(score) == 0:
        return {
            "ents_p": None,
            "ents_r": None,
            "ents_f": None,
            "ents_per_type": None,
        }
    per_type = {k: _prf_from_counts(c) for k, c in part["scores"]["per_type"].items()}
    if canonical:
        per_type_dict = _sorted_per_type(per_type)
    else:
        per_type_dict = {k: v.to_dict() for k, v in per_type.items()}
    return {
        "ents_p": score.precision,
        "ents_r": score.recall,
        "ents_f": score.fscore,
        "ents_per_type": per_type_dict,
    }


FINALIZERS = {
    "tokenization": _finalize_tokenization,
    "token_attr": _finalize_token_attr,
    "token_per_feat": _finalize_token_per_feat,
    "spans": _finalize_spans,
    "cats": _finalize_cats,
    "links": _finalize_links,
    "deps": _finalize_deps,
    "ner": _finalize_ner,
}


class Scorer:
    """Compute evaluation scores."""

    def __init__(
        self,
        nlp: Optional["Language"] = None,
        default_lang: str = "xx",
        default_pipeline: Iterable[str] = DEFAULT_PIPELINE,
        **cfg,
    ) -> None:
        """Initialize the Scorer.

        DOCS: https://spacy.io/api/scorer#init
        """
        self.cfg = cfg
        if nlp:
            self.nlp = nlp
        else:
            nlp = get_lang_class(default_lang)()
            for pipe in default_pipeline:
                nlp.add_pipe(pipe)
            self.nlp = nlp

    def score(
        self, examples: Iterable[Example], *, per_component: bool = False
    ) -> Dict[str, Any]:
        """Evaluate a list of Examples.

        examples (Iterable[Example]): The predicted annotations + correct annotations.
        per_component (bool): Whether to return the scores keyed by component
            name. Defaults to False.
        RETURNS (Dict): A dictionary of scores.

        DOCS: https://spacy.io/api/scorer#score
        """
        scores: Dict[str, Any] = {}
        tokenizer = self.nlp.tokenizer
        if hasattr(tokenizer, "score"):
            if hasattr(tokenizer, "score_state"):
                token_scores = _finalize_component_state(
                    tokenizer.score_state(examples, **self.cfg)
                )
            else:
                token_scores = tokenizer.score(examples, **self.cfg)
            if per_component:
                scores["tokenizer"] = token_scores
            else:
                scores.update(token_scores)
        for name, component in self.nlp.pipeline:
            if hasattr(component, "score"):
                # built-in scorers traverse the same state pathway used for
                # sharded evaluation; custom scorers without '_state' support
                # are called directly, so results stay identical to before
                if _scorer_supports_state(component):
                    component_scores = _finalize_component_state(
                        component.score_state(examples, **self.cfg)
                    )
                else:
                    component_scores = component.score(examples, **self.cfg)
                if per_component:
                    scores[name] = component_scores
                else:
                    scores.update(component_scores)
        return scores

    def accumulate(self, examples: Iterable[Example]) -> "ScorerState":
        """Evaluate a list of Examples and return the mergeable intermediate
        evaluation state instead of the final scores.

        examples (Iterable[Example]): The predicted annotations + correct annotations.
        RETURNS (ScorerState): The evaluation state, which can be saved to disk,
            transmitted and merged with states for other subsets.
        """
        states: Dict[str, Any] = {}
        if hasattr(self.nlp.tokenizer, "score"):
            states["tokenizer"] = self.nlp.tokenizer.score_state(examples, **self.cfg)
        for name, component in self.nlp.pipeline:
            if hasattr(component, "score"):
                states[name] = component.score_state(examples, **self.cfg)
        return ScorerState(
            states=states, pipeline=list(self.nlp.pipe_names), cfg=self.cfg
        )

    @staticmethod
    def score_tokenization(
        examples: Iterable[Example], *, _state: bool = False, **cfg
    ) -> Dict[str, Any]:
        """Returns accuracy and PRF scores for tokenization.
        * token_acc: # correct tokens / # gold tokens
        * token_p/r/f: PRF for token character spans

        examples (Iterable[Example]): Examples to score
        RETURNS (Dict[str, Any]): A dictionary containing the scores
        token_acc/p/r/f.

        DOCS: https://spacy.io/api/scorer#score_tokenization
        """
        part = Scorer._tokenization_part(examples)
        if _state:
            return {"parts": [part]}
        return _finalize_tokenization(part)

    @staticmethod
    def _tokenization_part(examples: Iterable[Example]):
        acc_score = PRFScore()
        prf_score = PRFScore()
        for example in examples:
            gold_doc = example.reference
            pred_doc = example.predicted
            if gold_doc.has_unknown_spaces:
                continue
            align = example.alignment
            gold_spans = set()
            pred_spans = set()
            for token in gold_doc:
                if token.orth_.isspace():
                    continue
                gold_spans.add((token.idx, token.idx + len(token)))
            for token in pred_doc:
                if token.orth_.isspace():
                    continue
                pred_spans.add((token.idx, token.idx + len(token)))
                if align.x2y.lengths[token.i] != 1:
                    acc_score.fp += 1
                else:
                    acc_score.tp += 1
            prf_score.score_set(pred_spans, gold_spans)
        return _make_part(
            "tokenization",
            {},
            {"acc": _prf_counts(acc_score), "prf": _prf_counts(prf_score)},
        )

    @staticmethod
    def score_token_attr(
        examples: Iterable[Example],
        attr: str,
        *,
        getter: Callable[[Token, str], Any] = getattr,
        missing_values: Set[Any] = MISSING_VALUES,  # type: ignore[assignment]
        _state: bool = False,
        **cfg,
    ) -> Dict[str, Any]:
        """Returns an accuracy score for a token-level attribute.

        examples (Iterable[Example]): Examples to score
        attr (str): The attribute to score.
        getter (Callable[[Token, str], Any]): Defaults to getattr. If provided,
            getter(token, attr) should return the value of the attribute for an
            individual token.
        missing_values (Set[Any]): Attribute values to treat as missing annotation
            in the reference annotation.
        RETURNS (Dict[str, Any]): A dictionary containing the accuracy score
        under the key attr_acc.

        DOCS: https://spacy.io/api/scorer#score_token_attr
        """
        part = Scorer._token_attr_part(examples, attr, getter, missing_values)
        if _state:
            return {"parts": [part]}
        return _finalize_token_attr(part)

    @staticmethod
    def _token_attr_part(examples, attr, getter, missing_values) -> Dict[str, Any]:
        tag_score = PRFScore()
        for example in examples:
            gold_doc = example.reference
            pred_doc = example.predicted
            align = example.alignment
            gold_tags = set()
            missing_indices = set()
            for gold_i, token in enumerate(gold_doc):
                value = getter(token, attr)
                if value not in missing_values:
                    gold_tags.add((gold_i, getter(token, attr)))
                else:
                    missing_indices.add(gold_i)
            pred_tags = set()
            for token in pred_doc:
                if token.orth_.isspace():
                    continue
                if align.x2y.lengths[token.i] == 1:
                    gold_i = align.x2y[token.i][0]
                    if gold_i not in missing_indices:
                        pred_tags.add((gold_i, getter(token, attr)))
            tag_score.score_set(pred_tags, gold_tags)
        part_cfg = {
            "attr": attr,
            "missing_values": _sorted_values(missing_values),
        }
        return _make_part("token_attr", part_cfg, {"score": _prf_counts(tag_score)})

    @staticmethod
    def score_token_attr_per_feat(
        examples: Iterable[Example],
        attr: str,
        *,
        getter: Callable[[Token, str], Any] = getattr,
        missing_values: Set[Any] = MISSING_VALUES,  # type: ignore[assignment]
        _state: bool = False,
        **cfg,
    ) -> Dict[str, Any]:
        """Return micro PRF and PRF scores per feat for a token attribute in
        UFEATS format.

        examples (Iterable[Example]): Examples to score
        attr (str): The attribute to score.
        getter (Callable[[Token, str], Any]): Defaults to getattr. If provided,
            getter(token, attr) should return the value of the attribute for an
            individual token.
        missing_values (Set[Any]): Attribute values to treat as missing
            annotation in the reference annotation.
        RETURNS (dict): A dictionary containing the micro PRF scores under the
        key attr_micro_p/r/f and the per-feat PRF scores under
        attr_per_feat.
        """
        part = Scorer._token_attr_per_feat_part(
            examples, attr, getter, missing_values
        )
        if _state:
            return {"parts": [part]}
        return _finalize_token_per_feat(part)

    @staticmethod
    def _token_attr_per_feat_part(
        examples, attr, getter, missing_values
    ) -> Dict[str, Any]:
        micro_score = PRFScore()
        per_feat = {}
        for example in examples:
            pred_doc = example.predicted
            gold_doc = example.reference
            align = example.alignment
            gold_per_feat: Dict[str, Set] = {}
            missing_indices = set()
            for gold_i, token in enumerate(gold_doc):
                value = getter(token, attr)
                morph = gold_doc.vocab.strings[value]
                if value not in missing_values and morph != Morphology.EMPTY_MORPH:
                    for feat in morph.split(Morphology.FEATURE_SEP):
                        field, values = feat.split(Morphology.FIELD_SEP)
                        if field not in per_feat:
                            per_feat[field] = PRFScore()
                        if field not in gold_per_feat:
                            gold_per_feat[field] = set()
                        gold_per_feat[field].add((gold_i, feat))
                else:
                    missing_indices.add(gold_i)
            pred_per_feat: Dict[str, Set] = {}
            for token in pred_doc:
                if token.orth_.isspace():
                    continue
                if align.x2y.lengths[token.i] == 1:
                    gold_i = align.x2y[token.i][0]
                    if gold_i not in missing_indices:
                        value = getter(token, attr)
                        morph = gold_doc.vocab.strings[value]
                        if (
                            value not in missing_values
                            and morph != Morphology.EMPTY_MORPH
                        ):
                            for feat in morph.split(Morphology.FEATURE_SEP):
                                field, values = feat.split(Morphology.FIELD_SEP)
                                if field not in per_feat:
                                    per_feat[field] = PRFScore()
                                if field not in pred_per_feat:
                                    pred_per_feat[field] = set()
                                pred_per_feat[field].add((gold_i, feat))
            for field in per_feat:
                micro_score.score_set(
                    pred_per_feat.get(field, set()), gold_per_feat.get(field, set())
                )
                per_feat[field].score_set(
                    pred_per_feat.get(field, set()), gold_per_feat.get(field, set())
                )
        part_cfg = {
            "attr": attr,
            "missing_values": _sorted_values(missing_values),
        }
        return _make_part(
            "token_per_feat",
            part_cfg,
            {
                "micro": _prf_counts(micro_score),
                "per_feat": {k: _prf_counts(v) for k, v in per_feat.items()},
            },
        )

    @staticmethod
    def score_spans(
        examples: Iterable[Example],
        attr: str,
        *,
        getter: Callable[[Doc, str], Iterable[Span]] = getattr,
        has_annotation: Optional[Callable[[Doc], bool]] = None,
        labeled: bool = True,
        allow_overlap: bool = False,
        _state: bool = False,
        **cfg,
    ) -> Dict[str, Any]:
        """Returns PRF scores for labeled spans.

        examples (Iterable[Example]): Examples to score
        attr (str): The attribute to score.
        getter (Callable[[Doc, str], Iterable[Span]]): Defaults to getattr. If
            provided, getter(doc, attr) should return the spans for the
            individual doc.
        has_annotation (Optional[Callable[[Doc], bool]]) should return whether a `Doc`
            has annotation for this `attr`. Docs without annotation are skipped for
            scoring purposes.
        labeled (bool): Whether or not to include label information in
            the evaluation. If set to 'False', two spans will be considered
            equal if their start and end match, irrespective of their label.
        allow_overlap (bool): Whether or not to allow overlapping spans.
            If set to 'False', the alignment will automatically resolve conflicts.
        RETURNS (Dict[str, Any]): A dictionary containing the PRF scores under
            the keys attr_p/r/f and the per-type PRF scores under attr_per_type.

        DOCS: https://spacy.io/api/scorer#score_spans
        """
        part = Scorer._spans_part(
            examples, attr, getter, has_annotation, labeled, allow_overlap
        )
        if _state:
            return {"parts": [part]}
        return _finalize_spans(part)

    @staticmethod
    def _spans_part(
        examples, attr, getter, has_annotation, labeled, allow_overlap
    ) -> Dict[str, Any]:
        score = PRFScore()
        score_per_type = dict()
        for example in examples:
            pred_doc = example.predicted
            gold_doc = example.reference
            # Option to handle docs without annotation for this attribute
            if has_annotation is not None and not has_annotation(gold_doc):
                continue
            # Find all labels in gold
            labels = set([k.label_ for k in getter(gold_doc, attr)])
            # If labeled, find all labels in pred
            if has_annotation is None or (
                has_annotation is not None and has_annotation(pred_doc)
            ):
                labels |= set([k.label_ for k in getter(pred_doc, attr)])
            # Set up all labels for per type scoring and prepare gold per type
            gold_per_type: Dict[str, Set] = {label: set() for label in labels}
            for label in labels:
                if label not in score_per_type:
                    score_per_type[label] = PRFScore()
            # Find all predidate labels, for all and per type
            gold_spans = set()
            pred_spans = set()
            for span in getter(gold_doc, attr):
                gold_span: Tuple
                if labeled:
                    gold_span = (span.label_, span.start, span.end - 1)
                else:
                    gold_span = (span.start, span.end - 1)
                gold_spans.add(gold_span)
                gold_per_type[span.label_].add(gold_span)
            pred_per_type: Dict[str, Set] = {label: set() for label in labels}
            if has_annotation is None or (
                has_annotation is not None and has_annotation(pred_doc)
            ):
                for span in example.get_aligned_spans_x2y(
                    getter(pred_doc, attr), allow_overlap
                ):
                    pred_span: Tuple
                    if labeled:
                        pred_span = (span.label_, span.start, span.end - 1)
                    else:
                        pred_span = (span.start, span.end - 1)
                    pred_spans.add(pred_span)
                    pred_per_type[span.label_].add(pred_span)
            # Scores per label
            if labeled:
                for k, v in score_per_type.items():
                    if k in pred_per_type:
                        v.score_set(pred_per_type[k], gold_per_type[k])
            # Score for all labels
            score.score_set(pred_spans, gold_spans)
        part_cfg = {
            "attr": attr,
            "labeled": labeled,
            "allow_overlap": allow_overlap,
        }
        part_scores: Dict[str, Any] = {"score": _prf_counts(score)}
        if labeled:
            part_scores["per_type"] = {
                k: _prf_counts(v) for k, v in score_per_type.items()
            }
        return _make_part("spans", part_cfg, part_scores)

    @staticmethod
    def score_cats(
        examples: Iterable[Example],
        attr: str,
        *,
        getter: Callable[[Doc, str], Any] = getattr,
        labels: Iterable[str] = SimpleFrozenList(),
        multi_label: bool = True,
        positive_label: Optional[str] = None,
        threshold: Optional[float] = None,
        _state: bool = False,
        **cfg,
    ) -> Dict[str, Any]:
        """Returns PRF and ROC AUC scores for a doc-level attribute with a
        dict with scores for each label like Doc.cats. The reported overall
        score depends on scorer settings.

        examples (Iterable[Example]): Examples to score
        attr (str): The attribute to score.
        getter (Callable[[Doc, str], Any]): Defaults to getattr. If provided,
            getter(doc, attr) should return the values for the individual doc.
        labels (Iterable[str]): The set of possible labels. Defaults to [].
        multi_label (bool): Whether the attribute allows multiple labels.
            Defaults to True. When set to False (exclusive labels), missing
            gold labels are interpreted as 0.0 and the threshold is set to 0.0.
        positive_label (str): The positive label for a binary task with
            exclusive classes. Defaults to None.
        threshold (float): Cutoff to consider a prediction "positive". Defaults
            to 0.5 for multi-label, and 0.0 (i.e. whatever's highest scoring)
            otherwise.
        RETURNS (Dict[str, Any]): A dictionary containing the scores, with
            inapplicable scores as None:
            for all:
                attr_score (one of attr_micro_f / attr_macro_f / attr_macro_auc),
                attr_score_desc (text description of the overall score),
                attr_micro_p,
                attr_micro_r,
                attr_micro_f,
                attr_macro_p,
                attr_macro_r,
                attr_macro_f,
                attr_macro_auc,
                attr_f_per_type,
                attr_auc_per_type

        DOCS: https://spacy.io/api/scorer#score_cats
        """
        part = Scorer._cats_part(
            examples,
            attr,
            getter,
            labels,
            multi_label,
            positive_label,
            threshold,
        )
        if _state:
            return {"parts": [part]}
        return _finalize_cats(part)

    @staticmethod
    def _cats_part(
        examples,
        attr,
        getter,
        labels,
        multi_label,
        positive_label,
        threshold,
    ) -> Dict[str, Any]:
        labels_list = list(labels)
        if threshold is None:
            threshold = 0.5 if multi_label else 0.0
        if not multi_label:
            threshold = 0.0
        f_per_type = {label: PRFScore() for label in labels_list}
        auc_per_type = {label: ROCAUCScore() for label in labels_list}
        labels_set = set(labels_list)
        for example in examples:
            # Through this loop, None in the gold_cats indicates missing label.
            pred_cats = getter(example.predicted, attr)
            pred_cats = {k: v for k, v in pred_cats.items() if k in labels_set}
            gold_cats = getter(example.reference, attr)
            gold_cats = {k: v for k, v in gold_cats.items() if k in labels_set}

            for label in labels_set:
                pred_score = pred_cats.get(label, 0.0)
                gold_score = gold_cats.get(label)
                if not gold_score and not multi_label:
                    gold_score = 0.0
                if gold_score is not None:
                    auc_per_type[label].score_set(pred_score, gold_score)
            if multi_label:
                for label in labels_set:
                    pred_score = pred_cats.get(label, 0.0)
                    gold_score = gold_cats.get(label)
                    if gold_score is not None:
                        if pred_score >= threshold and gold_score > 0:
                            f_per_type[label].tp += 1
                        elif pred_score >= threshold and gold_score == 0:
                            f_per_type[label].fp += 1
                        elif pred_score < threshold and gold_score > 0:
                            f_per_type[label].fn += 1
            elif pred_cats and gold_cats:
                # Get the highest-scoring for each.
                pred_label, pred_score = max(pred_cats.items(), key=lambda it: it[1])
                gold_label, gold_score = max(gold_cats.items(), key=lambda it: it[1])
                if pred_label == gold_label:
                    f_per_type[pred_label].tp += 1
                else:
                    f_per_type[gold_label].fn += 1
                    f_per_type[pred_label].fp += 1
            elif gold_cats:
                gold_label, gold_score = max(gold_cats, key=lambda it: it[1])
                if gold_score > 0:
                    f_per_type[gold_label].fn += 1
            elif pred_cats:
                pred_label, pred_score = max(pred_cats.items(), key=lambda it: it[1])
                f_per_type[pred_label].fp += 1
        micro_prf = PRFScore()
        for label_prf in f_per_type.values():
            micro_prf.tp += label_prf.tp
            micro_prf.fn += label_prf.fn
            micro_prf.fp += label_prf.fp
        part_cfg = {
            "attr": attr,
            "labels": labels_list,
            "multi_label": multi_label,
            "positive_label": positive_label,
            "threshold": threshold,
        }
        part_scores = {
            "micro": _prf_counts(micro_prf),
            "f_per_type": {k: _prf_counts(v) for k, v in f_per_type.items()},
            "auc_per_type": {k: {"points": _auc_points(v)} for k, v in auc_per_type.items()},
        }
        return _make_part("cats", part_cfg, part_scores)

    @staticmethod
    def score_links(
        examples: Iterable[Example],
        *,
        negative_labels: Iterable[str],
        _state: bool = False,
        **cfg,
    ) -> Dict[str, Any]:
        """Returns PRF for predicted links on the entity level.
        To disentangle the performance of the NEL from the NER,
        this method only evaluates NEL links for entities that overlap
        between the gold reference and the predictions.

        examples (Iterable[Example]): Examples to score
        negative_labels (Iterable[str]): The string values that refer to no annotation (e.g. "NIL")
        RETURNS (Dict[str, Any]): A dictionary containing the scores.

        DOCS: https://spacy.io/api/scorer#score_links
        """
        part = Scorer._links_part(examples, negative_labels)
        if _state:
            return {"parts": [part]}
        return _finalize_links(part)

    @staticmethod
    def _links_part(examples, negative_labels) -> Dict[str, Any]:
        f_per_type = {}
        for example in examples:
            gold_ent_by_offset = {}
            for gold_ent in example.reference.ents:
                gold_ent_by_offset[(gold_ent.start_char, gold_ent.end_char)] = gold_ent

            for pred_ent in example.predicted.ents:
                gold_span = gold_ent_by_offset.get(
                    (pred_ent.start_char, pred_ent.end_char), None
                )
                if gold_span is not None:
                    label = gold_span.label_
                    if label not in f_per_type:
                        f_per_type[label] = PRFScore()
                    gold = gold_span.kb_id_
                    # only evaluating entities that overlap between gold and pred,
                    # to disentangle the performance of the NEL from the NER
                    if gold is not None:
                        pred = pred_ent.kb_id_
                        if gold in negative_labels and pred in negative_labels:
                            # ignore true negatives
                            pass
                        elif gold == pred:
                            f_per_type[label].tp += 1
                        elif gold in negative_labels:
                            f_per_type[label].fp += 1
                        elif pred in negative_labels:
                            f_per_type[label].fn += 1
                        else:
                            # a wrong prediction (e.g. Q42 != Q3) counts as both a FP as well as a FN
                            f_per_type[label].fp += 1
                            f_per_type[label].fn += 1
        micro_prf = PRFScore()
        for label_prf in f_per_type.values():
            micro_prf.tp += label_prf.tp
            micro_prf.fn += label_prf.fn
            micro_prf.fp += label_prf.fp
        part_cfg = {"negative_labels": sorted(set(negative_labels))}
        part_scores = {
            "micro": _prf_counts(micro_prf),
            "f_per_type": {k: _prf_counts(v) for k, v in f_per_type.items()},
        }
        return _make_part("links", part_cfg, part_scores)

    @staticmethod
    def score_deps(
        examples: Iterable[Example],
        attr: str,
        *,
        getter: Callable[[Token, str], Any] = getattr,
        head_attr: str = "head",
        head_getter: Callable[[Token, str], Token] = getattr,
        ignore_labels: Iterable[str] = SimpleFrozenList(),
        missing_values: Set[Any] = MISSING_VALUES,  # type: ignore[assignment]
        _state: bool = False,
        **cfg,
    ) -> Dict[str, Any]:
        """Returns the UAS, LAS, and LAS per type scores for dependency
        parses.

        examples (Iterable[Example]): Examples to score
        attr (str): The attribute containing the dependency label.
        getter (Callable[[Token, str], Any]): Defaults to getattr. If provided,
            getter(token, attr) should return the value of the attribute for an
            individual token.
        head_attr (str): The attribute containing the head token. Defaults to
            'head'.
        head_getter (Callable[[Token, str], Token]): Defaults to getattr. If
            provided, head_getter(token, attr) should return the value of the head
            for an individual token.
        ignore_labels (Tuple): Labels to ignore while scoring (e.g., punct).
        missing_values (Set[Any]): Attribute values to treat as missing annotation
            in the reference annotation.
        RETURNS (Dict[str, Any]): A dictionary containing the scores:
            attr_uas, attr_las, and attr_las_per_type.

        DOCS: https://spacy.io/api/scorer#score_deps
        """
        part = Scorer._deps_part(
            examples,
            attr,
            getter,
            head_attr,
            head_getter,
            ignore_labels,
            missing_values,
        )
        if _state:
            return {"parts": [part]}
        return _finalize_deps(part)

    @staticmethod
    def _deps_part(
        examples,
        attr,
        getter,
        head_attr,
        head_getter,
        ignore_labels,
        missing_values,
    ) -> Dict[str, Any]:
        unlabelled = PRFScore()
        labelled = PRFScore()
        labelled_per_dep = dict()
        missing_indices = set()
        for example in examples:
            gold_doc = example.reference
            pred_doc = example.predicted
            align = example.alignment
            gold_deps = set()
            gold_deps_per_dep: Dict[str, Set] = {}
            for gold_i, token in enumerate(gold_doc):
                dep = getter(token, attr)
                head = head_getter(token, head_attr)
                if dep not in missing_values:
                    if dep not in ignore_labels:
                        gold_deps.add((gold_i, head.i, dep))
                        if dep not in labelled_per_dep:
                            labelled_per_dep[dep] = PRFScore()
                        if dep not in gold_deps_per_dep:
                            gold_deps_per_dep[dep] = set()
                        gold_deps_per_dep[dep].add((gold_i, head.i, dep))
                else:
                    missing_indices.add(gold_i)
            pred_deps = set()
            pred_deps_per_dep: Dict[str, Set] = {}
            for token in pred_doc:
                if token.orth_.isspace():
                    continue
                if align.x2y.lengths[token.i] != 1:
                    gold_i = None  # type: ignore
                else:
                    gold_i = align.x2y[token.i][0]
                if gold_i not in missing_indices:
                    dep = getter(token, attr)
                    head = head_getter(token, head_attr)
                    if dep not in ignore_labels and token.orth_.strip():
                        if align.x2y.lengths[head.i] == 1:
                            gold_head = align.x2y[head.i][0]
                        else:
                            gold_head = None
                        # None is indistinct, so we can't just add it to the set
                        # Multiple (None, None) deps are possible
                        if gold_i is None or gold_head is None:
                            unlabelled.fp += 1
                            labelled.fp += 1
                        else:
                            pred_deps.add((gold_i, gold_head, dep))
                            if dep not in labelled_per_dep:
                                labelled_per_dep[dep] = PRFScore()
                            if dep not in pred_deps_per_dep:
                                pred_deps_per_dep[dep] = set()
                            pred_deps_per_dep[dep].add((gold_i, gold_head, dep))
            labelled.score_set(pred_deps, gold_deps)
            for dep in labelled_per_dep:
                labelled_per_dep[dep].score_set(
                    pred_deps_per_dep.get(dep, set()), gold_deps_per_dep.get(dep, set())
                )
            unlabelled.score_set(
                set(item[:2] for item in pred_deps), set(item[:2] for item in gold_deps)
            )
        part_cfg = {
            "attr": attr,
            "ignore_labels": sorted(ignore_labels),
            "missing_values": _sorted_values(missing_values),
        }
        part_scores = {
            "unlabelled": _prf_counts(unlabelled),
            "labelled": _prf_counts(labelled),
            "per_dep": {k: _prf_counts(v) for k, v in labelled_per_dep.items()},
        }
        return _make_part("deps", part_cfg, part_scores)


class ScorerState:
    """A mergeable intermediate state of an evaluation.

    The state contains minimum-size integer counts (and ROC AUC point pairs)
    rather than final float scores, so states for arbitrary subsets can be
    merged deterministically and finalized to the same scores as a single full
    evaluation. States can be saved to disk and transmitted independently.
    """

    def __init__(
        self,
        *,
        states: Dict[str, Dict[str, Any]],
        pipeline: Iterable[str],
        cfg: Optional[Dict[str, Any]] = None,
        state_id: Optional[str] = None,
        canonical: bool = False,
    ) -> None:
        self.states = dict(states)
        self.pipeline = list(pipeline)
        self.cfg = dict(cfg) if cfg else {}
        # Make sure the scorer config is serializable before assigning an ID
        import srsly

        try:
            srsly.json_dumps(self.cfg)
        except (TypeError, ValueError) as e:
            raise ValueError(
                Errors.E1063.format(
                    msg=f"the scorer config is not JSON-serializable: {e}"
                )
            ) from e
        self.id = state_id or uuid.uuid4().hex
        self.canonical = canonical

    def finalize(self, *, per_component: bool = False) -> Dict[str, Any]:
        """Calculate the final scores from the state.

        per_component (bool): Whether to return the scores keyed by component
            name. Defaults to False.
        RETURNS (Dict[str, Any]): A dictionary of scores.
        """
        scores: Dict[str, Any] = {}
        for name, component in self.states.items():
            component_scores: Dict[str, Any] = {}
            for part in component["parts"]:
                finalizer = FINALIZERS[part["type"]]
                component_scores.update(
                    finalizer(part, canonical=self.canonical)
                )
            if per_component:
                scores[name] = component_scores
            else:
                scores.update(component_scores)
        return scores

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "id": self.id,
            "pipeline": self.pipeline,
            "cfg": self.cfg,
            "canonical": self.canonical,
            "states": self.states,
        }

    def to_disk(self, path: Any) -> None:
        import srsly

        path = ensure_path(path)
        data = self.to_dict()
        if str(path).endswith(".gz"):
            srsly.write_gzip_json(path, data)
        else:
            srsly.write_json(path, data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScorerState":
        if not isinstance(data, dict):
            raise ValueError(
                Errors.E1063.format(
                    msg=f"expected a JSON object, but got {type(data).__name__}"
                )
            )
        if data.get("version") != STATE_VERSION:
            raise ValueError(
                Errors.E1063.format(
                    msg=(
                        "unsupported state version "
                        f"{data.get('version')}, expected {STATE_VERSION}"
                    )
                )
            )
        for key in ("id", "pipeline", "states"):
            if key not in data:
                raise ValueError(
                    Errors.E1063.format(msg=f"missing required key '{key}'")
                )
        if not isinstance(data["states"], dict):
            raise ValueError(
                Errors.E1063.format(msg="'states' must be a JSON object")
            )
        return cls(
            states=data["states"],
            pipeline=data["pipeline"],
            cfg=data.get("cfg", {}),
            state_id=data["id"],
            canonical=data.get("canonical", False),
        )

    @classmethod
    def from_disk(cls, path: Any) -> "ScorerState":
        import srsly

        path = ensure_path(path)
        if not path.exists():
            raise ValueError(
                Errors.E1063.format(msg=f"state file not found: {path}")
            )
        if str(path).endswith(".gz"):
            data = srsly.read_gzip_json(path)
        else:
            data = srsly.read_json(path)
        return cls.from_dict(data)

    @classmethod
    def merge(cls, states: Iterable["ScorerState"]) -> "ScorerState":
        """Merge any number of evaluation states following deterministic
        rules. The merge is independent of the order of the states, merging
        the same state more than once raises an error, and states with
        inconsistent pipelines or scorer configurations raise an error
        instead of being mixed silently.

        states (Iterable[ScorerState]): The states to merge.
        RETURNS (ScorerState): The merged state.
        """
        states = list(states)
        if len(states) == 0:
            raise ValueError(
                Errors.E1063.format(msg="no states provided to merge")
            )
        # Reject duplicate states by ID
        seen_ids: Set[str] = set()
        for state in states:
            if not isinstance(state, ScorerState):
                raise ValueError(
                    Errors.E1063.format(
                        msg=f"expected a ScorerState, but got {type(state).__name__}"
                    )
                )
            if state.id in seen_ids:
                raise ValueError(Errors.E1058.format(state_id=state.id))
            seen_ids.add(state.id)
        first = states[0]
        for state in states[1:]:
            if state.pipeline != first.pipeline:
                raise ValueError(
                    Errors.E1059.format(
                        state_id=state.id,
                        pipeline=state.pipeline,
                        other_id=first.id,
                        other_pipeline=first.pipeline,
                    )
                )
            if state.cfg != first.cfg:
                raise ValueError(
                    Errors.E1060.format(
                        name="<scorer>",
                        cfg=state.cfg,
                        state_id=state.id,
                        other_cfg=first.cfg,
                        other_id=first.id,
                    )
                )
            if set(state.states.keys()) != set(first.states.keys()):
                raise ValueError(
                    Errors.E1063.format(
                        msg=(
                            "component states don't match: "
                            f"{sorted(state.states)} vs {sorted(first.states)}"
                        )
                    )
                )
        merged_states: Dict[str, Any] = {}
        for cname in first.states:
            components = [state.states[cname] for state in states]
            merged_states[cname] = _merge_component_states(cname, components, states)
        return cls(
            states=merged_states,
            pipeline=first.pipeline,
            cfg=first.cfg,
            canonical=True,
        )


def _merge_component_states(name, components, states):
    first = components[0]
    n_parts = len(first["parts"])
    for j, component in enumerate(components[1:], start=1):
        if len(component["parts"]) != n_parts:
            raise ValueError(
                Errors.E1061.format(
                    name=name,
                    n_parts=n_parts,
                    part_types=[p["type"] for p in first["parts"]],
                    state_id=states[0].id,
                    n_other=len(component["parts"]),
                    other_types=[p["type"] for p in component["parts"]],
                    other_id=states[j].id,
                )
            )
    merged_parts = []
    for i, part0 in enumerate(first["parts"]):
        for j, component in enumerate(components[1:], start=1):
            part = component["parts"][i]
            if part["type"] != part0["type"]:
                raise ValueError(
                    Errors.E1061.format(
                        name=name,
                        n_parts=n_parts,
                        part_types=[p["type"] for p in first["parts"]],
                        state_id=states[0].id,
                        n_other=len(component["parts"]),
                        other_types=[p["type"] for p in component["parts"]],
                        other_id=states[j].id,
                    )
                )
            if part["cfg"] != part0["cfg"]:
                raise ValueError(
                    Errors.E1060.format(
                        name=name,
                        cfg=part0["cfg"],
                        state_id=states[0].id,
                        other_cfg=part["cfg"],
                        other_id=states[j].id,
                    )
                )
        merged_scores = part0["scores"]
        for j, component in enumerate(components[1:], start=1):
            merged_scores = _merge_part_scores(
                part0["type"], merged_scores, component["parts"][i]["scores"]
            )
        merged_parts.append(
            {"type": part0["type"], "cfg": part0["cfg"], "scores": merged_scores}
        )
    return {"parts": merged_parts}


def get_ner_prf(examples: Iterable[Example], *, _state: bool = False, **kwargs):
    """Compute micro-PRF and per-entity PRF scores for a sequence of examples."""
    score_per_type = defaultdict(PRFScore)
    for eg in examples:
        if not eg.y.has_annotation("ENT_IOB"):
            continue
        golds = {(e.label_, e.start, e.end) for e in eg.y.ents}
        align_x2y = eg.alignment.x2y
        for pred_ent in eg.x.ents:
            if pred_ent.label_ not in score_per_type:
                score_per_type[pred_ent.label_] = PRFScore()
            indices = align_x2y[pred_ent.start : pred_ent.end]
            if len(indices):
                g_span = eg.y[indices[0] : indices[-1] + 1]
                # Check we aren't missing annotation on this span. If so,
                # our prediction is neither right nor wrong, we just
                # ignore it.
                if all(token.ent_iob != 0 for token in g_span):
                    key = (pred_ent.label_, indices[0], indices[-1] + 1)
                    if key in golds:
                        score_per_type[pred_ent.label_].tp += 1
                        golds.remove(key)
                    else:
                        score_per_type[pred_ent.label_].fp += 1
        for label, start, end in golds:
            score_per_type[label].fn += 1
    totals = PRFScore()
    for prf in score_per_type.values():
        totals += prf
    part = _make_part(
        "ner",
        {},
        {
            "score": _prf_counts(totals),
            "per_type": {k: _prf_counts(v) for k, v in score_per_type.items()},
        },
    )
    if _state:
        return {"parts": [part]}
    return _finalize_ner(part)


# The following implementation of trapezoid() is adapted from SciPy,
# which is distributed under the New BSD License.
# Copyright (c) 2001-2002 Enthought, Inc. 2003-2023, SciPy Developers.
# See licenses/3rd_party_licenses.txt
def trapezoid(y, x=None, dx=1.0, axis=-1):
    r"""
    Integrate along the given axis using the composite trapezoidal rule.

    If `x` is provided, the integration happens in sequence along its
    elements - they are not sorted.

    Integrate `y` (`x`) along each 1d slice on the given axis, compute
    :math:`\int y(x) dx`.
    When `x` is specified, this integrates along the parametric curve,
    computing :math:`\int_t y(t) dt =
    \int_t y(t) \left.\frac{dx}{dt}\right|_{x=x(t)} dt`.

    Parameters
    ----------
    y : array_like
        Input array.
    x : array_like, optional
        The sample points corresponding to the `y` values. If x is None,
        the sample points are assumed to be evenly spaced `dx` apart. The
        default is None.
    dx : scalar, optional
        The spacing between sample points when `x` is None. The default
        is 1.0.
    axis : int, optional
        The axis along which to integrate, compute :math:`\int_t y(t)
        dt`.

    Returns
    -------
    trapezoid : float or ndarray
        Definite integral of an N-dimensional array as approximated
        along a single axis by the trapezoidal rule. If `y` is a
        1-dimensional array, then the result is a float. If `n` is
        greater than 1, then the result is an `n`-1 dimensional array.

    See Also
    --------
    cumulative_trapezoid, simpson, romb

    Notes
    -----
    Image [2]_ illustrates trapezoidal rule -- y-axis locations of points
    will be taken from `y` array, by default x-axis distances between
    the red lines.

    References
    ----------
    .. [1] Wikipedia page: https://en.wikipedia.org/wiki/Trapezoidal_rule

    .. [2] Illustration image:
           https://en.wikipedia.org/wiki/File:Composite_trapezoidal_rule_illustration.png

    Examples
    --------
    Use the trapezoidal rule on evenly spaced:

    >>> import numpy as np
    >>> from scipy import integrate
    >>> integrate.trapezoid([1, 2, 3])
    4.0

    The spacing between sample points can be selected by either the
    ``x`` or ``dx`` arguments:

    >>> integrate.trapezoid([1, 2, 3], x=[4, 6, 8])
    8.0
    >>> integrate.trapezoid([1, 2, 3], dx=2)
    8.0

    Using a decreasing ``x`` corresponds to integrating in reverse:

    >>> integrate.trapezoid([1, 2, 3], x=[8, 6, 4])
    -8.0

    More generally ``x`` is used to integrate along a parametric curve. We can
    estimate the circle :math:`\int_0^1 x**2 = 1/3` using:

    >>> x = np.linspace(0, 1, num=50)
    >>> y = x**2
    >>> integrate.trapezoid(y, x)
    0.33340274885464394

    Or estimate a circle, noting we repeat the sample which closes
    the curve:

    >>> theta = np.linspace(0, 2 * np.pi, num=1000, endpoint=True)
    >>> integrate.trapezoid(np.cos(theta), x=np.sin(theta))
    3.141571941375841

    ``trapezoid`` can be applied along a specified axis to do multiple
    computations in one call:

    >>> a = np.arange(6).reshape(2, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> integrate.trapezoid(a, axis=0)
    array([1.5, 2.5, 3.5])
    >>> integrate.trapezoid(a, axis=1)
    array([2.,  8.])
    """
    y = np.asanyarray(y)
    if x is None:
        d = dx
    else:
        x = np.asanyarray(x)
        if x.ndim == 1:
            d = np.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = np.diff(x, axis=axis)
    nd = y.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    try:
        ret = (d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0).sum(axis)
    except ValueError:
        # Operations didn't work, cast to ndarray
        d = np.asarray(d)
        y = np.asarray(y)
        ret = np.add.reduce(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis)
    return ret


# The following implementation of roc_auc_score() is adapted from
# scikit-learn, which is distributed under the New BSD License
# Copyright (c) 2007–2019 The scikit-learn developers.
# See licenses/3rd_party_licenses.txt
def _roc_auc_score(y_true, y_score):
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC)
    from prediction scores.

    Note: this implementation is restricted to the binary classification task

    Parameters
    ----------
    y_true : array, shape = [n_samples] or [n_samples, n_classes]
        True binary labels or binary label indicators.

    y_score : array, shape = [n_samples] or [n_samples, n_classes]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some models). For binary
        y_true, y_score is supposed to be the score of the class with greater
        confidence. The multiclass case expects shape [n_samples, n_classes].

    Returns
    -------
    auc : float

    References
    ----------
    .. [1] `Wikipedia entry for the Receiver operating characteristic
            <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_

    .. [2] Fawcett T. An introduction to ROC analysis[J]. Pattern Recognition
           Letters, 2006, 27(8):861-874.

    .. [3] `Analyzing a portion of the ROC curve. McClish, 1989
    """
    if len(np.unique(y_true)) != 2:
        raise ValueError(Errors.E165.format(label=np.unique(y_true)))
    fpr, tpr, _ = _roc_curve(y_true, y_score)
    return _auc(fpr, tpr)


def _roc_curve(y_true, y_score):
    """Compute Receiver operating characteristic

    Note: this implementation is restricted to the binary classification task.

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        True labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

    y_score : array, shape = [n_samples]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function").

    Returns
    -------
    fpr : array, shape = [>2]
        Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= thresholds[i].

    tpr : array, shape = [>2]
        Increasing true positive rates such that element i is the true
        positive rate of predictions with score >= thresholds[i].

    thresholds : array, shape = [n_thresholds]
        Decreasing threshold values. `thresholds[0]` represents no instances
        being predicted and is arbitrarily set to `max(y_score) + 1`.

    Notes
    -----
    Since the thresholds are sorted from low to high values, they
    are reversed upon returning them to ensure they correspond to both ``fpr``
    and ``tpr``, which are sorted in reversed order during their calculation.

    References
    ----------
    .. [1] Wikipedia - ROC curve:
           https://en.wikipedia.org/wiki/Receiver_operating_characteristic

    .. [2] Fawcett T. An introduction to ROC analysis[J]. Pattern Recognition
           Letters, 2006, 27(8):861-874.
    """
    fps, tps, thresholds = _binary_clf_curve(y_true, y_score)

    # Add an extra threshold position
    # to make sure that the curve starts at (0, 0)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return fpr, tpr, thresholds


def _binary_clf_curve(y_true, y_score):
    """Calculate true and false positives per binary classification threshold.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification

    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function

    Returns
    -------
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds. The total number of negative
        samples is fps[-1] (thus true negatives are fps[-1] - fps).

    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i the number of
        positive samples assigned a score >= thresholds. The total number
        of positive samples is tps[-1] (thus false negatives are tps[-1]
        - tps).

    thresholds : array, shape = [n_thresholds]
        Decreasing score values.
    """
    pos_label = 1.0

    y_true = np.ravel(y_true)
    y_score = np.ravel(y_score)

    # make y_true a boolean vector
    y_true = y_true == pos_label

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    weight = 1.0

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = _stable_cumsum(y_true * weight)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]


def _stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum

    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    axis : int, optional
        Axis along which the cumulative sum is computed.
        The default is None to compute the cumsum over the flattened array.
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.all(
        np.isclose(
            out.take(-1, axis=axis), expected, rtol=rtol, atol=atol, equal_nan=True
        )
    ):
        raise ValueError(Errors.E163)
    return out


def _auc(x, y):
    """Compute Area Under the Curve (AUC) using the trapezoidal rule

    This is a general function, given points on a curve.  For computing the
    area under the ROC-curve, see :func:`roc_auc_score`.

    Parameters
    ----------
    x : array, shape = [n]
        x coordinates. These must be either monotonic increasing or monotonic
        decreasing.
    y : array, shape = [n]
        y coordinates.

    Returns
    -------
    auc : float
    """
    x = np.ravel(x)
    y = np.ravel(y)

    direction = 1
    dx = np.diff(x)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            raise ValueError(Errors.E164.format(x=x))

    area = direction * trapezoid(y, x)
    if isinstance(area, np.memmap):
        # Reductions such as .sum used internally in trapezoid do not return a
        # scalar by default for numpy.memmap instances contrary to regular
        # numpy.ndarray instances.
        area = area.dtype.type(area)
    return area
