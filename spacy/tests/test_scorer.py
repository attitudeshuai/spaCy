import uuid

import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pytest import approx

from spacy.lang.en import English
from spacy.scorer import (
    PRFScore,
    ROCAUCScore,
    STATE_VERSION,
    Scorer,
    ScorerState,
    _roc_auc_score,
    _roc_curve,
)
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.training.iob_utils import offsets_to_biluo_tags

from .util import make_tempdir

test_las_apple = [
    [
        "Apple is looking at buying U.K. startup for $ 1 billion",
        {
            "heads": [2, 2, 2, 2, 3, 6, 4, 4, 10, 10, 7],
            "deps": [
                "nsubj",
                "aux",
                "ROOT",
                "prep",
                "pcomp",
                "compound",
                "dobj",
                "prep",
                "quantmod",
                "compound",
                "pobj",
            ],
        },
    ]
]

test_ner_cardinal = [
    ["100 - 200", {"entities": [[0, 3, "CARDINAL"], [6, 9, "CARDINAL"]]}]
]

test_ner_apple = [
    [
        "Apple is looking at buying U.K. startup for $1 billion",
        {"entities": [(0, 5, "ORG"), (27, 31, "GPE"), (44, 54, "MONEY")]},
    ]
]


@pytest.fixture
def tagged_doc():
    text = "Sarah's sister flew to Silicon Valley via London."
    tags = ["NNP", "POS", "NN", "VBD", "IN", "NNP", "NNP", "IN", "NNP", "."]
    pos = [
        "PROPN",
        "PART",
        "NOUN",
        "VERB",
        "ADP",
        "PROPN",
        "PROPN",
        "ADP",
        "PROPN",
        "PUNCT",
    ]
    morphs = [
        "NounType=prop|Number=sing",
        "Poss=yes",
        "Number=sing",
        "Tense=past|VerbForm=fin",
        "",
        "NounType=prop|Number=sing",
        "NounType=prop|Number=sing",
        "",
        "NounType=prop|Number=sing",
        "PunctType=peri",
    ]
    nlp = English()
    doc = nlp(text)
    for i in range(len(tags)):
        doc[i].tag_ = tags[i]
        doc[i].pos_ = pos[i]
        doc[i].set_morph(morphs[i])
        if i > 0:
            doc[i].is_sent_start = False
    return doc


@pytest.fixture
def sented_doc():
    text = "One sentence. Two sentences. Three sentences."
    nlp = English()
    doc = nlp(text)
    for i in range(len(doc)):
        if i % 3 == 0:
            doc[i].is_sent_start = True
        else:
            doc[i].is_sent_start = False
    return doc


def test_tokenization(sented_doc):
    scorer = Scorer()
    gold = {"sent_starts": [t.sent_start for t in sented_doc]}
    example = Example.from_dict(sented_doc, gold)
    scores = scorer.score([example])
    assert scores["token_acc"] == 1.0

    nlp = English()
    example.predicted = Doc(
        nlp.vocab,
        words=["One", "sentence.", "Two", "sentences.", "Three", "sentences."],
        spaces=[True, True, True, True, True, False],
    )
    example.predicted[1].is_sent_start = False
    scores = scorer.score([example])
    assert scores["token_acc"] == 0.5
    assert scores["token_p"] == 0.5
    assert scores["token_r"] == approx(0.33333333)
    assert scores["token_f"] == 0.4

    # per-component scoring
    scorer = Scorer()
    scores = scorer.score([example], per_component=True)
    assert scores["tokenizer"]["token_acc"] == 0.5
    assert scores["tokenizer"]["token_p"] == 0.5
    assert scores["tokenizer"]["token_r"] == approx(0.33333333)
    assert scores["tokenizer"]["token_f"] == 0.4


def test_sents(sented_doc):
    scorer = Scorer()
    gold = {"sent_starts": [t.sent_start for t in sented_doc]}
    example = Example.from_dict(sented_doc, gold)
    scores = scorer.score([example])
    assert scores["sents_f"] == 1.0

    # One sentence start is moved
    gold["sent_starts"][3] = 0
    gold["sent_starts"][4] = 1
    example = Example.from_dict(sented_doc, gold)
    scores = scorer.score([example])
    assert scores["sents_f"] == approx(0.3333333)


def test_las_per_type(en_vocab):
    # Gold and Doc are identical
    scorer = Scorer()
    examples = []
    for input_, annot in test_las_apple:
        doc = Doc(
            en_vocab, words=input_.split(" "), heads=annot["heads"], deps=annot["deps"]
        )
        gold = {"heads": annot["heads"], "deps": annot["deps"]}
        example = Example.from_dict(doc, gold)
        examples.append(example)
    results = scorer.score(examples)

    assert results["dep_uas"] == 1.0
    assert results["dep_las"] == 1.0
    assert results["dep_las_per_type"]["nsubj"]["p"] == 1.0
    assert results["dep_las_per_type"]["nsubj"]["r"] == 1.0
    assert results["dep_las_per_type"]["nsubj"]["f"] == 1.0
    assert results["dep_las_per_type"]["compound"]["p"] == 1.0
    assert results["dep_las_per_type"]["compound"]["r"] == 1.0
    assert results["dep_las_per_type"]["compound"]["f"] == 1.0

    # One dep is incorrect in Doc
    scorer = Scorer()
    examples = []
    for input_, annot in test_las_apple:
        doc = Doc(
            en_vocab, words=input_.split(" "), heads=annot["heads"], deps=annot["deps"]
        )
        gold = {"heads": annot["heads"], "deps": annot["deps"]}
        doc[0].dep_ = "compound"
        example = Example.from_dict(doc, gold)
        examples.append(example)
    results = scorer.score(examples)

    assert results["dep_uas"] == 1.0
    assert_almost_equal(results["dep_las"], 0.9090909)
    assert results["dep_las_per_type"]["nsubj"]["p"] == 0
    assert results["dep_las_per_type"]["nsubj"]["r"] == 0
    assert results["dep_las_per_type"]["nsubj"]["f"] == 0
    assert_almost_equal(results["dep_las_per_type"]["compound"]["p"], 0.666666666)
    assert results["dep_las_per_type"]["compound"]["r"] == 1.0
    assert results["dep_las_per_type"]["compound"]["f"] == 0.8


def test_ner_per_type(en_vocab):
    # Gold and Doc are identical
    scorer = Scorer()
    examples = []
    for input_, annot in test_ner_cardinal:
        doc = Doc(
            en_vocab, words=input_.split(" "), ents=["B-CARDINAL", "O", "B-CARDINAL"]
        )
        entities = offsets_to_biluo_tags(doc, annot["entities"])
        example = Example.from_dict(doc, {"entities": entities})
        # a hack for sentence boundaries
        example.predicted[1].is_sent_start = False
        example.reference[1].is_sent_start = False
        examples.append(example)
    results = scorer.score(examples)

    assert results["ents_p"] == 1.0
    assert results["ents_r"] == 1.0
    assert results["ents_f"] == 1.0
    assert results["ents_per_type"]["CARDINAL"]["p"] == 1.0
    assert results["ents_per_type"]["CARDINAL"]["r"] == 1.0
    assert results["ents_per_type"]["CARDINAL"]["f"] == 1.0

    # Doc has one missing and one extra entity
    # Entity type MONEY is not present in Doc
    scorer = Scorer()
    examples = []
    for input_, annot in test_ner_apple:
        doc = Doc(
            en_vocab,
            words=input_.split(" "),
            ents=["B-ORG", "O", "O", "O", "O", "B-GPE", "B-ORG", "O", "O", "O"],
        )
        entities = offsets_to_biluo_tags(doc, annot["entities"])
        example = Example.from_dict(doc, {"entities": entities})
        # a hack for sentence boundaries
        example.predicted[1].is_sent_start = False
        example.reference[1].is_sent_start = False
        examples.append(example)
    results = scorer.score(examples)

    assert results["ents_p"] == approx(0.6666666)
    assert results["ents_r"] == approx(0.6666666)
    assert results["ents_f"] == approx(0.6666666)
    assert "GPE" in results["ents_per_type"]
    assert "MONEY" in results["ents_per_type"]
    assert "ORG" in results["ents_per_type"]
    assert results["ents_per_type"]["GPE"]["p"] == 1.0
    assert results["ents_per_type"]["GPE"]["r"] == 1.0
    assert results["ents_per_type"]["GPE"]["f"] == 1.0
    assert results["ents_per_type"]["MONEY"]["p"] == 0
    assert results["ents_per_type"]["MONEY"]["r"] == 0
    assert results["ents_per_type"]["MONEY"]["f"] == 0
    assert results["ents_per_type"]["ORG"]["p"] == 0.5
    assert results["ents_per_type"]["ORG"]["r"] == 1.0
    assert results["ents_per_type"]["ORG"]["f"] == approx(0.6666666)


def test_tag_score(tagged_doc):
    # Gold and Doc are identical
    scorer = Scorer()
    gold = {
        "tags": [t.tag_ for t in tagged_doc],
        "pos": [t.pos_ for t in tagged_doc],
        "morphs": [str(t.morph) for t in tagged_doc],
        "sent_starts": [1 if t.is_sent_start else -1 for t in tagged_doc],
    }
    example = Example.from_dict(tagged_doc, gold)
    results = scorer.score([example])

    assert results["tag_acc"] == 1.0
    assert results["pos_acc"] == 1.0
    assert results["morph_acc"] == 1.0
    assert results["morph_micro_f"] == 1.0
    assert results["morph_per_feat"]["NounType"]["f"] == 1.0

    # Gold annotation is modified
    scorer = Scorer()
    tags = [t.tag_ for t in tagged_doc]
    tags[0] = "NN"
    pos = [t.pos_ for t in tagged_doc]
    pos[1] = "X"
    morphs = [str(t.morph) for t in tagged_doc]
    morphs[1] = "Number=sing"
    morphs[2] = "Number=plur"
    gold = {
        "tags": tags,
        "pos": pos,
        "morphs": morphs,
        "sent_starts": gold["sent_starts"],
    }
    example = Example.from_dict(tagged_doc, gold)
    results = scorer.score([example])

    assert results["tag_acc"] == 0.9
    assert results["pos_acc"] == 0.9
    assert results["morph_acc"] == approx(0.8)
    assert results["morph_micro_f"] == approx(0.8461538)
    assert results["morph_per_feat"]["NounType"]["f"] == 1.0
    assert results["morph_per_feat"]["Poss"]["f"] == 0.0
    assert results["morph_per_feat"]["Number"]["f"] == approx(0.72727272)

    # per-component scoring
    scorer = Scorer()
    results = scorer.score([example], per_component=True)
    assert results["tagger"]["tag_acc"] == 0.9
    assert results["morphologizer"]["pos_acc"] == 0.9
    assert results["morphologizer"]["morph_acc"] == approx(0.8)


def test_partial_annotation(en_tokenizer):
    pred_doc = en_tokenizer("a b c d e")
    pred_doc[0].tag_ = "A"
    pred_doc[0].pos_ = "X"
    pred_doc[0].set_morph("Feat=Val")
    pred_doc[0].dep_ = "dep"

    # unannotated reference
    ref_doc = en_tokenizer("a b c d e")
    ref_doc.has_unknown_spaces = True
    example = Example(pred_doc, ref_doc)
    scorer = Scorer()
    scores = scorer.score([example])
    for key in scores:
        # cats doesn't have an unset state
        if key.startswith("cats"):
            continue
        assert scores[key] is None

    # partially annotated reference, not overlapping with predicted annotation
    ref_doc = en_tokenizer("a b c d e")
    ref_doc.has_unknown_spaces = True
    ref_doc[1].tag_ = "A"
    ref_doc[1].pos_ = "X"
    ref_doc[1].set_morph("Feat=Val")
    ref_doc[1].dep_ = "dep"
    example = Example(pred_doc, ref_doc)
    scorer = Scorer()
    scores = scorer.score([example])
    assert scores["token_acc"] is None
    assert scores["tag_acc"] == 0.0
    assert scores["pos_acc"] == 0.0
    assert scores["morph_acc"] == 0.0
    assert scores["dep_uas"] == 1.0
    assert scores["dep_las"] == 0.0
    assert scores["sents_f"] is None

    # partially annotated reference, overlapping with predicted annotation
    ref_doc = en_tokenizer("a b c d e")
    ref_doc.has_unknown_spaces = True
    ref_doc[0].tag_ = "A"
    ref_doc[0].pos_ = "X"
    ref_doc[1].set_morph("Feat=Val")
    ref_doc[1].dep_ = "dep"
    example = Example(pred_doc, ref_doc)
    scorer = Scorer()
    scores = scorer.score([example])
    assert scores["token_acc"] is None
    assert scores["tag_acc"] == 1.0
    assert scores["pos_acc"] == 1.0
    assert scores["morph_acc"] == 0.0
    assert scores["dep_uas"] == 1.0
    assert scores["dep_las"] == 0.0
    assert scores["sents_f"] is None


def test_roc_auc_score():
    # Binary classification, toy tests from scikit-learn test suite
    y_true = [0, 1]
    y_score = [0, 1]
    tpr, fpr, _ = _roc_curve(y_true, y_score)
    roc_auc = _roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 0, 1])
    assert_array_almost_equal(fpr, [0, 1, 1])
    assert_almost_equal(roc_auc, 1.0)

    y_true = [0, 1]
    y_score = [1, 0]
    tpr, fpr, _ = _roc_curve(y_true, y_score)
    roc_auc = _roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 1, 1])
    assert_array_almost_equal(fpr, [0, 0, 1])
    assert_almost_equal(roc_auc, 0.0)

    y_true = [1, 0]
    y_score = [1, 1]
    tpr, fpr, _ = _roc_curve(y_true, y_score)
    roc_auc = _roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 1])
    assert_array_almost_equal(fpr, [0, 1])
    assert_almost_equal(roc_auc, 0.5)

    y_true = [1, 0]
    y_score = [1, 0]
    tpr, fpr, _ = _roc_curve(y_true, y_score)
    roc_auc = _roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 0, 1])
    assert_array_almost_equal(fpr, [0, 1, 1])
    assert_almost_equal(roc_auc, 1.0)

    y_true = [1, 0]
    y_score = [0.5, 0.5]
    tpr, fpr, _ = _roc_curve(y_true, y_score)
    roc_auc = _roc_auc_score(y_true, y_score)
    assert_array_almost_equal(tpr, [0, 1])
    assert_array_almost_equal(fpr, [0, 1])
    assert_almost_equal(roc_auc, 0.5)

    # same result as above with ROCAUCScore wrapper
    score = ROCAUCScore()
    score.score_set(0.5, 1)
    score.score_set(0.5, 0)
    assert_almost_equal(score.score, 0.5)

    # check that errors are raised in undefined cases and score is -inf
    y_true = [0, 0]
    y_score = [0.25, 0.75]
    with pytest.raises(ValueError):
        _roc_auc_score(y_true, y_score)

    score = ROCAUCScore()
    score.score_set(0.25, 0)
    score.score_set(0.75, 0)
    with pytest.raises(ValueError):
        _ = score.score  # noqa: F841

    y_true = [1, 1]
    y_score = [0.25, 0.75]
    with pytest.raises(ValueError):
        _roc_auc_score(y_true, y_score)

    score = ROCAUCScore()
    score.score_set(0.25, 1)
    score.score_set(0.75, 1)
    with pytest.raises(ValueError):
        _ = score.score  # noqa: F841


def test_score_spans():
    nlp = English()
    text = "This is just a random sentence."
    key = "my_spans"
    gold = nlp.make_doc(text)
    pred = nlp.make_doc(text)
    spans = []
    spans.append(gold.char_span(0, 4, label="PERSON"))
    spans.append(gold.char_span(0, 7, label="ORG"))
    spans.append(gold.char_span(8, 12, label="ORG"))
    gold.spans[key] = spans

    def span_getter(doc, span_key):
        return doc.spans[span_key]

    # Predict exactly the same, but overlapping spans will be discarded
    pred.spans[key] = gold.spans[key].copy(doc=pred)
    eg = Example(pred, gold)
    scores = Scorer.score_spans([eg], attr=key, getter=span_getter)
    assert scores[f"{key}_p"] == 1.0
    assert scores[f"{key}_r"] < 1.0

    # Allow overlapping, now both precision and recall should be 100%
    pred.spans[key] = gold.spans[key].copy(doc=pred)
    eg = Example(pred, gold)
    scores = Scorer.score_spans([eg], attr=key, getter=span_getter, allow_overlap=True)
    assert scores[f"{key}_p"] == 1.0
    assert scores[f"{key}_r"] == 1.0

    # Change the predicted labels
    new_spans = [Span(pred, span.start, span.end, label="WRONG") for span in spans]
    pred.spans[key] = new_spans
    eg = Example(pred, gold)
    scores = Scorer.score_spans([eg], attr=key, getter=span_getter, allow_overlap=True)
    assert scores[f"{key}_p"] == 0.0
    assert scores[f"{key}_r"] == 0.0
    assert f"{key}_per_type" in scores

    # Discard labels from the evaluation
    scores = Scorer.score_spans(
        [eg], attr=key, getter=span_getter, allow_overlap=True, labeled=False
    )
    assert scores[f"{key}_p"] == 1.0
    assert scores[f"{key}_r"] == 1.0
    assert f"{key}_per_type" not in scores


def test_prf_score():
    cand = {"hi", "ho"}
    gold1 = {"yo", "hi"}
    gold2 = set()

    a = PRFScore()
    a.score_set(cand=cand, gold=gold1)
    assert (a.precision, a.recall, a.fscore) == approx((0.5, 0.5, 0.5))

    b = PRFScore()
    b.score_set(cand=cand, gold=gold2)
    assert (b.precision, b.recall, b.fscore) == approx((0.0, 0.0, 0.0))

    c = a + b
    assert (c.precision, c.recall, c.fscore) == approx((0.25, 0.5, 0.33333333))

    a += b
    assert (a.precision, a.recall, a.fscore) == approx(
        (c.precision, c.recall, c.fscore)
    )


@pytest.fixture
def rich_nlp():
    nlp = English()
    for factory in ("tagger", "morphologizer", "parser", "ner", "textcat", "spancat"):
        nlp.add_pipe(factory)
    for label in ("NN", "NNS", "VBZ"):
        nlp.get_pipe("tagger").add_label(label)
    for label in ("nsubj", "ROOT", "cc", "dobj"):
        nlp.get_pipe("parser").add_label(label)
    for label in ("PERSON", "ORG", "GPE"):
        nlp.get_pipe("ner").add_label(label)
    nlp.get_pipe("textcat").add_label("L1")
    nlp.get_pipe("textcat").add_label("L2")
    nlp.get_pipe("spancat").add_label("SP1")
    nlp.get_pipe("spancat").add_label("SP2")
    return nlp


def _make_rich_example(nlp, words, *, gold, pred=None):
    doc = Doc(nlp.vocab, words=words, spaces=[True] * (len(words) - 1) + [False])
    eg = Example.from_dict(doc, gold)
    if pred is not None:
        pred_doc = eg.predicted
        if "sent_starts" in pred:
            for token, value in zip(pred_doc, pred["sent_starts"]):
                token.is_sent_start = value
        if "tags" in pred:
            for token, tag in zip(pred_doc, pred["tags"]):
                token.tag_ = tag
        if "deps" in pred:
            for token, (dep, head_i) in zip(pred_doc, pred["deps"]):
                token.dep_ = dep
                token.head = pred_doc[head_i]
        if "cats" in pred:
            pred_doc.cats = dict(pred["cats"])
        if "ents" in pred:
            pred_doc.ents = [
                Span(pred_doc, start, end, label=label)
                for start, end, label in pred["ents"]
            ]
        if "spans" in pred:
            for key, spans in pred["spans"].items():
                pred_doc.spans[key] = [
                    Span(pred_doc, start, end, label=label)
                    for start, end, label in spans
                ]
    return eg


@pytest.fixture
def rich_examples(rich_nlp):
    examples = []
    examples.append(
        _make_rich_example(
            rich_nlp,
            ["Sarah", "likes", "cats"],
            gold={
                "tags": ["NNP", "VBZ", "NNS"],
                "pos": ["PROPN", "VERB", "NOUN"],
                "morphs": ["", "Tense=pres", "Number=plur"],
                "lemmas": ["Sarah", "like", "cat"],
                "heads": [1, 1, 1],
                "deps": ["nsubj", "ROOT", "dobj"],
                "sent_starts": [True, False, False],
                "entities": [[0, 5, "PERSON"]],
                "cats": {"L1": 1.0, "L2": 0.0},
                "spans": {"sc": [[0, 5, "SP1"]]},
            },
            pred={
                "tags": ["NNP", "VBZ", "NN"],
                "deps": [("nsubj", 1), ("ROOT", 1), ("nsubj", 1)],
                "sent_starts": [True, False, False],
                "cats": {"L1": 0.7, "L2": 0.3},
                "ents": [(0, 1, "PERSON")],
                "spans": {"sc": [(0, 1, "SP1")]},
            },
        )
    )
    examples.append(
        _make_rich_example(
            rich_nlp,
            ["John", "and", "Mary", "walked"],
            gold={
                "tags": ["NNP", "CC", "NNP", "VBD"],
                "heads": [3, 0, 3, 3],
                "deps": ["nsubj", "cc", "nsubj", "ROOT"],
                "sent_starts": [True, False, False, False],
                "entities": [[0, 4, "PERSON"], [9, 13, "PERSON"]],
                "cats": {"L1": 0.0, "L2": 1.0},
            },
            pred={
                "tags": ["NNP", "CC", "NN", "VBD"],
                "deps": [
                    ("nsubj", 3),
                    ("cc", 0),
                    ("nsubj", 3),
                    ("ROOT", 3),
                ],
                "sent_starts": [True, False, False, False],
                "cats": {"L1": 0.3, "L2": 0.7},
                "ents": [(0, 1, "PERSON"), (2, 3, "ORG")],
            },
        )
    )
    examples.append(
        _make_rich_example(
            rich_nlp,
            ["Google", "buys", "firms"],
            gold={
                "tags": ["NNP", "VBZ", "NNS"],
                "heads": [1, 1, 1],
                "deps": ["nsubj", "ROOT", "dobj"],
                "sent_starts": [True, False, False],
                "entities": [[0, 6, "ORG"]],
                "cats": {"L1": 1.0, "L2": 0.0},
                "spans": {"sc": [[0, 6, "SP2"]]},
            },
            pred={
                "tags": ["NN", "VBZ", "NNS"],
                "deps": [("nsubj", 1), ("ROOT", 1), ("dobj", 1)],
                "sent_starts": [True, False, False],
                "cats": {"L1": 0.4, "L2": 0.6},
                "ents": [],
                "spans": {"sc": [(0, 1, "SP1")]},
            },
        )
    )
    examples.append(
        _make_rich_example(
            rich_nlp,
            ["A", "small", "sentence"],
            gold={
                "tags": ["DT", "JJ", "NN"],
                "heads": [2, 2, 2],
                "deps": ["det", "amod", "ROOT"],
                "sent_starts": [True, False, False],
                "cats": {"L1": 0.0, "L2": 1.0},
                "spans": {"sc": [[8, 16, "SP1"]]},
            },
            pred={
                "tags": ["DT", "JJ", "NN"],
                "deps": [("det", 2), ("amod", 2), ("ROOT", 2)],
                "sent_starts": [True, False, False],
                "cats": {"L1": 0.1, "L2": 0.9},
                "spans": {"sc": [(2, 3, "SP1")]},
            },
        )
    )
    examples.append(
        _make_rich_example(
            rich_nlp,
            ["London", "is", "big"],
            gold={
                "tags": ["NNP", "VBZ", "JJ"],
                "heads": [0, 0, 0],
                "deps": ["ROOT", "aux", "amod"],
                "sent_starts": [True, False, False],
                "entities": [[0, 6, "GPE"]],
                "cats": {"L1": 1.0, "L2": 0.0},
            },
            pred={
                "tags": ["NNP", "VBZ", "JJ"],
                "deps": [("ROOT", 0), ("aux", 0), ("amod", 0)],
                "sent_starts": [True, False, False],
                "cats": {"L1": 0.8, "L2": 0.2},
                "ents": [(0, 1, "GPE")],
            },
        )
    )
    return rich_nlp, examples


def test_scorer_state_finalize_matches_score(rich_examples):
    nlp, examples = rich_examples
    scorer = Scorer(nlp=nlp)
    # a fresh state finalized directly produces the same scores as score()
    state = scorer.accumulate(examples)
    assert state.finalize() == scorer.score(examples)
    assert state.finalize(per_component=True) == scorer.score(
        examples, per_component=True
    )


def test_scorer_state_merge_equals_full(rich_examples):
    nlp, examples = rich_examples
    scorer = Scorer(nlp=nlp)
    full = scorer.score(examples)
    shard_subsets = [examples[:2], examples[2:3], examples[3:]]
    states = [scorer.accumulate(subset) for subset in shard_subsets]
    merged = ScorerState.merge(states).finalize()
    assert set(merged.keys()) == set(full.keys())
    # all covered keys are derived from counts with consistent label order,
    # so merged scores are exactly equal to a single full evaluation
    for key in full:
        assert merged[key] == full[key], key


def test_scorer_state_merge_order_independent(rich_examples):
    nlp, examples = rich_examples
    scorer = Scorer(nlp=nlp)
    subsets = [examples[:2], examples[2:3], examples[3:]]
    orders = [
        [0, 1, 2],
        [2, 0, 1],
        [1, 2, 0],
    ]
    results = []
    for order in orders:
        states = [scorer.accumulate(subsets[i]) for i in order]
        results.append(ScorerState.merge(states).finalize())
    for result in results[1:]:
        assert result == results[0]


def test_scorer_state_duplicate_rejected(rich_examples):
    nlp, examples = rich_examples
    scorer = Scorer(nlp=nlp)
    s1 = scorer.accumulate(examples[:2])
    s2 = scorer.accumulate(examples[2:])
    with pytest.raises(ValueError):
        ScorerState.merge([s1, s2, s1])


def test_scorer_state_pipeline_mismatch(rich_examples):
    nlp, examples = rich_examples
    scorer = Scorer(nlp=nlp)
    s1 = scorer.accumulate(examples[:2])
    s2 = ScorerState(states=s1.states, pipeline=["other", "pipes"], cfg={})
    with pytest.raises(ValueError):
        ScorerState.merge([s1, s2])


def test_scorer_state_cfg_mismatch():
    nlp = English()
    text = "some text"
    gold = nlp.make_doc(text)
    gold.cats = {"A": 1.0, "B": 0.0}
    pred = nlp.make_doc(text)
    pred.cats = {"A": 0.7, "B": 0.3}
    eg = Example(pred, gold)
    s1 = Scorer.score_cats(
        [eg], "cats", labels=["A", "B"], multi_label=False, _state=True
    )
    s2 = Scorer.score_cats(
        [eg], "cats", labels=["A", "C"], multi_label=False, _state=True
    )
    with pytest.raises(ValueError):
        ScorerState.merge(
            [ScorerState.from_dict(_state_with_parts(s1)),
             ScorerState.from_dict(_state_with_parts(s2))]
        )


def _state_with_parts(component_state, pipeline=["textcat"], cfg=None):
    return {
        "version": STATE_VERSION,
        "id": uuid.uuid4().hex,
        "pipeline": pipeline,
        "cfg": cfg or {},
        "canonical": False,
        "states": {"textcat": component_state},
    }


def test_scorer_state_empty_semantics(rich_nlp):
    scorer = Scorer(nlp=rich_nlp)
    empty = scorer.accumulate([]).finalize()
    # None semantics are preserved for counts-based keys, not turned into 0
    assert empty["token_acc"] is None
    assert empty["token_p"] is None
    assert empty["tag_acc"] is None
    assert empty["pos_acc"] is None
    assert empty["morph_acc"] is None
    assert empty["morph_per_feat"] is None
    assert Scorer.score_token_attr([], "lemma")["lemma_acc"] is None
    assert empty["dep_uas"] is None
    assert empty["dep_las"] is None
    assert empty["dep_las_per_type"] is None
    assert empty["ents_p"] is None
    assert empty["ents_per_type"] is None
    assert empty["sents_p"] is None
    assert empty["spans_sc_p"] is None
    # cats keep the existing semantics where empty data scores as 0.0
    assert empty["cats_score"] == 0.0
    assert empty["cats_micro_f"] == 0.0
    assert empty["cats_macro_f"] == 0.0
    assert empty["cats_auc_per_type"] == {"L1": None, "L2": None}


def test_scorer_state_links_empty_semantics():
    scores = Scorer.score_links([], negative_labels=["NIL"])
    assert scores["nel_score"] == 0.0
    assert scores["nel_micro_f"] == 0.0
    assert scores["nel_macro_f"] == 0.0
    assert scores["nel_f_per_type"] == {}


def test_scorer_state_empty_merges_as_real(rich_examples):
    nlp, examples = rich_examples
    scorer = Scorer(nlp=nlp)
    real = scorer.score(examples)
    empty_state = scorer.accumulate([])
    states = [scorer.accumulate(examples[:3]), scorer.accumulate(examples[3:])]
    merged = ScorerState.merge([empty_state] + states).finalize()
    for key in real:
        assert merged[key] == real[key], key


def test_scorer_state_missing_annotation_shard(rich_examples):
    nlp, examples = rich_examples
    scorer = Scorer(nlp=nlp)
    # copies of the first examples with reference NER marked missing
    missing_examples = []
    for eg in examples[:2]:
        # remove the entities from the JSON so from_json produces a reference
        # with missing NER annotation (not "O")
        ref_json = eg.reference.to_json()
        ref_json.pop("ents", None)
        ref = Doc(nlp.vocab).from_json(ref_json)
        missing_examples.append(Example(eg.predicted.copy(), ref))
    # a single evaluation over the same data skips examples missing NER
    mixed = missing_examples + examples[2:]
    full = scorer.score(mixed)
    states = [
        scorer.accumulate(missing_examples),
        scorer.accumulate(examples[2:]),
    ]
    # the shard with missing annotations does not score as 0, it stays null
    assert states[0].finalize()["ents_p"] is None
    merged = ScorerState.merge(states).finalize()
    assert merged["ents_p"] == full["ents_p"]
    assert merged["ents_r"] == full["ents_r"]
    assert merged["ents_f"] == full["ents_f"]


def test_scorer_state_disk_roundtrip(rich_examples):
    nlp, examples = rich_examples
    scorer = Scorer(nlp=nlp)
    state = scorer.accumulate(examples[:2])
    with make_tempdir() as tmpdir:
        path = tmpdir / "state.json"
        state.to_disk(path)
        loaded = ScorerState.from_disk(path)
        assert loaded.finalize() == state.finalize()
        gz_path = tmpdir / "state.json.gz"
        state.to_disk(gz_path)
        loaded_gz = ScorerState.from_disk(gz_path)
        assert loaded_gz.finalize() == state.finalize()


def _nel_example(nlp, words, *, pred_ents, gold_ents):
    pred = Doc(nlp.vocab, words=words, spaces=[True] * (len(words) - 1) + [False])
    gold = Doc(nlp.vocab, words=words, spaces=[True] * (len(words) - 1) + [False])
    for start, end, label, kb_id in gold_ents:
        span = Span(gold, start, end, label=label)
        span.kb_id_ = kb_id
        gold.ents = list(gold.ents) + [span]
    for start, end, label, kb_id in pred_ents:
        span = Span(pred, start, end, label=label)
        span.kb_id_ = kb_id
        pred.ents = list(pred.ents) + [span]
    return Example(pred, gold)


def test_scorer_state_links_merge_tolerance():
    nlp = English()
    examples = [
        _nel_example(
            nlp,
            ["a", "b"],
            gold_ents=[(0, 1, "PERSON", "Q1")],
            pred_ents=[(0, 1, "PERSON", "Q1")],
        ),
        _nel_example(
            nlp,
            ["c", "d"],
            gold_ents=[(0, 1, "ORG", "Q2")],
            pred_ents=[(0, 1, "ORG", "NIL")],
        ),
        _nel_example(
            nlp,
            ["e", "f"],
            gold_ents=[(0, 1, "ORG", "Q3")],
            pred_ents=[(0, 1, "ORG", "Q3")],
        ),
    ]
    full = Scorer.score_links(examples, negative_labels=["NIL"])
    states = [
        ScorerState(
            states={
                "nel": Scorer.score_links(
                    [eg], negative_labels=["NIL"], _state=True
                )
            },
            pipeline=["nel"],
        )
        for eg in examples
    ]
    merged = ScorerState.merge(states).finalize()
    assert merged["nel_micro_f"] == full["nel_micro_f"]
    assert merged["nel_macro_p"] == approx(full["nel_macro_p"])
    assert merged["nel_macro_r"] == approx(full["nel_macro_r"])
    assert merged["nel_macro_f"] == approx(full["nel_macro_f"])
    assert merged["nel_f_per_type"] == full["nel_f_per_type"]


def test_score_cats(en_tokenizer):
    text = "some text"
    gold_doc = en_tokenizer(text)
    gold_doc.cats = {"POSITIVE": 1.0, "NEGATIVE": 0.0}
    pred_doc = en_tokenizer(text)
    pred_doc.cats = {"POSITIVE": 0.75, "NEGATIVE": 0.25}
    example = Example(pred_doc, gold_doc)
    # threshold is ignored for multi_label=False
    scores1 = Scorer.score_cats(
        [example],
        "cats",
        labels=list(gold_doc.cats.keys()),
        multi_label=False,
        positive_label="POSITIVE",
        threshold=0.1,
    )
    scores2 = Scorer.score_cats(
        [example],
        "cats",
        labels=list(gold_doc.cats.keys()),
        multi_label=False,
        positive_label="POSITIVE",
        threshold=0.9,
    )
    assert scores1["cats_score"] == 1.0
    assert scores2["cats_score"] == 1.0
    assert scores1 == scores2
    # threshold is relevant for multi_label=True
    scores = Scorer.score_cats(
        [example],
        "cats",
        labels=list(gold_doc.cats.keys()),
        multi_label=True,
        threshold=0.9,
    )
    assert scores["cats_macro_f"] == 0.0
    # threshold is relevant for multi_label=True
    scores = Scorer.score_cats(
        [example],
        "cats",
        labels=list(gold_doc.cats.keys()),
        multi_label=True,
        threshold=0.1,
    )
    assert scores["cats_macro_f"] == 0.5
