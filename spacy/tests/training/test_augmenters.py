import random
from contextlib import contextmanager
from functools import partial

import pytest
from thinc.api import Config

from spacy.lang.en import English
from spacy.pipeline._parser_internals.nonproj import contains_cycle
from spacy.tokens import Doc, DocBin, Span
from spacy.training import (
    AugmentingScheduler,
    Corpus,
    Example,
    create_scheduled_augmenter,
    validate_variant,
)
from spacy.training.augment import (
    create_lower_casing_augmenter,
    create_orth_variants_augmenter,
    get_augmenting_rng,
    make_lowercase_variant,
    make_whitespace_variant,
)
from spacy.training.batchers import minibatch_by_words
from spacy.training.loop import create_train_batches
from spacy.util import registry

from ..util import make_tempdir


@contextmanager
def make_docbin(docs, name="roundtrip.spacy"):
    with make_tempdir() as tmpdir:
        output_file = tmpdir / name
        DocBin(docs=docs).to_disk(output_file)
        yield output_file


@pytest.fixture
def nlp():
    return English()


@pytest.fixture
def doc(nlp):
    # fmt: off
    words = ["Sarah", "'s", "sister", "flew", "to", "Silicon", "Valley", "via", "London", "."]
    tags = ["NNP", "POS", "NN", "VBD", "IN", "NNP", "NNP", "IN", "NNP", "."]
    pos = ["PROPN", "PART", "NOUN", "VERB", "ADP", "PROPN", "PROPN", "ADP", "PROPN", "PUNCT"]
    ents = ["B-PERSON", "I-PERSON", "O", "", "O", "B-LOC", "I-LOC", "O", "B-GPE", "O"]
    cats = {"TRAVEL": 1.0, "BAKING": 0.0}
    # fmt: on
    doc = Doc(nlp.vocab, words=words, tags=tags, pos=pos, ents=ents)
    doc.cats = cats
    return doc


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_make_orth_variants(nlp):
    single = [
        {"tags": ["NFP"], "variants": ["…", "..."]},
        {"tags": [":"], "variants": ["-", "—", "–", "--", "---", "——"]},
    ]
    # fmt: off
    words = ["\n\n", "A", "\t", "B", "a", "b", "…", "...", "-", "—", "–", "--", "---", "——"]
    tags = ["_SP", "NN", "\t", "NN", "NN", "NN", "NFP", "NFP", ":", ":", ":", ":", ":", ":"]
    # fmt: on
    spaces = [True] * len(words)
    spaces[0] = False
    spaces[2] = False
    doc = Doc(nlp.vocab, words=words, spaces=spaces, tags=tags)
    augmenter = create_orth_variants_augmenter(
        level=0.2, lower=0.5, orth_variants={"single": single}
    )
    with make_docbin([doc] * 10) as output_file:
        reader = Corpus(output_file, augmenter=augmenter)
        # Due to randomness, only test that it works without errors
        list(reader(nlp))

    # check that the following settings lowercase everything
    augmenter = create_orth_variants_augmenter(
        level=1.0, lower=1.0, orth_variants={"single": single}
    )
    with make_docbin([doc] * 10) as output_file:
        reader = Corpus(output_file, augmenter=augmenter)
        for example in reader(nlp):
            for token in example.reference:
                assert token.text == token.text.lower()

    # check that lowercasing is applied without tags
    doc = Doc(nlp.vocab, words=words, spaces=[True] * len(words))
    augmenter = create_orth_variants_augmenter(
        level=1.0, lower=1.0, orth_variants={"single": single}
    )
    with make_docbin([doc] * 10) as output_file:
        reader = Corpus(output_file, augmenter=augmenter)
        for example in reader(nlp):
            for ex_token, doc_token in zip(example.reference, doc):
                assert ex_token.text == doc_token.text.lower()

    # check that no lowercasing is applied with lower=0.0
    doc = Doc(nlp.vocab, words=words, spaces=[True] * len(words))
    augmenter = create_orth_variants_augmenter(
        level=1.0, lower=0.0, orth_variants={"single": single}
    )
    with make_docbin([doc] * 10) as output_file:
        reader = Corpus(output_file, augmenter=augmenter)
        for example in reader(nlp):
            for ex_token, doc_token in zip(example.reference, doc):
                assert ex_token.text == doc_token.text


def test_lowercase_augmenter(nlp, doc):
    augmenter = create_lower_casing_augmenter(level=1.0)
    with make_docbin([doc]) as output_file:
        reader = Corpus(output_file, augmenter=augmenter)
        corpus = list(reader(nlp))
    eg = corpus[0]
    assert eg.reference.text == doc.text.lower()
    assert eg.predicted.text == doc.text.lower()
    ents = [(e.start, e.end, e.label) for e in doc.ents]
    assert [(e.start, e.end, e.label) for e in eg.reference.ents] == ents
    for ref_ent, orig_ent in zip(eg.reference.ents, doc.ents):
        assert ref_ent.text == orig_ent.text.lower()
    assert [t.ent_iob for t in doc] == [t.ent_iob for t in eg.reference]
    assert [t.pos_ for t in eg.reference] == [t.pos_ for t in doc]

    # check that augmentation works when lowercasing leads to different
    # predicted tokenization
    words = ["A", "B", "CCC."]
    doc = Doc(nlp.vocab, words=words)
    with make_docbin([doc]) as output_file:
        reader = Corpus(output_file, augmenter=augmenter)
        corpus = list(reader(nlp))
    eg = corpus[0]
    assert eg.reference.text == doc.text.lower()
    assert eg.predicted.text == doc.text.lower()
    assert [t.text for t in eg.reference] == [t.lower() for t in words]
    assert [t.text for t in eg.predicted] == [
        t.text for t in nlp.make_doc(doc.text.lower())
    ]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_custom_data_augmentation(nlp, doc):
    def create_spongebob_augmenter(randomize: bool = False):
        def augment(nlp, example):
            text = example.text
            if randomize:
                ch = [c.lower() if random.random() < 0.5 else c.upper() for c in text]
            else:
                ch = [c.lower() if i % 2 else c.upper() for i, c in enumerate(text)]
            example_dict = example.to_dict()
            doc = nlp.make_doc("".join(ch))
            example_dict["token_annotation"]["ORTH"] = [t.text for t in doc]
            yield example
            yield example.from_dict(doc, example_dict)

        return augment

    with make_docbin([doc]) as output_file:
        reader = Corpus(output_file, augmenter=create_spongebob_augmenter())
        corpus = list(reader(nlp))
    orig_text = "Sarah 's sister flew to Silicon Valley via London . "
    augmented = "SaRaH 's sIsTeR FlEw tO SiLiCoN VaLlEy vIa lOnDoN . "
    assert corpus[0].text == orig_text
    assert corpus[0].reference.text == orig_text
    assert corpus[0].predicted.text == orig_text
    assert corpus[1].text == augmented
    assert corpus[1].reference.text == augmented
    assert corpus[1].predicted.text == augmented
    ents = [(e.start, e.end, e.label) for e in doc.ents]
    assert [(e.start, e.end, e.label) for e in corpus[0].reference.ents] == ents
    assert [(e.start, e.end, e.label) for e in corpus[1].reference.ents] == ents


def test_make_whitespace_variant(nlp):
    # fmt: off
    text = "They flew to New York City.\nThen they drove to Washington, D.C."
    words = ["They", "flew", "to", "New", "York", "City", ".", "\n", "Then", "they", "drove", "to", "Washington", ",", "D.C."]
    spaces = [True, True, True, True, True, False, False, False, True, True, True, True, False, True, False]
    tags = ["PRP", "VBD", "IN", "NNP", "NNP", "NNP", ".", "_SP", "RB", "PRP", "VBD", "IN", "NNP", ",", "NNP"]
    lemmas = ["they", "fly", "to", "New", "York", "City", ".", "\n", "then", "they", "drive", "to", "Washington", ",", "D.C."]
    heads = [1, 1, 1, 4, 5, 2, 1, 10, 10, 10, 10, 10, 11, 12, 12]
    deps = ["nsubj", "ROOT", "prep", "compound", "compound", "pobj", "punct", "dep", "advmod", "nsubj", "ROOT", "prep", "pobj", "punct", "appos"]
    ents = ["O", "", "O", "B-GPE", "I-GPE", "I-GPE", "O", "O", "O", "O", "O", "O", "B-GPE", "O", "B-GPE"]
    # fmt: on
    doc = Doc(
        nlp.vocab,
        words=words,
        spaces=spaces,
        tags=tags,
        lemmas=lemmas,
        heads=heads,
        deps=deps,
        ents=ents,
    )
    assert doc.text == text
    example = Example(nlp.make_doc(text), doc)
    # whitespace is only added internally in entity spans
    mod_ex = make_whitespace_variant(nlp, example, " ", 3)
    assert mod_ex.reference.ents[0].text == "New York City"
    mod_ex = make_whitespace_variant(nlp, example, " ", 4)
    assert mod_ex.reference.ents[0].text == "New  York City"
    mod_ex = make_whitespace_variant(nlp, example, " ", 5)
    assert mod_ex.reference.ents[0].text == "New York  City"
    mod_ex = make_whitespace_variant(nlp, example, " ", 6)
    assert mod_ex.reference.ents[0].text == "New York City"
    # add a space at every possible position
    for i in range(len(doc) + 1):
        mod_ex = make_whitespace_variant(nlp, example, " ", i)
        assert mod_ex.reference[i].is_space
        # adds annotation when the doc contains at least partial annotation
        assert [t.tag_ for t in mod_ex.reference] == tags[:i] + ["_SP"] + tags[i:]
        assert [t.lemma_ for t in mod_ex.reference] == lemmas[:i] + [" "] + lemmas[i:]
        assert [t.dep_ for t in mod_ex.reference] == deps[:i] + ["dep"] + deps[i:]
        # does not add partial annotation if doc does not contain this feature
        assert not mod_ex.reference.has_annotation("POS")
        assert not mod_ex.reference.has_annotation("MORPH")
        # produces well-formed trees
        assert not contains_cycle([t.head.i for t in mod_ex.reference])
        assert len(list(doc.sents)) == 2
        if i == 0:
            assert mod_ex.reference[i].head.i == 1
        else:
            assert mod_ex.reference[i].head.i == i - 1
        # adding another space also produces well-formed trees
        for j in (3, 8, 10):
            mod_ex2 = make_whitespace_variant(nlp, mod_ex, "\t\t\n", j)
            assert not contains_cycle([t.head.i for t in mod_ex2.reference])
            assert len(list(doc.sents)) == 2
            assert mod_ex2.reference[j].head.i == j - 1
        # entities are well-formed
        assert len(doc.ents) == len(mod_ex.reference.ents)
        # there is one token with missing entity information
        assert any(t.ent_iob == 0 for t in mod_ex.reference)
        for ent in mod_ex.reference.ents:
            assert not ent[0].is_space
            assert not ent[-1].is_space

    # no modifications if:
    # partial dependencies
    example.reference[0].dep_ = ""
    mod_ex = make_whitespace_variant(nlp, example, " ", 5)
    assert mod_ex.text == example.reference.text
    example.reference[0].dep_ = "nsubj"  # reset

    # spans
    example.reference.spans["spans"] = [example.reference[0:5]]
    mod_ex = make_whitespace_variant(nlp, example, " ", 5)
    assert mod_ex.text == example.reference.text
    del example.reference.spans["spans"]  # reset

    # links
    example.reference.ents = [Span(doc, 0, 2, label="ENT", kb_id="Q123")]
    mod_ex = make_whitespace_variant(nlp, example, " ", 5)
    assert mod_ex.text == example.reference.text


# ---------------------------------------------------------------------------
# Scheduled augmentation
# ---------------------------------------------------------------------------


def make_plain_example(nlp, text):
    doc = nlp.make_doc(text)
    return Example(doc, doc.copy())


def make_dep_example(nlp):
    # fmt: off
    words = ["She", "runs", "fast", "."]
    heads = [1, 1, 1, 1]
    deps = ["nsubj", "ROOT", "advmod", "punct"]
    # fmt: on
    doc = Doc(nlp.vocab, words=words, heads=heads, deps=deps)
    return Example(nlp.make_doc(doc.text), doc)


def random_choice_inner(nlp, example):
    rng = get_augmenting_rng()
    if rng.choice(["keep", "lower"]) == "keep":
        yield example
    else:
        yield make_lowercase_variant(nlp, example)


def random_whitespace_inner(nlp, example):
    rng = get_augmenting_rng()
    position = rng.randrange(len(example.reference) + 1)
    yield make_whitespace_variant(nlp, example, " ", position)


def run_scheduler_epoch(scheduler, nlp, examples, epoch):
    return list(
        scheduler.run_epoch(
            nlp, list(enumerate(examples)), epoch=epoch, n_inputs=len(examples)
        )
    )


def test_scheduler_quota_exact_counts_and_passthrough(nlp, doc):
    examples = [Example(nlp.make_doc(doc.text), doc.copy()) for _ in range(10)]
    inner = create_lower_casing_augmenter(level=1.0)
    scheduler = create_scheduled_augmenter(inner=inner, quota=3, seed=11)
    assert scheduler.enabled
    outputs = run_scheduler_epoch(scheduler, nlp, examples, 0)
    stats = scheduler.finish_epoch()
    assert stats["inputs"] == 10
    assert stats["quota"] == 3
    assert stats["augmented"] == 3
    assert stats["quota_remaining"] == 0
    assert stats["passed"] == 7
    assert stats["variants"] == 3
    assert stats["skipped_unaligned"] == 0
    assert stats["skipped_empty"] == 0
    assert stats["failed"] == 0
    # Output order follows the input order; non-selected inputs are the
    # original objects, selected ones are lowercased.
    n_lower = 0
    for example, output in zip(examples, outputs):
        if output is example:
            assert output.text == doc.text
        else:
            n_lower += 1
            assert output.text == doc.text.lower()
    assert n_lower == 3
    # Per-epoch reset: the next epoch gets a fresh quota.
    outputs = run_scheduler_epoch(scheduler, nlp, examples, 1)
    stats = scheduler.finish_epoch()
    assert stats["inputs"] == 10
    assert stats["augmented"] == 3
    assert stats["quota_remaining"] == 0
    assert stats["passed"] == 7
    assert len(scheduler.history) == 2
    assert scheduler.last_epoch is stats


def test_scheduler_quota_larger_than_inputs(nlp):
    examples = [make_plain_example(nlp, f"Sentence number {i} here.") for i in range(4)]
    scheduler = create_scheduled_augmenter(
        inner=create_lower_casing_augmenter(level=1.0), quota=10, seed=1
    )
    outputs = run_scheduler_epoch(scheduler, nlp, examples, 0)
    stats = scheduler.finish_epoch()
    assert stats["augmented"] == 4
    assert stats["passed"] == 0
    assert stats["quota_remaining"] == 6
    assert all(out.text.islower() for out in outputs)


def test_scheduler_ratio_exact(nlp):
    examples = [
        make_plain_example(nlp, f"Number {i} sentence here.") for i in range(10)
    ]
    for ratio in (0.5, 0.25, 0.3):
        scheduler = create_scheduled_augmenter(
            inner=create_lower_casing_augmenter(level=1.0), ratio=ratio, seed=3
        )
        run_scheduler_epoch(scheduler, nlp, examples, 0)
        stats = scheduler.finish_epoch()
        assert stats["augmented"] == int(round(ratio * len(examples)))
        assert stats["augmented"] + stats["passed"] == stats["inputs"]


def test_scheduler_disabled_is_identity_without_random_draws(nlp, doc):
    example = Example(nlp.make_doc(doc.text), doc.copy())
    scheduler = create_scheduled_augmenter(
        inner=create_lower_casing_augmenter(level=1.0), quota=0
    )
    assert not scheduler.enabled
    state = random.getstate()
    outputs = list(scheduler(nlp, example))
    assert outputs == [example]
    assert random.getstate() == state
    # run_epoch is a passthrough as well.
    examples = [example.copy() for _ in range(3)]
    outputs = run_scheduler_epoch(scheduler, nlp, examples, 0)
    assert outputs == examples
    assert random.getstate() == state


def test_scheduler_reproducible_and_independent_of_global_rng(nlp):
    examples = [
        make_plain_example(nlp, f"Independent sentence number {i} ok.")
        for i in range(8)
    ]

    def run():
        scheduler = create_scheduled_augmenter(
            inner=random_choice_inner, quota=5, seed=9
        )
        return [out.text for out in run_scheduler_epoch(scheduler, nlp, examples, 0)]

    random.seed(123)
    first = run()
    # Consume plenty of global randomness, then rerun: results must be
    # item-by-item identical because the scheduler never uses the global RNG.
    [random.random() for _ in range(1000)]
    [random.getrandbits(64) for _ in range(100)]
    second = run()
    assert first == second


def test_scheduler_streams_vary_by_epoch_but_repeat_across_runs(nlp):
    examples = [
        make_plain_example(
            nlp, f"Epoch independent sentence number {i} with extra tokens."
        )
        for i in range(8)
    ]
    epoch_texts = {}
    for epoch in range(3):
        scheduler = create_scheduled_augmenter(
            inner=random_whitespace_inner, quota=6, seed=5
        )
        texts = [
            out.text for out in run_scheduler_epoch(scheduler, nlp, examples, epoch)
        ]
        epoch_texts[epoch] = texts
    # Same epoch rerun reproduces exactly.
    scheduler = create_scheduled_augmenter(
        inner=random_whitespace_inner, quota=6, seed=5
    )
    assert [
        out.text for out in run_scheduler_epoch(scheduler, nlp, examples, 2)
    ] == epoch_texts[2]
    # Epoch mixing produces different variants (derived streams include the
    # epoch number).
    assert epoch_texts[0] != epoch_texts[1]
    assert epoch_texts[1] != epoch_texts[2]


def test_scheduler_parallel_iterators_do_not_cross_talk(nlp):
    examples_a = [
        make_plain_example(nlp, f"Shard alpha sentence number {i} now.")
        for i in range(8)
    ]
    examples_b = [
        make_plain_example(nlp, f"Shard beta sentence number {i} now.")
        for i in range(8)
    ]

    def isolated(examples):
        scheduler = create_scheduled_augmenter(
            inner=random_choice_inner, quota=5, seed=17
        )
        return [out.text for out in run_scheduler_epoch(scheduler, nlp, examples, 0)]

    isolated_a = isolated(examples_a)
    isolated_b = isolated(examples_b)

    # Now advance two epochs interleaved; per-item derived streams must not
    # interfere with each other.
    sched_a = create_scheduled_augmenter(inner=random_choice_inner, quota=5, seed=17)
    sched_b = create_scheduled_augmenter(inner=random_choice_inner, quota=5, seed=17)
    gen_a = sched_a.run_epoch(nlp, list(enumerate(examples_a)), epoch=0, n_inputs=8)
    gen_b = sched_b.run_epoch(nlp, list(enumerate(examples_b)), epoch=0, n_inputs=8)
    interleaved_a, interleaved_b = [], []
    for _ in range(8):
        interleaved_a.append(next(gen_a).text)
        interleaved_b.append(next(gen_b).text)
    assert interleaved_a == isolated_a
    assert interleaved_b == isolated_b
    # Different shard ids give independent streams.
    sched_0 = create_scheduled_augmenter(
        inner=random_whitespace_inner, quota=8, seed=17, shard=0
    )
    sched_1 = create_scheduled_augmenter(
        inner=random_whitespace_inner, quota=8, seed=17, shard=1
    )
    texts_0 = [out.text for out in run_scheduler_epoch(sched_0, nlp, examples_a, 0)]
    texts_1 = [out.text for out in run_scheduler_epoch(sched_1, nlp, examples_a, 0)]
    assert texts_0 != texts_1


def test_scheduler_multi_variant_outputs_respect_batch_budget(nlp):
    def two_variants(nlp, example):
        yield make_lowercase_variant(nlp, example)
        yield make_whitespace_variant(nlp, example, " ", 0)

    examples = [make_plain_example(nlp, " ".join(["word"] * 10)) for _ in range(4)]
    scheduler = create_scheduled_augmenter(
        inner=two_variants, quota=4, variants=1, seed=2
    )
    outputs = run_scheduler_epoch(scheduler, nlp, examples, 0)
    stats = scheduler.finish_epoch()
    # Every one of the 4 inputs produced 2 variants -> 8 items in the epoch.
    assert stats["inputs"] == 4
    assert stats["augmented"] == 4
    assert stats["variants"] == 8
    assert len(outputs) == 8
    # The expanded stream is fed to the batcher, which keeps the word budget.
    batcher = partial(minibatch_by_words, size=40, tolerance=0.25)
    for batch in batcher(iter(outputs)):
        assert sum(len(eg) for eg in batch) <= 40 * 1.25

    # Multiple independent variant sets per selected input.
    scheduler = create_scheduled_augmenter(
        inner=random_choice_inner, quota=4, variants=3, seed=2
    )
    outputs = run_scheduler_epoch(scheduler, nlp, examples, 0)
    stats = scheduler.finish_epoch()
    assert stats["augmented"] == 4
    assert 4 <= len(outputs) <= 12


def test_scheduler_skips_unaligned_entity_variant(nlp, doc):
    def drop_entity_inner(nlp, example):
        variant = make_lowercase_variant(nlp, example)
        variant.reference.ents = variant.reference.ents[1:]
        yield variant

    examples = [Example(nlp.make_doc(doc.text), doc.copy()) for _ in range(4)]
    # The standard lowercase variant is itself aligned.
    good = make_lowercase_variant(nlp, examples[0])
    assert validate_variant(examples[0], good) == []
    scheduler = create_scheduled_augmenter(inner=drop_entity_inner, quota=4, seed=4)
    outputs = run_scheduler_epoch(scheduler, nlp, examples, 0)
    stats = scheduler.finish_epoch()
    assert stats["variants"] == 0
    assert stats["skipped_unaligned"] == 4
    assert outputs == []
    # Failure policy "pass" keeps the original inputs instead.
    scheduler = create_scheduled_augmenter(
        inner=drop_entity_inner, quota=4, seed=4, on_failure="pass"
    )
    outputs = run_scheduler_epoch(scheduler, nlp, examples, 0)
    stats = scheduler.finish_epoch()
    assert stats["skipped_unaligned"] == 4
    assert stats["passed"] == 4
    assert outputs == examples


def test_scheduler_skips_unaligned_span_variant(nlp, doc):
    doc = doc.copy()
    doc.spans["group"] = [doc[0:2], doc[5:7]]
    example = Example(nlp.make_doc(doc.text), doc)

    def drop_spans_inner(nlp, example):
        variant = example.copy()
        variant.reference.spans["group"] = []
        yield variant

    scheduler = create_scheduled_augmenter(inner=drop_spans_inner, quota=1, seed=4)
    outputs = run_scheduler_epoch(scheduler, nlp, [example], 0)
    stats = scheduler.finish_epoch()
    assert outputs == []
    assert stats["skipped_unaligned"] == 1


def test_scheduler_skips_cyclic_dependency_variant(nlp):
    example = make_dep_example(nlp)

    def cyclic_dep_inner(nlp, example):
        variant = example.copy()
        variant.reference[0].head = variant.reference[2]
        variant.reference[2].head = variant.reference[0]
        yield variant

    scheduler = create_scheduled_augmenter(inner=cyclic_dep_inner, quota=1, seed=4)
    outputs = run_scheduler_epoch(scheduler, nlp, [example], 0)
    stats = scheduler.finish_epoch()
    assert outputs == []
    assert stats["skipped_unaligned"] == 1
    # The dependency check can be disabled via the align config.
    scheduler = create_scheduled_augmenter(
        inner=cyclic_dep_inner, quota=1, seed=4, align=["entities", "spans"]
    )
    outputs = run_scheduler_epoch(scheduler, nlp, [example], 0)
    stats = scheduler.finish_epoch()
    assert stats["variants"] == 1
    assert len(outputs) == 1


def test_scheduler_counts_failures_and_empty_outputs(nlp):
    def failing_inner(nlp, example):
        raise RuntimeError("boom")
        yield  # pragma: no cover

    def empty_inner(nlp, example):
        return
        yield  # makes this a generator

    examples = [
        make_plain_example(nlp, f"Failure sentence number {i} ok.") for i in range(2)
    ]
    scheduler = create_scheduled_augmenter(inner=failing_inner, quota=2, seed=4)
    assert run_scheduler_epoch(scheduler, nlp, examples, 0) == []
    stats = scheduler.finish_epoch()
    assert stats["failed"] == 2
    assert stats["variants"] == 0
    scheduler = create_scheduled_augmenter(inner=empty_inner, quota=2, seed=4)
    assert run_scheduler_epoch(scheduler, nlp, examples, 0) == []
    stats = scheduler.finish_epoch()
    assert stats["skipped_empty"] == 2
    # on_failure="pass" falls back to the originals.
    scheduler = create_scheduled_augmenter(
        inner=failing_inner, quota=2, seed=4, on_failure="pass"
    )
    assert run_scheduler_epoch(scheduler, nlp, examples, 0) == examples


def test_scheduler_streaming_mode(nlp):
    examples = [
        make_plain_example(nlp, f"Streaming sentence number {i} ok.") for i in range(6)
    ]

    def stream(quota=None, ratio=0.0, seed=8):
        scheduler = create_scheduled_augmenter(
            inner=create_lower_casing_augmenter(level=1.0),
            quota=quota,
            ratio=ratio,
            seed=seed,
        )
        # No n_inputs: streaming epoch.
        return list(scheduler.run_epoch(nlp, (eg.copy() for eg in examples), epoch=0))

    outputs = stream(quota=2)
    assert len(outputs) == 6
    assert sum(1 for out in outputs if out.text.islower()) == 2
    # First items are augmented until the quota is used.
    assert outputs[0].text.islower() and outputs[1].text.islower()
    assert not outputs[2].text.islower()
    # Ratio streaming is deterministic.
    first = stream(ratio=0.5)
    second = stream(ratio=0.5)
    assert [out.text for out in first] == [out.text for out in second]
    assert 0 < sum(1 for out in first if out.text.islower()) < 6


def test_scheduler_invalid_config(nlp):
    with pytest.raises(ValueError):
        create_scheduled_augmenter(quota=-1)
    with pytest.raises(ValueError):
        create_scheduled_augmenter(ratio=1.5)
    with pytest.raises(ValueError):
        create_scheduled_augmenter(quota=3, ratio=0.5)
    with pytest.raises(ValueError):
        create_scheduled_augmenter(quota=1, variants=0)
    with pytest.raises(ValueError):
        create_scheduled_augmenter(quota=1, on_failure="nope")
    with pytest.raises(ValueError):
        create_scheduled_augmenter(quota=1, align=["entities", "bogus"])


def test_scheduler_config_registry_resolution():
    config_str = """
    [corpus]
    @readers = "spacy.Corpus.v1"
    path = "/does/not/matter"
    gold_preproc = false
    max_length = 0
    limit = 0

    [corpus.augmenter]
    @augmenters = "spacy.augmenting_scheduler.v1"
    quota = 3
    ratio = 0.0
    variants = 2
    seed = 7

    [corpus.augmenter.inner]
    @augmenters = "spacy.lower_case.v1"
    level = 1.0
    """
    config = Config().from_str(config_str)
    resolved = registry.resolve(config)
    corpus = resolved["corpus"]
    assert isinstance(corpus, Corpus)
    assert isinstance(corpus.augmenter, AugmentingScheduler)
    assert corpus.augmenter.quota == 3
    assert corpus.augmenter.variants == 2
    assert corpus.augmenter._seed == 7
    assert corpus.augmenter.enabled


def test_loop_disabled_scheduler_matches_legacy_path(nlp, doc):
    docs = [doc.copy() for _ in range(6)]
    batcher = partial(minibatch_by_words, size=50, tolerance=0.2)

    def collect(corpus):
        random.seed(321)
        result = []
        for epoch, batch in create_train_batches(nlp, corpus, batcher, max_epochs=2):
            result.append((epoch, [eg.reference.text for eg in batch]))
        return result

    with make_docbin(docs) as path:
        legacy = Corpus(path)
        scheduled = Corpus(path, augmenter=create_scheduled_augmenter(quota=0))
        assert collect(legacy) == collect(scheduled)


def test_loop_scheduled_streaming_epoch(nlp, doc):
    # max_epochs=-1: the corpus is streamed and re-read per epoch.
    docs = [doc.copy() for _ in range(6)]
    scheduler = create_scheduled_augmenter(
        inner=create_lower_casing_augmenter(level=1.0), quota=2, seed=3
    )
    batcher = partial(minibatch_by_words, size=50, tolerance=0.2)
    with make_docbin(docs) as path:
        corpus = Corpus(path, augmenter=scheduler)
        gen = create_train_batches(nlp, corpus, batcher, max_epochs=-1)
        epoch_texts = []
        current_epoch = 0
        while True:
            epoch, batch = next(gen)
            if epoch != current_epoch:
                break
            epoch_texts.extend(eg.reference.text for eg in batch)
        gen.close()
    assert len(epoch_texts) == 6
    assert sum(1 for text in epoch_texts if text == doc.text.lower()) == 2
    stats = scheduler.history[0]
    assert stats["inputs"] == 6
    assert stats["augmented"] == 2
    assert stats["quota_remaining"] == 0
    assert stats["passed"] == 4


def test_loop_scheduled_epoch_stats(nlp, doc):
    docs = [doc.copy() for _ in range(6)]
    scheduler = create_scheduled_augmenter(
        inner=create_lower_casing_augmenter(level=1.0), quota=2, seed=12
    )
    batcher = partial(minibatch_by_words, size=50, tolerance=0.2)
    with make_docbin(docs) as path:
        corpus = Corpus(path, augmenter=scheduler)
        epochs = []
        all_texts = {0: [], 1: []}
        for epoch, batch in create_train_batches(nlp, corpus, batcher, max_epochs=2):
            all_texts[epoch].extend(eg.reference.text for eg in batch)
            epochs.append(epoch)
        assert epochs[:1] == [0]
        assert len(scheduler.history) == 2
        for stats in scheduler.history:
            assert stats["inputs"] == 6
            assert stats["augmented"] == 2
            assert stats["passed"] == 4
            assert stats["variants"] == 2
            assert stats["quota_remaining"] == 0
            assert stats["failed"] == 0
        # One stream per epoch, 6 items each (1:1 replacement), 2 lowercased.
        for epoch in (0, 1):
            assert len(all_texts[epoch]) == 6
            assert sum(1 for text in all_texts[epoch] if text == doc.text.lower()) == 2
