import re
from pathlib import Path
from timeit import default_timer as timer
from typing import Any, Dict, List, Optional

import srsly
from thinc.api import fix_random_seed
from wasabi import Printer

from .. import displacy, util
from ..scorer import Scorer, ScorerState
from ..tokens import Doc
from ..training import Corpus
from ._util import Arg, Opt, app, benchmark_cli, import_code, setup_gpu


@benchmark_cli.command(
    "accuracy",
)
@app.command("evaluate")
def evaluate_cli(
    # fmt: off
    model: str = Arg(..., help="Model name or path"),
    data_path: Path = Arg(
        ..., help="Location of binary evaluation data in .spacy format", exists=True
    ),
    output: Optional[Path] = Opt(
        None, "--output", "-o", help="Output JSON file for metrics", dir_okay=False
    ),
    code_path: Optional[Path] = Opt(
        None,
        "--code",
        "-c",
        help="Path to Python file with additional code (registered functions) to be imported",
    ),
    use_gpu: int = Opt(-1, "--gpu-id", "-g", help="GPU ID or -1 for CPU"),
    gold_preproc: bool = Opt(
        False, "--gold-preproc", "-G", help="Use gold preprocessing"
    ),
    displacy_path: Optional[Path] = Opt(
        None,
        "--displacy-path",
        "-dp",
        help="Directory to output rendered parses as HTML",
        exists=True,
        file_okay=False,
    ),
    displacy_limit: int = Opt(
        25, "--displacy-limit", "-dl", help="Limit of parses to render as HTML"
    ),
    per_component: bool = Opt(
        False,
        "--per-component",
        "-P",
        help="Return scores per component, only applicable when an output JSON file is specified.",
    ),
    spans_key: str = Opt(
        "sc", "--spans-key", "-sk", help="Spans key to use when evaluating Doc.spans"
    ),
    shards: int = Opt(
        0,
        "--shards",
        help="Number of shards to split the evaluation data into. Each shard produces a mergeable evaluation state, which is then merged into the final result.",
    ),
    state_dir: Optional[Path] = Opt(
        None,
        "--state-dir",
        "-sd",
        help="If sharded evaluation is enabled, write the per-shard evaluation states to this directory.",
        exists=False,
        file_okay=False,
    ),
    # fmt: on
):
    """
    Evaluate a trained pipeline. Expects a loadable spaCy pipeline and evaluation
    data in the binary .spacy format. The --gold-preproc option sets up the
    evaluation examples with gold-standard sentences and tokens for the
    predictions. Gold preprocessing helps the annotations align to the
    tokenization, and may result in sequences of more consistent length. However,
    it may reduce runtime accuracy due to train/test skew. To render a sample of
    dependency parses in a HTML file, set as output directory as the
    displacy_path argument.

    DOCS: https://spacy.io/api/cli#benchmark-accuracy
    """
    import_code(code_path)
    evaluate(
        model,
        data_path,
        output=output,
        use_gpu=use_gpu,
        gold_preproc=gold_preproc,
        displacy_path=displacy_path,
        displacy_limit=displacy_limit,
        per_component=per_component,
        silent=False,
        spans_key=spans_key,
        shards=shards,
        state_dir=state_dir,
    )


def evaluate(
    model: str,
    data_path: Path,
    output: Optional[Path] = None,
    use_gpu: int = -1,
    gold_preproc: bool = False,
    displacy_path: Optional[Path] = None,
    displacy_limit: int = 25,
    silent: bool = True,
    spans_key: str = "sc",
    per_component: bool = False,
    shards: int = 0,
    state_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    msg = Printer(no_print=silent, pretty=not silent)
    fix_random_seed()
    setup_gpu(use_gpu, silent=silent)
    data_path = util.ensure_path(data_path)
    output_path = util.ensure_path(output)
    displacy_path = util.ensure_path(displacy_path)
    state_dir = util.ensure_path(state_dir)
    if not data_path.exists():
        msg.fail("Evaluation data not found", data_path, exits=1)
    if displacy_path and not displacy_path.exists():
        msg.fail("Visualization output directory not found", displacy_path, exits=1)
    corpus = Corpus(data_path, gold_preproc=gold_preproc)
    nlp = util.load_model(model)
    dev_dataset = list(corpus(nlp))
    if shards:
        if per_component:
            msg.fail(
                "The per-component option can't be combined with sharded evaluation.",
                exits=1,
            )
        scores = _evaluate_shards(
            nlp, dev_dataset, shards, state_dir, spans_key, msg
        )
        data = _build_metrics_data(scores, msg=msg, spans_key=spans_key)
    else:
        scores = nlp.evaluate(dev_dataset, per_component=per_component)
        if per_component:
            data = scores
            if output is None:
                msg.warn(
                    "The per-component option is enabled but there is no output JSON file provided to save the scores to."
                )
            else:
                msg.info("Per-component scores will be saved to output JSON file.")
        else:
            data = _build_metrics_data(scores, msg=msg, spans_key=spans_key)

    if displacy_path:
        factory_names = [nlp.get_pipe_meta(pipe).factory for pipe in nlp.pipe_names]
        docs = list(nlp.pipe(ex.reference.text for ex in dev_dataset[:displacy_limit]))
        render_deps = "parser" in factory_names
        render_ents = "ner" in factory_names
        render_spans = "spancat" in factory_names

        render_parses(
            docs,
            displacy_path,
            model_name=model,
            limit=displacy_limit,
            deps=render_deps,
            ents=render_ents,
            spans=render_spans,
        )
        msg.good(f"Generated {displacy_limit} parses as HTML", displacy_path)

    if output_path is not None:
        srsly.write_json(output_path, data)
        msg.good(f"Saved results to {output_path}")
    return data


def _evaluate_shards(
    nlp,
    dev_dataset,
    n_shards: int,
    state_dir: Optional[Path],
    spans_key: str,
    msg: Printer,
) -> Dict[str, Any]:
    if n_shards < 1:
        msg.fail("The number of shards needs to be at least 1.", exits=1)
    scorer = Scorer(nlp=nlp)
    shard_examples = _make_shards(dev_dataset, n_shards)
    # The timing mirrors Language.evaluate (including the tokenizer pass).
    start_time = timer()
    # this is purely for timing
    for eg in dev_dataset:
        nlp.make_doc(eg.reference.text)
    states = []
    for i, shard in enumerate(shard_examples):
        docs = nlp.pipe((eg.predicted for eg in shard), batch_size=nlp.batch_size)
        for eg, doc in zip(shard, docs):
            eg.predicted = doc
        state = scorer.accumulate(shard)
        if state_dir is not None:
            state_dir.mkdir(parents=True, exist_ok=True)
            state_path = state_dir / f"shard-{i:04d}.json"
            state.to_disk(state_path)
            msg.good(f"Saved evaluation state to {state_path}")
        states.append(state)
    end_time = timer()
    merged = ScorerState.merge(states)
    scores = merged.finalize()
    n_words = sum(len(eg.predicted) for eg in dev_dataset)
    scores["speed"] = n_words / (end_time - start_time)
    return scores


def _make_shards(examples: List[Any], n_shards: int) -> List[List[Any]]:
    """Split examples into shards deterministically. Shard sizes differ by
    at most one example and if n_shards exceeds the number of examples, the
    remaining shards are empty."""
    n = len(examples)
    base, extra = divmod(n, n_shards)
    shards: List[List[Any]] = []
    start = 0
    for i in range(n_shards):
        size = base + (1 if i < extra else 0)
        shards.append(examples[start : start + size])
        start += size
    return shards


def _build_metrics_data(
    scores: Dict[str, Any], *, msg: Printer, spans_key: str = "sc"
) -> Dict[str, Any]:
    metrics = {
        "TOK": "token_acc",
        "TAG": "tag_acc",
        "POS": "pos_acc",
        "MORPH": "morph_acc",
        "LEMMA": "lemma_acc",
        "UAS": "dep_uas",
        "LAS": "dep_las",
        "NER P": "ents_p",
        "NER R": "ents_r",
        "NER F": "ents_f",
        "TEXTCAT": "cats_score",
        "SENT P": "sents_p",
        "SENT R": "sents_r",
        "SENT F": "sents_f",
        "SPAN P": f"spans_{spans_key}_p",
        "SPAN R": f"spans_{spans_key}_r",
        "SPAN F": f"spans_{spans_key}_f",
        "SPEED": "speed",
    }
    results = {}
    data = {}
    for metric, key in metrics.items():
        if key in scores:
            if key == "cats_score":
                metric = metric + " (" + scores.get("cats_score_desc", "unk") + ")"
            if isinstance(scores[key], (int, float)):
                if key == "speed":
                    results[metric] = f"{scores[key]:.0f}"
                else:
                    results[metric] = f"{scores[key] * 100:.2f}"
            else:
                results[metric] = "-"
            data[re.sub(r"[\s/]", "_", key.lower())] = scores[key]

    msg.table(results, title="Results")
    data = handle_scores_per_type(scores, data, spans_key=spans_key, msg=msg)
    return data


@benchmark_cli.command("merge-eval-states")
def merge_eval_states_cli(
    # fmt: off
    state_files: List[Path] = Arg(
        ..., help="Paths to evaluation state files to merge", exists=True
    ),
    output: Optional[Path] = Opt(
        None, "--output", "-o", help="Output JSON file for metrics", dir_okay=False
    ),
    spans_key: str = Opt(
        "sc", "--spans-key", "-sk", help="Spans key to use when reporting spans"
    ),
    # fmt: on
):
    """
    Merge evaluation states produced by sharded `spacy evaluate` runs (using
    `--state-dir`) and report the final metrics. The states don't need a model
    and can be produced on different machines.
    """
    msg = Printer(pretty=True)
    states = [ScorerState.from_disk(state_file) for state_file in state_files]
    merged = ScorerState.merge(states)
    scores = merged.finalize()
    data = _build_metrics_data(scores, msg=msg, spans_key=spans_key)
    output_path = util.ensure_path(output)
    if output_path is not None:
        srsly.write_json(output_path, data)
        msg.good(f"Saved results to {output_path}")


def handle_scores_per_type(
    scores: Dict[str, Any],
    data: Dict[str, Any] = {},
    *,
    spans_key: str = "sc",
    msg: Optional[Printer] = None,
    silent: bool = False,
) -> Dict[str, Any]:
    if msg is None:
        msg = Printer(no_print=silent, pretty=not silent)
    if "morph_per_feat" in scores:
        if scores["morph_per_feat"]:
            print_prf_per_type(msg, scores["morph_per_feat"], "MORPH", "feat")
            data["morph_per_feat"] = scores["morph_per_feat"]
    if "dep_las_per_type" in scores:
        if scores["dep_las_per_type"]:
            print_prf_per_type(msg, scores["dep_las_per_type"], "LAS", "type")
            data["dep_las_per_type"] = scores["dep_las_per_type"]
    if "ents_per_type" in scores:
        if scores["ents_per_type"]:
            print_prf_per_type(msg, scores["ents_per_type"], "NER", "type")
            data["ents_per_type"] = scores["ents_per_type"]
    if f"spans_{spans_key}_per_type" in scores:
        if scores[f"spans_{spans_key}_per_type"]:
            print_prf_per_type(
                msg, scores[f"spans_{spans_key}_per_type"], "SPANS", "type"
            )
            data[f"spans_{spans_key}_per_type"] = scores[f"spans_{spans_key}_per_type"]
    if "cats_f_per_type" in scores:
        if scores["cats_f_per_type"]:
            print_prf_per_type(msg, scores["cats_f_per_type"], "Textcat F", "label")
            data["cats_f_per_type"] = scores["cats_f_per_type"]
    if "cats_auc_per_type" in scores:
        if scores["cats_auc_per_type"]:
            print_textcats_auc_per_cat(msg, scores["cats_auc_per_type"])
            data["cats_auc_per_type"] = scores["cats_auc_per_type"]
    return scores


def render_parses(
    docs: List[Doc],
    output_path: Path,
    model_name: str = "",
    limit: int = 250,
    deps: bool = True,
    ents: bool = True,
    spans: bool = True,
):
    docs[0].user_data["title"] = model_name
    if ents:
        html = displacy.render(docs[:limit], style="ent", page=True)
        with (output_path / "entities.html").open("w", encoding="utf8") as file_:
            file_.write(html)
    if deps:
        html = displacy.render(
            docs[:limit], style="dep", page=True, options={"compact": True}
        )
        with (output_path / "parses.html").open("w", encoding="utf8") as file_:
            file_.write(html)

    if spans:
        html = displacy.render(docs[:limit], style="span", page=True)
        with (output_path / "spans.html").open("w", encoding="utf8") as file_:
            file_.write(html)


def print_prf_per_type(
    msg: Printer, scores: Dict[str, Dict[str, float]], name: str, type: str
) -> None:
    data = []
    for key, value in scores.items():
        row = [key]
        for k in ("p", "r", "f"):
            v = value[k]
            row.append(f"{v * 100:.2f}" if isinstance(v, (int, float)) else v)
        data.append(row)
    msg.table(
        data,
        header=("", "P", "R", "F"),
        aligns=("l", "r", "r", "r"),
        title=f"{name} (per {type})",
    )


def print_textcats_auc_per_cat(
    msg: Printer, scores: Dict[str, Dict[str, float]]
) -> None:
    msg.table(
        [
            (k, f"{v:.2f}" if isinstance(v, (float, int)) else v)
            for k, v in scores.items()
        ],
        header=("", "ROC AUC"),
        aligns=("l", "r"),
        title="Textcat ROC AUC (per label)",
    )
