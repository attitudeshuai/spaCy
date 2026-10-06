"""Tests for the transactional, manifest-validated artifact storage used by
spacy.util.to_disk / spacy.util.from_disk (and therefore Language.to_disk).

Covers:
- staging + atomic commit and item-by-item on-disk layout
- whole-version overwrite, rollback on commit failure
- distinct failure semantics: version / checksum / incomplete / interrupted
- legacy directories (no manifest) keep reading item-by-item
- concurrent saves are serialized or explicitly rejected, readers never see
  a half-written directory
"""

import json
import os
import threading
from pathlib import Path

import pytest
import srsly

import spacy
from spacy import util
from spacy.errors import (
    ArtifactCommitError,
    ArtifactCommitInterruptedError,
    ArtifactIncompleteError,
    ArtifactIntegrityError,
    ArtifactLockError,
    ArtifactVersionError,
)
from spacy.lang.en import English

from ..util import make_tempdir

MANIFEST = util.ARTIFACT_MANIFEST_NAME

PAYLOAD_A = {
    "meta.json": json.dumps({"version": 1}),
    "config.cfg": "v = 1\n",
}
PAYLOAD_B = {
    "meta.json": json.dumps({"version": 2}),
    "config.cfg": "v = 2222222222\n",
}


def _file_writer(key, text):
    def writer(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    return writer


def _dir_writer(key, files, nested_util_call=False):
    def writer(path):
        path.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (path / name).write_text(content, encoding="utf-8")
        if nested_util_call:
            # Mirrors Vectors.to_disk writing into the vocab item directory.
            util.to_disk(path, {"extra.cfg": lambda p: p.write_text("x")}, [])

    return writer


def _writers(payload, nested=False, delay_event=None, fail_key=None):
    def maybe_delay(key):
        if delay_event is not None and key == "meta.json":
            delay_event.wait(timeout=10)

    def make(key, text):
        def writer(path):
            if fail_key == key:
                raise RuntimeError("serializer exploded")
            maybe_delay(key)
            _file_writer(key, text)(path)

        return writer

    return {
        "meta.json": make("meta.json", payload["meta.json"]),
        "config.cfg": make("config.cfg", payload["config.cfg"]),
        "vocab": _dir_writer(
            "vocab", payload.get("vocab", {"strings.json": "s"}), nested
        ),
    }


def _read_guarded(path, readers):
    captured = {}

    def read_file(key):
        def reader(p):
            if p.exists():
                captured[key] = p.read_text(encoding="utf-8")

        return reader

    def read_vocab(p):
        captured["vocab"] = {}
        if p.exists():
            for child in p.iterdir():
                if child.is_file():
                    captured["vocab"][child.name] = child.read_text(encoding="utf-8")
            # Nested util.from_disk must join the outer verified read.
            util.from_disk(p, {"extra.cfg": read_file("extra.cfg")}, [])

    all_readers = {
        "meta.json": read_file("meta.json"),
        "config.cfg": read_file("config.cfg"),
        "vocab": read_vocab,
        "optional": read_file("optional"),
    }
    util.from_disk(path, all_readers, [])
    return captured


def _manifest(path):
    return srsly.json_loads((path / MANIFEST).read_bytes())


def _no_markers(directory):
    return not [
        p for p in directory.iterdir() if ".staging-" in p.name or ".backup-" in p.name
    ]


@pytest.fixture
def short_lock_timeout(monkeypatch):
    monkeypatch.setenv("SPACY_ARTIFACT_LOCK_TIMEOUT", "2")


def test_to_disk_is_atomic_and_item_by_item():
    with make_tempdir() as d:
        model = d / "model"
        util.to_disk(model, _writers(PAYLOAD_A), [])
        # Same item layout as the old item-by-item writer, plus the manifest.
        assert sorted(p.name for p in model.iterdir()) == sorted(
            [MANIFEST, "meta.json", "config.cfg", "vocab"]
        )
        assert _no_markers(d)
        data = _read_guarded(model, None)
        assert data["meta.json"] == PAYLOAD_A["meta.json"]
        assert data["config.cfg"] == PAYLOAD_A["config.cfg"]
        assert data["vocab"]["strings.json"] == "s"
        # The manifest records a checksum and type for every item.
        items = _manifest(model)["items"]
        assert set(items) == {"meta.json", "config.cfg", "vocab"}
        assert items["meta.json"]["type"] == "file"
        assert items["vocab"]["type"] == "dir"


def test_nested_to_disk_joins_enclosing_transaction():
    with make_tempdir() as d:
        model = d / "model"
        util.to_disk(model, _writers(PAYLOAD_A, nested=True), [])
        # No nested manifest/lock inside an item and no sibling leftovers.
        assert not (model / "vocab" / MANIFEST).exists()
        assert not (model / "vocab" / ".vocab.lock").exists()
        assert (model / "vocab" / "extra.cfg").exists()
        assert _no_markers(d)
        data = _read_guarded(model, None)
        assert data["vocab"]["extra.cfg"] == "x"


def test_overwrite_commits_whole_new_version():
    with make_tempdir() as d:
        model = d / "model"
        util.to_disk(model, _writers(PAYLOAD_A), [])
        (model / "user_file.txt").write_text("keep")
        util.to_disk(model, _writers(PAYLOAD_B), [])
        assert _no_markers(d)
        data = _read_guarded(model, None)
        assert data["meta.json"] == PAYLOAD_B["meta.json"]
        assert data["config.cfg"] == PAYLOAD_B["config.cfg"]
        # Files not owned by the transaction are preserved.
        assert (model / "user_file.txt").exists()


def test_writer_failure_before_commit_leaves_old_version():
    with make_tempdir() as d:
        model = d / "model"
        util.to_disk(model, _writers(PAYLOAD_A), [])
        with pytest.raises(RuntimeError):
            util.to_disk(model, _writers(PAYLOAD_B, fail_key="meta.json"), [])
        data = _read_guarded(model, None)
        assert data["meta.json"] == PAYLOAD_A["meta.json"]
        assert _no_markers(d)


def test_serializer_without_output_is_allowed():
    # Some serializers legitimately produce no on-disk item (e.g. empty
    # vectors); such items are omitted instead of aborting the transaction.
    with make_tempdir() as d:
        model = d / "model"
        writers = {
            "ghost": lambda p: None,
            "real": _file_writer("real", "data"),
        }
        util.to_disk(model, writers, [])
        assert (model / "real").exists()
        assert not (model / "ghost").exists()
        items = _manifest(model)["items"]
        assert "real" in items and "ghost" not in items
        seen = []
        util.from_disk(
            model,
            {
                "real": lambda p: seen.append(p.read_text(encoding="utf-8")),
                "ghost": lambda p: seen.append("ghost" if p.exists() else None),
            },
            [],
        )
        assert seen == ["data", None]


def test_checksum_mismatch_raises_integrity_error():
    with make_tempdir() as d:
        model = d / "model"
        util.to_disk(model, _writers(PAYLOAD_A), [])
        (model / "meta.json").write_text(json.dumps({"version": 999}))
        with pytest.raises(ArtifactIntegrityError) as exc:
            util.from_disk(model, {"meta.json": lambda p: p.read_text()}, [])
        assert exc.value.item == "meta.json"
        assert exc.value.expected != exc.value.actual


def test_missing_item_raises_incomplete_error():
    with make_tempdir() as d:
        model = d / "model"
        util.to_disk(model, _writers(PAYLOAD_A), [])
        (model / "config.cfg").unlink()
        with pytest.raises(ArtifactIncompleteError) as exc:
            util.from_disk(model, {"config.cfg": lambda p: None}, [])
        assert exc.value.item == "config.cfg"
        assert exc.value.actual_type == "missing"


def test_integrity_and_incomplete_are_distinct_from_legacy():
    # The three failure modes must not collapse onto the legacy read path.
    assert not issubclass(ArtifactIntegrityError, ArtifactIncompleteError)
    assert ArtifactVersionError is not ArtifactIntegrityError


@pytest.mark.parametrize(
    "version,kind",
    [(999, "artifact-newer"), (0, "artifact-older")],
)
def test_manifest_version_mismatch_is_diagnosable(version, kind):
    with make_tempdir() as d:
        model = d / "model"
        util.to_disk(model, _writers(PAYLOAD_A), [])
        manifest = _manifest(model)
        manifest["manifest_version"] = version
        (model / MANIFEST).write_text(srsly.json_dumps(manifest))
        with pytest.raises(ArtifactVersionError) as exc:
            util.from_disk(model, {"meta.json": lambda p: None}, [])
        assert exc.value.kind == kind
        assert exc.value.found == version
        assert exc.value.supported == util.ARTIFACT_MANIFEST_VERSION


def test_truncated_manifest_means_interrupted_commit():
    with make_tempdir() as d:
        model = d / "model"
        util.to_disk(model, _writers(PAYLOAD_A), [])
        (model / MANIFEST).write_text("{truncated")
        with pytest.raises(ArtifactCommitInterruptedError):
            util.from_disk(model, {"meta.json": lambda p: None}, [])


def test_legacy_directory_without_manifest_keeps_old_behavior():
    with make_tempdir() as d:
        model = d / "model"
        model.mkdir()
        (model / "meta.json").write_text(PAYLOAD_A["meta.json"])
        (model / "vocab").mkdir()
        (model / "vocab" / "strings.json").write_text("s")
        # No verification, no error even though content is later modified.
        data = _read_guarded(model, None)
        assert data["meta.json"] == PAYLOAD_A["meta.json"]
        (model / "meta.json").write_text(json.dumps({"version": 42}))
        data = _read_guarded(model, None)
        assert json.loads(data["meta.json"])["version"] == 42


def test_interrupted_commit_is_rolled_back_on_next_open():
    with make_tempdir() as d:
        model = d / "model"
        util.to_disk(model, _writers(PAYLOAD_A), [])
        # Simulate a writer killed after parking meta.json in the backup but
        # before moving the new one into place (old manifest still present).
        backup = d / ".model.backup-0123456789abcdef"
        backup.mkdir()
        os.replace(model / "meta.json", backup / "meta.json")
        assert not (model / "meta.json").exists()
        data = _read_guarded(model, None)
        assert data["meta.json"] == PAYLOAD_A["meta.json"]
        assert not backup.exists()


def test_dead_commit_with_new_item_is_rolled_back_completely():
    with make_tempdir() as d:
        model = d / "model"
        util.to_disk(model, _writers(PAYLOAD_A), [])
        (model / "user_file.txt").write_text("keep")
        txid = "deadbeefdeadbeef"
        staging = d / f".model.staging-{txid}"
        staging.mkdir()
        backup = d / f".model.backup-{txid}"
        backup.mkdir()
        # Phase 1: every old item (incl. manifest) was parked in the backup.
        for name in [MANIFEST, "meta.json", "config.cfg", "vocab"]:
            os.replace(model / name, backup / name)
        # Phase 2: new items were moved in, including a brand-new component.
        (model / "new_pipe").mkdir()
        (model / "new_pipe" / "weights").write_text("half")
        (model / "meta.json").write_text(PAYLOAD_B["meta.json"])
        (model / "config.cfg").write_text(PAYLOAD_B["config.cfg"])
        # The draft manifest records the new component (as a real draft would).
        (model_vocab := model / "vocab").mkdir()
        (model_vocab / "strings.json").write_text("s2")
        staged_manifest = {
            "manifest_version": util.ARTIFACT_MANIFEST_VERSION,
            "txid": txid,
            "items": {
                "meta.json": {"type": "file", "sha256": "x"},
                "config.cfg": {"type": "file", "sha256": "x"},
                "vocab": {"type": "dir", "sha256": "x"},
                "new_pipe": {"type": "dir", "sha256": "x"},
            },
        }
        (staging / MANIFEST).write_text(srsly.json_dumps(staged_manifest))
        # Killed before phase 3 (manifest publish). Opening the artifact must
        # restore the complete old version and drop the half-new component.
        data = _read_guarded(model, None)
        assert data["meta.json"] == PAYLOAD_A["meta.json"]
        assert data["config.cfg"] == PAYLOAD_A["config.cfg"]
        assert not (model / "new_pipe").exists()
        assert not backup.exists()
        assert not staging.exists()
        assert (model / "user_file.txt").exists()
        assert (model / MANIFEST).exists()


def test_stale_staging_dirs_are_swept_on_next_save():
    with make_tempdir() as d:
        model = d / "model"
        util.to_disk(model, _writers(PAYLOAD_A), [])
        stale = d / ".model.staging-deadbeef"
        stale.mkdir()
        (stale / "junk").write_text("x")
        util.to_disk(model, _writers(PAYLOAD_B), [])
        assert not stale.exists()
        data = _read_guarded(model, None)
        assert data["meta.json"] == PAYLOAD_B["meta.json"]


def test_concurrent_save_is_rejected_without_touching_target(short_lock_timeout):
    with make_tempdir() as d:
        model = d / "model"
        util.to_disk(model, _writers(PAYLOAD_A), [])
        with util._artifact_lock(model, exclusive=True):
            with pytest.raises(ArtifactLockError):
                util.to_disk(model, _writers(PAYLOAD_B), [])
        data = _read_guarded(model, None)
        assert data["meta.json"] == PAYLOAD_A["meta.json"]


def test_concurrent_writers_are_serialized(short_lock_timeout):
    with make_tempdir() as d:
        model = d / "model"
        errors = []

        def save(payload):
            try:
                util.to_disk(model, _writers(payload), [])
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        t1 = threading.Thread(target=save, args=(PAYLOAD_A,))
        t2 = threading.Thread(target=save, args=(PAYLOAD_B,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert errors == []
        data = _read_guarded(model, None)
        # One complete version, never a mix of the two.
        assert data["meta.json"] in (PAYLOAD_A["meta.json"], PAYLOAD_B["meta.json"])
        assert data["config.cfg"] in (
            PAYLOAD_A["config.cfg"],
            PAYLOAD_B["config.cfg"],
        )
        if data["meta.json"] == PAYLOAD_B["meta.json"]:
            assert data["config.cfg"] == PAYLOAD_B["config.cfg"]
        else:
            assert data["config.cfg"] == PAYLOAD_A["config.cfg"]


def test_reader_blocks_during_commit_and_sees_complete_version(short_lock_timeout):
    with make_tempdir() as d:
        model = d / "model"
        util.to_disk(model, _writers(PAYLOAD_A), [])
        proceed = threading.Event()
        result = {}

        def writer():
            util.to_disk(model, _writers(PAYLOAD_B, delay_event=proceed), [])

        def reader():
            result["data"] = _read_guarded(model, None)

        tw = threading.Thread(target=writer)
        tr = threading.Thread(target=reader)
        tw.start()
        tr.start()
        try:
            # The reader must wait instead of observing a partial directory.
            assert tr.is_alive() or result == {}
        finally:
            proceed.set()
        tw.join(timeout=10)
        tr.join(timeout=10)
        assert result["data"]["meta.json"] == PAYLOAD_B["meta.json"]
        assert result["data"]["config.cfg"] == PAYLOAD_B["config.cfg"]


def test_commit_failure_rolls_back_to_previous_version(monkeypatch):
    with make_tempdir() as d:
        model = d / "model"
        util.to_disk(model, _writers(PAYLOAD_A), [])
        real_replace = os.replace

        def fail_manifest_publish(src, dst):
            if Path(dst).resolve() == (model / MANIFEST).resolve() and Path(
                src
            ).parent.name.startswith(".model.staging-"):
                raise OSError("simulated commit failure")
            return real_replace(src, dst)

        monkeypatch.setattr(os, "replace", fail_manifest_publish)
        with pytest.raises(ArtifactCommitError):
            util.to_disk(model, _writers(PAYLOAD_B), [])
        data = _read_guarded(model, None)
        assert data["meta.json"] == PAYLOAD_A["meta.json"]
        assert data["config.cfg"] == PAYLOAD_A["config.cfg"]
        assert _no_markers(d)


def test_language_roundtrip_with_pipeline():
    nlp = English()
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([{"label": "ORG", "pattern": "MyCorp"}])
    before = [(e.text, e.label_) for e in nlp("Hello MyCorp").ents]
    with make_tempdir() as model_dir:
        nlp.to_disk(model_dir)
        assert (model_dir / MANIFEST).exists()
        assert (model_dir / "meta.json").exists()
        assert (model_dir / "config.cfg").exists()
        assert (model_dir / "tokenizer").exists()
        assert (model_dir / "vocab").is_dir()
        assert (model_dir / "entity_ruler").is_dir()
        # In-place roundtrip on the same object (keeps its components).
        same = English()
        same.add_pipe("entity_ruler").add_patterns(
            [{"label": "ORG", "pattern": "MyCorp"}]
        )
        same.from_disk(model_dir)
        assert [(e.text, e.label_) for e in same("Hello MyCorp").ents] == before
        # Loading through the public API rebuilds the pipeline from config.
        loaded = spacy.load(model_dir)
        assert [(e.text, e.label_) for e in loaded("Hello MyCorp").ents] == before


def test_language_overwrite_and_excluded_items():
    with make_tempdir() as model_dir:
        nlp = English()
        nlp.add_pipe("entity_ruler").add_patterns(
            [{"label": "ORG", "pattern": "MyCorp"}]
        )
        nlp.to_disk(model_dir)
        # Saving again with a component excluded keeps its old on-disk item
        # (same behavior as the legacy item-by-item writer).
        nlp.meta["description"] = "updated"
        nlp.to_disk(model_dir, exclude=["entity_ruler"])
        assert (model_dir / "entity_ruler").exists()
        reloaded = spacy.load(model_dir)
        assert reloaded.meta["description"] == "updated"
        assert [(e.text, e.label_) for e in reloaded("Hi MyCorp").ents] == [
            ("MyCorp", "ORG")
        ]
