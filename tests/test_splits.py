"""Explicit split contract + one-time migration tests."""
import json
from pathlib import Path

import pytest

import assign_splits
import run_eval
import score
import splits


def test_assign_rows_is_deterministic_exact_and_has_no_train():
    rows = [{"id": f"fake-{i:03d}", "request_sha256": f"hash-{i}"} for i in range(1, 101)]
    a, changed = assign_splits.assign_rows("fake", rows, 0.20, "seed")
    b, changed2 = assign_splits.assign_rows("fake", rows, 0.20, "seed")
    assert changed and changed2
    assert [x["split"] for x in a] == [x["split"] for x in b]
    assert sum(x["split"] == "calibrate" for x in a) == 20
    assert sum(x["split"] == "test" for x in a) == 80
    assert all(x["split"] != "train" for x in a)
    # source row order/fields are not otherwise rewritten semantically
    assert all(x["id"] == y["id"] for x, y in zip(a, rows))


def test_existing_complete_split_assignment_is_preserved_without_force():
    rows = [{"id": "a", "split": "train"}, {"id": "b", "split": "test"},
            {"id": "c", "split": "calibrate"}]
    out, changed = assign_splits.assign_rows("fake", rows, 0.20, "seed")
    assert out == rows and changed is False


def test_partial_split_assignment_fails_closed():
    with pytest.raises(ValueError, match="partial/invalid"):
        assign_splits.assign_rows("fake", [{"id": "a", "split": "test"}, {"id": "b"}], 0.20, "seed")


def _fixture(root: Path):
    (root / "requests").mkdir()
    (root / "responses").mkdir()
    (root / "metadata").mkdir()
    manifest = {"capabilities": {"fake": {
        "requests": "requests/fake.jsonl",
        "responses": "responses/fake.jsonl",
        "metadata": "metadata/fake.jsonl",
        "cases": 4,
    }}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    reqs = [
        '{"model":"REPLACED_BY_TYPESAFE_MODEL","x":1}',
        '{"model":"REPLACED_BY_TYPESAFE_MODEL","x":2}',
        '{"model":"REPLACED_BY_TYPESAFE_MODEL","x":3}',
        '{"model":"REPLACED_BY_TYPESAFE_MODEL","x":4}',
    ]
    (root / "requests" / "fake.jsonl").write_text("\n".join(reqs) + "\n")
    gold = {"answers": {"q": {"type": "noul", "noul": 1.0}}}
    (root / "responses" / "fake.jsonl").write_text("\n".join(json.dumps(gold) for _ in range(4)) + "\n")
    metas = [{"id": f"fake-{i}", "split": split} for i, split in enumerate(
        ["test", "calibrate", "train", "test"], 1)]
    (root / "metadata" / "fake.jsonl").write_text("\n".join(json.dumps(x) for x in metas) + "\n")
    return manifest


def test_runner_defaults_to_test_plus_calibrate_and_preserves_source_line_numbers(tmp_path: Path):
    manifest = _fixture(tmp_path)
    tasks, counts, first = run_eval.build_tasks(
        tmp_path, ["fake"], manifest["capabilities"], None, "model-x")
    assert counts == {"fake": 4}
    assert [(cap, no) for cap, no, _ in tasks] == [("fake", 1), ("fake", 2), ("fake", 4)]
    assert all('"model":"model-x"' in body for _, _, body in tasks)
    assert "fake" in first

    test_only, _, _ = run_eval.build_tasks(
        tmp_path, ["fake"], manifest["capabilities"], None, "model-x", ("test",))
    assert [no for _, no, _ in test_only] == [1, 4]


def test_scorer_publishes_test_only(tmp_path: Path):
    manifest = _fixture(tmp_path)
    response = {"model": "m", "answers": {"q": {"type": "noul", "noul": 0.9}}}
    log = tmp_path / "run.jsonl"
    log.write_text("\n".join(json.dumps({
        "request_id": f"fake-{i:03d}", "response": response, "error_type": None,
        "error": None, "duration_ms": float(i), "endpoint": "x",
    }) for i in range(1, 5)) + "\n")

    test_keys = score.split_keys(tmp_path, manifest, "test")
    assert test_keys == {("fake", 1), ("fake", 4)}
    lines, cases = score.build_report(
        "run", tmp_path, manifest, score.log_predictions(log), score.run_stats(log, test_keys), "test")
    assert len(cases) == 2
    assert {c["case_id"] for c in cases} == {
        "typed-decisions-bench-v1:fake:0001",
        "typed-decisions-bench-v1:fake:0004",
    }
    assert lines[-1]["capability"] == "micro" and lines[-1]["n"] == 2


def test_parse_splits_rejects_unknown_and_deduplicates():
    assert splits.parse_splits("test,calibrate,test") == ("test", "calibrate")
    with pytest.raises(ValueError):
        splits.parse_splits("test,validation")
