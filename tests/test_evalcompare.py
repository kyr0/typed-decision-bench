"""/ Unit tests for the evalcompare library in src/ (io -> analysis -> report).

Covers the load guarantees (run uniqueness, duplicate rejection), the shared-set
summary rule, performance-aligned deltas for lower-is-better metrics, and the
end-to-end HTML report build. Run via `make test`; conftest.py maps the flat
src/ modules onto the `evalcompare` package name.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalcompare.analysis import build_comparison
from evalcompare.loader import load_eval_files
from evalcompare.metrics import metric_spec
from evalcompare.report import build_report


def _write(path: Path, run: str, values: dict[str, float], *, missing: set[str] | None = None) -> None:
    """/ Fixture writer: one single-run stats JSONL with the given capability scores
    (skipping `missing` caps) plus a micro aggregate line."""
    missing = missing or set()
    with path.open("w", encoding="utf-8") as fh:
        for cap, value in values.items():
            if cap in missing:
                continue
            fh.write(json.dumps({
                "run": run,
                "capability": cap,
                "n": 100,
                "accuracy": value,
                "soft_accuracy": value,
                "nll": 1.0 - value,
                "ece_15": 0.05,
                "latency_ms_p50": 100.0,
                "latency_ms_p95": 150.0,
            }) + "\n")
        fh.write(json.dumps({"run": run, "capability": "micro", "n": 300, "soft_accuracy": 0.8}) + "\n")


def test_load_compare_and_report(tmp_path: Path) -> None:
    """/ Full pipeline smoke test: two runs load in order, the comparison computes
    the expected delta and summary ranking, and build_report writes the dashboard."""
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    values = {"reasoning": 0.7, "routing": 0.8, "code": 0.9}
    _write(a, "A", values)
    _write(b, "B", {k: v + 0.05 for k, v in values.items()})

    data = load_eval_files([a, b])
    assert data.runs == ("A", "B")
    assert len(data.capabilities) == 6
    assert len(data.aggregates) == 2

    c = build_comparison(data, metric_spec("soft_accuracy"), baseline="A")
    assert len(c.common_capabilities) == 3
    assert c.deltas.loc["reasoning", "B"] == pytest.approx(0.05)
    assert c.summary.loc["B", "macro"] > c.summary.loc["A", "macro"]

    # the discriminators figure must not carry its own title: the report's
    # section heading already names it
    from evalcompare.report import discriminating_capabilities
    assert discriminating_capabilities(c).layout.title.text is None

    out = tmp_path / "report"
    report = build_report(c, out, title="test")
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "reasoning" in text
    assert "Plotly" in text


def test_missing_capability_uses_shared_set(tmp_path: Path) -> None:
    """/ A capability only one run covers stays in the union but leaves the common
    set, so macro summaries cannot be gamed by coverage differences."""
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    values = {"x": 0.2, "y": 0.8, "z": 0.9}
    _write(a, "A", values)
    _write(b, "B", values, missing={"z"})
    c = build_comparison(load_eval_files([a, b]), metric_spec("soft_accuracy"), baseline="A")
    assert set(c.common_capabilities) == {"x", "y"}
    assert len(c.union_capabilities) == 3


def test_lower_is_better_delta_is_performance_aligned(tmp_path: Path) -> None:
    """/ For NLL (lower is better) a decrease vs baseline must still be a positive delta."""
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write(a, "A", {"x": 0.8})
    _write(b, "B", {"x": 0.9})
    data = load_eval_files([a, b])
    # _write sets NLL=1-value, so B has 0.1 vs A 0.2 => +0.1 improvement.
    c = build_comparison(data, metric_spec("nll"), baseline="A")
    assert c.deltas.loc["x", "B"] == pytest.approx(0.1)


def test_duplicate_capability_rejected(tmp_path: Path) -> None:
    """/ Repeated (run, capability) rows would make the pivot ambiguous — loader refuses."""
    p = tmp_path / "dup.jsonl"
    row = {"run": "A", "capability": "x", "n": 1, "soft_accuracy": 0.5}
    p.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_eval_files([p])
