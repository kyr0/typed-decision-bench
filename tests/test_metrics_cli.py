"""End-to-end checks for scripts/metrics.py: per-run export and the
baseline-gated per-capability comparison (no baseline -> refuse to compare).

Run: via `make test` (the uv env provides pandas/plotly).
"""
import json
import subprocess
import sys
from pathlib import Path

METRICS_CLI = Path(__file__).resolve().parents[1] / "scripts" / "metrics.py"


def _write_stats(path: Path, run: str, values: dict[str, float]) -> None:
    lines = [json.dumps({"run": run, "capability": cap, "n": 10,
                         "accuracy": v, "soft_accuracy": v, "nll": 1.0 - v})
             for cap, v in values.items()]
    lines.append(json.dumps({"run": run, "capability": "micro", "n": 30,
                             "accuracy": 0.8, "soft_accuracy": 0.8}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_metrics_cli_export_and_compare(tmp_path: Path) -> None:
    out = tmp_path / "output"
    out.mkdir()
    _write_stats(out / "a_stats.jsonl", "a", {"cap1": 0.7, "cap2": 0.9})
    _write_stats(out / "b_stats.jsonl", "b", {"cap1": 0.8, "cap2": 0.6})

    # default: per-run export, one subfolder per named eval result
    p = subprocess.run([sys.executable, str(METRICS_CLI), "--benchmark", str(tmp_path)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert (out / "a" / "metrics.csv").exists() and (out / "b" / "metrics.json").exists()

    # comparison refuses to run without an explicit baseline
    p = subprocess.run([sys.executable, str(METRICS_CLI), "--benchmark", str(tmp_path), "--compare"],
                       capture_output=True, text=True)
    assert p.returncode != 0

    # unknown baseline is a hard error listing the available runs
    p = subprocess.run([sys.executable, str(METRICS_CLI), "--benchmark", str(tmp_path),
                        "--compare", "--baseline", "nope"], capture_output=True, text=True)
    assert p.returncode != 0 and "a" in p.stderr and "b" in p.stderr

    # per-capability comparison vs baseline lands in output/comparison/
    p = subprocess.run([sys.executable, str(METRICS_CLI), "--benchmark", str(tmp_path),
                        "--compare", "--baseline", "a"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    comp = out / "comparison"
    assert (comp / "capability_soft_accuracy.csv").exists()
    assert (comp / "delta_vs_a_soft_accuracy.csv").exists()
    assert "cap1" in (comp / "delta_vs_a_soft_accuracy.csv").read_text(encoding="utf-8")
    assert (comp / "model_summary.csv").exists() and (comp / "summary.json").exists()
    assert "cap2" in (comp / "report.html").read_text(encoding="utf-8")
