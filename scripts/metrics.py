#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "plotly"]
# ///
"""Per-run metrics exporter + baseline-gated comparison.

Default: turns every output/<run>_stats.jsonl into a subfolder output/<run>/
holding that single run's metrics files (metrics.csv + metrics.json) — no
comparison without a baseline.

--compare --baseline <run>: per-capability comparison of ALL scored runs
against that baseline into output/comparison/ (capability x run matrix,
per-capability deltas/spread, model summary and a plotly dashboard), built on
evalcompare.analysis/report.
"""
import argparse
import json
import sys
from pathlib import Path

# import the evalcompare package via the standard src layout: src/ on sys.path
# exposes exactly one top-level name, the qualified `evalcompare` package
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evalcompare.analysis import build_comparison, write_tables
from evalcompare.loader import load_eval_files
from evalcompare.metrics import METRICS, metric_spec
from evalcompare.report import build_report


def _jsonable(value):
    """numpy scalars -> plain Python, NaN -> None, so metrics.json stays valid JSON."""
    if value != value:
        return None
    return value.item() if hasattr(value, "item") else value


def export_run(stats_path: Path, out_dir: Path) -> None:
    """/ One run's stats file -> output/<run>/metrics.csv + metrics.json.
    Metric columns follow the evalcompare registry order so every run folder
    shares one schema; unknown extra columns are appended unchanged."""
    data = load_eval_files([stats_path])
    known = [c for c in METRICS if c in data.rows.columns]
    extra = [c for c in data.rows.columns
             if c not in METRICS and not c.startswith("_") and c not in ("run", "capability")]
    out_dir.mkdir(parents=True, exist_ok=True)
    data.rows[["run", "capability", *known, *extra]].to_csv(out_dir / "metrics.csv", index=False)
    micro = data.aggregates[data.aggregates["capability"] == "micro"]
    payload = {
        "run": data.runs[0],
        "source": stats_path.name,
        "capabilities": int(len(data.capabilities)),
        "micro": {k: _jsonable(v) for k, v in micro.iloc[0].items() if not str(k).startswith("_")}
        if not micro.empty else None,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=_jsonable) + "\n",
                                          encoding="utf-8")


def compare_runs(root: Path, baseline: str, metric: str, out_dir: Path) -> None:
    """/ Per-capability comparison of every scored run vs an explicit baseline:
    pivots all <run>_stats.jsonl into a capability x run matrix plus
    per-capability deltas, spread, model summary and an HTML dashboard.
    Only ever called with a user-provided baseline — no baseline, no compare."""
    stats = sorted((root / "output").glob("*_stats.jsonl"))
    by_run = {p.stem[:-len("_stats")]: p for p in stats}
    if baseline not in by_run:
        raise SystemExit(f"baseline {baseline!r} not found; available runs: "
                         f"{', '.join(sorted(by_run)) or 'none'}")
    others = [p for name, p in sorted(by_run.items()) if name != baseline]
    if not others:
        raise SystemExit(f"comparison needs at least one run besides the baseline {baseline!r}")
    data = load_eval_files([by_run[baseline], *others])  # baseline first => sensible default ordering
    c = build_comparison(data, metric_spec(metric), baseline=baseline)
    write_tables(c, out_dir)
    build_report(c, out_dir, title=f"Per-capability comparison vs {baseline}")
    print(f"wrote comparison of {len(data.runs)} runs (baseline {baseline!r}, metric {metric!r}) "
          f"-> {out_dir}/", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-run metrics exporter (output/<run>/metrics.csv + "
                                             "metrics.json per scored run) and, with --compare "
                                             "--baseline <run>, a per-capability comparison into "
                                             "output/comparison/")
    ap.add_argument("--benchmark", default=".")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="Restrict the per-run export to these run names (default: every output/*_stats.jsonl)")
    ap.add_argument("--compare", action="store_true",
                    help="Per-capability comparison of all scored runs (requires --baseline)")
    ap.add_argument("--baseline", default=None, help="Baseline run name for --compare")
    ap.add_argument("--metric", default="soft_accuracy",
                    help="Primary comparison metric (default: soft_accuracy)")
    a = ap.parse_args()
    root = Path(a.benchmark)
    out = root / "output"
    if a.compare:
        if not a.baseline:
            ap.error("--compare requires --baseline <run> (no baseline, no comparison)")
        compare_runs(root, a.baseline, a.metric, out / "comparison")
        return
    stats = sorted(out.glob("*_stats.jsonl"))
    if a.runs is not None:
        wanted = set(a.runs)
        stats = [p for p in stats if p.stem[:-len("_stats")] in wanted]
    if not stats:
        raise SystemExit(f"no scored runs found in {out}/ (run an eval or `make score` first)")
    for p in stats:
        run_dir = out / p.stem[:-len("_stats")]
        export_run(p, run_dir)
        print(f"wrote {run_dir}/metrics.csv + metrics.json (from {p.name})", file=sys.stderr)


if __name__ == "__main__":
    main()
