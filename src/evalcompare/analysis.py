"""/ Cross-run comparison built on loaded EvalData.

The core idea: pivot every metric into a capability x run matrix, then derive
per-run summaries, per-capability deltas vs a baseline and a model-spread table.
Summary aggregates (macro/weighted/median/p10) use only the capabilities ALL
runs share, so a model that skipped suites cannot "win" by avoiding hard ones;
the full union still appears in the matrices (NaN where a run has no data).
Deltas and spread are performance-aligned: multiplied by the metric's sign so
positive always means better, regardless of metric direction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .loader import EvalData
from .metrics import MetricSpec


@dataclass(frozen=True)
class Comparison:
    """/ Everything report.py and write_tables() need from one comparison run:
    the source data, the chosen metric/baseline, the common vs union capability
    sets, the capability x run matrices, and the derived summary/delta/spread
    tables (all performance-aligned)."""
    data: EvalData
    metric: MetricSpec
    baseline: str
    common_capabilities: tuple[str, ...]
    union_capabilities: tuple[str, ...]
    metric_matrix: pd.DataFrame
    n_matrix: pd.DataFrame
    summary: pd.DataFrame
    deltas: pd.DataFrame
    spread: pd.DataFrame


def _pivot(data: EvalData, field: str) -> pd.DataFrame:
    """/ capability x run matrix of one field (coerced to numeric); empty frame with
    the run columns when the field is absent, so callers can stay unconditional."""
    if field not in data.capabilities.columns:
        return pd.DataFrame(index=[], columns=list(data.runs), dtype=float)
    frame = data.capabilities[["capability", "run", field]].copy()
    frame[field] = pd.to_numeric(frame[field], errors="coerce")
    return frame.pivot(index="capability", columns="run", values=field).reindex(columns=data.runs)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """/ n-weighted mean over paired non-NaN entries (weights <= 0 ignored); NaN when nothing pairs."""
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask].astype(float), weights=weights[mask].astype(float)))


def build_comparison(data: EvalData, metric: MetricSpec, baseline: str | None = None) -> Comparison:
    """/ Pivots + summarizes; baseline defaults to the first run (load order).

    Raises when the metric column is missing from the input or the baseline is
    not one of the loaded runs. Summary stats and deltas are computed on the
    shared-capability set only (see module docstring); spread keeps NaN rows so
    partially covered capabilities still show how far apart the covering models
    are.
    """
    if metric.name not in data.capabilities.columns:
        raise ValueError(f"metric {metric.name!r} is not present in the input")
    baseline = baseline or data.runs[0]
    if baseline not in data.runs:
        raise ValueError(f"baseline {baseline!r} not found; choices: {', '.join(data.runs)}")

    metric_matrix = _pivot(data, metric.name)
    n_matrix = _pivot(data, "n")
    union_caps = tuple(metric_matrix.index.tolist())
    common_mask = metric_matrix.notna().all(axis=1)
    common_caps = tuple(metric_matrix.index[common_mask].tolist())

    summary_rows: list[dict] = []
    common_set = set(common_caps)
    for run in data.runs:
        r = data.capabilities[data.capabilities["run"] == run].copy()
        r[metric.name] = pd.to_numeric(r[metric.name], errors="coerce")
        r["n"] = pd.to_numeric(r["n"], errors="coerce")
        rc = r[r["capability"].isin(common_set)].copy()

        row: dict[str, float | int | str] = {
            "run": run,
            "capabilities_present": int(r["capability"].nunique()),
            "common_capabilities": len(common_caps),
            "coverage": float(r["capability"].nunique() / len(union_caps)) if union_caps else float("nan"),
            "macro": float(rc[metric.name].mean()) if not rc.empty else float("nan"),
            "weighted": _weighted_mean(rc[metric.name], rc["n"]),
            "median": float(rc[metric.name].median()) if not rc.empty else float("nan"),
            "p10": float(rc[metric.name].quantile(0.10)) if not rc.empty else float("nan"),
            "examples": int(rc["n"].sum()) if not rc.empty else 0,
            "min_n": int(rc["n"].min()) if not rc.empty and rc["n"].notna().any() else 0,
        }
        micro = data.aggregates[(data.aggregates["run"] == run) & (data.aggregates["capability"] == "micro")]
        if not micro.empty:
            mr = micro.iloc[0]
            row["reported_micro"] = pd.to_numeric(pd.Series([mr.get(metric.name)]), errors="coerce").iloc[0]
            row["reported_micro_n"] = pd.to_numeric(pd.Series([mr.get("n")]), errors="coerce").iloc[0]
        else:
            row["reported_micro"] = float("nan")
            row["reported_micro_n"] = float("nan")
        for aux in ("accuracy", "soft_accuracy", "nll", "ece_15", "brier", "latency_ms_p50", "latency_ms_p95", "error_rate"):
            if aux in rc.columns:
                vals = pd.to_numeric(rc[aux], errors="coerce")
                row[f"macro_{aux}"] = float(vals.mean()) if vals.notna().any() else float("nan")
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).set_index("run").reindex(data.runs)

    baseline_values = metric_matrix[baseline]
    delta_cols: dict[str, pd.Series] = {}
    for run in data.runs:
        if run == baseline:
            continue
        raw = metric_matrix[run] - baseline_values
        delta_cols[run] = raw * metric.performance_sign
    deltas = pd.DataFrame(delta_cols, index=metric_matrix.index)

    # Spread is always performance-aligned; larger means models disagree more on this capability.
    perf = metric_matrix * metric.performance_sign
    spread = pd.DataFrame(
        {
            "min": perf.min(axis=1, skipna=True),
            "max": perf.max(axis=1, skipna=True),
            "spread": perf.max(axis=1, skipna=True) - perf.min(axis=1, skipna=True),
            "models_present": perf.notna().sum(axis=1),
        }
    )
    spread = spread.sort_values("spread", ascending=False)

    return Comparison(
        data=data,
        metric=metric,
        baseline=baseline,
        common_capabilities=common_caps,
        union_capabilities=union_caps,
        metric_matrix=metric_matrix,
        n_matrix=n_matrix,
        summary=summary,
        deltas=deltas,
        spread=spread,
    )


def write_tables(c: Comparison, out_dir: Path) -> None:
    """/ Machine-readable companion of the HTML report: raw rows, model summary, the
    capability x run matrices, per-baseline deltas and spread as CSVs plus a
    summary.json (metric metadata + summary records, NaN -> null)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    c.data.rows.to_csv(out_dir / "all_rows.csv", index=False)
    c.summary.reset_index().to_csv(out_dir / "model_summary.csv", index=False)
    c.metric_matrix.to_csv(out_dir / f"capability_{c.metric.name}.csv")
    c.n_matrix.to_csv(out_dir / "capability_n.csv")
    c.deltas.to_csv(out_dir / f"delta_vs_{safe_name(c.baseline)}_{c.metric.name}.csv")
    c.spread.to_csv(out_dir / f"capability_spread_{c.metric.name}.csv")

    payload = {
        "metric": c.metric.name,
        "metric_direction": c.metric.direction,
        "baseline": c.baseline,
        "runs": list(c.data.runs),
        "common_capabilities": len(c.common_capabilities),
        "union_capabilities": len(c.union_capabilities),
        "summary": c.summary.reset_index().replace({np.nan: None}).to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def safe_name(value: str) -> str:
    """/ Filesystem-safe form of a run name for output file names (baseline names
    come from user data and may contain anything)."""
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value).strip("_") or "run"
