"""/ The metric registry: one source of truth for names, directions and formatting.

``METRICS`` keys are exactly the column names ``scripts/score.py`` writes into
``output/<run>_stats.jsonl``, so a metric name means the same thing in scoring,
export, comparison and reporting. ``direction`` encodes which way is better;
``performance_sign`` turns any metric into "higher = better" so deltas and
spreads can be compared across metrics. ``format_kind`` drives rendering in
tables and plotly axes. ``metric_spec()`` resolves unknown names gracefully so
custom score columns still plot (neutral direction, plain floats).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["higher", "lower", "neutral"]
FormatKind = Literal["probability", "float", "count", "milliseconds"]


@dataclass(frozen=True)
class MetricSpec:
    """/ Descriptor for one metric; ``name`` doubles as the stats-JSONL column key."""
    name: str
    label: str
    direction: Direction
    format_kind: FormatKind = "float"
    description: str = ""

    @property
    def performance_sign(self) -> int:
        """Multiply raw (model-baseline) deltas by this so positive always means better."""
        return -1 if self.direction == "lower" else 1


METRICS: dict[str, MetricSpec] = {
    "accuracy": MetricSpec("accuracy", "Accuracy", "higher", "probability"),
    "soft_accuracy": MetricSpec(
        "soft_accuracy",
        "Soft accuracy",
        "higher",
        "probability",
        "Probability-aware correctness; often more discriminative than saturated hard accuracy.",
    ),
    "nll": MetricSpec("nll", "NLL", "lower", "float"),
    "brier": MetricSpec("brier", "Brier score", "lower", "float"),
    "confidence": MetricSpec("confidence", "Confidence", "neutral", "probability"),
    "ece_15": MetricSpec("ece_15", "ECE (15 bins)", "lower", "probability"),
    "score_mae": MetricSpec("score_mae", "Score MAE", "lower", "float"),
    "score_within_one": MetricSpec("score_within_one", "Score within one", "higher", "probability"),
    "errors": MetricSpec("errors", "Errors", "lower", "count"),
    "error_rate": MetricSpec("error_rate", "Error rate", "lower", "probability"),
    "latency_ms_mean": MetricSpec("latency_ms_mean", "Mean latency", "lower", "milliseconds"),
    "latency_ms_p50": MetricSpec("latency_ms_p50", "p50 latency", "lower", "milliseconds"),
    "latency_ms_p95": MetricSpec("latency_ms_p95", "p95 latency", "lower", "milliseconds"),
    "n": MetricSpec("n", "Examples", "neutral", "count"),
}


def metric_spec(name: str, direction: Direction | None = None) -> MetricSpec:
    """/ Registry lookup with two fallbacks: an explicit direction override wins over
    the registered one, and unknown names become neutral/float specs instead of
    errors so ad-hoc metric columns still work in the analysis pipeline."""
    if name in METRICS:
        spec = METRICS[name]
        if direction is None or direction == spec.direction:
            return spec
        return MetricSpec(spec.name, spec.label, direction, spec.format_kind, spec.description)
    return MetricSpec(name, name.replace("_", " ").title(), direction or "neutral", "float")


def fmt_value(value: float | int | None, spec: MetricSpec) -> str:
    """/ Renders one value per the spec's format_kind; NaN/None become an em dash
    so tables never show raw nan."""
    if value is None:
        return "-"
    try:
        if value != value:  # NaN
            return "-"
    except TypeError:
        return str(value)
    if spec.format_kind == "probability":
        return f"{float(value) * 100:.2f}%"
    if spec.format_kind == "milliseconds":
        return f"{float(value):,.1f} ms"
    if spec.format_kind == "count":
        return f"{int(round(float(value))):,}"
    return f"{float(value):.4f}"


def axis_tickformat(spec: MetricSpec) -> str | None:
    """/ plotly tickformat for axes showing this metric (None = plotly default)."""
    if spec.format_kind == "probability":
        return ".0%"
    if spec.format_kind == "milliseconds":
        return ",.0f"
    return None
