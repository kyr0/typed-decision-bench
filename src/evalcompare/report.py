"""/ Self-contained HTML dashboard for a Comparison.

build_report() embeds the plotly bundle once and lays out the sections readers
actually need to judge a model suite: a numeric summary, a full capability
heatmap, baseline deltas, the most discriminating capabilities (biggest model
spread), per-run distributions, quality/latency trade-off, pairwise parity and
a calibration overview, followed by an exact-value table with live filtering
and links to the machine-readable CSVs written by analysis.write_tables().

Every figure is data-driven: direction-aware sorting (hardest capabilities
first), probability metrics get fixed 0..1 axes, and figures that cannot be
drawn from the available data (single run, missing latency column) return None
and are simply omitted from the report.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline.offline import get_plotlyjs

from .analysis import Comparison, safe_name
from .metrics import MetricSpec, axis_tickformat, fmt_value


def _base_layout(fig: go.Figure, *, height: int = 560, title: str | None = None) -> go.Figure:
    """/ Shared visual language for all figures: white template, compact margins,
    horizontal legend above the plot."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=32, r=24, t=72 if title else 34, b=52),
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=20)) if title else None,
        font=dict(family="Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size=12),
        hoverlabel=dict(font_size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _metric_axis(fig: go.Figure, spec: MetricSpec, *, axis: str = "y") -> None:
    """/ Applies the metric's label + tick format to one axis; probability metrics
    get a fixed 0..1 range so panels stay comparable."""
    kwargs: dict = {"title": spec.label}
    tickformat = axis_tickformat(spec)
    if tickformat:
        kwargs["tickformat"] = tickformat
    if spec.format_kind == "probability":
        kwargs["range"] = [0, 1]
    if axis == "y":
        fig.update_yaxes(**kwargs)
    else:
        fig.update_xaxes(**kwargs)


def summary_bar(c: Comparison) -> go.Figure:
    """/ Horizontal macro-score bars, ranked best-first (direction-aware); hover
    shows weighted macro, p10, coverage and example counts."""
    s = c.summary.reset_index().copy()
    ascending = c.metric.direction == "lower"
    if c.metric.direction != "neutral":
        s = s.sort_values("macro", ascending=ascending)
    custom = np.column_stack([
        s["weighted"].to_numpy(),
        s["p10"].to_numpy(),
        s["coverage"].to_numpy(),
        s["examples"].to_numpy(),
        s["min_n"].to_numpy(),
    ])
    fig = go.Figure(
        go.Bar(
            x=s["macro"],
            y=s["run"],
            orientation="h",
            customdata=custom,
            hovertemplate=(
                "<b>%{y}</b><br>Macro: %{x:.4f}<br>Weighted: %{customdata[0]:.4f}"
                "<br>p10: %{customdata[1]:.4f}<br>Coverage: %{customdata[2]:.1%}"
                "<br>Examples on shared set: %{customdata[3]:,.0f}<br>Min n/capability: %{customdata[4]:,.0f}<extra></extra>"
            ),
        )
    )
    _base_layout(fig, height=max(330, 75 + 52 * len(s)), title=f"Shared-capability macro — {c.metric.label}")
    _metric_axis(fig, c.metric, axis="x")
    fig.update_yaxes(title=None, autorange="reversed")
    return fig


def _ordered_caps(c: Comparison) -> list[str]:
    """/ Capabilities ordered by mean metric value with the hardest first, so weak
    spots surface at the top of heatmaps and tables regardless of metric direction."""
    m = c.metric_matrix.copy()
    mean = m.mean(axis=1, skipna=True)
    # Put weakest/hardest capabilities first.
    if c.metric.direction == "lower":
        return mean.sort_values(ascending=False).index.tolist()
    return mean.sort_values(ascending=True).index.tolist()


def capability_heatmap(c: Comparison) -> go.Figure:
    """/ Runs as columns, every capability as a row (hardest first); gaps render as
    holes so missing coverage is visible rather than interpolated."""
    caps = _ordered_caps(c)
    m = c.metric_matrix.reindex(caps)
    n = c.n_matrix.reindex(caps)
    custom = np.empty((len(caps), len(c.data.runs), 2), dtype=object)
    for i, cap in enumerate(caps):
        for j, run in enumerate(c.data.runs):
            custom[i, j, 0] = cap
            custom[i, j, 1] = n.loc[cap, run] if cap in n.index and run in n.columns else np.nan

    zmin = 0 if c.metric.format_kind == "probability" else None
    zmax = 1 if c.metric.format_kind == "probability" else None
    fig = go.Figure(
        go.Heatmap(
            z=m.to_numpy(dtype=float),
            x=list(c.data.runs),
            y=caps,
            customdata=custom,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title=c.metric.label),
            hoverongaps=False,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Model: %{x}<br>"
                + c.metric.label
                + ": %{z:.4f}<br>n: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    inner_height = max(520, 25 * len(caps) + 170)
    _base_layout(fig, height=inner_height, title=f"Every capability — {c.metric.label}")
    fig.update_xaxes(side="top", title=None)
    fig.update_yaxes(title=None, autorange="reversed", tickfont=dict(size=11))
    return fig


def delta_heatmap(c: Comparison) -> go.Figure | None:
    """/ Signed improvement over the baseline (symmetric color scale centred at 0,
    probability deltas in percentage points), rows sorted by largest absolute gap.
    None when the baseline is the only run."""
    if c.deltas.shape[1] == 0:
        return None
    d = c.deltas.copy()
    order = d.abs().max(axis=1, skipna=True).sort_values(ascending=False).index.tolist()
    d = d.reindex(order)
    scale = 100.0 if c.metric.format_kind == "probability" else 1.0
    z = d.to_numpy(dtype=float) * scale
    suffix = " pp" if c.metric.format_kind == "probability" else ""
    max_abs = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 1.0
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=list(d.columns),
            y=list(d.index),
            zmid=0,
            zmin=-max_abs,
            zmax=max_abs,
            colorscale="RdBu",
            colorbar=dict(title=f"Δ{suffix}"),
            hoverongaps=False,
            hovertemplate=f"<b>%{{y}}</b><br>Model: %{{x}}<br>Improvement vs {html.escape(c.baseline)}: %{{z:+.2f}}{suffix}<extra></extra>",
        )
    )
    inner_height = max(520, 25 * len(d) + 170)
    _base_layout(fig, height=inner_height, title=f"Improvement vs baseline: {c.baseline}")
    fig.update_xaxes(side="top", title=None)
    fig.update_yaxes(title=None, autorange="reversed", tickfont=dict(size=11))
    return fig


def discriminating_capabilities(c: Comparison, top_n: int = 40) -> go.Figure | None:
    """/ Per-model points on the capabilities with the largest model spread — where
    the suite separates models most. None for a single run or no multi-model rows."""
    if len(c.data.runs) < 2:
        return None
    top = c.spread[c.spread["models_present"] >= 2].head(top_n)
    if top.empty:
        return None
    m = c.metric_matrix.reindex(top.index)
    # Plot each model as points/lines over capabilities sorted by spread.
    fig = go.Figure()
    for run in c.data.runs:
        fig.add_trace(
            go.Scatter(
                x=m[run],
                y=m.index,
                mode="markers",
                name=run,
                customdata=np.column_stack([c.n_matrix.reindex(m.index)[run].to_numpy()]),
                hovertemplate="<b>%{y}</b><br>" + html.escape(run) + ": %{x:.4f}<br>n: %{customdata[0]}<extra></extra>",
            )
        )
    # no in-chart title: the report section heading already names this figure
    _base_layout(fig, height=max(580, 28 * len(top) + 160))
    _metric_axis(fig, c.metric, axis="x")
    fig.update_yaxes(title=None, autorange="reversed")
    return fig


def distribution_plot(c: Comparison) -> go.Figure:
    """/ Box plot of each run's per-capability values on the shared set (mean line +
    outliers), showing whether a macro average hides a heavy tail."""
    common = set(c.common_capabilities)
    fig = go.Figure()
    for run in c.data.runs:
        r = c.data.capabilities[
            (c.data.capabilities["run"] == run) & c.data.capabilities["capability"].isin(common)
        ]
        y = pd.to_numeric(r[c.metric.name], errors="coerce")
        fig.add_trace(
            go.Box(
                y=y,
                name=run,
                boxmean=True,
                boxpoints="outliers",
                customdata=r["capability"],
                hovertemplate="%{customdata}<br>" + c.metric.label + ": %{y:.4f}<extra>" + html.escape(run) + "</extra>",
            )
        )
    _base_layout(fig, height=520, title=f"Capability distribution — {c.metric.label}")
    _metric_axis(fig, c.metric, axis="y")
    fig.update_xaxes(title=None)
    return fig


def latency_quality(c: Comparison) -> go.Figure | None:
    """/ p50 latency (log axis) vs the primary metric, one point per capability per
    run — the speed/quality trade-off view. None without a latency column."""
    if "latency_ms_p50" not in c.data.capabilities.columns:
        return None
    common = set(c.common_capabilities)
    fig = go.Figure()
    for run in c.data.runs:
        r = c.data.capabilities[
            (c.data.capabilities["run"] == run) & c.data.capabilities["capability"].isin(common)
        ].copy()
        x = pd.to_numeric(r["latency_ms_p50"], errors="coerce")
        y = pd.to_numeric(r[c.metric.name], errors="coerce")
        n = pd.to_numeric(r["n"], errors="coerce")
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="markers",
                name=run,
                customdata=np.column_stack([r["capability"], n]),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>Model: " + html.escape(run) +
                    "<br>p50 latency: %{x:.1f} ms<br>" + c.metric.label + ": %{y:.4f}<br>n: %{customdata[1]}<extra></extra>"
                ),
                marker=dict(opacity=0.68, size=7),
            )
        )
    _base_layout(fig, height=620, title="Quality / latency trade-off by capability")
    fig.update_xaxes(title="p50 latency (ms)", type="log")
    _metric_axis(fig, c.metric, axis="y")
    return fig


def pairwise_baseline(c: Comparison) -> go.Figure | None:
    """/ Baseline value on x, model value on y (equal axes + parity diagonal): points
    above the line beat the baseline on that capability. None with a single run."""
    if len(c.data.runs) < 2:
        return None
    common = list(c.common_capabilities)
    base = c.metric_matrix.loc[common, c.baseline]
    finite = c.metric_matrix.loc[common].to_numpy(dtype=float)
    vals = finite[np.isfinite(finite)]
    if not len(vals):
        return None
    lo, hi = float(vals.min()), float(vals.max())
    if c.metric.format_kind == "probability":
        lo, hi = 0.0, 1.0
    pad = (hi - lo) * 0.04 if hi > lo else 0.05
    fig = go.Figure()
    for run in c.data.runs:
        if run == c.baseline:
            continue
        y = c.metric_matrix.loc[common, run]
        fig.add_trace(
            go.Scattergl(
                x=base,
                y=y,
                mode="markers",
                name=run,
                customdata=np.array(common, dtype=object),
                hovertemplate="<b>%{customdata}</b><br>Baseline: %{x:.4f}<br>Model: %{y:.4f}<extra>" + html.escape(run) + "</extra>",
                marker=dict(opacity=0.65, size=7),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[lo - pad, hi + pad],
            y=[lo - pad, hi + pad],
            mode="lines",
            name="Parity",
            hoverinfo="skip",
            line=dict(dash="dash"),
        )
    )
    _base_layout(fig, height=620, title=f"Pairwise capability parity vs {c.baseline}")
    fig.update_xaxes(title=f"{c.baseline} — {c.metric.label}", range=[lo - pad, hi + pad])
    fig.update_yaxes(title=f"Compared model — {c.metric.label}", range=[lo - pad, hi + pad], scaleanchor="x", scaleratio=1)
    tickformat = axis_tickformat(c.metric)
    if tickformat:
        fig.update_xaxes(tickformat=tickformat)
        fig.update_yaxes(tickformat=tickformat)
    return fig


def calibration_summary(c: Comparison) -> go.Figure | None:
    """/ Macro ECE (x) vs macro soft accuracy (y), labelled by run — the upper-left
    corner is the target. None unless the inputs carry calibration columns."""
    if "macro_ece_15" not in c.summary.columns or "macro_soft_accuracy" not in c.summary.columns:
        return None
    s = c.summary.reset_index()
    fig = go.Figure(
        go.Scatter(
            x=s["macro_ece_15"],
            y=s["macro_soft_accuracy"],
            mode="markers+text",
            text=s["run"],
            textposition="top center",
            customdata=np.column_stack([s["macro_nll"] if "macro_nll" in s else np.full(len(s), np.nan)]),
            hovertemplate="<b>%{text}</b><br>Macro ECE: %{x:.4f}<br>Macro soft accuracy: %{y:.4f}<br>Macro NLL: %{customdata[0]:.4f}<extra></extra>",
            marker=dict(size=12),
        )
    )
    _base_layout(fig, height=520, title="Calibration / correctness summary")
    fig.update_xaxes(title="Macro ECE (lower is better)", tickformat=".1%")
    fig.update_yaxes(title="Macro soft accuracy (higher is better)", tickformat=".1%", range=[0, 1])
    return fig


def _fig_fragment(fig: go.Figure, *, scroll: bool = False, div_id: str) -> str:
    """/ plotly div without the JS bundle (included once per report); optionally
    wrapped in a scroll container for very tall charts."""
    frag = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        config={"displaylogo": False, "responsive": True, "toImageButtonOptions": {"format": "svg"}},
        div_id=div_id,
    )
    if scroll:
        return f'<div class="plot-scroll"><div class="plot-scroll-inner">{frag}</div></div>'
    return frag


def _summary_table(c: Comparison) -> str:
    """/ Numeric summary as an HTML table; the reported-micro columns only appear
    when the stats files carried a micro aggregate."""
    show_micro = "reported_micro" in c.summary.columns and c.summary["reported_micro"].notna().any()
    cols = ["macro", "weighted", "median", "p10"]
    head = ["Model", "Macro", "Weighted", "Median", "p10"]
    if show_micro:
        cols += ["reported_micro", "reported_micro_n"]
        head += ["Reported micro", "Micro n"]
    cols += ["coverage", "examples", "min_n"]
    head += ["Coverage", "Shared examples", "Min n"]
    rows = []
    for run, r in c.summary.iterrows():
        cells = [html.escape(str(run))]
        for col in cols:
            v = r[col]
            if col in {"macro", "weighted", "median", "p10", "reported_micro"}:
                cells.append(fmt_value(v, c.metric))
            elif col == "coverage":
                cells.append("—" if pd.isna(v) else f"{float(v) * 100:.1f}%")
            else:
                cells.append("—" if pd.isna(v) else f"{int(v):,}")
        rows.append("<tr>" + "".join(f"<td>{x}</td>" for x in cells) + "</tr>")
    return (
        '<div class="table-wrap"><table class="summary"><thead><tr>'
        + "".join(f"<th>{html.escape(h)}</th>" for h in head)
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _capability_table(c: Comparison) -> str:
    """/ Exact per-capability values + n + deltas with a client-side filter box;
    values are formatted per the metric spec (pp for probability deltas)."""
    caps = _ordered_caps(c)
    headers = ["Capability"] + list(c.data.runs)
    if c.deltas.shape[1]:
        headers += [f"Δ {run} vs {c.baseline}" for run in c.deltas.columns]
    body = []
    for cap in caps:
        cells = [html.escape(cap)]
        for run in c.data.runs:
            v = c.metric_matrix.loc[cap, run] if run in c.metric_matrix.columns else np.nan
            n = c.n_matrix.loc[cap, run] if cap in c.n_matrix.index and run in c.n_matrix.columns else np.nan
            label = fmt_value(v, c.metric)
            if not pd.isna(n):
                label += f'<span class="n">n={int(n)}</span>'
            cells.append(label)
        for run in c.deltas.columns:
            v = c.deltas.loc[cap, run]
            if pd.isna(v):
                cells.append("—")
            elif c.metric.format_kind == "probability":
                cells.append(f"{float(v) * 100:+.2f} pp")
            else:
                cells.append(f"{float(v):+.4f}")
        body.append('<tr data-filter="' + html.escape(cap.lower(), quote=True) + '">' + "".join(f"<td>{v}</td>" for v in cells) + "</tr>")
    prefix = (
        '<input id="cap-search" class="search" type="search" placeholder="Filter capabilities…" aria-label="Filter capabilities">'
        '<div class="table-wrap capability-table"><table><thead><tr>'
    )
    return prefix + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr></thead><tbody id=\"cap-body\">" + "".join(body) + "</tbody></table></div>"


def _warnings(c: Comparison) -> list[str]:
    """/ Honest caveats rendered at the top of the report: unequal coverage, small
    per-capability n, saturated hard accuracy, direction-less metrics."""
    warnings: list[str] = []
    if len(c.common_capabilities) < len(c.union_capabilities):
        warnings.append(
            f"Coverage differs across models: {len(c.common_capabilities)} capabilities are shared out of "
            f"{len(c.union_capabilities)} in the union. Macro/weighted summaries use only the shared set."
        )
    min_n = int(c.summary["min_n"].min()) if len(c.summary) else 0
    if min_n < 20:
        warnings.append(
            f"At least one shared capability has n={min_n}. Per-capability point estimates are noisy at small n; "
            "use them diagnostically rather than as precise rankings."
        )
    if "accuracy" in c.data.capabilities.columns:
        acc = pd.to_numeric(c.data.capabilities["accuracy"], errors="coerce")
        if acc.notna().any() and acc.nunique(dropna=True) <= 2:
            warnings.append("Hard accuracy has very low variance in these inputs; probability-aware metrics may separate models better.")
    if c.metric.direction == "neutral":
        warnings.append(f"{c.metric.label} has no configured better/worse direction; delta signs are raw model-minus-baseline values.")
    return warnings


def build_report(c: Comparison, out_dir: Path, *, title: str, top_n: int = 40) -> Path:
    """/ Writes the single-file dashboard to out_dir/report.html (offline-capable:
    the plotly bundle is inlined) and returns its path. Figures that return None
    are skipped; None in the figures list never produces an empty section."""
    out_dir.mkdir(parents=True, exist_ok=True)

    figures: list[tuple[str, str, go.Figure | None, bool]] = [
        ("summary", "Model summary", summary_bar(c), False),
        ("heatmap", "All capabilities", capability_heatmap(c), True),
        ("delta", "Baseline deltas", delta_heatmap(c), True),
        ("discriminators", f"Most discriminating capabilities (top {top_n} by model spread)",
         discriminating_capabilities(c, top_n=top_n), False),
        ("distribution", "Capability distribution", distribution_plot(c), False),
        ("pairwise", "Pairwise parity", pairwise_baseline(c), False),
        ("latency", "Quality / latency", latency_quality(c), False),
        ("calibration", "Calibration", calibration_summary(c), False),
    ]

    cards = [
        ("Models", str(len(c.data.runs))),
        ("Shared capabilities", f"{len(c.common_capabilities):,}"),
        ("Union capabilities", f"{len(c.union_capabilities):,}"),
        ("Primary metric", c.metric.label),
        ("Baseline", c.baseline),
    ]
    warning_html = "".join(f'<div class="warning">{html.escape(w)}</div>' for w in _warnings(c))

    sections = []
    for key, heading, fig, scroll in figures:
        if fig is None:
            continue
        sections.append(
            f'<section id="{key}"><h2>{html.escape(heading)}</h2>'
            + _fig_fragment(fig, scroll=scroll, div_id=f"fig-{key}")
            + "</section>"
        )

    js = get_plotlyjs()
    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light; --bg:#f6f7f9; --card:#fff; --text:#16181d; --muted:#626a76; --line:#e3e6eb; --warn:#fff8dc; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ width:min(1540px, calc(100% - 32px)); margin:0 auto 64px; }}
header {{ padding:42px 0 20px; }}
h1 {{ font-size:30px; letter-spacing:-.025em; margin:0 0 8px; }}
.subtitle {{ color:var(--muted); margin:0; }}
.nav {{ display:flex; flex-wrap:wrap; gap:8px 14px; margin:14px 0 0; font-size:13px; }}
.nav a {{ color:#2456a6; text-decoration:none; }}
.nav a:hover {{ text-decoration:underline; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:20px 0; }}
.card, section {{ background:var(--card); border:1px solid var(--line); border-radius:14px; box-shadow:0 1px 2px rgba(16,24,40,.035); }}
.card {{ padding:16px 18px; }}
.card .k {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.05em; }}
.card .v {{ font-weight:680; font-size:20px; margin-top:6px; overflow-wrap:anywhere; }}
.warning {{ border:1px solid #eadb92; background:var(--warn); padding:11px 14px; border-radius:10px; margin:8px 0; font-size:13px; line-height:1.45; }}
section {{ margin:16px 0; padding:16px 18px 18px; overflow:hidden; }}
h2 {{ font-size:18px; margin:2px 0 8px; letter-spacing:-.01em; }}
.plot-scroll {{ max-height:860px; overflow:auto; border-top:1px solid var(--line); border-bottom:1px solid var(--line); }}
.plot-scroll-inner {{ min-width:760px; }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); border-radius:10px; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; background:white; }}
th, td {{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
th:first-child, td:first-child {{ text-align:left; position:sticky; left:0; background:white; z-index:1; }}
th {{ position:sticky; top:0; background:#fafbfc; z-index:2; font-weight:650; }}
th:first-child {{ z-index:3; background:#fafbfc; }}
.n {{ display:block; color:var(--muted); font-size:10px; margin-top:2px; }}
.search {{ width:min(520px,100%); padding:10px 12px; border:1px solid var(--line); border-radius:9px; margin:4px 0 10px; font:inherit; }}
.capability-table {{ max-height:720px; }}
.downloads a {{ display:inline-block; margin:4px 10px 4px 0; color:#2456a6; text-decoration:none; }}
.downloads a:hover {{ text-decoration:underline; }}
footer {{ color:var(--muted); font-size:12px; padding:12px 2px; }}
@media (max-width:720px) {{ main {{ width:min(100% - 16px,1540px); }} header {{ padding-top:24px; }} section {{ padding:10px; }} }}
</style>
<script>{js}</script>
</head>
<body>
<main>
<header>
<h1>{html.escape(title)}</h1>
<p class="subtitle">Per-capability comparison · primary metric: {html.escape(c.metric.label)} · positive baseline deltas mean improvement.</p>
<nav class="nav"><a href="#table-summary">Summary</a><a href="#heatmap">Capabilities</a><a href="#delta">Deltas</a><a href="#discriminators">Discriminators</a><a href="#latency">Latency</a><a href="#details">Exact values</a></nav>
<div class="cards">{''.join(f'<div class="card"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div></div>' for k,v in cards)}</div>
{warning_html}
</header>
<section id="table-summary"><h2>Numeric summary</h2>{_summary_table(c)}</section>
{''.join(sections)}
<section id="details"><h2>Capability values</h2>{_capability_table(c)}</section>
<section class="downloads"><h2>Machine-readable outputs</h2>
<a href="model_summary.csv">model_summary.csv</a>
<a href="capability_{html.escape(c.metric.name)}.csv">capability_{html.escape(c.metric.name)}.csv</a>
<a href="delta_vs_{html.escape(safe_name(c.baseline))}_{html.escape(c.metric.name)}.csv">baseline deltas</a>
<a href="capability_spread_{html.escape(c.metric.name)}.csv">capability spread</a>
<a href="all_rows.csv">all_rows.csv</a>
<a href="summary.json">summary.json</a>
</section>
<footer>Generated by evalcompare. Shared-set summaries prevent models with missing capabilities from benefiting from coverage differences.</footer>
</main>
<script>
const search = document.getElementById('cap-search');
const rows = [...document.querySelectorAll('#cap-body tr')];
search.addEventListener('input', () => {{
  const q = search.value.trim().toLowerCase();
  for (const row of rows) row.style.display = !q || row.dataset.filter.includes(q) ? '' : 'none';
}});
</script>
</body>
</html>"""
    path = out_dir / "report.html"
    path.write_text(report, encoding="utf-8")
    return path


def export_static_figures(c: Comparison, out_dir: Path, formats: Iterable[str], *, top_n: int = 40) -> list[Path]:
    """/ Renders the same figures as static images (svg/png/pdf/webp/jpg) into
    out_dir/figures/ — requires plotly's kaleido exporter; returns written paths.
    Not part of the make pipeline; provided for embedding decks/papers."""
    formats = tuple(dict.fromkeys(f.lower() for f in formats if f))
    if not formats:
        return []
    figs: dict[str, go.Figure | None] = {
        "summary": summary_bar(c),
        "heatmap": capability_heatmap(c),
        "delta": delta_heatmap(c),
        "discriminators": discriminating_capabilities(c, top_n=top_n),
        "distribution": distribution_plot(c),
        "pairwise": pairwise_baseline(c),
        "latency": latency_quality(c),
        "calibration": calibration_summary(c),
    }
    dest = out_dir / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, fig in figs.items():
        if fig is None:
            continue
        for fmt in formats:
            if fmt not in {"svg", "png", "pdf", "webp", "jpg", "jpeg"}:
                raise ValueError(f"unsupported static format: {fmt}")
            path = dest / f"{name}.{fmt}"
            pio.write_image(fig, path)
            written.append(path)
    return written
