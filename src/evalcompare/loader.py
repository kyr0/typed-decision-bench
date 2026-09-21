"""/ Input gate of the evalcompare library: loads stats JSONL files into DataFrames.

Each input file must carry exactly one ``run`` (model) name and one row per
capability; rows need at least ``run``, ``capability`` and ``n`` (``REQUIRED``).
Duplicate ``(run, capability)`` pairs and negative ``n`` are rejected so the
downstream capability x run pivots are unambiguous. Rows whose capability is one
of ``DEFAULT_AGGREGATES`` (e.g. the scorer's ``micro`` line) are split out into
``EvalData.aggregates`` so per-capability analysis never mixes them in.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED = {"run", "capability", "n"}
DEFAULT_AGGREGATES = frozenset({"micro", "macro", "overall", "summary"})


@dataclass(frozen=True)
class EvalData:
    """/ Immutable load result: every row, per-capability vs aggregate rows split,
    run order matching the input file order, and the resolved source paths."""
    rows: pd.DataFrame
    capabilities: pd.DataFrame
    aggregates: pd.DataFrame
    runs: tuple[str, ...]
    source_files: tuple[Path, ...]


def _read_file(path: Path) -> list[dict]:
    """/ Parses one JSONL stats file into dicts tagged with ``_source_file``/``_line``
    for precise error messages; every line must be an object with the REQUIRED fields."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            missing = REQUIRED - obj.keys()
            if missing:
                raise ValueError(f"{path}:{line_no}: missing fields: {sorted(missing)}")
            obj = dict(obj)
            obj["_source_file"] = str(path)
            obj["_line"] = line_no
            rows.append(obj)
    if not rows:
        raise ValueError(f"{path}: empty JSONL")
    return rows


def load_eval_files(
    paths: Iterable[str | Path],
    *,
    aggregate_names: Iterable[str] = DEFAULT_AGGREGATES,
) -> EvalData:
    """/ Loads one or more single-run stats files into an EvalData.

    Hard guarantees (each raises): >= 1 file, one run name per file, run names
    unique across files, no duplicate (run, capability) rows, n >= 0. File order
    defines run order, which matters: comparison code puts the baseline first
    for a sensible default ordering.
    """
    source_files = tuple(Path(p).expanduser().resolve() for p in paths)
    if len(source_files) < 1:
        raise ValueError("at least one JSONL file is required")

    all_rows: list[dict] = []
    run_order: list[str] = []
    for path in source_files:
        if not path.is_file():
            raise FileNotFoundError(path)
        file_rows = _read_file(path)
        file_runs = list(dict.fromkeys(str(r["run"]) for r in file_rows))
        if len(file_runs) != 1:
            raise ValueError(
                f"{path}: expected exactly one run/model name, found {file_runs}. "
                "Keep one model run per JSONL file."
            )
        run = file_runs[0]
        if run in run_order:
            raise ValueError(
                f"run/model name {run!r} appears in more than one input file; run names must be unique"
            )
        run_order.append(run)
        all_rows.extend(file_rows)

    df = pd.DataFrame(all_rows)
    df["run"] = df["run"].astype(str)
    df["capability"] = df["capability"].astype(str)

    dupes = df.duplicated(["run", "capability"], keep=False)
    if dupes.any():
        details = df.loc[dupes, ["run", "capability", "_source_file", "_line"]]
        raise ValueError("duplicate (run, capability) rows:\n" + details.to_string(index=False))

    if (pd.to_numeric(df["n"], errors="coerce") < 0).any():
        raise ValueError("n must be >= 0")

    aggregates_set = set(aggregate_names)
    aggregate_mask = df["capability"].isin(aggregates_set)
    capabilities = df.loc[~aggregate_mask].copy()
    aggregates = df.loc[aggregate_mask].copy()

    return EvalData(
        rows=df,
        capabilities=capabilities,
        aggregates=aggregates,
        runs=tuple(run_order),
        source_files=source_files,
    )


def numeric_metrics(df: pd.DataFrame) -> list[str]:
    """/ Columns with at least one parseable numeric value (identity/metadata columns excluded)."""
    ignored = {"run", "capability", "_source_file", "_line"}
    out: list[str] = []
    for col in df.columns:
        if col in ignored:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().any():
            out.append(col)
    return out
