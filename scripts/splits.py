"""Explicit train/calibrate/test split helpers for typed-decision-bench.

The benchmark persists split membership in line-aligned metadata/*.jsonl files.
Runtime tools read that field; they never invent or re-hash split membership.
Missing metadata is tolerated only by callers that explicitly opt into legacy
fallback behavior (used by small synthetic unit-test fixtures).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

VALID_SPLITS = ("train", "calibrate", "test")
DEFAULT_EVAL_SPLITS = ("test", "calibrate")


def parse_splits(value: str | Iterable[str] | None) -> tuple[str, ...]:
    """Parse/validate a split selection while preserving user order."""
    if value is None:
        return DEFAULT_EVAL_SPLITS
    raw = value.split(",") if isinstance(value, str) else list(value)
    selected = []
    for item in raw:
        split = str(item).strip()
        if not split:
            continue
        if split not in VALID_SPLITS:
            raise ValueError(f"invalid split {split!r}; expected one of {VALID_SPLITS}")
        if split not in selected:
            selected.append(split)
    if not selected:
        raise ValueError("at least one split must be selected")
    return tuple(selected)


def metadata_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def capability_splits(root: Path, capability: str, entry: dict, *, strict: bool = False,
                      expected_cases: int | None = None) -> list[str]:
    """Return one split per source line for a manifest capability.

    strict=True is for repository validation and requires a metadata file plus
    an explicit valid split on every line. strict=False deliberately supports
    old/minimal fixtures by treating missing metadata/split as `test`.
    """
    meta_rel = entry.get("metadata") or entry.get("metadata_file")
    if not meta_rel:
        if strict:
            raise AssertionError(f"{capability}: manifest has no metadata path")
        return ["test"] * int(expected_cases or entry.get("cases") or 0)
    path = root / meta_rel
    if not path.exists():
        if strict:
            raise AssertionError(f"{capability}: metadata file missing: {meta_rel}")
        return ["test"] * int(expected_cases or entry.get("cases") or 0)

    rows = metadata_rows(path)
    expected = expected_cases if expected_cases is not None else entry.get("cases")
    if expected is not None and len(rows) != int(expected):
        raise AssertionError(f"{capability}: metadata line count {len(rows)} != {expected}")

    out = []
    for i, row in enumerate(rows, 1):
        split = row.get("split")
        if split is None and not strict:
            split = "test"
        if split not in VALID_SPLITS:
            raise AssertionError(
                f"{capability}:{i}: metadata split must be one of {VALID_SPLITS}, got {split!r}"
            )
        out.append(split)
    return out


def split_map(root: Path, manifest: dict, *, capabilities: Iterable[str] | None = None,
              strict: bool = False) -> dict[tuple[str, int], str]:
    caps = manifest["capabilities"]
    names = list(capabilities) if capabilities is not None else list(caps)
    result: dict[tuple[str, int], str] = {}
    for cap in names:
        entry = caps[cap]
        splits = capability_splits(root, cap, entry, strict=strict)
        result.update({(cap, i): split for i, split in enumerate(splits, 1)})
    return result


def keys_for_split(mapping: dict[tuple[str, int], str], split: str) -> set[tuple[str, int]]:
    if split not in VALID_SPLITS:
        raise ValueError(f"invalid split {split!r}")
    return {key for key, value in mapping.items() if value == split}


def split_counts(mapping: dict[tuple[str, int], str]) -> dict[str, int]:
    return {name: sum(value == name for value in mapping.values()) for name in VALID_SPLITS}
