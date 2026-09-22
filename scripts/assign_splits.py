#!/usr/bin/env python3
"""One-time explicit split migration for typed-decision-bench metadata.

Assigns an exact per-suite calibration share using deterministic SHA-256 rank,
then persists `split` on every metadata row. Runtime benchmark/calibration code
only reads this persisted field; it never recomputes membership.

Current benchmark policy: no training rows are created here. Existing explicit
splits are preserved unless --force is passed.
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
from pathlib import Path

from gpqa_zip import cleanup_unlocked, unlock_paths
from splits import VALID_SPLITS

DEFAULT_FRACTION = 0.20
DEFAULT_SEED = "typed-decision-explicit-splits-v1"


def rank_key(seed: str, capability: str, line_no: int, row: dict) -> tuple[bytes, int]:
    stable_id = str(row.get("request_sha256") or row.get("id") or f"{capability}:{line_no}")
    digest = hashlib.sha256(f"{seed}\0{capability}\0{stable_id}".encode("utf-8")).digest()
    return digest, line_no


def assign_rows(capability: str, rows: list[dict], fraction: float, seed: str, *, force: bool = False):
    if len(rows) < 2:
        raise ValueError(f"{capability}: need at least 2 cases for test/calibrate split")
    present = [row.get("split") for row in rows]
    valid_present = [value in VALID_SPLITS for value in present]
    if all(valid_present) and not force:
        return rows, False
    if any(value is not None for value in present) and not force:
        raise ValueError(f"{capability}: partial/invalid existing split assignment; use --force to replace")

    n_cal = int(math.floor(len(rows) * fraction + 0.5))
    n_cal = min(len(rows) - 1, max(1, n_cal))
    ranked = sorted(range(len(rows)), key=lambda i: rank_key(seed, capability, i + 1, rows[i]))
    calibrate = set(ranked[:n_cal])
    out = []
    for i, row in enumerate(rows):
        updated = dict(row)
        updated["split"] = "calibrate" if i in calibrate else "test"
        out.append(updated)
    return out, True


def main() -> None:
    ap = argparse.ArgumentParser(description="Persist explicit train/calibrate/test split metadata.")
    ap.add_argument("--benchmark", default=".")
    ap.add_argument("--calibrate-fraction", type=float, default=DEFAULT_FRACTION)
    ap.add_argument("--seed", default=DEFAULT_SEED)
    ap.add_argument("--fix", action="store_true", help="Write metadata files; default is report-only")
    ap.add_argument("--force", action="store_true", help="Replace already-present split assignments")
    args = ap.parse_args()
    if not 0.0 < args.calibrate_fraction < 1.0:
        ap.error("--calibrate-fraction must be between 0 and 1")

    root = Path(args.benchmark)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    meta_paths = [root / (entry.get("metadata") or entry.get("metadata_file"))
                  for entry in manifest["capabilities"].values()
                  if entry.get("metadata") or entry.get("metadata_file")]
    atexit.register(cleanup_unlocked, unlock_paths(meta_paths))

    totals = {name: 0 for name in VALID_SPLITS}
    changed = 0
    for capability, entry in manifest["capabilities"].items():
        rel = entry.get("metadata") or entry.get("metadata_file")
        if not rel:
            raise SystemExit(f"{capability}: manifest has no metadata path")
        path = root / rel
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected = entry.get("cases")
        if expected is not None and len(rows) != int(expected):
            raise SystemExit(f"{capability}: metadata line count {len(rows)} != manifest cases {expected}")
        try:
            updated, did_change = assign_rows(capability, rows, args.calibrate_fraction, args.seed, force=args.force)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        counts = {name: sum(row["split"] == name for row in updated) for name in VALID_SPLITS}
        for name in VALID_SPLITS:
            totals[name] += counts[name]
        if did_change:
            changed += 1
            if args.fix:
                path.write_text("\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in updated) + "\n",
                                encoding="utf-8")
        print(f"{capability}: test={counts['test']} calibrate={counts['calibrate']} train={counts['train']}"
              + (" [write]" if args.fix and did_change else ""))

    print(f"TOTAL: test={totals['test']} calibrate={totals['calibrate']} train={totals['train']} "
          f"changed_suites={changed} mode={'fix' if args.fix else 'report'}")


if __name__ == "__main__":
    main()
