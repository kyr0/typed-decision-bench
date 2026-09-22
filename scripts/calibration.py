#!/usr/bin/env python3
"""Fit temperature calibration from explicit metadata split membership.

Only metadata rows with split="calibrate" are used to fit T. split="test" is
held out completely from fitting and is used only for offline raw-vs-calibrated
reporting. split="train" is ignored.

The transform is q_i = normalize(p_i ** (1 / T)), exactly equivalent to
softmax(logits / T) for a single softmax distribution. It therefore works from
System One probability responses without exposing raw logits, and can be loaded
by an inference server later.

Artifacts are always schema v2: `temperatures` maps every answer type that
reached the per-group data floor to its own T, and the top-level `temperature`
stays the global fit and the fallback for every group without its own T
(missing/unknown qtype -> `temperature`). `scope` distinguishes the mode
("per_qtype" if any group was fitted, else "global"). Loader contract:
`T = temperatures.get(qtype, artifact["temperature"])`, requiring
`schema_version == 2` (v1 artifacts are no longer produced).
"""
from __future__ import annotations

import argparse
import atexit
import datetime as _dt
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    from .gpqa_zip import cleanup_unlocked, unlock_paths
    from .score import CASE_ID_PREFIX, distributions, last_records, newest_log, rows, run_name
    from .splits import capability_splits
except ImportError:
    from gpqa_zip import cleanup_unlocked, unlock_paths
    from score import CASE_ID_PREFIX, distributions, last_records, newest_log, rows, run_name
    from splits import capability_splits

SCHEMA_VERSION = 2  # the only emitted shape: v1 keys + mandatory `temperatures` map
KIND = "typed-decision-temperature-calibration"
METHOD = "temperature_scaling"
DEFAULT_MIN_CASES = 100
DEFAULT_QTYPE_MIN_CASES = 100  # per-group floor: below this a qtype falls back to global T
EPS = 1e-12
MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 20.0


@dataclass(frozen=True)
class CalibrationExample:
    case_id: str
    family: str
    qtype: str
    split: str
    gold: tuple[float, ...]
    predicted: tuple[float, ...]


def _normalized(values: Sequence[float]) -> tuple[float, ...]:
    xs = [max(0.0, float(x)) for x in values]
    total = math.fsum(xs)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("probability vector has no finite positive mass")
    return tuple(x / total for x in xs)


def temperature_scale(probabilities: Sequence[float], temperature: float) -> tuple[float, ...]:
    """Temperature-scale a probability vector without source logits."""
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and > 0")
    p = _normalized(probabilities)
    beta = 1.0 / temperature
    scaled = [beta * math.log(max(EPS, x)) for x in p]
    peak = max(scaled)
    weights = [math.exp(x - peak) for x in scaled]
    z = math.fsum(weights)
    return tuple(x / z for x in weights)


def _gradient(examples: Sequence[CalibrationExample], beta: float) -> float:
    """d mean soft-label cross-entropy / d beta, beta=1/T."""
    total = 0.0
    for ex in examples:
        p = _normalized(ex.predicted)
        y = _normalized(ex.gold)
        logp = [math.log(max(EPS, x)) for x in p]
        scaled = [beta * x for x in logp]
        peak = max(scaled)
        weights = [math.exp(x - peak) for x in scaled]
        z = math.fsum(weights)
        eq = math.fsum((w / z) * lp for w, lp in zip(weights, logp))
        ey = math.fsum(yy * lp for yy, lp in zip(y, logp))
        total += eq - ey
    return total / len(examples)


def fit_temperature(examples: Sequence[CalibrationExample]) -> tuple[float, bool]:
    """Return (temperature, at_bound) minimizing calibration-split NLL."""
    if not examples:
        raise ValueError("cannot fit calibration with zero examples")
    lo = 1.0 / MAX_TEMPERATURE
    hi = 1.0 / MIN_TEMPERATURE
    g_lo = _gradient(examples, lo)
    g_hi = _gradient(examples, hi)
    if g_lo >= 0.0:
        return MAX_TEMPERATURE, True
    if g_hi <= 0.0:
        return MIN_TEMPERATURE, True
    for _ in range(80):
        mid = (lo + hi) * 0.5
        if _gradient(examples, mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 1.0 / ((lo + hi) * 0.5), False


def _ece(rows_: Sequence[tuple[float, int]], bins: int = 15) -> float | None:
    if not rows_:
        return None
    value = 0.0
    n = len(rows_)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        xs = [(conf, correct) for conf, correct in rows_
              if lo <= conf < hi or (b == bins - 1 and conf >= lo)]
        if not xs:
            continue
        mean_conf = math.fsum(x[0] for x in xs) / len(xs)
        mean_acc = math.fsum(x[1] for x in xs) / len(xs)
        value += len(xs) / n * abs(mean_conf - mean_acc)
    return value


def _temperature_for(temperature, qtype: str) -> float:
    """Resolve the temperature for one qtype: scalar, or per-qtype dict with
    the global T under the reserved `__default__` key."""
    if isinstance(temperature, dict):
        return float(temperature.get(qtype, temperature.get("__default__", 1.0)))
    return float(temperature)


def qtype_groups(examples: Sequence[CalibrationExample]) -> dict[str, list[CalibrationExample]]:
    """Group examples by answer type (the natural miscalibration axis)."""
    groups: dict[str, list[CalibrationExample]] = {}
    for ex in examples:
        groups.setdefault(ex.qtype, []).append(ex)
    return groups


def fit_qtype_temperatures(fit_examples: Sequence[CalibrationExample], min_cases: int):
    """Per-qtype T for groups with >= min_cases calibrate examples.

    Small groups are deliberately NOT fitted (their T would be noise, e.g. 18
    score cases fitted a wrong-direction 0.93 on the bonsai run); they fall back
    to the global T. Returns (temperatures, per-group info incl. fallbacks).
    """
    temps: dict[str, float] = {}
    info: dict[str, dict] = {}
    for qtype, exs in sorted(qtype_groups(fit_examples).items()):
        entry = {"calibration_n": len(exs), "temperature": None,
                 "temperature_at_search_bound": None, "case_ids_sha256": _ids_digest(exs)}
        if len(exs) >= min_cases:
            entry["temperature"], entry["temperature_at_search_bound"] = fit_temperature(exs)
            temps[qtype] = entry["temperature"]
        info[qtype] = entry
    return temps, info


def metrics(examples: Sequence[CalibrationExample], temperature=1.0) -> dict:
    """Mean calibration metrics; `temperature` is a scalar or a per-qtype dict."""
    if not examples:
        return {"n": 0, "nll": None, "brier": None, "ece_15": None,
                "accuracy": None, "soft_accuracy": None, "confidence": None}
    nll = brier = soft_accuracy = confidence = 0.0
    hard = []
    for ex in examples:
        y = _normalized(ex.gold)
        p = temperature_scale(ex.predicted, _temperature_for(temperature, ex.qtype))
        yi = max(range(len(y)), key=y.__getitem__)
        pi = max(range(len(p)), key=p.__getitem__)
        conf = max(p)
        correct = int(pi == yi)
        hard.append((conf, correct))
        confidence += conf
        nll += -math.fsum(yy * math.log(max(EPS, pp)) for yy, pp in zip(y, p))
        brier += math.fsum((pp - yy) ** 2 for pp, yy in zip(p, y))
        soft_accuracy += math.fsum(pp * yy for pp, yy in zip(p, y))
    n = len(examples)
    return {
        "n": n,
        "nll": nll / n,
        "brier": brier / n,
        "ece_15": _ece(hard, 15),
        "accuracy": math.fsum(x[1] for x in hard) / n,
        "soft_accuracy": soft_accuracy / n,
        "confidence": confidence / n,
    }


def reliability_bins(examples: Sequence[CalibrationExample], temperature=1.0,
                     bins: int = 15) -> list[dict]:
    bucketed = [[] for _ in range(bins)]
    for ex in examples:
        y = _normalized(ex.gold)
        p = temperature_scale(ex.predicted, _temperature_for(temperature, ex.qtype))
        yi = max(range(len(y)), key=y.__getitem__)
        pi = max(range(len(p)), key=p.__getitem__)
        conf = max(p)
        bucketed[min(bins - 1, int(conf * bins))].append((conf, int(pi == yi)))
    result = []
    for b, xs in enumerate(bucketed):
        mean_conf = math.fsum(x[0] for x in xs) / len(xs) if xs else None
        accuracy = math.fsum(x[1] for x in xs) / len(xs) if xs else None
        result.append({
            "bin": b, "lower": b / bins, "upper": (b + 1) / bins, "n": len(xs),
            "mean_confidence": mean_conf, "accuracy": accuracy,
            "gap": abs(mean_conf - accuracy) if xs else None,
        })
    return result


def qtype_metrics(examples: Sequence[CalibrationExample], temperature: float) -> dict:
    return {
        qtype: metrics([ex for ex in examples if ex.qtype == qtype], temperature)
        for qtype in sorted({ex.qtype for ex in examples})
    }


def _ids_digest(examples: Iterable[CalibrationExample]) -> str:
    h = hashlib.sha256()
    for case_id in sorted(ex.case_id for ex in examples):
        h.update(case_id.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def collect_examples(root: Path, manifest: dict, log_path: Path) -> tuple[dict[str, list[CalibrationExample]], dict]:
    """Join successful run responses to gold + persisted metadata split."""
    records = last_records(log_path)
    by_split: dict[str, list[CalibrationExample]] = {"train": [], "calibrate": [], "test": []}
    models, endpoints = set(), set()
    for record in records.values():
        if record.get("error_type") or not isinstance(record.get("response"), dict):
            continue
        response = record["response"]
        if isinstance(response.get("model"), str) and response["model"]:
            models.add(response["model"])
        if isinstance(record.get("endpoint"), str) and record["endpoint"]:
            endpoints.add(record["endpoint"])

    covered_caps = {cap for cap, _ in records}
    for cap, entry in manifest["capabilities"].items():
        if cap not in covered_caps:
            continue
        gold_rows = rows(root / entry["responses"])
        splits = capability_splits(root, cap, entry, strict=True, expected_cases=len(gold_rows))
        for i, (gold_response, split) in enumerate(zip(gold_rows, splits), 1):
            if split == "train":
                continue
            record = records.get((cap, i))
            if not record or record.get("error_type") or not isinstance(record.get("response"), dict):
                continue
            predicted_response = record["response"]
            gold_answers = gold_response.get("answers")
            pred_answers = predicted_response.get("answers")
            if not isinstance(gold_answers, dict) or not gold_answers or not isinstance(pred_answers, dict):
                continue
            qid = next(iter(gold_answers))
            if qid not in pred_answers:
                continue
            ga, pa = gold_answers[qid], pred_answers[qid]
            _, y, p = distributions(ga, pa)
            by_split[split].append(CalibrationExample(
                case_id=f"{CASE_ID_PREFIX}:{cap}:{i:04d}",
                family=cap,
                qtype=ga["type"],
                split=split,
                gold=_normalized(y),
                predicted=_normalized(p),
            ))

    provenance = {
        "models": sorted(models),
        "endpoints": sorted(endpoints),
        "successful_scored_cases": sum(len(v) for v in by_split.values()),
    }
    return by_split, provenance


def build_artifact(root: Path, manifest: dict, log_path: Path, *, min_cases: int = DEFAULT_MIN_CASES,
                   qtype_min_cases: int = DEFAULT_QTYPE_MIN_CASES) -> dict:
    by_split, provenance = collect_examples(root, manifest, log_path)
    fit = by_split["calibrate"]
    test = by_split["test"]
    enough = len(fit) >= min_cases
    temperature, at_bound = fit_temperature(fit) if enough else (1.0, False)

    # per-qtype layer: fitted only from calibrate rows with enough data per group
    temps, qinfo = fit_qtype_temperatures(fit, qtype_min_cases) if enough else ({}, {})
    per_qtype = bool(temps)  # scope only: the artifact is always schema v2
    scope = "per_qtype" if per_qtype else "global"
    cal_t = {**temps, "__default__": temperature} if per_qtype else temperature

    raw_fit = metrics(fit, 1.0)
    calibrated_fit = metrics(fit, cal_t)
    raw_test = metrics(test, 1.0)
    calibrated_test = metrics(test, cal_t)
    test_nll_improved = bool(
        raw_test["nll"] is not None and calibrated_test["nll"] is not None
        and calibrated_test["nll"] < raw_test["nll"] - 1e-12
    )

    provenance_ok = len(provenance["models"]) == 1 and len(provenance["endpoints"]) == 1
    deployable = bool(enough and provenance_ok)
    if not enough:
        status = "insufficient_calibration_data"
    elif not provenance_ok:
        status = "mixed_provenance"
    else:
        status = "ok"
    provenance["model"] = provenance["models"][0] if len(provenance["models"]) == 1 else None
    provenance["endpoint"] = provenance["endpoints"][0] if len(provenance["endpoints"]) == 1 else None

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "method": METHOD,
        "scope": scope,
        "deployable": deployable,
        "status": status,
        "temperature": temperature,
        "inverse_temperature": 1.0 / temperature,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "source": {
            "benchmark": CASE_ID_PREFIX,
            "run": run_name(log_path),
            "log": log_path.name,
            **provenance,
        },
        "split": {
            "strategy": "metadata_field_v1",
            "field": "split",
            "allowed_values": ["train", "calibrate", "test"],
            "fit": "calibrate",
            "held_out_report": "test",
            "excluded": "train",
            "minimum_calibration_cases": min_cases,
            "calibration_n": len(fit),
            "test_n": len(test),
            "calibration_case_ids_sha256": _ids_digest(fit),
            "test_case_ids_sha256": _ids_digest(test),
        },
        "fit": {
            "objective": "mean_cross_entropy_full_gold_distribution",
            "temperature_at_search_bound": at_bound,
            "temperature_bounds": [MIN_TEMPERATURE, MAX_TEMPERATURE],
            "uses_test_data": False,
            "qtype_minimum_cases": qtype_min_cases,
            "qtype_fitted_groups": sorted(temps),
            "qtype_fallback_temperature": temperature,
        },
        "metrics": {
            "calibrate": {"raw": raw_fit, "calibrated": calibrated_fit},
            "test": {"raw": raw_test, "calibrated": calibrated_test},
        },
        "analysis": {
            "test_nll_improved": test_nll_improved,
            "test_by_qtype": {
                "raw": qtype_metrics(test, 1.0),
                "calibrated": qtype_metrics(test, cal_t),
            },
            "reliability_15": {
                "calibrate": {
                    "raw": reliability_bins(fit, 1.0),
                    "calibrated": reliability_bins(fit, cal_t),
                },
                "test": {
                    "raw": reliability_bins(test, 1.0),
                    "calibrated": reliability_bins(test, cal_t),
                },
            },
        },
    }
    artifact["temperatures"] = temps  # always present (possibly empty) in v2
    artifact["qtype_calibration"] = {
        "fallback": "global",
        "minimum_cases": qtype_min_cases,
        "groups": qinfo,
    }
    return artifact


def calibrate_log(root: Path, log_path: Path, *, out: Path | None = None,
                  min_cases: int = DEFAULT_MIN_CASES,
                  qtype_min_cases: int = DEFAULT_QTYPE_MIN_CASES) -> tuple[Path, dict]:
    """Fit T from split=calibrate; the artifact lands next to the run log it came from."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    artifact = build_artifact(root, manifest, log_path, min_cases=min_cases,
                              qtype_min_cases=qtype_min_cases)
    # the artifact belongs with its source log (e.g. --log somewhere/other/run.jsonl),
    # so the default follows the log directory instead of hardcoding output/
    out = out or log_path.parent / f"{run_name(log_path)}_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out, artifact


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit temperature calibration from metadata split=calibrate.")
    ap.add_argument("--benchmark", default=".")
    ap.add_argument("--log", default=None, help="Combined run log; default: newest output/*.jsonl")
    ap.add_argument("--out", default=None, help="Default: output/<run>_calibration.json")
    ap.add_argument("--min-cases", type=int, default=DEFAULT_MIN_CASES,
                    help=f"Minimum successful calibrate cases required (default: {DEFAULT_MIN_CASES})")
    ap.add_argument("--qtype-min-cases", type=int, default=DEFAULT_QTYPE_MIN_CASES,
                    help=f"Minimum calibrate cases per answer type to fit its own temperature "
                         f"(default: {DEFAULT_QTYPE_MIN_CASES}; smaller groups fall back to global T)")
    args = ap.parse_args()
    if args.min_cases < 1:
        ap.error("--min-cases must be >= 1")
    if args.qtype_min_cases < 2:
        ap.error("--qtype-min-cases must be >= 2")
    root = Path(args.benchmark)
    if args.log:
        log_path = Path(args.log)
        # a missing --log is almost always a run name that was never run: say exactly that
        if not log_path.exists():
            ap.error(f"run log not found: {log_path} (run `make eval` first, or point --log at an existing output/*.jsonl)")
    else:
        log_path = newest_log(root)
        if log_path is None:
            ap.error("no run logs in output/; run `make eval` first or pass --log")

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    covered = {cap for cap, _ in last_records(log_path)}
    paths = []
    for cap in covered:
        entry = manifest["capabilities"].get(cap)
        if not entry:
            continue
        paths.extend(root / entry[k] for k in ("responses", "metadata") if entry.get(k))
    atexit.register(cleanup_unlocked, unlock_paths(paths))

    out, artifact = calibrate_log(root, log_path, out=Path(args.out) if args.out else None,
                                  min_cases=args.min_cases, qtype_min_cases=args.qtype_min_cases)
    print(json.dumps({
        "path": str(out),
        "deployable": artifact["deployable"],
        "status": artifact["status"],
        "scope": artifact["scope"],
        "temperature": artifact["temperature"],
        "temperatures": artifact.get("temperatures"),
        "calibrate": artifact["metrics"]["calibrate"],
        "test": artifact["metrics"]["test"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
