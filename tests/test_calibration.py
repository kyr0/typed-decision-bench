"""Calibration fitting/artifact regression tests for explicit metadata splits."""
import json
from pathlib import Path

import pytest

import calibration


def _ex(i, gold, predicted, split="calibrate"):
    return calibration.CalibrationExample(
        str(i), "fake", "choice", split, tuple(gold), tuple(predicted))


def test_temperature_fit_softens_overconfident_predictions():
    examples = []
    # Always predicts class 0 at 99%, but class 0 is correct only 75% of the time.
    for i in range(100):
        examples.append(_ex(i, (0.0, 1.0) if i % 4 == 0 else (1.0, 0.0), (0.99, 0.01)))
    temperature, at_bound = calibration.fit_temperature(examples)
    assert temperature > 1.0 and not at_bound
    assert calibration.metrics(examples, temperature)["nll"] < calibration.metrics(examples, 1.0)["nll"]


def test_temperature_fit_is_identity_when_prediction_matches_soft_gold():
    examples = [_ex(i, (0.8, 0.2), (0.8, 0.2)) for i in range(50)]
    temperature, _ = calibration.fit_temperature(examples)
    assert temperature == pytest.approx(1.0, abs=1e-8)


def test_qtype_fallback_respects_group_minimum():
    """Groups below the per-group floor must NOT get their own T (small-group T
    is noise, e.g. 18 score cases fitting a wrong-direction sharpening)."""
    def _ex_t(qtype, n):  # qtype-tagged variant of the module fixture helper
        return [calibration.CalibrationExample(f"{qtype}{i}", "fake", qtype, "calibrate",
                                               (0.8, 0.2), (0.9, 0.1)) for i in range(n)]
    fit = _ex_t("choice", 120) + _ex_t("noul", 60)
    temps, info = calibration.fit_qtype_temperatures(fit, 100)
    assert set(temps) == {"choice"} and info["choice"]["temperature"] is not None
    assert info["noul"]["temperature"] is None  # fallback: below the 100-case floor
    assert info["noul"]["calibration_n"] == 60 and info["noul"]["case_ids_sha256"]
    temps_low, _ = calibration.fit_qtype_temperatures(fit, 50)
    assert set(temps_low) == {"choice", "noul"}


def test_metrics_temperature_dict_applies_per_qtype_T():
    """metrics() with a per-qtype dict must apply that qtype's T (== the scalar
    path) and fall back to __default__ for other groups."""
    overconf = [calibration.CalibrationExample(str(i), "fake", "noul", "test",
                                               (0.0, 1.0) if i % 4 == 0 else (1.0, 0.0),
                                               (0.01, 0.99) if i % 4 != 0 else (0.99, 0.01))
                for i in range(100)]
    with_dict = calibration.metrics(overconf, {"noul": 2.0, "__default__": 1.0})
    assert with_dict == calibration.metrics(overconf, 2.0)
    assert with_dict["nll"] < calibration.metrics(overconf, 1.0)["nll"]  # T>1 softens
    with_default = calibration.metrics(overconf, {"other": 2.0, "__default__": 1.0})
    assert with_default == calibration.metrics(overconf, 1.0)  # unknown qtype -> fallback


def test_artifact_always_v2_with_global_fallback(tmp_path: Path):
    """noul (120 overconfident calibrate cases) gets its own T while the smaller
    choice group falls back and the top-level `temperature` keeps the diluted
    global fit; with an unreachable qtype floor the artifact stays schema v2 but
    scope=global and `temperatures` is empty (loaders fall back to top-level T)."""
    (tmp_path / "responses").mkdir()
    (tmp_path / "metadata").mkdir()
    (tmp_path / "output").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"capabilities": {
        "fake": {"responses": "responses/fake.jsonl", "metadata": "metadata/fake.jsonl", "cases": 430}}}))
    gold_rows, meta_rows, log = [], [], []
    for i in range(1, 431):
        split = "calibrate" if i <= 210 else "test"
        qtype = "noul" if (i - 1) % 210 < 120 else "choice"  # 120 noul + 90 choice per split
        if qtype == "noul":
            # overconfident AND wrong 25% of the time: always claims false@0.99, gold true 25%
            gold = {"type": "noul", "noul": 1.0 if i % 4 == 0 else 0.0}
            pred = {"type": "noul", "noul": 0.01}
        else:
            gold = {"type": "choice", "choice": "a", "probabilities": {"a": 1.0, "b": 0.0}}
            pred = {"type": "choice", "probabilities": {"a": 0.9, "b": 0.1}}  # calibrated
        gold_rows.append({"answers": {"q": gold}})
        meta_rows.append({"id": f"fake-{i:03d}", "split": split})
        log.append({"request_id": f"fake-{i:03d}", "response": {"model": "m", "answers": {"q": pred}},
                    "error_type": None, "error": None, "duration_ms": 1.0, "endpoint": "http://x/v1/systemone"})
    (tmp_path / "responses" / "fake.jsonl").write_text("\n".join(map(json.dumps, gold_rows)) + "\n")
    (tmp_path / "metadata" / "fake.jsonl").write_text("\n".join(map(json.dumps, meta_rows)) + "\n")
    log_path = tmp_path / "output" / "run.jsonl"
    log_path.write_text("\n".join(map(json.dumps, log)) + "\n")

    _, v2 = calibration.calibrate_log(tmp_path, log_path, min_cases=20, qtype_min_cases=100)
    assert v2["schema_version"] == 2 and v2["scope"] == "per_qtype"
    assert "noul" in v2["temperatures"] and v2["temperatures"]["noul"] > 1.0
    assert "choice" not in v2["temperatures"]  # 90 < 100 floor -> fallback
    assert v2["qtype_calibration"]["fallback"] == "global"
    assert v2["qtype_calibration"]["groups"]["noul"]["calibration_n"] == 120
    # the global scalar is diluted by the calibrated choice group -> differs from noul's T
    assert v2["temperature"] < v2["temperatures"]["noul"]
    # per-group calibrated metrics used the GROUP temperature (nll improved on test)
    raw_q = v2["analysis"]["test_by_qtype"]["raw"]["noul"]
    cal_q = v2["analysis"]["test_by_qtype"]["calibrated"]["noul"]
    assert cal_q["nll"] < raw_q["nll"] and cal_q["ece_15"] < raw_q["ece_15"]
    assert v2["deployable"] is True  # unchanged policy: calibrate-only counts + provenance

    _, v1 = calibration.calibrate_log(tmp_path, log_path, min_cases=20, qtype_min_cases=100000)
    assert v1["schema_version"] == 2 and v1["scope"] == "global" and v1["temperatures"] == {}
    # groups are still reported (with fallbacks) so the data need is visible for the next run
    assert v1["qtype_calibration"]["groups"]["noul"]["temperature"] is None


def test_artifact_fits_only_calibrate_and_test_never_controls_deployability(tmp_path: Path):
    (tmp_path / "responses").mkdir()
    (tmp_path / "metadata").mkdir()
    (tmp_path / "output").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"capabilities": {
        "fake": {
            "responses": "responses/fake.jsonl",
            "metadata": "metadata/fake.jsonl",
            "cases": 200,
        },
        # Deliberately absent on disk: a partial run must not touch suites it did not run.
        "not_in_run": {
            "responses": "responses/not_in_run.jsonl",
            "metadata": "metadata/not_in_run.jsonl",
            "cases": 100,
        },
    }}))

    gold_rows, meta_rows, log = [], [], []
    for i in range(1, 201):
        split = "calibrate" if i <= 40 else "test"
        actual = i % 4 != 0
        gold_rows.append({"answers": {"q": {"type": "noul", "noul": 1.0 if actual else 0.0}}})
        meta_rows.append({"id": f"fake-{i:03d}", "split": split})

        if split == "calibrate":
            # deliberately overconfident => fitted T > 1
            predicted = actual if i % 5 != 0 else not actual
            p_true = 0.995 if predicted else 0.005
        else:
            # already much softer test probabilities; the fitted T can worsen test NLL.
            p_true = 0.70 if actual else 0.30
        response = {"model": "bonsai-test", "answers": {"q": {"type": "noul", "noul": p_true}}}
        log.append({
            "request_id": f"fake-{i:03d}", "response": response, "error_type": None,
            "error": None, "duration_ms": 1.0,
            "endpoint": "http://localhost:5380/v1/systemone",
        })

    (tmp_path / "responses" / "fake.jsonl").write_text(
        "\n".join(json.dumps(x) for x in gold_rows) + "\n")
    (tmp_path / "metadata" / "fake.jsonl").write_text(
        "\n".join(json.dumps(x) for x in meta_rows) + "\n")
    log_path = tmp_path / "output" / "run.jsonl"
    log_path.write_text("\n".join(json.dumps(x) for x in log) + "\n")

    out, artifact = calibration.calibrate_log(tmp_path, log_path, min_cases=20)
    assert out.exists()
    assert artifact["schema_version"] == 2 and artifact["scope"] == "global"
    assert artifact["temperatures"] == {}  # 40 cases < per-group floor -> global T only
    assert artifact["kind"] == "typed-decision-temperature-calibration"
    assert artifact["source"]["model"] == "bonsai-test"
    assert artifact["source"]["endpoint"] == "http://localhost:5380/v1/systemone"
    assert artifact["split"]["calibration_n"] == 40
    assert artifact["split"]["test_n"] == 160
    assert artifact["fit"]["uses_test_data"] is False
    assert artifact["temperature"] > 1.0
    assert artifact["metrics"]["calibrate"]["calibrated"]["nll"] <= artifact["metrics"]["calibrate"]["raw"]["nll"]
    # Deployment is decided without peeking at test improvement.
    assert artifact["deployable"] is True
    assert artifact["analysis"]["test_nll_improved"] is False
