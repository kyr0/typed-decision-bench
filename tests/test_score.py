"""/ E2E tests for scripts/score.py, driven through the CLI as a subprocess —
exactly the way `make score` invokes it.

Covers single-run mode (--responses dir joined with the newest run log) and
--all-logs backfill: per-case records (case IDs, calibration against the full
gold distribution, joined latency), per-capability + micro stats lines, the
last-record-wins resume rule, auto log-pick skipping the scorer's own outputs,
and idempotence. Run via `make test`.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

SCORE = Path(__file__).resolve().parents[1] / 'scripts' / 'score.py'


def noul(v):
    """/ Fixture payload: one noul answer with probability v."""
    return {'answers': {'q1': {'type': 'noul', 'noul': v}}}


def score3(probs):
    """/ Fixture payload: one score answer over three rubric levels."""
    return {'answers': {'q1': {'type': 'score', 'probabilities': probs}}}


def test_single_run_scoring(tmp_path: Path) -> None:
    """/ Single-run mode: predictions dir + newest log -> _cases/_stats with exact
    calibration math (nll/brier/soft_accuracy vs the full gold distribution),
    transport errors excluded from latency, stdout mirroring the stats file."""
    root = tmp_path
    (root / 'responses').mkdir()
    (root / 'preds').mkdir()
    (root / 'output').mkdir()
    (root / 'manifest.json').write_text(json.dumps({'capabilities': {
        'fake_cap': {'responses': 'responses/fake_cap.jsonl'},
        'scorey_cap': {'responses': 'responses/scorey_cap.jsonl'},
    }}))
    (root / 'responses' / 'fake_cap.jsonl').write_text(
        '\n'.join(json.dumps(x) for x in (noul(0.9), noul(0.1))) + '\n')
    (root / 'preds' / 'fake_cap.jsonl').write_text(
        '\n'.join(json.dumps(x) for x in (noul(0.9), noul(0.9))) + '\n')
    # gold peaks at level 1 (expected 1.0); prediction peaks at 0 (expected 0.5)
    (root / 'responses' / 'scorey_cap.jsonl').write_text(
        json.dumps(score3({'0': 0.1, '1': 0.8, '2': 0.1})) + '\n')
    (root / 'preds' / 'scorey_cap.jsonl').write_text(
        json.dumps(score3({'0': 0.6, '1': 0.3, '2': 0.1})) + '\n')
    # one success (100 ms) + one service error (300 ms, must NOT count as latency)
    log = [{'request_id': f'fake_cap-00{i}', 'response': None, 'error_type': et,
            'error': None, 'duration_ms': dur, 'endpoint': 'x'}
           for i, (dur, et) in enumerate(((100.0, None), (300.0, 'service_error')), 1)]
    log_path = root / 'output' / 'mymodel_2026-09-21_19_00.jsonl'
    log_path.write_text('\n'.join(json.dumps(r) for r in log) + '\n')

    proc = subprocess.run([sys.executable, str(SCORE), '--benchmark', str(root),
                           '--responses', str(root / 'preds')],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    # per-case records: typed-decisions-bench-v1:<family>:<line>, calibration vs full gold distribution
    cases = [json.loads(l) for l in (root / 'output' / 'mymodel_cases.jsonl').read_text().splitlines()]
    c1, c2, c3 = cases
    assert c1['case_id'] == 'typed-decisions-bench-v1:fake_cap:0001' and c1['family'] == 'fake_cap'
    assert c1['qtype'] == 'noul' and c1['correct'] == 1 and c1['confidence'] == 0.9
    assert abs(c1['nll'] - -(0.1 * math.log(0.1) + 0.9 * math.log(0.9))) < 1e-9
    assert c1['brier'] == 0.0 and abs(c1['soft_accuracy'] - 0.82) < 1e-9
    assert c1['prediction_label'] == 'true' and c1['gold_label'] == 'true' and c1['latency_ms'] == 100.0
    assert c2['correct'] == 0 and abs(c2['nll'] - -(0.9 * math.log(0.1) + 0.1 * math.log(0.9))) < 1e-9
    assert abs(c2['brier'] - 1.28) < 1e-9 and abs(c2['soft_accuracy'] - 0.18) < 1e-9
    assert c2['prediction_label'] == 'true' and c2['gold_label'] == 'false' and c2['latency_ms'] is None
    assert c3['qtype'] == 'score' and c3['correct'] == 0 and c3['prediction_label'] == '0' \
        and c3['gold_label'] == '1' and abs(c3['score_error'] - 0.5) < 1e-9

    # per-capability stats: one flat line each + micro
    lines = [json.loads(l) for l in (root / 'output' / 'mymodel_stats.jsonl').read_text().splitlines()]
    assert len(lines) == 3 and lines[-1]['capability'] == 'micro'
    cap = lines[0]
    assert cap['run'] == 'mymodel' and cap['n'] == 2 and cap['accuracy'] == 0.5
    assert cap['soft_accuracy'] == 0.5 and cap['nll'] == 1.204 and cap['brier'] == 0.64
    assert cap['errors'] == 1 and cap['error_rate'] == 0.5
    assert cap['latency_ms_mean'] == 100.0 and cap['latency_ms_p50'] == 100.0
    assert cap['ece_15'] == 0.4 and cap['score_mae'] is None
    scap = lines[1]
    assert scap['accuracy'] == 0.0 and scap['score_mae'] == 0.5 and scap['score_within_one'] == 1.0
    assert scap['latency_ms_mean'] is None
    micro = lines[2]
    assert micro['n'] == 3 and micro['accuracy'] == 0.3333 and micro['errors'] == 1
    # stdout mirrors the stats file line-by-line
    assert [json.loads(l) for l in proc.stdout.splitlines()] == lines

    # auto log-pick must skip score.py's own *_stats/_cases outputs even when newer
    assert subprocess.run([sys.executable, str(SCORE), '--benchmark', str(root),
                           '--responses', str(root / 'preds')],
                          capture_output=True, text=True).returncode == 0
    assert (root / 'output' / 'mymodel_stats.jsonl').exists()  # still derived from the run log
    assert not (root / 'output' / 'mymodel_cases_stats.jsonl').exists()

    # explicit --log/--out still work
    proc2 = subprocess.run([sys.executable, str(SCORE), '--benchmark', str(root),
                            '--responses', str(root / 'preds'), '--log', str(log_path),
                            '--out', str(root / 'custom.jsonl')],
                           capture_output=True, text=True)
    assert proc2.returncode == 0, proc2.stderr
    assert json.loads((root / 'custom.jsonl').read_text().splitlines()[0])['capability'] == 'fake_cap'
    assert (root / 'custom_cases.jsonl').exists()


def test_all_logs_backfill(tmp_path: Path) -> None:
    """/ --all-logs: every unscored output/*.jsonl gets a _stats/_cases pair scored
    from the responses embedded in the log; already-scored runs are skipped,
    recovered retries win over their failed tries, and golden-only capabilities
    (log covers none of their cases) become n=0 null-metric lines, not crashes."""
    root = tmp_path
    (root / 'responses').mkdir()
    (root / 'output').mkdir()
    # ghost_cap has golden data but the log covers none of it: its stat line
    # must be n=0 with null metrics instead of crashing (aborted-run logs)
    (root / 'manifest.json').write_text(json.dumps({'capabilities': {
        'fake_cap': {'responses': 'responses/fake_cap.jsonl'},
        'ghost_cap': {'responses': 'responses/ghost_cap.jsonl'}}}))
    (root / 'responses' / 'ghost_cap.jsonl').write_text(json.dumps(noul(0.9)) + '\n')
    (root / 'responses' / 'fake_cap.jsonl').write_text(
        '\n'.join(json.dumps(x) for x in (noul(0.9), noul(0.1))) + '\n')
    # run log: case 1 ok, case 2 first timed out then recovered on a resumed run
    # (last record per case wins), case 3 failed (no case, counts as error)
    log = [{'request_id': 'fake_cap-001', 'response': noul(0.9), 'error_type': None, 'error': None,
            'duration_ms': 100.0, 'endpoint': 'x'},
           {'request_id': 'fake_cap-002', 'response': None, 'error_type': 'timeout_retries_exhausted',
            'error': 'timeout after 5s', 'duration_ms': 999.0, 'endpoint': 'x'},
           {'request_id': 'fake_cap-002', 'response': noul(0.9), 'error_type': None, 'error': None,
            'duration_ms': 120.0, 'endpoint': 'x'},
           {'request_id': 'fake_cap-003', 'response': None, 'error_type': 'service_error',
            'error': 'HTTP 500', 'duration_ms': 300.0, 'endpoint': 'x'}]
    (root / 'output' / 'three.jsonl').write_text('\n'.join(json.dumps(x) for x in log) + '\n')
    # an already-scored run must be left untouched
    (root / 'output' / 'e2e.jsonl').write_text(json.dumps(log[0]) + '\n')
    (root / 'output' / 'e2e_stats.jsonl').write_text('{"run": "e2e"}\n')

    proc = subprocess.run([sys.executable, str(SCORE), '--benchmark', str(root), '--all-logs'],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert 'skip e2e.jsonl' in proc.stderr and proc.stdout == ''

    cases = [json.loads(l) for l in (root / 'output' / 'three_cases.jsonl').read_text().splitlines()]
    assert len(cases) == 2
    assert cases[0]['correct'] == 1 and cases[0]['latency_ms'] == 100.0
    assert cases[1]['correct'] == 0 and cases[1]['gold_label'] == 'false' and cases[1]['latency_ms'] == 120.0
    lines = [json.loads(l) for l in (root / 'output' / 'three_stats.jsonl').read_text().splitlines()]
    assert len(lines) == 3 and lines[0]['run'] == 'three' and lines[0]['n'] == 2
    assert lines[0]['accuracy'] == 0.5 and lines[0]['errors'] == 1 and lines[0]['error_rate'] == 0.5
    # the recovered case neither counts as an error nor leaks its failed try's 999 ms latency
    assert lines[0]['latency_ms_mean'] == 110.0
    assert lines[1]['capability'] == 'ghost_cap' and lines[1]['n'] == 0 and lines[1]['accuracy'] is None
    assert lines[2]['capability'] == 'micro' and lines[2]['n'] == 2 and lines[2]['errors'] == 1
    assert (root / 'output' / 'e2e_stats.jsonl').read_text() == '{"run": "e2e"}\n'
    assert not (root / 'output' / 'e2e_cases.jsonl').exists()

    # --all-logs is idempotent: nothing left to score
    proc = subprocess.run([sys.executable, str(SCORE), '--benchmark', str(root), '--all-logs'],
                          capture_output=True, text=True)
    assert proc.returncode == 0 and 'skip three.jsonl' in proc.stderr and 'skip e2e.jsonl' in proc.stderr

    # single-run mode still requires --responses
    assert subprocess.run([sys.executable, str(SCORE), '--benchmark', str(root)],
                          capture_output=True).returncode != 0
