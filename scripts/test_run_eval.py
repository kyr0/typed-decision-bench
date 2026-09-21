#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "plotly"]
# ///
"""Self-checks for run_eval's log-path naming: --log wins, --name stays
timestamp-free, auto-naming sanitises the model and stamps the time.

Run: uv run scripts/test_run_eval.py (or python3 scripts/test_run_eval.py)
"""
import http.server
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_eval


def check_naming():
    """resolve_log_path must honour --log, use --name verbatim (no timestamp)
    and fall back to <sanitised-model>_<timestamp>."""
    explicit = run_eval.resolve_log_path(
        Namespace(log='x.jsonl', output='output', name=None, model='m'), 'm')
    assert explicit == (Path('x.jsonl'), None), explicit

    named = run_eval.resolve_log_path(
        Namespace(log=None, output='output', name='nightly', model='m'), 'm')
    assert named == (Path('output/nightly.jsonl'), 'nightly'), named

    auto = run_eval.resolve_log_path(
        Namespace(log=None, output='output', name=None, model='GPT 4/x'), 'GPT 4/x')
    assert auto[1] == 'GPT-4-x', auto
    assert re.fullmatch(r'output/GPT-4-x_\d{4}-\d{2}-\d{2}_\d{2}_\d{2}\.jsonl', str(auto[0])), auto
    print('ok: log path naming')


def check_score_reports():
    """write_score_reports must turn a finished run log into <run>_stats/_cases.jsonl
    using only the responses embedded in the log, and must never raise."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / 'responses').mkdir()
        (root / 'output').mkdir()
        (root / 'manifest.json').write_text(json.dumps({'capabilities': {
            'fake_cap': {'responses': 'responses/fake_cap.jsonl'}}}))
        (root / 'responses' / 'fake_cap.jsonl').write_text(
            json.dumps({'answers': {'q1': {'type': 'noul', 'noul': 0.9}}}) + '\n')
        log = root / 'output' / 'four.jsonl'
        log.write_text(json.dumps({'request_id': 'fake_cap-001',
                                   'response': {'answers': {'q1': {'type': 'noul', 'noul': 0.9}}},
                                   'error_type': None, 'error': None,
                                   'duration_ms': 50.0, 'endpoint': 'x'}) + '\n')
        run_eval.write_score_reports(root, log)
        lines = [json.loads(l) for l in (root / 'output' / 'four_stats.jsonl').read_text().splitlines()]
        assert len(lines) == 2 and lines[0]['n'] == 1 and lines[0]['accuracy'] == 1.0
        assert lines[1]['capability'] == 'micro' and lines[1]['n'] == 1
        cases = [json.loads(l) for l in (root / 'output' / 'four_cases.jsonl').read_text().splitlines()]
        assert cases[0]['correct'] == 1 and cases[0]['latency_ms'] == 50.0
        # a broken log must only warn, never raise (the run log is the source of truth)
        run_eval.write_score_reports(root, root / 'output' / 'missing.jsonl')
    print('ok: post-run scoring')


def check_resume():
    """log_successes must return only successful {(cap, line)} records with later
    records winning, so reusing a run name re-sends exactly the pending cases."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / 'output').mkdir()
        log = root / 'output' / 'five.jsonl'
        recs = [
            {'request_id': 'fake_cap-001', 'response': {'answers': {}}, 'error_type': None,
             'error': None, 'duration_ms': 10.0, 'endpoint': 'x'},
            {'request_id': 'fake_cap-002', 'response': None, 'error_type': 'timeout_retries_exhausted',
             'error': 'timeout', 'duration_ms': 9.0, 'endpoint': 'x'},
            {'request_id': 'fake_cap-002', 'response': {'answers': {}}, 'error_type': None,
             'error': None, 'duration_ms': 11.0, 'endpoint': 'x'},  # recovered retry wins
            {'request_id': 'fake_cap-003', 'response': None, 'error_type': 'aborted',
             'error': 'not sent', 'duration_ms': 0.0, 'endpoint': 'x'},
        ]
        log.write_text('\n'.join(json.dumps(r) for r in recs) + '\n')
        done = run_eval.log_successes(log)
        assert set(done) == {('fake_cap', 1), ('fake_cap', 2)}
        assert done[('fake_cap', 2)]['duration_ms'] == 11.0
    print('ok: resume set')


def check_resume_e2e():
    """Full resume round-trip against a stub endpoint: a run that fails
    everywhere is re-invoked under the same --name and must only re-send the
    pending cases, append to the same log, merge the responses files and
    refresh stats to errors=0."""
    class Stub(http.server.BaseHTTPRequestHandler):
        fail = True

        def do_POST(self):
            self.rfile.read(int(self.headers.get('Content-Length', 0)))
            if Stub.fail:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b'boom')
                return
            body = json.dumps({'model': 'stub', 'answers': {'q1': {'type': 'noul', 'noul': 0.9}},
                               'usage': {}}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(('127.0.0.1', 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    root_dir = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / 'requests').mkdir()
        (root / 'responses').mkdir()
        (root / 'output').mkdir()
        (root / 'manifest.json').write_text(json.dumps({'capabilities': {
            'fake_cap': {'requests': 'requests/fake_cap.jsonl',
                         'responses': 'responses/fake_cap.jsonl',
                         'metadata': 'metadata/fake_cap.jsonl'}}}))
        (root / 'requests' / 'fake_cap.jsonl').write_text('{"q": "why"}\n{"q": "again"}\n')
        (root / 'responses' / 'fake_cap.jsonl').write_text(
            '\n'.join(json.dumps({'answers': {'q1': {'type': 'noul', 'noul': 0.9}}}) for _ in range(2)) + '\n')
        # a pre-scored reference run: oldest stats file => the auto-comparison baseline
        (root / 'output' / 'e2e_stats.jsonl').write_text(
            json.dumps({'run': 'e2e', 'capability': 'fake_cap', 'n': 2, 'accuracy': 0.5,
                        'soft_accuracy': 0.5, 'nll': 0.5}) + '\n'
            + json.dumps({'run': 'e2e', 'capability': 'micro', 'n': 2, 'accuracy': 0.5,
                          'soft_accuracy': 0.5}) + '\n')
        url = f'http://127.0.0.1:{srv.server_port}/v1/systemone'
        base = [sys.executable, str(root_dir / 'run_eval.py'), '--benchmark', str(root),
                '--url', url, '--model', 'stub', '--name', 'five', '--n', '2',
                '--responses', str(root / 'model_responses'), '--retries', '1',
                '--output', str(root / 'output')]

        Stub.fail = True  # phase 1: every request fails, run exits non-zero
        p1 = subprocess.run(base, capture_output=True, text=True, cwd=root_dir)
        assert p1.returncode == 1, p1.stderr
        log = root / 'output' / 'five.jsonl'
        assert len(log.read_text().splitlines()) == 2  # one error record per case

        Stub.fail = False  # phase 2: same name resumes and recovers everything
        p2 = subprocess.run(base, capture_output=True, text=True, cwd=root_dir)
        assert p2.returncode == 0, p2.stderr
        assert 'resuming five.jsonl' in p2.stdout and '0 case(s) already succeeded' in p2.stdout
        assert len(log.read_text().splitlines()) == 4  # appended retries, successes kept
        # responses files merged: no error placeholders, full payload rows
        rows = [json.loads(l) for l in (root / 'model_responses' / 'fake_cap.jsonl').read_text().splitlines()]
        assert len(rows) == 2 and all('answers' in r for r in rows)
        # refreshed stats: recovered cases count as successes, not errors
        stats = [json.loads(l) for l in (root / 'output' / 'five_stats.jsonl').read_text().splitlines()]
        assert stats[0]['n'] == 2 and stats[0]['errors'] == 0 and stats[0]['accuracy'] == 1.0
        # post-run metrics + comparison (baseline = oldest scored run, e2e)
        assert (root / 'output' / 'five' / 'metrics.csv').exists()
        delta = root / 'output' / 'comparison' / 'delta_vs_e2e_soft_accuracy.csv'
        assert delta.exists() and 'fake_cap' in delta.read_text()
        assert (root / 'output' / 'comparison' / 'report.html').exists()
    srv.shutdown()
    print('ok: resume end-to-end')


def check_quota_buster():
    """BurstGate must arm randomized cool-downs (25±5 requests, wait ±10%) and
    be fully disabled at 0 ms."""
    g = run_eval.BurstGate(0)
    t0 = time.monotonic()
    for _ in range(100):
        g.wait()
    assert time.monotonic() - t0 < 0.1, '0 ms must disable the gate'

    g = run_eval.BurstGate(50)  # 50 ms cool-down keeps the test fast
    assert 20 <= g.next_at <= 30
    tripped = False
    for _ in range(30):
        g.wait()
        if g.pause_until > 0:  # a cool-down was scheduled and the counter re-armed
            assert g.count == 0 and g.pause_until > time.monotonic() - 1
            assert 20 <= g.next_at <= 30
            tripped = True
            break
    assert tripped, 'a cool-down must trip within 30 waits'
    print('ok: quota buster')


if __name__ == '__main__':
    check_naming()
    check_score_reports()
    check_resume()
    check_resume_e2e()
    check_quota_buster()
