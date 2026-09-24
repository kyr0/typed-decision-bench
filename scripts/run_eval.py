#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "plotly"]
# ///
"""Benchmark runner: sends every selected request to System One and records outcomes.

Pipeline: read manifest.json -> build (capability, case) task list -> POST each
request in parallel at a bounded rate -> write results.

Configuration (CLI arg > environment variable > .env, parsed without dependencies):
  endpoint   --url or TYPESAFE_BASE_URL (no trailing slash, e.g. http://host:5380)
              -> <base>/v1/systemone
  auth       --api-key or TYPESAFE_API_KEY (bearer token)
  model      --model or TYPESAFE_MODEL; request lines carry the placeholder
              "model":"REPLACED_BY_TYPESAFE_MODEL" which is substituted at send time

Outputs:
  combined log (always):   one JSON line per case:
    {request_id, response, error_type, error, duration_ms, endpoint}
    error_type is null on success, otherwise "timeout_retries_exhausted",
    "service_error" or "aborted" (not sent because the run was cancelled).
    Path: --log (explicit, overwrites); output/<name>.jsonl when --name is given
    (stable name, no timestamp postfix); otherwise output/<model>_<YYYY-MM-DD_HH_MM>.jsonl.
    Reusing a run name RESUMES that run: only cases without a successful record
    in the existing log are (re)sent, new records are appended, and the stats/
    cases reports are refreshed - so old results are never silently discarded.
  per-capability files (only with an explicit --responses DIR, never by
  default): <DIR>/<capability>.jsonl, line-aligned with the request files;
  failed cases become {"error": ...} placeholder lines. The golden responses/
  dir is the scoring answer key - writing there is refused outright.
  stats/cases (always): each finished run scores ONLY metadata split=test
  into output/<run>_stats.jsonl + output/<run>_cases.jsonl (scripts/score.py);
  scoring failures only warn - the log is kept and `make score` re-derives them.
  calibration (default): fit a scalar temperature ONLY from metadata
  split=calibrate and report raw vs calibrated metrics on held-out split=test;
  writes output/<run>_calibration.json. split=train is ignored.
  metrics/comparison (always): the run's output/<run>/ metrics folder refreshes
  and, once >=2 runs are scored, the per-capability comparison in
  output/comparison/ regenerates against --baseline (default: jev-1.13.0 if scored,
  else the oldest scored run).

Reliability: per-request --timeout (default 10s) and --retries (default 3); a
circuit breaker aborts the run when the first requests ALL fail, instead of
hammering an unresponsive endpoint; --quota-buster (default 750 ms) pauses all
senders after every ~25 (±5) requests for the given wait (±10%) so sustained
bursts don't trip quota limits (0 disables). The process exits 1 if any case failed.
"""
import argparse, atexit, datetime, json, os, random, re, signal, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# sibling imports: this file lives in scripts/ and is run standalone
# (sys.path[0] = scripts/) or imported as top-level `run_eval` by its self-test
from gpqa_zip import cleanup_unlocked, unlock_paths
from score import last_records, score_log
from splits import DEFAULT_EVAL_SPLITS, capability_splits, parse_splits

MODEL_PLACEHOLDER = 'REPLACED_BY_TYPESAFE_MODEL'
ENDPOINT_PATH = '/v1/systemone'
DEFAULT_BASE_URL = 'https://api.typesafe.ai'
ENV_FILE = '.env'
# circuit breaker threshold: abort when this many requests failed and none succeeded yet
ABORT_AFTER_CONSECUTIVE_ERRORS = 10
SENDABLE_ERRORS = ('timeout_retries_exhausted', 'service_error')
# quota-buster: pause ALL senders after every ~QUOTA_BURST dispatched requests
# for --quota-buster ms; both numbers jitter so the pattern isn't recognizable
QUOTA_BURST = 25           # requests between cool-downs (± QUOTA_BURST_JITTER, re-rolled per cycle)
QUOTA_BURST_JITTER = 5
QUOTA_WAIT_JITTER = 0.1    # wait jitter as a fraction: ±10% (750 ms -> ±75 ms)
PROGRESS_EVERY = 50        # print an ETA line after every N completed cases
DEFAULT_BASELINE = 'jev-1.13.0'  # hosted reference run; auto-baseline when scored


def load_dotenv(path=ENV_FILE):
    """/ Minimal .env parser (KEY=VALUE, '#' comments, optional quotes) so no dependency is needed."""
    env = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def setting(name, cli_value, dotenv):
    """/ Resolution order: CLI arg > real environment > .env file."""
    return cli_value or os.environ.get(name) or dotenv.get(name)


def post(url, headers, body, timeout, retries, label=''):
    """/ POST one request with timeout + retries; returns (response|None, error_type, error_text).

    HTTP errors keep status plus a body excerpt for the run log; timeouts are
    classified separately so `timeout_retries_exhausted` stays distinguishable
    from real `service_error` responses.
    """
    last_type, last_text = None, None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, data=body.encode('utf-8'), headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read()), None, None
        except urllib.error.HTTPError as e:
            detail = e.read(500).decode('utf-8', 'replace')
            last_type, last_text = 'service_error', f'HTTP {e.code} {e.reason}: {detail}'
        except Exception as e:
            reason = getattr(e, 'reason', e)  # urllib wraps the cause in URLError.reason
            if isinstance(reason, TimeoutError):
                last_type, last_text = 'timeout_retries_exhausted', f'timeout after {timeout}s: {e}'
            else:
                last_type, last_text = 'service_error', f'{type(e).__name__}: {e}'
        print(f'  {label} retry {attempt}/{retries}: {last_text}', flush=True)
    return None, last_type, last_text


class BurstGate:
    """/ Randomized cool-down against quota-style rate limits: after every
    ~QUOTA_BURST (± jitter) dispatched requests, all senders pause for the
    configured wait (± QUOTA_WAIT_JITTER) so sustained bursts get a gap
    without a recognizable fixed pattern. wait_ms <= 0 disables the gate."""

    def __init__(self, wait_ms: float):
        self.wait_s = max(0.0, float(wait_ms)) / 1000.0
        self.lock = threading.Lock()
        self.count = 0
        self.next_at = random.randint(QUOTA_BURST - QUOTA_BURST_JITTER, QUOTA_BURST + QUOTA_BURST_JITTER)
        self.pause_until = 0.0

    def wait(self):
        """/ Blocks while a cool-down is active; counts this request and trips
        the next cool-down when the (re-rolled) burst threshold is reached.
        Sleeping happens outside the lock so other workers can still check in."""
        if self.wait_s <= 0:
            return
        while True:
            with self.lock:
                now = time.monotonic()
                if now >= self.pause_until:
                    self.count += 1
                    if self.count >= self.next_at:
                        # trip: everyone pauses, the counter re-arms with a fresh threshold
                        self.count = 0
                        self.next_at = random.randint(QUOTA_BURST - QUOTA_BURST_JITTER,
                                                       QUOTA_BURST + QUOTA_BURST_JITTER)
                        self.pause_until = now + self.wait_s * random.uniform(1 - QUOTA_WAIT_JITTER,
                                                                              1 + QUOTA_WAIT_JITTER)
                    return
                remain = self.pause_until - now
            time.sleep(min(remain, 0.25))


class Pace:
    """/ Thread-safe rate limiter: spaces request starts exactly 1/rps apart.

    Workers reserve a start time under the lock, then sleep outside it, so the
    lock is never held while waiting and the dispatch rate stays bounded no
    matter how many workers exist.
    """

    def __init__(self, rps: float):
        self.interval = 1.0 / rps
        self.next_start = time.monotonic()
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            start = max(time.monotonic(), self.next_start)
            self.next_start = start + self.interval
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def selected_capabilities(csv, manifest_caps):
    """/ Resolves --capabilities (comma-separated, whitespace-trimmed) against the manifest."""
    if not csv:
        return list(manifest_caps)
    picked = [c.strip() for c in csv.split(',') if c.strip()]
    unknown = [c for c in picked if c not in manifest_caps]
    if unknown:
        raise SystemExit(f'unknown capabilities: {unknown} (see manifest.json)')
    return picked


def build_tasks(root, caps, manifest_caps, limit, model, selected_splits=DEFAULT_EVAL_SPLITS):
    """/ Flattens selected persisted splits into (capability, source_line, body).

    `split` lives only in line-aligned metadata, never in the System One wire
    payload. The default sends test+calibrate and excludes train. `--n` limits
    the selected rows after split filtering while preserving original source
    line numbers so scoring/calibration rejoin exactly.
    """
    tasks, counts, first_body = [], {}, {}
    selected_splits = tuple(selected_splits)
    for cap in caps:
        entry = manifest_caps[cap]
        req_path = root / entry['requests']
        lines = [l.strip() for l in req_path.read_text(encoding='utf-8').splitlines() if l.strip()]
        splits = capability_splits(root, cap, entry, strict=False, expected_cases=len(lines))
        counts[cap] = len(lines)  # total source rows; response materialization stays line-aligned
        picked = [(i, body) for i, (body, split) in enumerate(zip(lines, splits), 1)
                  if split in selected_splits]
        if limit is not None:
            picked = picked[:limit]
        for line_no, body in picked:
            if MODEL_PLACEHOLDER in body:
                if not model:
                    raise SystemExit(f'{cap}:{line_no}: {MODEL_PLACEHOLDER} present but no model configured '
                                     f'(set TYPESAFE_MODEL or pass --model)')
                body = body.replace(MODEL_PLACEHOLDER, model)
            first_body.setdefault(cap, body)
            tasks.append((cap, line_no, body))
    return tasks, counts, first_body


def make_record(cap, line_no, url, obj=None, error_type=None, error=None, duration_ms=0.0):
    """/ One combined-log line; the single place that defines the record schema.

    request_id is '<capability>-<line:03d>' - the key every downstream tool
    (scoring, resume, e2e validation) uses to re-join a record with its case.
    """
    return {
        'request_id': f'{cap}-{line_no:03d}',
        'response': obj,
        'error_type': error_type,
        'error': error,
        'duration_ms': round(duration_ms, 1),
        'endpoint': url,
    }


def run_batch(tasks, url, headers, args, log_file):
    """/ Dispatches all tasks through a rate-limited thread pool; returns {(cap, line): record}.

    Results are consumed in submission order so the combined log and progress
    output stay deterministic. The circuit breaker fires when the first
    ABORT_AFTER_CONSECUTIVE_ERRORS requests all failed (`consecutive == done`
    means no success has been seen yet); queued tasks then short-circuit to
    'aborted' records. The abort flag is re-checked after the rate-limit wait
    because workers pre-queue on the limiter before the breaker can fire.
    """
    pace = Pace(args.parallel)
    gate = BurstGate(args.quota_buster)
    # ponytail: worker heuristic rps*5 sustains the target rate at ~5s request latency; cap 512
    workers = min(512, max(8, int(args.parallel * 5)))
    results = {}
    done = consecutive_errors = 0
    abort = threading.Event()
    # ETA baseline: measured rate (not --parallel) since quota-buster + latency
    # always deliver below the configured rps
    run_started = time.monotonic()

    def work(task):
        cap, line_no, body = task
        if abort.is_set():
            return make_record(cap, line_no, url, error_type='aborted',
                               error='not sent: run aborted (endpoint unresponsive)')
        gate.wait()
        if abort.is_set():  # the cool-down may have lasted a while; don't send into a dead run
            return make_record(cap, line_no, url, error_type='aborted',
                               error='not sent: run aborted (endpoint unresponsive)')
        pace.wait()
        if abort.is_set():
            return make_record(cap, line_no, url, error_type='aborted',
                               error='not sent: run aborted (endpoint unresponsive)')
        started = time.monotonic()
        obj, err_type, err_text = post(url, headers, body, args.timeout, args.retries, label=f'{cap}:{line_no}')
        return make_record(cap, line_no, url, obj, err_type, err_text,
                           (time.monotonic() - started) * 1000)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (cap, line_no), record in pool.map(lambda t: ((t[0], t[1]), work(t)), tasks):
            results[(cap, line_no)] = record
            log_file.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
            done += 1
            if record['error_type'] in SENDABLE_ERRORS:
                consecutive_errors += 1
                if consecutive_errors >= ABORT_AFTER_CONSECUTIVE_ERRORS and consecutive_errors == done:
                    abort.set()
                    print(f'ABORT: first {done} requests all failed - check endpoint {url}, '
                          f'--timeout ({args.timeout}s) and --retries', flush=True)
            elif not record['error_type']:
                consecutive_errors = 0
            status = f'ERROR {record["error_type"]}' if record['error_type'] else 'ok'
            print(f'[{done}/{len(tasks)}] {cap}:{line_no} {status} ({record["duration_ms"]} ms)', flush=True)
            if done % PROGRESS_EVERY == 0 and done < len(tasks):
                elapsed = time.monotonic() - run_started
                rate = done / max(elapsed, 1e-6)
                m, s = divmod(int((len(tasks) - done) / rate) if rate > 0 else 0, 60)
                print(f'    ETA {done / len(tasks) * 100:.1f}% | {m}m{s:02d}s left '
                      f'({rate:.1f} completed req/s)', flush=True)
    return results


def log_successes(log_path):
    """/ {(capability, line): record} of every case with a successful response in
    a run log - the resume set: reusing a run name (re)sends exactly the cases
    NOT in here (failures, timeouts, aborted and never-reached cases)."""
    return {k: r for k, r in last_records(log_path).items() if not r.get('error_type')}


def write_metrics_reports(root, log_path, baseline=None, metric='soft_accuracy', compare=True):
    """/ Metrics + comparison follow-ups after scoring: refreshes this run's
    output/<run>/metrics.{csv,json} and, once at least two runs are scored, the
    per-capability comparison in output/comparison/ against the explicit
    baseline (default: `jev-1.13.0` when it is scored - the hosted reference -
    else the oldest scored run; never a fabricated one). compare=False stops
    after the single-run metrics export (`make eval-only` / `--no-compare`):
    useful when a baseline comparison is not wanted or would be misleading.
    Post-step only: failures warn,
    `make metrics` and `make compare` re-derive everything."""
    try:
        from metrics import compare_runs, export_run
        from score import run_name
        out = root / 'output'
        name = run_name(log_path)
        stats = out / f'{name}_stats.jsonl'
        if stats.exists():
            export_run(stats, out / name)
            print(f'wrote metrics -> {out / name}/', file=sys.stderr)
        if not compare:
            print('skipped comparison (--no-compare)', file=sys.stderr)
            return
        scored = sorted(out.glob('*_stats.jsonl'), key=lambda p: p.stat().st_mtime)
        names = [p.stem[:-len('_stats')] for p in scored]
        # jev is the reference deployment; if it is among the scored runs it wins
        # the auto-baseline role, otherwise the oldest scored run stays stable
        ref = DEFAULT_BASELINE if DEFAULT_BASELINE in names else (names[0] if names else None)
        base = baseline or (ref if len(scored) >= 2 else None)
        if base:
            compare_runs(root, base, metric, out / 'comparison')
        else:
            print('skipped comparison (needs >= 2 scored runs; no baseline to compare against)',
                  file=sys.stderr)
    except SystemExit as e:
        print(f'WARNING: post-run comparison stopped ({e})', file=sys.stderr)
    except Exception as e:  # post-step only: a finished run must not be lost over a reporting bug
        print(f'WARNING: post-run metrics/comparison failed ({type(e).__name__}: {e}); '
              f'run `make metrics` / `make compare` to retry', file=sys.stderr)


def write_score_reports(root, log_path):
    """/ Scores the just-written run log into output/<run>_stats.jsonl +
    output/<run>_cases.jsonl (predictions from the log itself) so every eval
    run leaves ready-to-compare artifacts behind. Deliberately never fatal:
    the log is the source of truth, `make score` can always re-derive reports."""
    try:
        man = json.loads((root / 'manifest.json').read_text())
        score_log(root, man, log_path)
    except Exception as e:  # post-step only: a finished run must not be lost over a scoring bug
        print(f'WARNING: post-run scoring failed ({type(e).__name__}: {e}); '
              f'run `make score` to retry once fixed', file=sys.stderr)


def write_calibration_report(root, log_path, min_cases=100):
    """Fit T from calibrate only; test is held out and never controls deployability."""
    try:
        from calibration import calibrate_log
        out, artifact = calibrate_log(root, log_path, min_cases=min_cases)
        print(f'wrote calibration -> {out} (T={artifact["temperature"]:.8g}, '
              f'deployable={artifact["deployable"]}, status={artifact["status"]})',
              file=sys.stderr)
        if not artifact['deployable']:
            print(f'WARNING: calibration artifact is not deployable ({artifact["status"]})',
                  file=sys.stderr)
    except Exception as e:  # post-step only: never lose a completed benchmark log
        print(f'WARNING: post-run calibration failed ({type(e).__name__}: {e}); '
              f'run `make calibrate` to retry once fixed', file=sys.stderr)


def write_response_files(caps, counts, results, out_dir):
    """/ Materialises responses/<capability>.jsonl, line-aligned with the request files.

    Failed cases are written as {"error": ...} placeholders at their original
    position so downstream scorers can detect them without losing alignment.
    """
    for cap in caps:
        out = out_dir / f'{cap}.jsonl'
        sent = cap_errors = 0
        with open(out, 'w', encoding='utf-8') as dst:
            for line_no in range(1, counts[cap] + 1):
                record = results.get((cap, line_no))
                if record is None:
                    dst.write(json.dumps({'skipped': True, 'capability': cap, 'line': line_no},
                                         ensure_ascii=False, separators=(',', ':')) + '\n')
                elif record['error_type']:
                    dst.write(json.dumps({'error': record['error'], 'capability': cap, 'line': line_no},
                                         ensure_ascii=False, separators=(',', ':')) + '\n')
                    cap_errors += 1
                else:
                    dst.write(json.dumps(record['response'], ensure_ascii=False, separators=(',', ':')) + '\n')
                    sent += 1
        print(f'wrote {sent} responses + {cap_errors} error(s) -> {out}', flush=True)


def resolve_log_path(args, model):
    """/ Returns (log_path, run_name); run_name is None for explicit --log paths.

    Explicit --log paths may overwrite. --name is used verbatim (no timestamp
    postfix) so reruns land in a stable, predictable file; when that file (or a
    log of the same auto-generated model name) already exists the caller resumes
    it instead of starting over. Without --name the sanitised model name plus
    a timestamp keeps every run unique.
    """
    if args.log:
        return Path(args.log), None
    if args.name:
        return Path(args.output) / f'{args.name}.jsonl', args.name
    run_name = re.sub(r'[^A-Za-z0-9._-]+', '-', model or 'unnamed')
    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M')
    return Path(args.output) / f'{run_name}_{stamp}.jsonl', run_name


def main():
    ap = argparse.ArgumentParser(description='Run benchmark capabilities against System One.')
    ap.add_argument('--benchmark', default='.', help='Benchmark root directory')
    ap.add_argument('--responses', default='none', help="Directory for per-capability <capability>.jsonl response files "
                                                        "(default: none - the run log already embeds every response). "
                                                        "MUST NOT be the golden responses/ dir; that is refused.")
    ap.add_argument('--capabilities', default=None, help='Comma-separated capability slugs to run (default: all)')
    ap.add_argument('--splits', default='test,calibrate',
                    help='Comma-separated persisted metadata splits to send (default: test,calibrate; train is opt-in)')
    ap.add_argument('--n', type=int, default=None,
                    help='Limit to the first N selected cases per capability after split filtering')
    ap.add_argument('--url', default=None, help='Defaults to $TYPESAFE_BASE_URL' + ENDPOINT_PATH)
    ap.add_argument('--api-key', default=None)
    ap.add_argument('--model', default=None, help='Value substituted for ' + MODEL_PLACEHOLDER + '; defaults to $TYPESAFE_MODEL')
    ap.add_argument('--timeout', type=float, default=10.0, help='Per-request timeout in seconds')
    ap.add_argument('--retries', type=int, default=3, help='Retries per request before noting an error')
    ap.add_argument('--parallel', type=float, default=4.0, help='Target requests per second (rate limit for parallel execution)')
    ap.add_argument('--quota-buster', type=float, default=750.0, metavar='MS',
                    help='Cool-down in ms after every ~25 (±5, re-rolled) requests, jittered ±10%% '
                         '(default: 750, i.e. ±75 ms); 0 disables')
    ap.add_argument('--name', default=None, help='Run name used verbatim as output/<name>.jsonl (no timestamp); '
                                                 'reusing an existing run name resumes it (only cases without a '
                                                 'successful record are re-sent); defaults to <model>_<timestamp>.jsonl')
    ap.add_argument('--output', default='output', help='Directory for the combined run log')
    ap.add_argument('--log', default=None, help='Explicit combined-log path; overwrites and skips the run-name collision check')
    ap.add_argument('--baseline', default=None, help='Baseline run for the post-run per-capability comparison; '
                                                     'default: jev-1.13.0 when scored, else oldest scored run (comparison only runs once >= 2 '
                                                     'runs are scored)')
    ap.add_argument('--no-compare', action='store_true',
                    help='Skip the post-run output/comparison/ report (log, stats and calibration still run)')
    ap.add_argument('--no-calibration', action='store_true',
                    help='Do not fit output/<run>_calibration.json after this run')
    ap.add_argument('--calibration-min-cases', type=int, default=100,
                    help='Minimum successful split=calibrate cases required for deployable T (default: 100)')
    ap.add_argument('--dry-run', action='store_true', help='Show the first request per capability and totals without sending.')
    args = ap.parse_args()

    if args.calibration_min_cases < 1:
        ap.error('--calibration-min-cases must be >= 1')
    try:
        selected_splits = parse_splits(args.splits)
    except ValueError as exc:
        ap.error(str(exc))

    root = Path(args.benchmark)
    manifest_caps = json.loads((root / 'manifest.json').read_text())['capabilities']
    caps = selected_capabilities(args.capabilities, manifest_caps)

    # GOLD GUARD: responses/ is the scoring answer key (manifest 'responses'
    # paths). Eval output must never land there - an accidental default
    # silently replaced the whole benchmark's gold once; never again.
    if args.responses.lower() != 'none':
        gold_root = (root / next(iter(manifest_caps.values()))['responses']).parent
        if Path(args.responses).resolve() == gold_root.resolve():
            raise SystemExit(f'refusing --responses {args.responses}: it is the golden responses dir '
                             f'({gold_root}), the scoring answer key. Use another directory or --responses none.')

    # gated capability files (gpqa_diamond) exist only as password-protected
    # <name>.jsonl.zip at rest: extract the ones this run needs, then remove (and
    # re-encrypt, if a run rewrote them) the plaintext copies when the process
    # ends. atexit covers normal exits, errors and KeyboardInterrupt; the SIGTERM
    # handler turns terminate signals into an exit so the cleanup still runs.
    atexit.register(cleanup_unlocked, unlock_paths(
        [root / manifest_caps[c][k] for c in caps for k in ('requests', 'responses', 'metadata')]))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    dotenv = load_dotenv()
    base = (setting('TYPESAFE_BASE_URL', None, dotenv) or DEFAULT_BASE_URL).rstrip('/')
    url = args.url or base + ENDPOINT_PATH
    api_key = setting('TYPESAFE_API_KEY', args.api_key, dotenv)
    model = setting('TYPESAFE_MODEL', args.model, dotenv)

    write_responses = args.responses.lower() != 'none'
    out_dir = Path(args.responses)
    if write_responses:
        out_dir.mkdir(parents=True, exist_ok=True)  # safe: the gold guard above already rejected responses/

    log_path, run_name = resolve_log_path(args, model)
    resume_path = None
    if not args.dry_run:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if run_name:
            # same run name again = resume: append to the existing log (never
            # rewrite it) and let the task filter below skip already-succeeded cases.
            # the exact-path check covers --name runs, whose log_path carries no timestamp
            existing = sorted(log_path.parent.glob(f'{run_name}_*.jsonl'))
            if log_path.exists() and log_path not in existing:
                existing.append(log_path)
            if existing:
                resume_path = log_path if log_path.exists() else existing[-1]
                log_path = resume_path
        log_file = log_path.open('a' if resume_path else 'w', encoding='utf-8')
    headers = {'Content-Type': 'application/json'}
    if api_key: headers['Authorization'] = 'Bearer ' + api_key

    tasks, counts, first_body = build_tasks(root, caps, manifest_caps, args.n, model, selected_splits)
    done = log_successes(resume_path) if resume_path else {}
    if resume_path:
        total = len(tasks)
        tasks = [t for t in tasks if (t[0], t[1]) not in done]
        print(f'resuming {resume_path.name}: {total - len(tasks)} case(s) already succeeded, '
              f'{len(tasks)} left to (re)run', flush=True)

    if args.dry_run:
        for cap in caps:
            if cap in first_body:
                print(f'{cap}: {first_body[cap][:200]}')
        results = {}
    else:
        results = run_batch(tasks, url, headers, args, log_file)
        log_file.close()
        results.update(done)  # carry over prior successes so responses files stay complete on resume
        if write_responses:
            write_response_files(caps, counts, results, out_dir)
        write_score_reports(root, log_path)
        if not args.no_calibration:
            write_calibration_report(root, log_path, args.calibration_min_cases)
        write_metrics_reports(root, log_path, args.baseline, compare=not args.no_compare)

    errors = sum(1 for r in results.values() if r['error_type'])
    mode = 'dry-run: would send' if args.dry_run else 'sent'
    log_note = '' if args.dry_run else f', log: {log_path}'
    print(f'{mode} {len(tasks)} cases across {len(caps)} capabilities to {url} '
          f'(model={model or "as-is"}, splits={",".join(selected_splits)}), errors: {errors}{log_note}')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__': main()
