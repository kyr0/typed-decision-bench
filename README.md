![typed-decision-bench logo](logo.png)

# typed-decision-bench

A held-out synthetic benchmark for all System One-compatible models and API inference systems, exercising `POST /v1/systemone` (OpenAPI spec in [`openapi/`](openapi/), hosted "Jev" reference at `https://api.typesafe.ai`).

- **275 suites / 27,598 cases** — for details see [`CAPABILITIES.md`](CAPABILITIES.md) and [`METHODOLOGY.md`](METHODOLOGY.md)
- Line-aligned `requests/<suite>.jsonl` ↔ `responses/<suite>.jsonl`; each line is a literal `SystemOneRequest` / `SystemOneResponse` with one atomic question; no gold leaks
- Question types: choice 18,140 · noul 7,250 · score 2,208
- Zero-install tooling: every script is self-contained Python 3.11+ with PEP 723 inline dependencies, run via `uv run`

## Repository layout

| Path | Contents |
|---|---|
| [`requests/`](requests/) | one `<capability>.jsonl` per suite — literal request payloads (model placeholder inside) |
| [`responses/`](responses/) | one `<capability>.jsonl` per suite — model responses, line-aligned with the requests |
| [`metadata/`](metadata/) | gold labels + per-case provenance |
| [`manifest.json`](manifest.json) | capability → file paths (the runner's source of truth for suite selection) |
| [`openapi/`](openapi/) | SystemOne OpenAPI spec, schema-checked by `make validate` |
| [`scripts/`](scripts/) | [`run_eval.py`](scripts/run_eval.py) (benchmark runner — full CLI reference below), plus `validate.py`, `score.py`, `metrics.py`, `sync-metadata.py`, `gpqa_zip.py` |
| [`src/evalcompare/`](src/evalcompare/) | the `evalcompare` library package — `loader.py` (stats JSONL → DataFrames), `metrics.py`, `analysis.py`, `report.py` |
| [`tests/`](tests/) | pytest suite: evalcompare library, scorer CLI, runner (naming/resume/rate limiting vs a stub endpoint), gpqa zip lifecycle — run via `make test` |
| `output/` | run logs + generated reports (created on first run) |

## Quick start

```bash
cp env-example .env        # set TYPESAFE_MODEL / TYPESAFE_API_KEY / TYPESAFE_BASE_URL
make eval ARGS="--n 2 --capabilities gpqa_diamond,contains_pii"   # smoke run first
make eval                  # full benchmark + scoring + metrics + comparison
```

## Configuration

Request lines carry the model placeholder `REPLACED_BY_TYPESAFE_MODEL`, substituted at send time. Every setting resolves **CLI flag > process environment > `.env`** (repo root, parsed without dependencies — `KEY=VALUE`, `#` comments, optional quotes):

| Setting | CLI flag | Environment variable | Example |
|---|---|---|---|
| Model | `--model` | `TYPESAFE_MODEL` | `XHToken/Spark-X2.5` |
| Base URL | `--url` (*full* endpoint URL) | `TYPESAFE_BASE_URL` | `http://baradcuda:5380` (no trailing slash; `/v1/systemone` is appended) |
| API key | `--api-key` | `TYPESAFE_API_KEY` | bearer token (sent as `Authorization: Bearer …`) |
| GPQA zip password | — (per-file via `gpqa_zip.py`) | `GPQA_ZIP_PASSWORD` | see [Gated data](#gated-data-gpqa_diamond) |

Without `TYPESAFE_BASE_URL` the hosted endpoint `https://api.typesafe.ai/v1/systemone` is used. `--url` overrides the whole endpoint (base + path), so it accepts the complete URL.

## Make targets

| Target | What it does |
|---|---|
| `make eval` | run all suites (`ARGS` forwarded to `scripts/run_eval.py`, e.g. `ARGS="--capabilities gpqa_diamond --n 2"`); after the run: score → metrics → comparison |
| `make e2e` | 1 case × all 275 suites, `--responses none --timeout 120 --log output/e2e.jsonl` (endpoint smoke test) |
| `make score` | backfill: score every `output/*.jsonl` log that has no `_stats.jsonl` yet (`scripts/score.py --all-logs`) |
| `make metrics` | project every `output/<run>_stats.jsonl` → `output/<run>/metrics.{csv,json}` |
| `make compare` | per-capability comparison of all scored runs, e.g. `ARGS="--baseline three [--metric accuracy]"` → `output/comparison/` |
| `make validate` | schema-check all benchmark data + the latest e2e log → `validation_status.json` |
| `make sync-metadata` | reconcile `capability_index.json`, `suite_hashes.json`, `manifest.json`, `stats.json`, `golden_selftest.json`, `preservation_receipt.json`, `quality_report.json` with disk (`--fix` applies) |
| `make test` | pytest suite in [`tests/`](tests/): evalcompare library, scorer, runner, gpqa zip |
| `make gpqa-status` / `gpqa-lock` / `gpqa-unlock` | inspect / encrypt / decrypt the gated GPQA files |

## CLI reference

### `scripts/run_eval.py` — benchmark runner

`uv run scripts/run_eval.py [options]` (the Makefile adds `--benchmark . --responses responses`).

| Flag | Default | Description |
|---|---|---|
| `--benchmark PATH` | `.` | benchmark root directory (must contain `manifest.json`) |
| `--capabilities CSV` | all | comma-separated capability slugs to run; unknown slugs abort with the valid list |
| `--n N` | all | limit to the first N cases per capability |
| `--url URL` | `$TYPESAFE_BASE_URL` + `/v1/systemone` | full endpoint URL, overrides base + path |
| `--api-key KEY` | `$TYPESAFE_API_KEY` | bearer token |
| `--model NAME` | `$TYPESAFE_MODEL` | value substituted for `REPLACED_BY_TYPESAFE_MODEL` |
| `--timeout SEC` | `10` | per-request timeout (`make e2e` raises it to `120`) |
| `--retries N` | `3` | attempts per request before recording an error |
| `--parallel RPS` | `18` | target requests per second (workers = `min(512, max(8, rps × 5))`) |
| `--quota-buster MS` | `750` | cool-down after every ~25 (±5, re-rolled) requests, jittered ±10 % (`0` disables) |
| `--responses DIR` | `responses` | per-capability response files; `none` skips them (predictions stay embedded in the log, scoring still works) |
| `--output DIR` | `output` | directory for the combined run log |
| `--log PATH` | — | explicit combined-log path; **overwrites** and skips resume detection |
| `--name NAME` | `<model>_<timestamp>` | stable log name `output/<name>.jsonl`; reusing an existing name **resumes** that run |
| `--baseline RUN` | oldest scored run | baseline for the automatic post-run comparison (needs ≥ 2 scored runs) |
| `--dry-run` | — | print the first request body per capability + totals, send nothing |

Exit status is non-zero if any case failed. A run **aborts early** when its first 10 requests all fail (circuit breaker) — queued cases are recorded as `aborted` instead of hammering a dead endpoint.

### `scripts/score.py` — scorer

| Flag | Default | Description |
|---|---|---|
| `--benchmark PATH` | `.` | benchmark root |
| `--all-logs` | — | backfill mode: score every unscored `output/*.jsonl` log from the responses embedded in it (mutually exclusive with `--responses`/`--log`/`--out`) |
| `--responses DIR` | — | single-run mode: directory of `<capability>.jsonl` (required without `--all-logs`) |
| `--log PATH` | newest `output/*.jsonl` | combined log supplying latency/error rows |
| `--out PATH` | `output/<run>_stats.jsonl` | stats path; the cases file goes next to it as `<stem>_cases.jsonl` |

Run names are derived from the log, e.g. `mymodel` from `mymodel_<timestamp>.jsonl`. Deleting a run's `_stats.jsonl` marks it unscored so `make score` re-derives it.

### `scripts/metrics.py` — metrics exporter & comparison

| Flag | Default | Description |
|---|---|---|
| `--benchmark PATH` | `.` | benchmark root |
| `--runs NAME…` | all | restrict the per-run export to these run names |
| `--compare` | — | switch from per-run export to the cross-run comparison (requires `--baseline`) |
| `--baseline RUN` | — | baseline run name; the comparison never fabricates one — no baseline, no comparison |
| `--metric NAME` | `soft_accuracy` | primary comparison metric (see [metrics registry](#metrics-registry)) |

### `scripts/validate.py`

`--benchmark PATH` (default `.`) — validates every request/response/metadata line against the OpenAPI schema and semantic rules, prints the status document to stdout (`make validate` redirects it to `validation_status.json`).

### `scripts/sync-metadata.py`

No flag = report drift (missing/stale entries, wrong aggregates); `--fix` = append missing entries and recompute hashes/aggregates for `capability_index.json`, `suite_hashes.json`, `manifest.json`, `stats.json`, `golden_selftest.json`, `preservation_receipt.json`, `quality_report.json`.

### `scripts/gpqa_zip.py`

Subcommands `lock`, `unlock`, `status`, each accepting repeatable `--file JSONL` (default: the three protected `gpqa_diamond.jsonl` paths). Password from `GPQA_ZIP_PASSWORD`, falling back to the public constant in the script.

## Metrics registry

The `--metric` values and `*_stats.jsonl` columns come from one registry in [`src/metrics.py`](src/metrics.py), so a metric name always means the same thing (and deltas always point "better = positive"):

| Metric | Better | Meaning |
|---|---|---|
| `accuracy` | higher | hard correctness against the gold label |
| `soft_accuracy` | higher | probability-weighted correctness — more discriminative than saturated hard accuracy |
| `nll` | lower | negative log loss vs the full gold distribution |
| `brier` | lower | Brier score vs the full gold distribution |
| `confidence` | neutral | mean top-probability of the prediction |
| `ece_15` | lower | expected calibration error, 15 bins |
| `score_mae` | lower | mean absolute error on `score` questions |
| `score_within_one` | higher | share of `score` answers within ±1 |
| `errors`, `error_rate` | lower | transport errors (timeout/service) and their share |
| `latency_ms_mean`, `latency_ms_p50`, `latency_ms_p95` | lower | wall time of successful requests only |
| `n` | neutral | scored examples |

## Output artifacts

Every eval/e2e run finishes with the full pipeline: its log is scored, its metrics folder refreshed, and — once at least two runs are scored — the per-capability comparison regenerated.

- **Run log** `output/<model>_<timestamp>.jsonl` (or `output/<name>.jsonl` with `--name`) — one record per case: `{request_id, response, error_type, error, duration_ms, endpoint}`. `error_type` is `null` on success, else `timeout_retries_exhausted`, `service_error` or `aborted` (never sent because the run was cancelled).
- **`output/<run>_stats.jsonl`** — one flat line per capability plus a `micro` line: `{run, capability, n, accuracy, soft_accuracy, nll, brier, confidence, ece_15, score_mae, score_within_one, errors, error_rate, latency_ms_mean, latency_ms_p50, latency_ms_p95}` — lines from different models can simply be concatenated and compared.
- **`output/<run>_cases.jsonl`** — one line per case: `{case_id: "typed-decisions-bench-v1:<family>:<line>", family, qtype, correct, confidence, nll, brier, soft_accuracy, prediction_label, gold_label, latency_ms[, score_error]}`.
- **`output/<run>/`** — single-run projection via the evalcompare registry: `metrics.csv` (registry column order, identical schema for every run) + `metrics.json` (micro aggregate + metadata).
- **`output/comparison/`** — cross-run pivot (capability × run): `capability_<metric>.csv`, `capability_n.csv`, `delta_vs_<baseline>_<metric>.csv`, `capability_spread_<metric>.csv`, `model_summary.csv`, `all_rows.csv`, `summary.json` and a plotly `report.html` dashboard (heatmap, deltas, discriminators, latency trade-off).

Calibration (`nll`, `brier`, `soft_accuracy`) is computed against the full gold probability distribution, `confidence` is the prediction's top probability, and only successful requests count toward latency. Because predictions are the response payloads embedded in the log, runs recorded with `--responses none` (like `make e2e`) score too, and partial/aborted runs produce `n=0` null-metric lines instead of failing.

**Resuming:** reusing a run name resumes that run — cases with a successful record in the existing log are skipped and only pending ones (failed, timed out, aborted, never reached) are (re)sent; new records are appended to the same log, the responses files are merged, and the `_stats`/`_cases` reports are refreshed. Recovered cases don't count as errors — the last record per case wins.

## Rate limiting & reliability

Three deliberate layers keep a sustained benchmark run from tripping endpoint quotas or masking outages:

1. **Pacer** (`--parallel`): request starts are spaced exactly `1/rps` apart under one lock, so the dispatch rate is bounded regardless of worker count.
2. **Quota buster** (`--quota-buster`): after every ~25 ±5 dispatched requests all senders pause for the configured wait ±10 %; both numbers re-roll per cycle so the traffic pattern stays unrecognizable to burst heuristics. `0` disables it.
3. **Circuit breaker**: if the first 10 requests all fail, the run aborts with a clear message — an unresponsive endpoint should fail fast, not consume hours of retries. Queued tasks get `aborted` records so line alignment and resume stay intact.

`make validate` schema-checks all benchmark data and the latest e2e log; `make sync-metadata --fix` reconciles the index/hash/stats/receipt files with disk (`SHA256SUMS.txt` is verifiable via `shasum -c SHA256SUMS.txt`).

## Gated data (gpqa_diamond)

The GPQA Diamond files are stored encrypted at rest — each of `requests/gpqa_diamond.jsonl`, `responses/gpqa_diamond.jsonl` and `metadata/gpqa_diamond.jsonl` exists only as a password-protected `<name>.zip` sibling (`gpqa_diamond.jsonl.zip`), never as plaintext on disk. The zip password is a public constant in [`scripts/gpqa_zip.py`](scripts/gpqa_zip.py) (overridable via `GPQA_ZIP_PASSWORD`): the lock keeps plaintext out of checkouts and context windows, it is not a secrecy boundary.

Every consumer decrypts transparently: `scripts/run_eval.py` (eval/e2e), `scripts/validate.py`, `scripts/score.py` and `scripts/sync-metadata.py` extract the files they need at startup and remove the plaintext copies when the process ends — files a run rewrote (e.g. `responses/gpqa_diamond.jsonl` after an eval) are re-encrypted into the zip first, so nothing is lost. The archives use classic PKZIP ZipCrypto (readable by the stdlib and `unzip -P` without third-party dependencies).

```bash
make gpqa-status   # show locked/unlocked state
make gpqa-lock     # encrypt *.jsonl -> *.jsonl.zip, remove plaintext
make gpqa-unlock   # decrypt for manual maintenance (remember to lock again)
```

The zip interop + lifecycle self-checks are part of `make test` (`tests/test_gpqa_zip.py`).

## Generation & status

Gold is deterministically derivable (arithmetic, state machines, rubrics, supplied evidence) — see [`METHODOLOGY.md`](METHODOLOGY.md). Boolean→`noul`, category/ranking→`choice`, ordinal→`score`. Requests are globally unique with varied option order and label-free context.

Current build (see `quality_report.json` / `validation_status.json`): OpenAPI + semantic validation **PASS**; golden self-score **1.000** micro accuracy; **0** duplicate request bodies; every generated suite ≥ 2 distinct gold outcomes (max single-outcome share 86%); preserved suite files: 20 / 30 byte-identical (details in `preservation_receipt.json`).

## Interpretation

Cases are synthetic and parameterized, not IID production samples — prefer paired comparisons on identical cases, regression detection, per-suite error analysis, and invariance tests over absolute IID confidence intervals. Intentionally separate from model training corpora.
