# typed-decision-bench

A held-out synthetic benchmark for all System One-compatible models and API inference systems, exercising `POST /v1/systemone` (OpenAPI spec in `openapi/`, hosted reference at `https://api.typesafe.ai`).

- **275 suites / 27,598 cases** — 207 generated suites (20,700 cases) built from [`CAPABILITIES.md`](CAPABILITIES.md), 67 preserved legacy suites (6,700), GPQA Diamond (198)
- Line-aligned `requests/<suite>.jsonl` ↔ `responses/<suite>.jsonl`; each line is a literal `SystemOneRequest` / `SystemOneResponse` with one atomic question; no gold leaks
- Question types: choice 18,140 · noul 7,250 · score 2,208

## Layout

```text
requests/<suite>.jsonl          responses/<suite>.jsonl      metadata/<suite>.jsonl
manifest.json  capability_index.json  stats.json  quality_report.json
openapi/typesafe-systemone-openapi-v0.2.0.json
run_eval.py  scripts/{score,validate,sync-metadata,gpqa_zip,test_gpqa_zip}.py  Makefile
```

## Configuration

Request lines carry the model placeholder `REPLACED_BY_TYPESAFE_MODEL`, substituted at send time. Resolution order: CLI flag > environment variable > `.env` (repo root):

| Setting | CLI flag | Variable | Example |
|---|---|---|---|
| Model | `--model` | `TYPESAFE_MODEL` | `XHToken/Spark-X2.5` |
| Base URL | `--url` (full URL) | `TYPESAFE_BASE_URL` | `http://baradcuda:5380` (no trailing slash) |
| API key | `--api-key` | `TYPESAFE_API_KEY` | bearer token |

Without `TYPESAFE_BASE_URL`, the hosted endpoint is used.

## Running

```bash
make eval                                            # all suites; run log -> stats/cases -> metrics -> comparison
make eval ARGS="--capabilities gpqa_diamond --n 2"   # subset
make e2e                                             # 1 case x 275 suites -> output/e2e.jsonl (+ stats/cases)
make score                                           # backfill unscored output/*.jsonl logs
make metrics                                         # every run -> output/<run>/metrics.csv + metrics.json
make compare ARGS="--baseline three"                 # per-capability comparison -> output/comparison/
make validate                                        # rewrites validation_status.json
make test                                            # unit tests for the evalcompare library in src/
```

Every eval/e2e run finishes with the full pipeline: its log is scored
(`output/<run>_stats.jsonl`/`_cases.jsonl`), its metrics folder refreshed
(`output/<run>/`), and — once at least two runs are scored — the per-capability
comparison regenerated in `output/comparison/`. The comparison baseline is the
run's `--baseline` flag, defaulting to the oldest scored run (a stable
reference — with fewer than two runs it is skipped: no baseline, no
comparison).

`make metrics` (`scripts/metrics.py`) projects each `output/<run>_stats.jsonl`
through the evalcompare metric registry in `src/` into a per-run subfolder
`output/<run>/` (`metrics.csv` + `metrics.json`, one schema for all runs).
Single-run only — no baseline, no comparison. For a per-capability comparison
across runs, `make compare ARGS="--baseline <run> [--metric accuracy]"` loads
every scored run, pivots it into a capability x run matrix and writes
`output/comparison/`: `capability_<metric>.csv`, per-capability
`delta_vs_<baseline>_<metric>.csv`, `capability_spread_<metric>.csv`,
`model_summary.csv`, `all_rows.csv`, `summary.json` and a plotly
`report.html` dashboard (heatmap, deltas, discriminators, latency trade-off).
No baseline -> no comparison.

Every run (`make eval`, `make e2e`) scores its own log on completion and writes
both artifacts for it — predictions are the response payloads embedded in the
log, so runs recorded with `--responses none` (like `make e2e`) score too, and
partial/aborted runs produce `n=0` null-metric lines instead of failing.
`make score` (`scripts/score.py --all-logs`) backfills any older combined run
log in `output/` that has no `<run>_stats.jsonl` yet; delete a run's
`_stats.jsonl` to re-score it. The single-run mode joins a responses dir with
the newest run log (or `--log`, e.g. `uv run scripts/score.py --benchmark . --responses responses`)
and derives the run name from the log, e.g. `mymodel` from `mymodel_<timestamp>.jsonl`:

- `output/<run>_stats.jsonl` — one flat line per capability plus a `micro` line:
  `{run, capability, n, accuracy, soft_accuracy, nll, brier, confidence, ece_15,
  score_mae, score_within_one, errors, error_rate, latency_ms_mean, latency_ms_p50,
  latency_ms_p95}` — lines from different models can simply be concatenated and compared.
- `output/<run>_cases.jsonl` — one line per case:
  `{case_id: "typed-decisions-bench-v1:<family>:<line>", family, qtype, correct, confidence,
  nll, brier, soft_accuracy, prediction_label, gold_label, latency_ms[, score_error]}`.

Calibration (`nll`, `brier`, `soft_accuracy`) is computed against the full gold
probability distribution, `confidence` is the prediction's top probability, and only
successful requests count toward latency.

Responses go to `responses/<capability>.jsonl` plus a combined run log — `output/<model>_<timestamp>.jsonl` by default, or `output/<name>.jsonl` (no timestamp) with `--name` (one record per case: `{request_id, response, error_type, error, duration_ms, endpoint}`). Tuning: `--timeout` (default 5 s; `make e2e` uses 120), `--retries` (3), `--parallel` req/s (20), `--quota-buster <ms>` (all senders cool down after every ~25 ±5 requests for the given wait ±10%; default 500 ms ±50 ms, `0` disables), `--name` (stable file name), `--dry-run`. Failures keep line alignment and exit non-zero; a run aborts early if its first 10 requests all fail. `make validate` schema-checks all benchmark data and the latest e2e log; `make sync-metadata --fix` reconciles the index/hash/stats/receipt files with disk (see `SHA256SUMS.txt`, verifiable via `shasum -c`).

**Resuming:** reusing a run name resumes that run — cases with a successful record in the existing log are skipped and only pending ones (failed, timed out, aborted, never reached) are (re)sent; new records are appended to the same log, the responses files are merged, and the `_stats`/`_cases` reports are refreshed. Recovered cases don't count as errors — the last record per case wins.

## Gated data (gpqa_diamond)

The GPQA Diamond files are stored encrypted at rest — each of
`requests/gpqa_diamond.jsonl`, `responses/gpqa_diamond.jsonl` and
`metadata/gpqa_diamond.jsonl` exists only as a password-protected
`<name>.zip` sibling (`gpqa_diamond.jsonl.zip`), never as plaintext on disk.
The zip password is a public constant in `scripts/gpqa_zip.py` (overridable via
`GPQA_ZIP_PASSWORD`): the lock keeps plaintext out of checkouts and context
windows, it is not a secrecy boundary.

Every consumer decrypts transparently: `run_eval.py` (eval/e2e),
`scripts/validate.py`, `scripts/score.py` and `scripts/sync-metadata.py`
extract the files they need at startup and remove the plaintext copies when
the process ends — files a run rewrote (e.g. `responses/gpqa_diamond.jsonl`
after an eval) are re-encrypted into the zip first, so nothing is lost. The
archives use classic PKZIP ZipCrypto (readable by the stdlib and `unzip -P`
without third-party dependencies).

```bash
make gpqa-status   # show locked/unlocked state
make gpqa-lock     # encrypt *.jsonl -> *.jsonl.zip, remove plaintext
make gpqa-unlock   # decrypt for manual maintenance (remember to lock again)
uv run scripts/test_gpqa_zip.py   # self-checks: interop + lifecycle
```

## Generation & status

Gold is deterministically derivable (arithmetic, state machines, rubrics, supplied evidence) — see [`GENERATION_METHODOLOGY.md`](GENERATION_METHODOLOGY.md). Boolean→`noul`, category/ranking→`choice`, ordinal→`score`. Requests are globally unique with varied option order and label-free context.

Current build (see `quality_report.json` / `validation_status.json`): OpenAPI + semantic validation **PASS**; golden self-score **1.000** micro accuracy; **0** duplicate request bodies; every generated suite ≥ 2 distinct gold outcomes (max single-outcome share 86%); preserved suite files: 20 / 30 byte-identical (details in `preservation_receipt.json`).

## Interpretation

Cases are synthetic and parameterized, not IID production samples — prefer paired comparisons on identical cases, regression detection, per-suite error analysis, and invariance tests over absolute IID confidence intervals. Intentionally separate from model training corpora.
