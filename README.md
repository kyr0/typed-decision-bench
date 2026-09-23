![typed-decision-bench logo](logo.png)

# 🧪 typed-decision-bench

> The eval bench (and novel `calibration.json` standard) that fixes the **overconfident decisions** problem in **any** open System One reproduction model - without touching a single model weight.

**TL;DR:**
- 🎯 Evaluate **any** model and inference system that speaks the System One typed decisions API (`POST /v1/systemone`) — 275 capabilities, 27,598 held-out cases.
- 🌡️ Calibrate **any** model and inference System One inference system! **Calibration** changes what the system claims, never what it chooses. Any  **_confidently_ wrong model** becomes an **_honestly_ wrong model** via the new [**Qtype-Stratified Temperature Scaling method**](#-calibration) with **no re-training and no weight changes required**: Just run the `calibrate` split to fit a `calibration.json` (I propose this as a standard, see below); a compliant engine loading it provably states confidences that match its observed accuracy (see [`Bonsai-Llama-Jev`](https://github.com/kyr0/Bonsai-Llama-Jev) where I implemented this novel method).

## ✨ What's inside

- 📊 **275 suites / 27,598 cases** on an explicit `test`/`calibrate`/`train` split contract — details in [`CAPABILITIES.md`](CAPABILITIES.md) and [`METHODOLOGY.md`](METHODOLOGY.md)
- 🌡️ **Calibration standard** ([`CALIBRATION.md`](CALIBRATION.md)): run the bench, configure your inference engine to load the `calibration.json` produced by this bench. Internally, the inference engine must apply the temperature transform described below: `temperatures.get(question_type, artifact.temperature)` at serve time (that's extremely cheap!); and this fixes the **overconfident decisions** issues right away! It upgrades any model from "making typed decisions" to "making typed decisions and knowing how much to trust each decision". See my [`Bonsai-Llama-Jev`](https://github.com/kyr0/Bonsai-Llama-Jev) reference implementation. It's a just a few lines of code!
- 📝 **Line-aligned** `requests/<suite>.jsonl` ↔ `responses/<suite>.jsonl`; each line is a literal `SystemOneRequest` / `SystemOneResponse` with one atomic question; no gold leaks
- 🔤 **Question types:** __choice:__ 18,140 · __noul:__ 7,250 · __score:__ 2,208
- 🚫 **Zero-install tooling:** every script is self-contained Python 3.11+ with PEP 723 inline dependencies, run via `uv run`

## 🚀 Quick start

**Step 1 — configure:**

```bash
cp .env.example .env        # set TYPESAFE_MODEL / TYPESAFE_API_KEY / TYPESAFE_BASE_URL
```

**Step 2 — smoke run** (2 cases, before you burn an hour):

```bash
make eval ARGS="--n 2 --capabilities gpqa_diamond,contains_pii"
```

**Step 3 — the full benchmark** (run → score → metrics → comparison, all automatic):

```bash
make eval
```

## ⚙️ Configuration

Request lines carry the model placeholder `REPLACED_BY_TYPESAFE_MODEL`, substituted at send time. Every setting resolves **CLI flag > process environment > `.env`** (repo root, parsed without dependencies — `KEY=VALUE`, `#` comments, optional quotes):

| Setting | CLI flag | Environment variable | Example |
|---|---|---|---|
| Model | `--model` | `TYPESAFE_MODEL` | `Jev-1.13.0` |
| Base URL | `--url` (*full* endpoint URL) | `TYPESAFE_BASE_URL` | `https://api.typesafe.ai` (no trailing slash; `/v1/systemone` is appended) |
| API key | `--api-key` | `TYPESAFE_API_KEY` | bearer token (sent as `Authorization: Bearer ...`) |
| GPQA zip password | — (per-file via `gpqa_zip.py`) | `GPQA_ZIP_PASSWORD` | see [Gated data](#-gated-data-gpqa_diamond) |

Set `TYPESAFE_BASE_URL` to use a local endpoint like `http://localhost:5380` in case you run this with my [`Bonsai-Llama-Jev`](https://github.com/kyr0/Bonsai-Llama-Jev) reference implementation.

## 🎛️ Make targets

| Target | What it does |
|---|---|
| `make eval` | run all suites (`ARGS` forwarded to `scripts/run_eval.py`, e.g. `ARGS="--capabilities gpqa_diamond --n 2"`); after the run: score → metrics → comparison |
| `make eval-only` | same run, **no** `output/comparison/` report (`--no-compare`; stats/metrics/calibration still written) |
| `make e2e` | 1 case × all 275 suites, `--responses none --timeout 120 --log output/e2e.jsonl` (endpoint smoke test) |
| `make score` | backfill: score **split=test only** for every unscored run log |
| `make calibrate` | fit temperature scaling from **split=calibrate only** — qtype-affine: one T per answer type (`choice`/`noul`/`score`) where ≥100 calibrate cases exist, global T otherwise; reports held-out **split=test** raw/calibrated metrics and nothing else (no `_stats`/`_cases`/metrics/comparison files). Requires an existing run log (`make eval` first; default: newest `output/*.jsonl`). Example with an explicit artifact location: `make calibrate ARGS="--log output/my-model-20 --out calibrations/my-model.json"` (default output: `<log dir>/<run>_calibration.json`) |
| `make metrics` | project every `output/<run>_stats.jsonl` → `output/<run>/metrics.{csv,json}` |
| `make compare` | re-create `output/comparison/` for all scored runs; baseline defaults to `jev-1.13.0` (override: `BASELINE=<run>` or `ARGS="--baseline <run> [--metric accuracy]"`) |
| `make validate` | schema-check all benchmark data + the latest e2e log → `validation_status.json` |
| `make sync-metadata` | reconcile `capability_index.json`, `suite_hashes.json`, `manifest.json`, `stats.json`, `golden_selftest.json`, `preservation_receipt.json`, `quality_report.json` with disk (`--fix` applies) |
| `make test` | pytest suite in [`tests/`](tests/): evalcompare library, scorer, runner, gpqa zip |
| `make gpqa-status` / `gpqa-lock` / `gpqa-unlock` | inspect / encrypt / decrypt the gated GPQA files |

## 📖 CLI reference

### 🏃 `scripts/run_eval.py` — benchmark runner

`uv run scripts/run_eval.py [options]` (the Makefile adds `--benchmark .`).

| Flag | Default | Description |
|---|---|---|
| `--name NAME` | `<model>_<timestamp>` | stable log name `output/<name>.jsonl`; reusing an existing name **resumes** that run |
| `--benchmark PATH` | `.` | benchmark root directory (must contain `manifest.json`) |
| `--capabilities CSV` | all | comma-separated capability slugs to run; unknown slugs abort with the valid list |
| `--splits CSV` | `test,calibrate` | persisted metadata splits to send; `train` is opt-in |
| `--n N` | all | limit to first N selected cases per capability **after** split filtering |
| `--url URL` | `$TYPESAFE_BASE_URL` + `/v1/systemone` | full endpoint URL, overrides base + path |
| `--api-key KEY` | `$TYPESAFE_API_KEY` | bearer token |
| `--model NAME` | `$TYPESAFE_MODEL` | value substituted for `REPLACED_BY_TYPESAFE_MODEL` |
| `--timeout SEC` | `10` | per-request timeout (`make e2e` raises it to `120`) |
| `--retries N` | `3` | attempts per request before recording an error |
| `--parallel RPS` | `18` | target requests per second (workers = `min(512, max(8, rps × 5))`) |
| `--quota-buster MS` | `750` | cool-down after every ~25 (±5, re-rolled) requests, jittered ±10 % (`0` disables) |
| `--responses DIR` | `none` | opt-in per-capability response files; pointing this at the golden `responses/` dir is **refused** (answer key!) — the run log already embeds every response, so scoring never needs them |
| `--output DIR` | `output` | directory for the combined run log |
| `--log PATH` | — | explicit combined-log path; **overwrites** and skips resume detection |
| `--baseline RUN` | `jev-1.13.0` if scored, else oldest scored run | baseline for the automatic post-run comparison (needs ≥ 2 scored runs) |
| `--dry-run` | — | print the first request body per capability + totals, send nothing |

Exit status is non-zero if any case failed. A run **aborts early** when its first 10 requests all fail (circuit breaker) — queued cases are recorded as `aborted` instead of hammering a dead endpoint.

### ✅ `scripts/score.py` — scorer

| Flag | Default | Description |
|---|---|---|
| `--benchmark PATH` | `.` | benchmark root |
| `--all-logs` | — | backfill mode: score every unscored `output/*.jsonl` log from the responses embedded in it (mutually exclusive with `--responses`/`--log`/`--out`) |
| `--responses DIR` | — | single-run mode: directory of `<capability>.jsonl` (required without `--all-logs`) |
| `--log PATH` | newest `output/*.jsonl` | combined log supplying latency/error rows |
| `--out PATH` | `output/<run>_stats.jsonl` | stats path; the cases file goes next to it as `<stem>_cases.jsonl` |

Run names are derived from the log, e.g. `mymodel` from `mymodel_<timestamp>.jsonl`. Deleting a run's `_stats.jsonl` marks it unscored so `make score` re-derives it.

### 📊 `scripts/metrics.py` — metrics exporter & comparison

| Flag | Default | Description |
|---|---|---|
| `--benchmark PATH` | `.` | benchmark root |
| `--runs NAME…` | all | restrict the per-run export to these run names |
| `--compare` | — | switch from per-run export to the cross-run comparison (requires `--baseline`) |
| `--baseline RUN` | — | baseline run name; the comparison never fabricates one — no baseline, no comparison |
| `--metric NAME` | `soft_accuracy` | primary comparison metric (see [metrics definitions](#-metrics-definitions)) |

### 🔍 `scripts/validate.py`

`--benchmark PATH` (default `.`) — validates every request/response/metadata line against the OpenAPI schema and semantic rules, prints the status document to stdout (`make validate` redirects it to `validation_status.json`).

### 🔄 `scripts/sync-metadata.py`

No flag = report drift (missing/stale entries, wrong aggregates); `--fix` = append missing entries and recompute hashes/aggregates for `capability_index.json`, `suite_hashes.json`, `manifest.json`, `stats.json`, `golden_selftest.json`, `preservation_receipt.json`, `quality_report.json`.

### 🔐 `scripts/gpqa_zip.py`

Subcommands `lock`, `unlock`, `status`, each accepting repeatable `--file JSONL` (default: the three protected `gpqa_diamond.jsonl` paths). Password from `GPQA_ZIP_PASSWORD`, falling back to the public constant in the script.

## 📐 Metrics definitions

The `--metric` values and `*_stats.jsonl` columns come from one registry in [`src/metrics.py`](src/metrics.py), so a metric name always means the same thing (and deltas always point "better = positive"):

Throughout, $p$ is the predicted probability vector and $y$ the gold distribution over the same labels (for `noul`, $y=(1-v,\,v)$ on *(false, true)* from the gold $v$; for `score`/`choice`, the gold's own `probabilities`) — all calibration metrics use the **full** $y$, never its argmax one-hot:

| Metric | Better | Definition / meaning |
|---|---|---|
| `accuracy` | higher | hard correctness: $\frac{1}{n}\sum_i \mathbb{1}[\arg\max p^{(i)} = \arg\max y^{(i)}]$ |
| `soft_accuracy` | higher | gold-distribution agreement: $\frac{1}{n}\sum_i \sum_k p_k^{(i)} y_k^{(i)}$ — the probability mass $p$ puts on the gold (for a one-hot gold this is simply the gold label's probability); rewards "close and spread right", so it stays discriminative where hard accuracy saturates |
| `nll` | lower | negative log-loss / cross-entropy vs the full gold: $\frac{1}{n}\sum_i -\sum_k y_k^{(i)} \log p_k^{(i)}$ — proper (truthful), punishes wrong confidence hardest |
| `brier` | lower | squared Euclidean distance to gold: $\frac{1}{n}\sum_i \sum_k (p_k^{(i)} - y_k^{(i)})^2$ — also proper, quadratically gentler than NLL |
| `confidence` | neutral | mean claimed certainty: $\frac{1}{n}\sum_i \max_k p_k^{(i)}$ — read *against* accuracy; the gap is the miscalibration temperature fixes |
| `ece_15` | lower | expected calibration error over 15 equal-width confidence bins: $\sum_b \frac{n_b}{n}\bigl\lvert \overline{\mathrm{conf}}_b - \mathrm{acc}_b \bigr\rvert$ — the acceptance metric for calibration (accuracy is blind to it) |
| `score_mae` | lower | $\frac{1}{n}\sum \lvert \mathbb{E}_p[i] - \mathbb{E}_y[i] \rvert$ on `score` questions (expected-level error) |
| `score_within_one` | higher | share of `score` answers with `score_mae` $\le 1$ |
| `errors`, `error_rate` | lower | transport errors (timeout/service) and their share |
| `latency_ms_mean`, `latency_ms_p50`, `latency_ms_p95` | lower | wall time of successful requests only |
| `n` | neutral | scored examples |

## 🌡️ Calibration

Published benchmark metrics are computed from **split=test only**. Temperature calibration is fitted from **split=calibrate only** and may report raw-vs-calibrated test metrics offline; **split=train is excluded from both**. The fit is **qtype-affine** — one temperature $T$ per answer type (`choice`/`noul`/`score`), global $T$ as fallback for groups below the data floor (default ≥ 100 calibrate cases each). Artifacts are always **schema v2**: a `temperatures` map (empty until a qtype earns its own $T$) plus the global `temperature` fallback and a `scope` flag; loaders resolve `temperatures.get(question_type, artifact.temperature)` and must require `schema_version == 2` (v1 artifacts are no longer produced).

The transform, broken down ($p$ = the model's reported probability vector, $q$ = the calibrated one, $T$ = the fitted temperature):

$$
q_i \;=\; \frac{p_i^{\,1/T}}{\sum_j p_j^{\,1/T}} \;\equiv\; \mathrm{softmax}\!\left(\frac{\log p}{T}\right)_i
$$

1. **Raise each probability to $1/T$** — this takes the odds between any two options to the power $1/T$, nothing else.
2. **Renormalize** so the vector $\sum_i q_i = 1$ again.
3. **Effect:** $T>1$ flattens, $T<1$ sharpens, $T=1$ is the identity; the ordering of the $q_i$ (and therefore the predicted label) never changes — only confidence moves.

Worked example on an overconfident `noul` answer: $p(\text{true}) = 0.9$ (odds $9:1$ for *true*). With $T=2$ every probability becomes its square root, so the odds are square-rooted:

$$
\sqrt{0.9} : \sqrt{0.1} \;=\; 3 : 1 \quad\Rightarrow\quad q(\text{true}) = \tfrac{3}{4} = 0.75
$$

The model still says *true*, but its claimed certainty $q(\text{true})$ drops from $0.9$ to $0.75$ — exactly the amount of confidence the calibrate split showed it cannot keep. The fitted $T$ is simply "the exponent that makes held-out confidence match observed accuracy".

The statistics behind the estimator — convexity, the $\mathrm{SE}(\hat T) \propto 1/\sqrt n$ law that sets the ≥ 100-case floor, and why the independent `calibrate` split is a requirement rather than a convention — are derived in [`CALIBRATION.md`](CALIBRATION.md#why-the-fit-is-statistically-sound-and-when-it-isnt).

## 📄 Output artifacts

Every eval/e2e run finishes with the full pipeline: its log is scored, its metrics folder refreshed, and — once at least two runs are scored — the per-capability comparison regenerated.

- 🧾 **Run log** `output/<model>_<timestamp>.jsonl` (or `output/<name>.jsonl` with `--name`) — one record per case: `{request_id, response, error_type, error, duration_ms, endpoint}`. `error_type` is `null` on success, else `timeout_retries_exhausted`, `service_error` or `aborted` (never sent because the run was cancelled).
- 📈 **`output/<run>_stats.jsonl`** — one flat line per capability plus a `micro` line: `{run, capability, n, accuracy, soft_accuracy, nll, brier, confidence, ece_15, score_mae, score_within_one, errors, error_rate, latency_ms_mean, latency_ms_p50, latency_ms_p95}` — lines from different models can simply be concatenated and compared.
- 🔬 **`output/<run>_cases.jsonl`** — one line per case: `{case_id: "typed-decisions-bench-v1:<family>:<line>", family, qtype, correct, confidence, nll, brier, soft_accuracy, prediction_label, gold_label, latency_ms[, score_error]}`.
- 🗂️ **`output/<run>/`** — single-run projection via the evalcompare registry: `metrics.csv` (registry column order, identical schema for every run) + `metrics.json` (micro aggregate + metadata).
- 🆚 **`output/comparison/`** — cross-run pivot (capability × run): `capability_<metric>.csv`, `capability_n.csv`, `delta_vs_<baseline>_<metric>.csv`, `capability_spread_<metric>.csv`, `model_summary.csv`, `all_rows.csv`, `summary.json` and a plotly `report.html` dashboard (heatmap, deltas, discriminators, latency trade-off).
- 🌡️ **`output/<run>_calibration.json`** — temperature calibration artifact fitted from this run's `calibrate` split (always schema v2: `temperatures` map + global `temperature` fallback; written unless `--no-calibration`) — format and loader contract in [`CALIBRATION.md`](CALIBRATION.md).

Per-metric definitions are in the [metrics definitions](#-metrics-definitions) above; only successful requests count toward latency. Because predictions are the response payloads embedded in the log, runs recorded with `--responses none` (like `make e2e`) score too, and partial/aborted runs produce `n=0` null-metric lines instead of failing.

**🔁 Resuming:** reusing a run name resumes that run — cases with a successful record in the existing log are skipped and only pending ones (failed, timed out, aborted, never reached) are (re)sent; new records are appended to the same log, the responses files are merged, and the `_stats`/`_cases` reports are refreshed. Recovered cases don't count as errors — the last record per case wins.

## 🛡️ Rate limiting & reliability

Three deliberate layers keep a sustained benchmark run from tripping endpoint quotas or masking outages:

1. **Pacer** (`--parallel`): request starts are spaced exactly `1/rps` apart under one lock, so the dispatch rate is bounded regardless of worker count.
2. **Quota buster** (`--quota-buster`): after every ~25 ±5 dispatched requests all senders pause for the configured wait ±10 %; both numbers re-roll per cycle so the traffic pattern stays unrecognizable to burst heuristics. `0` disables it.
3. **Circuit breaker**: if the first 10 requests all fail, the run aborts with a clear message — an unresponsive endpoint should fail fast, not consume hours of retries. Queued tasks get `aborted` records so line alignment and resume stay intact.

`make validate` schema-checks all benchmark data and the latest e2e log; `make sync-metadata --fix` reconciles the index/hash/stats/receipt files with disk (`SHA256SUMS.txt` is verifiable via `shasum -c SHA256SUMS.txt`).

## 🔒 Gated data (gpqa_diamond)

The GPQA Diamond files are stored encrypted at rest — each of `requests/gpqa_diamond.jsonl`, `responses/gpqa_diamond.jsonl` and `metadata/gpqa_diamond.jsonl` exists only as a password-protected `<name>.zip` sibling (`gpqa_diamond.jsonl.zip`), never as plaintext on disk. The zip password is a public constant in [`scripts/gpqa_zip.py`](scripts/gpqa_zip.py) (overridable via `GPQA_ZIP_PASSWORD`): the lock keeps plaintext out of checkouts and context windows, it is not a secrecy boundary.

Every consumer decrypts transparently: `scripts/run_eval.py` (eval/e2e), `scripts/validate.py`, `scripts/score.py` and `scripts/sync-metadata.py` extract the files they need at startup and remove the plaintext copies when the process ends — files a run rewrote are re-encrypted into the zip first, so nothing is lost. The archives use classic PKZIP ZipCrypto (readable by the stdlib and `unzip -P` without third-party dependencies).

```bash
make gpqa-status   # show locked/unlocked state
make gpqa-lock     # encrypt *.jsonl -> *.jsonl.zip, remove plaintext
make gpqa-unlock   # decrypt for manual maintenance (remember to lock again)
```

The zip interop + lifecycle self-checks are part of `make test` (`tests/test_gpqa_zip.py`).

## 🏗️ Generation & status

Gold is deterministically derivable (arithmetic, state machines, rubrics, supplied evidence) — see [`METHODOLOGY.md`](METHODOLOGY.md). Boolean→`noul`, category/ranking→`choice`, ordinal→`score`. Requests are globally unique with varied option order and label-free context.

Current build (see `quality_report.json` / `validation_status.json`): OpenAPI + semantic validation **PASS**; golden self-score **1.000** micro accuracy; **0** duplicate request bodies; every generated suite ≥ 2 distinct gold outcomes (max single-outcome share 86%); preserved suite files: 20 / 30 byte-identical (details in `preservation_receipt.json`).

## 🧭 Interpretation

Cases are synthetic and parameterized, not IID production samples — prefer paired comparisons on identical cases, regression detection, per-suite error analysis, and invariance tests over absolute IID confidence intervals. Intentionally separate from model training corpora.

## 📦 Repository layout

| Path | Contents |
|---|---|
| [`requests/`](requests/) | one `<capability>.jsonl` per suite — literal request payloads (model placeholder inside) |
| [`responses/`](responses/) | one `<capability>.jsonl` per suite — **golden reference responses: the scoring answer key** (line-aligned with requests; read-only — the runner refuses to write here) |
| [`metadata/`](metadata/) | gold labels + per-case provenance + explicit `split: train|calibrate|test` (line-aligned) |
| [`manifest.json`](manifest.json) | capability → file paths (the runner's source of truth for suite selection) |
| [`models.json`](models.json) | per-run model registry (model name, VRAM @ 8k KV, license, max context, image support, pareto flag, kyr0-project flag, inference repo) — deployment columns and `kyr0/` org branding in the comparison report |
| *(no local dir)* | the SystemOne OpenAPI spec is **not vendored** — `make validate` fetches the canonical live spec from <https://api.typesafe.ai/openapi.json> (offline: `--openapi path/to/spec.json`) |
| [`scripts/`](scripts/) | [`run_eval.py`](scripts/run_eval.py) (benchmark runner — full CLI reference below), plus `validate.py`, `score.py`, `calibration.py`, `metrics.py`, `sync-metadata.py`, `assign_splits.py`, `gpqa_zip.py` |
| [`src/evalcompare/`](src/evalcompare/) | the `evalcompare` library package — `loader.py` (stats JSONL → DataFrames), `metrics.py`, `analysis.py`, `report.py` |
| [`tests/`](tests/) | pytest suite: evalcompare library, scorer CLI, runner (naming/resume/rate limiting vs a stub endpoint), gpqa zip lifecycle — run via `make test` |
| `output/` | run logs + generated reports (created on first run) |

## 📎 Citation

If you use this benchmark, its split contract, or the Qtype-Stratified Temperature Scaling standard (`calibration.json`), please like this repository and cite it in your work. The preferred citation format is BibTeX:

```bibtex
@software{homberg_typed_decision_bench,
  author = {Homberg, Aron},
  title  = {typed-decision-bench: A Held-Out Benchmark And Qtype-Stratified Temperature Scaling Method For Language Models Turned Into Typed Decision Engines},
  year   = {2026},
  version = {5},
  url    = {https://github.com/kyr0/typed-decision-bench},
  license = {MIT}
}
```

Machine-readable: [`CITATION.cff`](CITATION.cff) (GitHub renders it as "Cite this repository").

## 📝 License

This repository is licensed under the MIT License. See [`LICENSE`](LICENSE) for details. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for third-party notices (GPQA Diamond, OpenAPI spec, etc.).