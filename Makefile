# every target below shells out to `uv`; the installer puts it in ~/.local/bin
# (or ~/.cargo/bin on older installs), so make sure non-login shells find it
export PATH := $(HOME)/.local/bin:$(HOME)/.cargo/bin:$(PATH)

.PHONY: setup assign-splits sync-metadata validate eval eval-only bench e2e score calibrate metrics compare test gpqa-lock gpqa-unlock gpqa-status

# One-command bootstrap for a fresh clone: verify the toolchain (installs uv
# when missing), create .env from the template, create output/, pre-warm every
# dependency cache the targets below need (PEP 723 envs + the pytest suite),
# check the gated GPQA archives, validate all benchmark data and print a dry-run
# send plan. Idempotent — re-run it any time something feels broken.
setup:
	@set -e; \
	echo '[1/6] toolchain…'; \
	if ! command -v uv >/dev/null 2>&1; then \
		echo '  uv not found — installing from https://astral.sh/uv/install.sh'; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		command -v uv >/dev/null 2>&1 || { echo '  error: uv not on PATH after install (installer puts it in ~/.local/bin) — open a new shell or extend PATH, then re-run make setup'; exit 1; }; \
	fi; \
	echo "  $$(uv --version)"; \
	echo '[2/6] configuration…'; \
	if [ ! -f .env ]; then cp .env.example .env; echo '  created .env from .env.example — now edit it (TYPESAFE_MODEL / TYPESAFE_API_KEY / TYPESAFE_BASE_URL)'; else echo '  .env already exists (CLI flag > process env > .env)'; fi; \
	mkdir -p output; \
	echo '[3/6] dependency caches (PEP 723 scripts + pytest suite)…'; \
	uv run scripts/run_eval.py --help >/dev/null; \
	uv run scripts/validate.py --help >/dev/null; \
	uv run --with pytest --with pandas --with numpy --with plotly pytest --version >/dev/null; \
	echo '  warmed'; \
	echo '[4/6] gated GPQA archives…'; \
	uv run scripts/gpqa_zip.py status; \
	echo '[5/6] benchmark validation (needs network: fetches the live OpenAPI spec)…'; \
	uv run scripts/validate.py > validation_status.json; \
	echo '  PASS (document written to validation_status.json)'; \
	echo '[6/6] send plan (dry-run, nothing sent)…'; \
	uv run scripts/run_eval.py --benchmark . --dry-run --n 1 >/dev/null; \
	echo 'setup complete — smoke run next: make eval ARGS="--n 2 --capabilities gpqa_diamond,contains_pii"'

# ONE-TIME DATA MIGRATION: persist explicit metadata split membership, then
# refresh every derived checksum/index and validate the resulting benchmark.
assign-splits:
	uv run scripts/assign_splits.py --benchmark . --fix
	uv run scripts/sync-metadata.py --fix
	uv run scripts/validate.py --benchmark . > validation_status.json

# Syncs capability_index.json, suite_hashes.json, manifest.json, stats.json,
# golden_selftest.json, preservation_receipt.json, quality_report.json with disk
sync-metadata:
	uv run scripts/sync-metadata.py --fix

# Regenerates validation_status.json (uv auto-provides jsonschema via PEP 723 script metadata)
validate:
	uv run scripts/validate.py > validation_status.json

# Runs the benchmark, e.g. make eval ARGS="--capabilities gpqa_diamond --n 2"
# (per-capability response files stay off by default — the run log embeds them;
# the golden responses/ dir is never written to)
eval:
	uv run scripts/run_eval.py --benchmark . $(ARGS)

# Same run but the post-run output/comparison/ report is skipped (log, test-split
# stats, metrics export and calibration artifacts are still produced); e.g. when
# an auto-baseline comparison would be misleading or costs more than it tells you
eval-only:
	uv run scripts/run_eval.py --benchmark . --no-compare $(ARGS)

# Real benchmark run against the endpoint configured in .env. The run name
# defaults to <TYPESAFE_MODEL>[-N] (the name doubles as the resume key: re-run
# the same target to (re)send only pending cases). N caps cases per suite;
# omit it for all 27,598, e.g. `make bench N=20` -> bonsai-2-27b-20.
-include .env
N ?=
NAME ?= $(if $(TYPESAFE_MODEL),$(TYPESAFE_MODEL),bench)$(if $(N),-$(N))
# reference run for `make compare` (override: make compare BASELINE=<run>);
# jev is the hosted reference deployment, so it doubles as the default baseline
BASELINE ?=jev-1.13.0
bench:
	uv run scripts/run_eval.py --benchmark . $(if $(N),--n $(N)) --name "$(NAME)" $(ARGS)

# One case per capability (all 275) against the endpoint configured via .env;
# parallel at the runner default of 4 req/s (override via ARGS, e.g. ARGS="--parallel 10")
e2e:
	uv run scripts/run_eval.py --benchmark . --responses none --splits test --n 1 --timeout 120 --no-calibration --log output/e2e.jsonl $(ARGS)

# Backfill: score every output/*.jsonl run log that has no <run>_stats.jsonl/
# _cases.jsonl pair yet (eval/e2e runs score themselves automatically)
score:
	uv run scripts/score.py --benchmark . --all-logs

# Fit a deployable scalar temperature from persisted split=calibrate cases in an
# existing run. split=test is held out and reported only; split=train is ignored.
calibrate:
	uv run scripts/calibration.py --benchmark . $(ARGS)

# Per-run metrics: every output/<run>_stats.jsonl -> output/<run>/metrics.csv +
# metrics.json (single-run projection through the evalcompare metric registry;
# no baseline, so no comparison report)
metrics:
	uv run scripts/metrics.py --benchmark .

# Per-capability comparison of ALL scored runs into output/comparison/
# (matrix/deltas/spread CSVs + report.html). Default baseline is jev-1.13.0;
# override with BASELINE=<run> (or pass --baseline inside ARGS), metric with
# ARGS="--metric accuracy".
compare:
ifeq ($(findstring --baseline,$(ARGS)),)
	uv run scripts/metrics.py --benchmark . --compare --baseline $(BASELINE) $(ARGS)
else
	uv run scripts/metrics.py --benchmark . --compare $(ARGS)
endif

# Pytest suite in tests/: evalcompare library (src/evalcompare/), the scorer,
# the runner (naming/resume/rate limiting, stub endpoint) and gpqa zip lifecycle
test:
	uv run --with pytest --with pandas --with numpy --with plotly pytest tests -q

# Encrypt the gated gpqa_diamond.jsonl files into <file>.jsonl.zip (requests/,
# responses/, metadata/) and remove the plaintext copies; eval/validate/score
# decrypt them transparently for the duration of their run
gpqa-lock:
	uv run scripts/gpqa_zip.py lock

gpqa-unlock:
	uv run scripts/gpqa_zip.py unlock

gpqa-status:
	uv run scripts/gpqa_zip.py status
