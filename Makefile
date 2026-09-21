.PHONY: sync-metadata validate eval e2e score metrics compare test gpqa-lock gpqa-unlock gpqa-status

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

# One case per capability (all 275) against the endpoint configured via .env;
# parallel at the runner default of 18 req/s (override via ARGS, e.g. ARGS="--parallel 10")
e2e:
	uv run scripts/run_eval.py --benchmark . --responses none --n 1 --timeout 120 --log output/e2e.jsonl $(ARGS)

# Backfill: score every output/*.jsonl run log that has no <run>_stats.jsonl/
# _cases.jsonl pair yet (eval/e2e runs score themselves automatically)
score:
	uv run scripts/score.py --benchmark . --all-logs

# Per-run metrics: every output/<run>_stats.jsonl -> output/<run>/metrics.csv +
# metrics.json (single-run projection through the evalcompare metric registry;
# no baseline, so no comparison report)
metrics:
	uv run scripts/metrics.py --benchmark .

# Per-capability comparison of all scored runs vs an explicit baseline, e.g.
# make compare ARGS="--baseline three"   (optional: --metric accuracy);
# writes output/comparison/ (matrix/deltas/spread CSVs + report.html)
compare:
	uv run scripts/metrics.py --benchmark . --compare $(ARGS)

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
