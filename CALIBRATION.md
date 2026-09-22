# Explicit benchmark splits and probability calibration

`typed-decision-bench` persists split membership in the already line-aligned
`metadata/*.jsonl` files. The System One request/response payloads stay pure
wire objects and therefore remain directly POSTable/schema-valid.

Each metadata row must contain exactly one:

```json
{"split":"test"}
{"split":"calibrate"}
{"split":"train"}
```

## Semantics

| split | model training | calibration fit | published benchmark metrics |
|---|---:|---:|---:|
| `train` | allowed externally | no | no |
| `calibrate` | no | **yes** | no |
| `test` | no | no | **yes** |

The benchmark itself creates no `train` rows by default. `train` exists in the
contract so externally fine-tuned checkpoints can stay honest: a model trained
on `train` rows is still legitimately evaluated on `test`, because those rows
were never in its gradient — and so future datasets stay explicit without
inventing another field.

## One-time migration

Run once after applying this implementation:

```bash
make assign-splits
```

This runs `scripts/assign_splits.py --fix`, then refreshes derived hashes/indexes
and validates the repository. For each suite, the migration deterministically
ranks rows by SHA-256 and assigns approximately 20% (exactly 20/100 for normal
100-case suites, 40/158 for GPQA's 198) to `calibrate`, with the remainder
`test`. It changes metadata only and is idempotent; runtime tools never
recompute or infer membership afterward. The script is report-only unless
`--fix` is passed, refuses partial existing assignments, and preserves an
already-complete assignment (`--force` replaces it intentionally).

## One benchmark run

Normal `make eval` sends `test,calibrate` by default and skips `train`:

```text
request rows
├─ split=test       -> inference -> published *_stats.jsonl / *_cases.jsonl
├─ split=calibrate  -> inference -> fit temperatures (schema-v2 artifact)
└─ split=train      -> not sent by default
```

Override collection explicitly when needed:

```bash
make eval ARGS="--splits test"
make eval ARGS="--splits calibrate"
make eval ARGS="--splits train,test,calibrate"
```

`--n` applies **after** split filtering. Source line numbers are preserved in
request IDs, so every prediction rejoins its exact metadata/gold row.

## Qtype-affine temperature calibration

Every run (unless `--no-calibration`) fits and writes
`output/<run>_calibration.json` from the **`calibrate` split only**.

### The method, step by step

1. **The problem: raw confidences are claims, not promises.** A System One
   probability is the next-token softmax over the answer labels. The softmax is
   an excellent *ranking* device but a badly behaved *probability*: sharper
   models saturate it toward 1, so a system claiming `0.93` may in fact be right
   ~78 % of the time. Calibration asks the purely empirical question — *when you
   say 90 %, how often are you actually right?* — and rescales the claim into a
   promise the observed data supports.
2. **The transform is affine in logit space** — it scales log-probabilities
   without shifting them (no intercept term):

$$
q_i \;=\; \frac{p_i^{\,1/T}}{\sum_j p_j^{\,1/T}} \;\equiv\; \mathrm{softmax}\!\left(\frac{\log p}{T}\right)_i
$$

   Raising probabilities to $1/T$ takes the odds between any two options to the
   power $1/T$; renormalizing restores the sum to 1. $T>1$ flattens an
   over-confident model, $T<1$ sharpens an under-confident one, $T=1$ is the
   identity. **The argmax — and the whole ordering — never changes**: the model
   keeps making the same decisions, only the confidence numbers move.
3. **Worked example.** An `noul` claim of $p(\text{true})=0.9$ is odds $9:1$.
   With $T=2$ the odds are square-rooted: $9:1 \to 3:1$, i.e. $q(\text{true})=0.75$.
   If the calibrate split showed the system right ~75 % of the time it claims
   90 %, that $T$ converts an inflated claim into an honest one.
4. **One $T$ per answer type (`choice` / `noul` / `score`).** Answer type — not
   capability, not domain — is the axis along which miscalibration actually
   varies, because each System One output type is a structurally different
   decision task: `noul` is one binary claim (a 2-point distribution with free
   support), `choice` compares an arbitrary candidate set (a shared max-logit
   slope is the natural degree of freedom), `score` spreads mass over an ordered
   scale (expected-level errors matter, not just the argmax level). Types below
   the data floor keep the global $T$.

   **Why not a single global $T$:** opposing miscalibration cancels in the
   pooled fit. Measured on the local bonsai deployment (259 calibrate / 1,116
   test cases): `noul` was *under*-confident (gap −0.058) while `score` was
   *over*-confident (gap +0.107); the pooled micro gap was −0.009 and the fitted
   global T=1.0005 — a provable no-op. Per-qtype fitting recovered the real
   headroom on the held-out test split: `noul` ECE-15 **0.070 → 0.043** and NLL
   **0.247 → 0.218** with T=0.653. Aggregate ECE therefore cannot be the
   acceptance criterion; the per-qtype metrics in the artifact
   (`analysis.test_by_qtype`) are.
5. **The fit is a one-parameter maximum-likelihood search** over the calibrate
   split (cross-entropy objective, next section). The objective is convex on the
   $\beta = 1/T$ scale, so plain gradient bisection (80 iterations, bracketing
   $T \in [0.05, 20]$, candidate set including $T=1$) finds the unique global
   optimum — no seeds, no tuning, no local minima.
6. **The gate is separate from the fit.** `deployable: true` requires ≥
   `--calibration-min-cases` successful calibrate cases (global), exactly one
   model and one endpoint in the run, and a finite in-bounds $T$. Fitted ≠
   deployable, and the test split never participates in either (see contract).

### Benefits

- **Zero-weight-touch, engine-agnostic.** The artifact is one number per answer
  type plus metadata; applying it is one line of serve-time code
  (`temperatures.get(qtype, temperature)`). Any compliant engine — C++, Rust,
  Python, a proxy — can certify the same model without touching its weights.
- **Decisions untouched.** Because the transform is strictly monotone per
  distribution, accuracy and all ranking-based metrics are mathematically
  unchanged — this can only fix confidence, never break a decision. (Compare
  fine-tuning, which can regress capability it was never asked to change.)
- **Optimal within its family, by construction.** Cross-entropy is a proper
  scoring rule and the objective is convex, so $\hat T$ is the unique MLE — there
  is a precise, falsifiable sense in which no other temperature is better.
- **Auditable.** The artifact embeds provenance (model, endpoint, run), split
  sizes, SHA-256 digests of both case-ID sets, and held-out test evidence, so a
  reviewer can verify disjointness and deployability conditions without the logs.
- **Self-verifying in production.** Temperature scaling has a fixed point
  ($\hat T \approx 1$ on fresh served data): deployment correctness is testable
  with the bench alone, no access to internals required
  ([workflow](#fixed-point-verification-workflow)).
- **Cheap.** One benchmark run per (model, endpoint, prompt-template) triple;
  serve-time cost is a power and a normalization per answer — orders of
  magnitude below a retry of the same question.

### Limits

- **One slope per type.** The family can only *rotate* a reliability curve
  around $(0, 0)$–$(1, 1)$ — it fixes uniformly over- or under-confident
  systems. A **crossing** curve (under-confident at low confidence,
  over-confident at high, or vice versa) needs a non-affine map (isotonic
  regression, vector/instance-based methods); the residual after fitting shows
  up as ECE that is flat in $T$ and is reported honestly in `metrics`/`analysis`.
- **Not a capability fix.** Calibration changes what the system *claims*, never
  what it *chooses*: `accuracy` and `soft_accuracy` of the artifact's own metrics
  are identical raw vs. calibrated for the argmax. A confidently wrong model
  becomes an honestly wrong model.
- **It is the *artifact's* deployment that is calibrated.** $\hat T$ is fitted on
  one endpoint with one model name, one prompt template, and deterministic
  (temperature-0) label scoring. A different endpoint (different quantization,
  sampling temperature, template or adapter) is a different distribution —
  refit for it (`deployable` deliberately encodes single provenance for exactly
  this reason; the fixed-point refit catches double-application, missing types
  and drift).
- **Data floor.** Groups below `--qtype-min-cases` (default 100 successful
  calibrate cases) fall back to the global $T$ and are reported under
  `qtype_calibration.groups` with `temperature: null` — the artifact states
  exactly how much data the next run needs. The guard is empirical, not
  theoretical: an 18-case `score` fit produced a wrong-direction T=0.93
  (sharpening an already over-confident group). Yields scale at ~20 % of sent
  rows: `--n 30` clears all three floors (score ≈ 131), a full run gives score
  ≈ 443.
- **Optimal in average log-loss, not for every consumer.** The MLE minimizes
  expected NLL over the whole calibrate corpus; a downstream use that only acts
  on `noul > 0.9` cares about a different functional, and per-bin reliability
  (right tail in particular) can remain imperfect even at the global optimum.

### Why the fit is statistically sound (and when it isn't)

The estimator is a one-parameter maximum-likelihood fit of the cross-entropy
risk over the calibration corpus ($n$ = successful `calibrate` cases of that
group, $\beta \equiv 1/T$):

$$
\hat T \;=\; \arg\min_T\; \frac{1}{n}\sum_{i=1}^{n} \mathrm{CE}\!\left(y_i,\; p_i^{1/T}/\textstyle\sum_j p_{ij}^{1/T}\right)
$$

Two theorems make this sound — *given a large enough corpus*:

- **Unique global optimum.** On the $\beta$ scale the objective is convex: its
  gradient is an expectation difference and its curvature is a variance,

$$
\frac{\partial\,\mathrm{CE}}{\partial\beta} = \mathbb{E}_{q_\beta}[\log p] - \mathbb{E}_{y}[\log p], \qquad \frac{\partial^2\,\mathrm{CE}}{\partial\beta^2} = \mathrm{Var}_{q_\beta}[\log p] \;\ge\; 0
$$

  so there are no local minima — which is why the fitter is plain bisection on
  the gradient. For `noul` the transform is exactly Platt scaling with tied
  slope ($q = \sigma(\mathrm{logit}/T)$), i.e. one-parameter logistic MLE; and
  cross-entropy is a *proper scoring rule*, so minimizing it maximizes the
  expected log-quality of the distribution itself, not a proxy.
- **Precision scales as $1/\sqrt{n}$.** As an M-estimator,
  $\sqrt{n}(\hat T - T^*) \to \mathcal{N}(0,\, 1/I(T^*))$ with $I$ the Fisher
  information per case, hence $\mathrm{SE}(\hat T) \propto 1/\sqrt{n}$ —
  quadrupling the corpus only halves the noise. This is the arithmetic of the
  ≥ 100-case floor: roughly where $\mathrm{SE}(\hat T)$ drops below the smallest
  correction worth making ($|T^* - 1| \gtrsim 0.15$). The same `score` group at
  443 cases sits within $\approx \pm 0.03$ of 1.

**Why `calibrate` must be an independent split.** If $\hat T$ were fitted on
the cases it is then scored on, the reported improvement would be structurally
optimistic: any fit exploits sample noise, and for log-loss the expected
in-sample optimism is $k/n$ nats ($k$ = fitted parameters). The deeper damage is
to the *decision*: `deployable` and the raw-vs-calibrated comparison are
evaluations, and choosing $T$ on evaluation data is selection on the outcome —
the "held-out" number then estimates nothing about deployment. With disjoint
splits the chain is clean: $\hat T(\text{calibrate})$ is statistically
independent of the `test` metrics, so they are an unbiased estimate of deployed
behavior — and the fixed-point test (see below) is a genuine hypothesis test
instead of an echo of the fit. Disjointness is enforced line-by-line (exactly
one `split` per metadata row) and *cryptographically checkable*: the artifact
stores `calibration_case_ids_sha256` and `test_case_ids_sha256` separately.

The 80/20 ratio is the same $1/\sqrt{n}$ arithmetic applied to the *report*
side: published metric error shrinks as $1/\sqrt{n_{\text{test}}}$, so `test`
keeps 80 % (≈22 k cases → published ECE/NLL precise to a few $10^{-3}$) while
the ≈5.5 k-case `calibrate` pool drives per-group $\mathrm{SE}(\hat T)$ to
$\approx 0.02$ — precise enough that a refit landing materially away from 1
means the server isn't applying the artifact, not that the estimator was noisy.

## Artifact contract

`schema_version` is **`2`**.
`temperatures` is always present (possibly an object without entries); `scope` describes the mode:

| scope | `temperatures` | applies |
|---|---|---|
| `per_qtype` | non-empty per fitted group | `temperatures[question.type]`, falling back to top-level `temperature` for unknown/small groups |
| `global` | `{}` | top-level `temperature` for every question (nothing earned its own T yet) |

Loader logic (one line — empty map or missing type both degrade to the global T):

```python
T = artifact.get("temperatures", {}).get(qtype, artifact["temperature"])
```

Loaders must validate `schema_version == 2`; v1 artifacts (no `temperatures`
key) are no longer produced and are not part of the contract.

**Only `split=calibrate` affects $T$ or `deployable`** (`fit.uses_test_data=false`
is recorded); `test` rows produce the held-out `raw` vs `calibrated` evidence in
`metrics` / `analysis` but never gate deployment; `train` rows are ignored.

An artifact is `deployable: true` when:

1. ≥ `--calibration-min-cases` (default 100) successful `calibrate` cases exist —
   note this is a **global** count while `--n` caps **per suite**: since ~20 % of
   rows are `calibrate`, `make eval ARGS="--n 3"` (~820 sent rows) already clears
   it, while `--n 1` (~50) needs `--min-cases 50`;
2. the run has exactly one model and one endpoint (single provenance);
3. the fitted temperature is finite and within the search bounds (the optimizer
   always includes $T=1$).

## Standalone fitting

```bash
make calibrate ARGS="--log output/my-run.jsonl"
```

Re-fits from any existing run log; the artifact is written next to it as
`<run>_calibration.json` (override with `--out`). Same contract as the
post-run artifact, including the fixed-point property below.

## Fixed-point verification workflow

Temperature scaling has a verifiable fixed point: apply $T$, serve, then refit
on fresh data — every group's new $T$ must return to ≈ 1. After loading the
artifact server-side, run a fresh pass under a **distinct `--name`** (reusing
the name would resume and send nothing) and check:

```bash
python3 -c "import json; a=json.load(open('output/<run>_calibration.json')); print(a['temperatures'])"
```

All group temperatures ≈ 1 (± ~0.05 at n≈500) ⇒ the applied temperatures are
confirmed optimal and that artifact is the production one. A group materially
≠ 1 ⇒ its temperature is not applied server-side, or the deployment drifted.
