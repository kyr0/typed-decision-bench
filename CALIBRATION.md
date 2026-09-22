# Probability Calibration for Typed Decisions

This document specifies the calibration protocol used by `typed-decision-bench`,
explains the mathematics behind it, and separates four kinds of statements:

- **PROVED** — follows mathematically from the transform or from an implementation invariant.
- **EMPIRICALLY SUPPORTED** — supported by held-out measurements or established calibration literature.
- **EXPECTED** — a reasonable prior expectation that must still be tested on the target deployment.
- **NOT CLAIMED** — stronger conclusions that the method does not establish.

The goal is practical: an engineer should be able to understand what the calibration
artifact does, why it is mathematically well behaved, when it is likely to generalize,
how to implement it correctly in an inference server, and what evidence is required
before making stronger claims about a deployment.

---

## TL;DR

`typed-decision-bench` uses **post-hoc temperature scaling**:

$$
q_i(T)=\frac{p_i^{1/T}}{\sum_j p_j^{1/T}}
$$

where:

- $p_i$ is the model's raw probability for answer $i$,
- $q_i$ is the calibrated probability,
- $T>0$ is fitted on a dedicated `calibrate` split,
- published benchmark metrics use a disjoint `test` split.

The benchmark can fit separate temperatures for `choice`, `noul`, and `score`
questions when enough calibration examples are available.

The method does **not** retrain the model and does **not** change its weights.

For every finite $T>0$, the transform preserves candidate ordering:

$$
p_i > p_j \iff q_i > q_j.
$$

Therefore the predicted answer is unchanged; only the probability distribution
around that answer changes.

The fitting objective is mean cross-entropy / negative log-likelihood (NLL).
When parameterized by inverse temperature

$$
\beta=\frac1T,
$$

that objective is convex. Consequently, the implementation can find a global
minimum with deterministic one-dimensional gradient bisection rather than a
general-purpose stochastic optimizer.

Temperature scaling is not an experimental idea invented for this benchmark.
It is an established post-hoc calibration method. Guo et al. (ICML 2017) found
that a single fitted temperature was surprisingly effective across most of the
classification datasets and neural-network architectures they studied.

That historical evidence, combined with the very low capacity of this calibrator
(one scalar per sufficiently represented question type), gives a reasonable
prior expectation that a temperature fitted on a large calibration split will
generalize to a **similar held-out distribution from the same deployment**.

That expectation is deliberately testable rather than assumed:

```text
calibrate split
    ↓
fit T
    ↓
freeze calibration.json
    ↓
test split
    ↓
measure held-out NLL / Brier / top-label ECE
```

What does **not** follow is that one temperature remains optimal under arbitrary
distribution shift. Ovadia et al. (NeurIPS 2019) showed that uncertainty and
post-hoc calibration can degrade as the evaluation distribution moves away from
the calibration distribution. A materially different model, quantization,
prompt template, adapter, endpoint, or production distribution should therefore
be treated as a new calibration target.

---

## Empirical reference result: Bonsai-2-27B

The method has now been exercised end to end on a real typed-decision deployment,
not only on synthetic calibration-unit tests.

Reference deployment:

- model: `bonsai-2-27b`
- System One endpoint: `/v1/systemone`
- frozen calibration artifact: schema v2, `per_qtype`
- server implementation: `Bonsai-Llama-Jev`, commit
  [`c805487`](https://github.com/kyr0/Bonsai-Llama-Jev/commit/c8054874473399fa32868257a0f7d05985a863ea)
- benchmark invocation: all 275 suites with `--n 100`
- total requests: **27,500**
- calibration rows: **5,499**
- held-out test rows: **22,001**

The run is slightly smaller than the repository's complete 27,598-row corpus
because `--n 100` caps GPQA Diamond at 100 of its 198 stored rows.

The frozen temperatures were:

| parameter | fitted value | interpretation |
|---|---:|---|
| pooled/global fallback | 1.011720 | close to identity |
| `choice` | 1.037240 | very mild flattening |
| `noul` | 0.765169 | substantial sharpening |
| `score` | 1.162158 | moderate flattening |

All three qtype fits were interior solutions, not search-bound results.

This is useful evidence for qtype stratification: the pooled scalar is close to
1 even though `noul` and `score` require corrections in opposite directions.
A single pooled slope cannot represent both effects simultaneously.

### Held-out generalization

The temperatures were fitted on `split=calibrate` only. Applying that frozen
artifact offline to the 22,001 disjoint `split=test` examples produced:

| metric | raw test | calibrated test | relative change |
|---|---:|---:|---:|
| NLL | 0.438015 | **0.434996** | **-0.689%** |
| Brier | 0.235513 | **0.233939** | **-0.668%** |
| hard accuracy | 0.832962 | 0.832962 | exactly unchanged offline |
| top-label ECE-15 | **0.011998** | 0.014646 | +22.1% |

The two proper full-distribution scores improved on rows that were never used
to fit the temperatures. This is direct empirical evidence that the fitted
correction generalized from the calibration partition to the held-out benchmark
partition for this deployment.

The ECE result is deliberately shown rather than hidden: top-label ECE did not
improve globally. Temperature scaling was fitted to NLL, and ECE is a binned
top-label diagnostic rather than the optimized full-distribution objective.

### Held-out result by qtype

| qtype | T | test NLL raw → calibrated | test Brier raw → calibrated | ECE-15 raw → calibrated |
|---|---:|---:|---:|---:|
| `choice` | 1.037240 | 0.488071 → **0.487981** | 0.251921 → **0.251900** | 0.015138 → **0.013787** |
| `noul` | 0.765169 | 0.277873 → **0.267841** | 0.169659 → **0.165212** | 0.046100 → **0.025605** |
| `score` | 1.162158 | 0.555333 → **0.551416** | 0.317909 → **0.313074** | **0.068788** → 0.073269 |

`noul` is the clearest calibration win: held-out NLL fell by about **3.61%**,
Brier by about **2.62%**, and top-label ECE by about **44.5%**.

`score` demonstrates why proper scores and ECE must not be conflated: its
full-distribution NLL and Brier improved while its binned top-label ECE became
slightly worse.

### Independent server-side reproduction

The frozen artifact was then loaded into the same inference server configuration
with the C++ calibration path enabled, and the benchmark was run again.

The served result almost exactly reproduced the offline prediction:

| metric | offline prediction from raw Run A | actual calibrated Run B | absolute difference |
|---|---:|---:|---:|
| NLL | 0.4349960051 | 0.4349837998 | 0.0000122052 |
| Brier | 0.2339390976 | 0.2339318835 | 0.0000072142 |
| soft accuracy | 0.7645820850 | 0.7645868428 | 0.0000047578 |
| mean confidence | 0.8319436708 | 0.8319458355 | 0.0000021647 |
| top-label ECE-15 | 0.0146457356 | 0.0150875306 | 0.0004417949 |

The larger relative difference in ECE is expected to be more sensitive than NLL
or Brier because fixed-width bin membership is discontinuous: a tiny probability
change can move an example across a bin boundary.

Hard accuracy differed by one test case between the two independent server runs
(0.832962 versus 0.833008). Positive temperature scaling cannot cause such an
argmax change for an identical raw vector; therefore this is evidence of a tiny
cross-run raw-inference difference near a decision boundary, not a property of
the calibration transform.

The C++ implementation itself also has a direct unit test that compares
server-returned calibrated probabilities against the analytic transform.

### Fixed-point reproduction

After Run B — whose probabilities were already calibrated by the frozen artifact
— the benchmark fitted another temperature artifact from the same calibration
rows.

The residual temperatures were:

| parameter | residual T | deviation from identity |
|---|---:|---:|
| pooled/global | 0.99987869 | -0.0121% |
| `choice` | 0.99991057 | -0.0089% |
| `noul` | 0.99959046 | -0.0410% |
| `score` | 0.99999629 | -0.00037% |

The additional residual fit changed held-out NLL only from

$$
0.4349837998
$$

to

$$
0.4349823236,
$$

a relative change of roughly **0.00034%**.

This is the expected temperature-composition fixed point in practice: after the
frozen qtype temperatures are applied, the best additional multiplicative
temperature correction is essentially identity.

Because the residual fit reuses the same benchmark calibration partition, this
is an **implementation consistency / fixed-point check**, not a second
independent generalization experiment.

Taken together, the reference experiment establishes four different facts:

1. the low-capacity temperatures fit successfully on the calibration partition;
2. proper probabilistic scores improve on the disjoint test partition;
3. the C++ deployment reproduces the offline-calibrated aggregate behavior to
   very small numerical error;
4. refitting the already calibrated endpoint returns essentially identity
   temperatures.

That combination is materially stronger evidence than calibration-set
improvement alone.

## 1. What calibration means

A model is not well calibrated merely because it is accurate.

Suppose a binary system emits confidence $0.80$ for 1,000 decisions. A common
notion of confidence calibration asks whether decisions reported near $0.80$
are actually correct approximately 80% of the time.

For multiclass predictions the full object is a probability vector,

$$
p=(p_1,\ldots,p_K),\qquad p_k\ge0,\qquad \sum_k p_k=1.
$$

Calibration of the complete probability distribution is stronger than calibration
of only the largest probability.

This distinction matters in this benchmark.

For a `choice` response such as

```json
{
  "probabilities": {
    "billing": 0.70,
    "technical": 0.20,
    "sales": 0.10
  }
}
```

the **top-label confidence** is

$$
\max_k p_k = 0.70.
$$

A top-label reliability diagram or ECE examines whether examples whose maximum
probability is approximately $0.70$ are correct approximately 70% of the time.

But the full distribution additionally claims that `technical` is twice as likely
as `sales`. Top-label ECE does not test that claim.

For this reason the benchmark reports multiple complementary metrics:

- **NLL / cross-entropy** — primary optimization objective; evaluates the full distribution.
- **Brier score** — another proper score for the full distribution.
- **top-label ECE** — useful reliability diagnostic for the winning probability.
- **hard accuracy** — evaluates only the chosen label.
- **soft accuracy** — descriptive probability/gold agreement, but **not** a proper scoring rule.

Useful introductions:

- Probability calibration: https://en.wikipedia.org/wiki/Calibration_%28statistics%29
- Probabilistic classification: https://en.wikipedia.org/wiki/Probabilistic_classification
- Scoring rules: https://en.wikipedia.org/wiki/Scoring_rule
- Brier score: https://en.wikipedia.org/wiki/Brier_score

For a more rigorous treatment of multiclass calibration evaluation, see
Vaicenavicius et al. (AISTATS 2019):

https://proceedings.mlr.press/v89/vaicenavicius19a.html

---

## 2. Why typed decisions need calibration

A System One typed-decision endpoint returns a structured decision rather than a
free-form generated answer. Typical answer types are:

- `choice` — a categorical probability distribution over named candidates,
- `noul` — a binary probability $p(\text{true})$,
- `score` — a categorical probability distribution over ordered score levels,
  from which an expected score is derived.

The inference engine usually obtains these values from model scores or logits.
A softmax converts scores $z_i$ into probabilities:

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}.
$$

Softmax is excellent at turning relative scores into an ordered probability
simplex, but the numerical scale of the logits determines how sharp the resulting
distribution is.

If all logit differences are too large, the model can be systematically
over-confident while still choosing the correct answers.

If all logit differences are too small, the model can be systematically
under-confident while still choosing the same answers.

Temperature scaling directly targets that degree of freedom.

Accessible softmax/temperature primer:

https://en.wikipedia.org/wiki/Softmax_function

---

## 3. Explicit `train` / `calibrate` / `test` split contract

Split membership is stored in the line-aligned `metadata/*.jsonl` files, not in
the System One wire payload.

Each metadata row contains exactly one:

```json
{"split":"train"}
{"split":"calibrate"}
{"split":"test"}
```

| split | may train model | may fit calibration | published benchmark metrics |
|---|---:|---:|---:|
| `train` | yes | no | no |
| `calibrate` | no | yes | no |
| `test` | no | no | yes |

The benchmark currently creates no `train` rows by default.

The one-time migration assigns approximately 20% of each suite to `calibrate`
and the remainder to `test` by deterministic SHA-256 rank. Runtime tools then
read the persisted field; they do not recompute split membership.

```bash
make assign-splits
```

The current repository validation reports:

```text
train:          0
calibrate:  5,520
test:      22,078
total:     27,598
```

For ordinary 100-case suites this normally means 20 calibration rows and 80 test
rows. GPQA Diamond has 198 rows and therefore receives 40 calibration rows and
158 test rows.

## What the split guarantees

**PROVED by repository structure:** the same row is not simultaneously assigned
to `calibrate` and `test`.

This prevents direct calibration/test row reuse.

## What the split does not guarantee

Disjoint rows are not automatically statistically independent.

Synthetic cases can share generators, templates, source families, semantic
structure, or latent difficulty factors.

`METHODOLOGY.md` already notes that templated synthetic cases can be correlated.

Therefore the scientifically correct interpretation is:

> The test split is held out from direct calibration fitting.

It is stronger than evaluating on the calibration rows themselves, but it is not
a proof that calibration and test observations are independent random variables,
nor that benchmark test performance is an unbiased estimate of every possible
production workload.

---

## 4. The temperature transform

For one probability vector $p$ and temperature $T>0$,

$$
\boxed{
q_i(T)=\frac{p_i^{1/T}}{\sum_j p_j^{1/T}}
}
$$

The same operation can be written

$$
q(T)=\operatorname{softmax}\left(\frac{\log p}{T}\right)
$$

when every $p_i>0$.

If the inference engine has the original logits, it is preferable to apply the
temperature there:

$$
\boxed{q(T)=\operatorname{softmax}(z/T)}
$$

because this avoids taking logarithms of rounded or underflowed probabilities.

If only probabilities are available, a numerically stable implementation can use:

```python
def temperature_scale(p, T, eps=1e-12):
    logp = [math.log(max(eps, x)) / T for x in p]
    m = max(logp)
    w = [math.exp(x - m) for x in logp]
    z = sum(w)
    return [x / z for x in w]
```

The probability-space and logit-space forms are mathematically identical for
strictly positive probabilities. With an implementation epsilon, zero-valued
serialized probabilities are approximated rather than exactly inverted.

---

## 5. Proof: probability-space scaling equals logit temperature scaling

Assume

$$
p_i=\frac{e^{z_i}}{Z},
\qquad
Z=\sum_j e^{z_j}.
$$

Then

$$
p_i^{1/T}
=
\left(\frac{e^{z_i}}{Z}\right)^{1/T}
=
\frac{e^{z_i/T}}{Z^{1/T}}.
$$

After normalizing across $i$,

$$
q_i
=
\frac{e^{z_i/T}/Z^{1/T}}
{\sum_j e^{z_j/T}/Z^{1/T}}
=
\frac{e^{z_i/T}}
{\sum_j e^{z_j/T}}.
$$

Therefore:

$$
\boxed{
\operatorname{normalize}(p^{1/T})
=
\operatorname{softmax}(z/T)
}
$$

for $p_i>0$.

**PROVED.**

---

## 6. Proof: pairwise odds are exponentiated by $1/T$

For any two candidates $i,j$,

$$
\frac{q_i}{q_j}
=
\frac{p_i^{1/T}}{p_j^{1/T}}
=
\left(\frac{p_i}{p_j}\right)^{1/T}.
$$

Therefore temperature scaling changes the *strength* of pairwise preference but
not its direction.

For $T>1$, $1/T<1$, so odds move toward $1:1$.

For $0<T<1$, $1/T>1$, so odds move farther away from $1:1$.

Hence:

- $T>1$ **flattens** the distribution,
- $T<1$ **sharpens** the distribution,
- $T=1$ is exactly the identity.

---

## 7. Proof: candidate ordering and argmax are unchanged

For $T>0$, the function

$$
f(x)=x^{1/T}
$$

is strictly increasing for $x>0$.

Therefore:

$$
p_i>p_j
\iff
p_i^{1/T}>p_j^{1/T}
\iff
q_i>q_j.
$$

So:

$$
\boxed{
\arg\max_i p_i
=
\arg\max_i q_i
}
$$

except for pre-existing exact ties, whose tie-breaking behavior is unchanged by
the ideal mathematical transform.

This gives an important invariant:

> Positive temperature scaling cannot change the winning candidate of one
> decision.

Consequently **hard argmax accuracy is mathematically unchanged**.

This guarantee applies to candidate ordering *within a decision*. It does not
mean every conceivable ranking metric across different examples is unchanged,
because confidence values can move by different amounts across examples.

---

## 8. Worked binary example: $0.90 \rightarrow 0.75$

Suppose a `noul` answer reports

$$
p(\text{true})=0.9,
\qquad
p(\text{false})=0.1.
$$

Its odds are

$$
\frac{0.9}{0.1}=9.
$$

Set $T=2$. Pairwise odds become:

$$
9^{1/2}=3.
$$

So the calibrated probabilities have odds $3:1$:

$$
q(\text{true})=\frac3{3+1}=0.75.
$$

Equivalently:

$$
q
=
\frac{(\sqrt{0.9},\sqrt{0.1})}
{\sqrt{0.9}+\sqrt{0.1}}
=
(0.75,0.25).
$$

The binary decision remains `true`, but its probability changes from $0.90$
to $0.75$.

In binary form temperature scaling is:

$$
q
=
\sigma\left(\frac{\operatorname{logit}(p)}{T}\right).
$$

Platt-scaling primer:

https://en.wikipedia.org/wiki/Platt_scaling

Temperature scaling is more restrictive than full Platt scaling: it uses one
positive scale parameter and no learned intercept.

---

## 9. Why fit separate temperatures by question type?

The benchmark can fit one temperature for each sufficiently represented answer
type:

```text
choice
noul
score
```

This is a **predeclared modeling choice**, not a theorem that answer type is the
only possible source of calibration heterogeneity.

Other plausible sources include number of candidates, domain, capability,
difficulty, prompt template, quantization, adapter, and model version.

The reason to begin with question type is engineering parsimony:

1. there are only three groups;
2. they correspond to structurally different API outputs;
3. each group can contain examples from many capabilities;
4. one scalar per group remains extremely low capacity;
5. the grouping is fixed before inspecting test results.

This is substantially less flexible than capability-by-capability calibration
and correspondingly harder to overfit.

The repository has historically called this **qtype-affine calibration**.
Strictly speaking, the transform used here is a **zero-intercept linear scaling
of logits** within each qtype:

$$
z \mapsto z/T.
$$

"Qtype-stratified temperature scaling" is the more precise mathematical name.

## Current schema-v2 fallback behavior

A qtype receives its own fitted $T$ only when it reaches the configured
minimum number of successful calibration examples.

Underrepresented qtypes use the top-level global temperature.

The current v2 implementation fits that global temperature on the pooled
calibration corpus. This gives a stable fallback estimate, but when some qtypes
also have dedicated temperatures it is **not mathematically the same as fitting
an optimum only on the remaining fallback subset**.

Accordingly, claims of exact within-family optimality apply to:

- each dedicated qtype fit on its own calibration examples, and
- the pooled global model considered by itself.

The pooled global fallback in a mixed artifact is a stability-oriented fallback,
not a theorem that it is the exact optimum for the residual small-qtype subset.

---

## 10. What is actually fitted?

For calibration example $r$, let:

- $p^{(r)}$ be the raw predicted distribution,
- $y^{(r)}$ be the gold distribution,
- $q^{(r)}(T)$ be the temperature-scaled distribution.

The empirical calibration objective is:

$$
\boxed{
L(T)
=
\frac1n
\sum_{r=1}^{n}
\operatorname{CE}
\left(
y^{(r)}, q^{(r)}(T)
\right)
}
$$

with:

$$
\operatorname{CE}(y,q)
=
-\sum_k y_k\log q_k.
$$

For the benchmark's deterministic one-hot gold labels this is ordinary mean
categorical negative log-likelihood.

For soft gold targets the same expression is empirical cross-entropy risk.

Cross-entropy primer:

https://en.wikipedia.org/wiki/Cross-entropy

Because the implementation supports soft target distributions, **empirical
cross-entropy minimization** is the most general description.

For current one-hot benchmark targets, the same objective coincides with
categorical maximum-likelihood estimation under the usual independent-case
conditional model.

---

## 11. Proof: the objective is convex in inverse temperature

Let:

$$
\beta=\frac1T
$$

and for one example define:

$$
a_k=\log p_k.
$$

Then:

$$
q_k(\beta)
=
\frac{e^{\beta a_k}}
{\sum_j e^{\beta a_j}}.
$$

The cross-entropy for that example is:

$$
L_r(\beta)
=
-\sum_k y_k\log q_k(\beta).
$$

Expanding:

$$
L_r(\beta)
=
\log\left(\sum_j e^{\beta a_j}\right)
-
\beta\sum_k y_k a_k.
$$

Differentiate:

$$
\frac{\partial L_r}{\partial\beta}
=
\frac{\sum_j a_j e^{\beta a_j}}
{\sum_j e^{\beta a_j}}
-
\sum_k y_k a_k.
$$

The first term is an expectation under $q_\beta$:

$$
\boxed{
\frac{\partial L_r}{\partial\beta}
=
\mathbb E_{q_\beta}[a]
-
\mathbb E_y[a]
}
$$

or equivalently:

$$
\boxed{
\frac{\partial L_r}{\partial\beta}
=
\mathbb E_{q_\beta}[\log p]
-
\mathbb E_y[\log p].
}
$$

Differentiate again:

$$
\boxed{
\frac{\partial^2 L_r}{\partial\beta^2}
=
\operatorname{Var}_{q_\beta}(a)
=
\operatorname{Var}_{q_\beta}(\log p)
\ge0.
}
$$

A mean of convex functions is convex, therefore the complete calibration
objective is convex in $\beta$.

**PROVED.**

This gives the implementation an important property:

> There are no non-global local minima in inverse-temperature space.

The derivative is monotone non-decreasing, so one-dimensional gradient bisection
is sufficient once a search interval is chosen.

---

## 12. Global optimum does not always mean unique optimum

Convexity guarantees that every local optimum is global.

It does **not** guarantee that the minimizer is always unique.

Consider a prediction that is uniform:

$$
p=(1/K,\ldots,1/K).
$$

Then every $\log p_k$ is equal, so:

$$
\operatorname{Var}_{q_\beta}(\log p)=0.
$$

The objective can therefore be flat in $\beta$.

The rigorous statement is:

> The objective has a global minimum. It is unique when the aggregate objective
> is strictly convex over the relevant interval; degenerate data can produce a
> flat set of minimizers.

The implementation searches within:

$$
T\in[0.05,20].
$$

A fitted value at one of those search bounds should be interpreted cautiously:
the unconstrained optimum may lie outside the supported interval, or the
objective may be insufficiently informative near that direction.

The artifact records whether the fitted temperature is at a search bound.

---

## 13. Why optimize NLL?

Negative log-likelihood / cross-entropy is a **proper scoring rule**.

Informally, a proper scoring rule is constructed so that, in expectation, a
forecaster minimizes its loss by reporting its actual predictive distribution
rather than strategically distorting it.

For formal background:

Tilmann Gneiting and Adrian E. Raftery,
**Strictly Proper Scoring Rules, Prediction, and Estimation**, JASA 2007:

https://doi.org/10.1198/016214506000001437

Accessible primer:

https://en.wikipedia.org/wiki/Scoring_rule

NLL is also the exact fitting objective, which makes held-out NLL the most
direct measure of whether the fitted transform generalized beyond calibration
rows.

---

## 14. Metrics: what each one does and does not establish

## 14.1 NLL / cross-entropy

$$
\operatorname{NLL}
=
-\frac1n
\sum_r\sum_k
y_k^{(r)}
\log p_k^{(r)}.
$$

Lower is better.

It evaluates the complete reported distribution and strongly penalizes assigning
very little probability to the true outcome.

**Use:** primary fitting objective and primary held-out calibration comparison.

## 14.2 Brier score

$$
\operatorname{Brier}
=
\frac1n
\sum_r\sum_k
\left(
p_k^{(r)}-y_k^{(r)}
\right)^2.
$$

Lower is better.

Brier score is also a proper scoring rule.

**Use:** complementary full-distribution evaluation.

Primer:

https://en.wikipedia.org/wiki/Brier_score

## 14.3 Top-label ECE

The implementation's ECE uses:

$$
c_r=\max_k p_k^{(r)}
$$

and:

$$
a_r=
\mathbf 1[
\arg\max_k p_k^{(r)}
=
\arg\max_k y_k^{(r)}
].
$$

Examples are placed into confidence bins $B_b$:

$$
\operatorname{ECE}
=
\sum_b
\frac{|B_b|}{n}
\left|
\operatorname{mean}_{r\in B_b}(c_r)
-
\operatorname{mean}_{r\in B_b}(a_r)
\right|.
$$

This is useful and intuitive, but evaluates only the winning confidence.

It does not fully test the multiclass distribution and is bin-dependent.

Vaicenavicius et al. discuss these subtleties:

https://proceedings.mlr.press/v89/vaicenavicius19a.html

ECE should therefore be treated as a **diagnostic**, not as the sole acceptance
criterion.

## 14.4 Hard accuracy

$$
\operatorname{Accuracy}
=
\frac1n
\sum_r
\mathbf 1[
\arg\max p^{(r)}
=
\arg\max y^{(r)}
].
$$

Positive temperature scaling preserves argmax, therefore:

$$
\boxed{
\operatorname{Accuracy}_{raw}
=
\operatorname{Accuracy}_{calibrated}
}
$$

up to implementation/tie-breaking bugs.

This is a useful invariant test.

## 14.5 Soft accuracy

The benchmark also reports:

$$
\operatorname{SoftAccuracy}
=
\frac1n
\sum_r
p^{(r)\top}y^{(r)}.
$$

This can be useful as a descriptive agreement measure.

It is **not invariant under temperature scaling**.

Example:

$$
y=(1,0),\qquad p=(0.9,0.1).
$$

Then:

$$
p^\top y=0.9.
$$

After $T=2$:

$$
q=(0.75,0.25)
$$

and:

$$
q^\top y=0.75.
$$

It is also not a proper scoring rule for general soft targets. For example:

$$
y=(0.6,0.4).
$$

Reporting $p=y$ gives:

$$
p^\top y=0.52,
$$

whereas reporting:

$$
p=(1,0)
$$

gives:

$$
p^\top y=0.60.
$$

So soft accuracy should not be used as the mathematical calibration objective.

---

## 15. What temperature scaling can fix

Temperature scaling is a one-dimensional shape correction.

It is well suited to a common failure mode:

> the model's ranking is useful, but the logit scale is systematically too
> sharp or too flat.

Suppose an ideal model would produce logits $z^*$, but the deployed model
approximately produces:

$$
z=\alpha z^* + c\mathbf 1.
$$

Softmax is invariant to adding the same constant $c$ to every candidate:

$$
\operatorname{softmax}(z)
=
\operatorname{softmax}(\alpha z^*).
$$

Choosing:

$$
T=\alpha
$$

gives:

$$
\operatorname{softmax}(z/T)
=
\operatorname{softmax}(z^*).
$$

In this idealized failure mode, temperature scaling exactly corrects the
probability distortion without changing the decision ordering.

This is one reason temperature scaling can generalize well when miscalibration
is dominated by a stable logit-scale effect.

---

## 16. What temperature scaling cannot fix

One scalar cannot express every calibration map.

Examples it cannot generally repair perfectly:

```text
low confidence  → under-confident
high confidence → over-confident
```

or:

```text
2-way choices   → calibrated
12-way choices  → over-confident
```

or:

```text
domain A → over-confident
domain B → under-confident
```

when all of those cases share the same fitted temperature.

A richer calibrator may be required if held-out diagnostics show structured
residual miscalibration.

Examples include isotonic regression, vector scaling, class-specific scaling,
beta calibration, and instance-conditioned calibration.

Kull, Silva Filho and Flach (AISTATS 2017) show why a richer parametric family
can outperform simpler logistic calibration when the simple shape is
misspecified:

https://proceedings.mlr.press/v54/kull17a.html

The engineering trade-off is capacity:

> richer calibrators can fit richer distortions, but they also require more
> calibration data and create more opportunities to overfit.

The benchmark intentionally begins with a very low-capacity family.

---

## 17. Why we expect the fitted temperature to generalize

This section is deliberately stronger than "it might work" but weaker than a
universal guarantee.

There are now two sources of support:

1. prior calibration literature gives a reasonable expectation of
   same-distribution transfer for low-capacity temperature scaling;
2. the Bonsai-2-27B reference experiment above directly observed
   `calibrate` → held-out `test` generalization on this benchmark.

## 17.1 Historical empirical evidence

Guo, Pleiss, Sun and Weinberger studied post-hoc calibration for modern neural
networks in ICML 2017.

Their practical conclusion was that temperature scaling — a single-parameter
variant of Platt scaling — was **surprisingly effective on most of the datasets
they evaluated**.

Paper:

https://proceedings.mlr.press/v70/guo17a.html

That result does not prove temperature scaling will work for every language model
or every typed-decision endpoint.

It does establish substantial historical empirical evidence that a
one-parameter temperature correction can generalize from a held-out calibration
set to a same-distribution test set.

## 17.2 Very low estimator capacity

A qtype temperature fit learns one scalar.

Even with three fitted question types, the deployed calibrator contains only a
handful of learned degrees of freedom.

It cannot memorize individual benchmark examples, capability names, prompts, or
labels.

It can only express statements like:

```text
choice distributions are systematically too sharp/flat by this scale
noul distributions are systematically too sharp/flat by this scale
score distributions are systematically too sharp/flat by this scale
```

That low capacity makes severe row-level overfitting much less plausible than
with a flexible nonlinear calibrator.

This is not a formal finite-sample guarantee, especially because benchmark cases
are correlated, but it is a strong practical reason to expect better
generalization behavior than a high-capacity post-hoc map with the same data
budget.

## 17.3 Broad calibration support

The current benchmark reserves 5,520 rows for calibration across hundreds of
capabilities.

A qtype temperature is therefore learned from many semantically different
tasks rather than one narrow classification problem.

If a similar fitted temperature improves held-out test NLL across heterogeneous
capabilities, that is evidence that the correction captures a deployment-wide
property rather than one task-specific accident.

## 17.4 The expectation is strongest under distributional continuity

The strongest expected transfer is:

```text
same model weights
same quantization
same inference code
same prompt template
same typed-decision readout
similar task distribution
calibrate rows → held-out test rows
```

Under those conditions, expecting temperature calibration to generalize is
reasonable.

The expectation becomes progressively weaker as those conditions change.

---

## 18. Three different meanings of 'generalization'

## Level A — calibration split → benchmark test split

Fit only on `split=calibrate`, freeze the artifact, and evaluate on
`split=test`.

If held-out test NLL and Brier improve, then the correction **generalized beyond
the rows used to fit it on this benchmark**.

This is the primary generalization claim the benchmark can directly establish.

**EXPECTED before measurement. EMPIRICALLY SUPPORTED after measurement.**

For the Bonsai-2-27B reference deployment this Level-A claim has now been
observed: held-out NLL and Brier both improved on 22,001 test examples after
fitting only on 5,499 calibration examples.

## Level B — benchmark → new but similar typed decisions

If test improvement is broad across many capabilities, it becomes plausible that
the same correction will help new typed decisions from a similar operational
distribution.

This is stronger than Level A and should remain an expected-transfer claim
unless tested on genuinely new cases.

## Level C — benchmark → shifted production traffic

This is not guaranteed.

Ovadia et al. studied predictive uncertainty under dataset shift and found that
traditional post-hoc calibration can deteriorate as the evaluation distribution
moves away from the calibration distribution:

https://papers.nips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html

Therefore:

$$
T_{\text{benchmark}}
\not\equiv
T_{\text{all future production distributions}}.
$$

Calibration should be treated as part of the exact inference configuration and
rechecked when that configuration or its traffic distribution changes
materially.

---

## 19. What held-out evidence should we require?

The fit itself minimizes calibration-set NLL, so calibration-set improvement is
not evidence of generalization.

The important evidence is held out.

For each qtype, inspect at least:

```text
test NLL:       raw → calibrated
test Brier:     raw → calibrated
top-label ECE:  raw → calibrated
accuracy:       raw = calibrated   [invariant check]
```

Also inspect reliability bins, capability-level breakdowns, candidate-count
breakdowns when relevant, raw-confidence distributions, and whether the fitted
temperature lies at a search bound.

A particularly convincing pattern is:

```text
calibrate NLL improves
test NLL improves
test Brier improves
test ECE improves or remains reasonable
accuracy is unchanged
improvement appears across many capabilities
```

ECE alone is not enough because it is binned and only tests top-label
confidence.


The Bonsai-2-27B reference experiment satisfies the central proper-score part of
this pattern: held-out NLL and Brier improve while hard accuracy is effectively
unchanged. It also provides a useful counterexample to treating ECE as the only
criterion: aggregate ECE worsens slightly even while both full-distribution
proper scores improve.

---

## 20. Statistical precision: what $1/\sqrt n$ does and does not mean

The fitted temperature is a one-dimensional empirical risk estimator.

Under standard regularity conditions, nondegenerate M-estimators often have
sampling error that decreases on the order of:

$$
O(n^{-1/2}).
$$

This gives the familiar rule:

> multiplying effective sample size by four roughly halves estimator noise.

But that scaling law does **not** justify a universal numeric confidence interval
from raw case count alone.

The benchmark cases are not guaranteed IID. Examples within one synthetic family
may be correlated, reducing effective sample size.

For a general scalar M-estimator $\theta$, the asymptotic variance has the
sandwich form:

$$
\operatorname{Var}
\left(
\sqrt n(\hat\theta-\theta^*)
\right)
\approx
\frac{J}{H^2},
$$

where:

- $H$ is expected objective curvature,
- $J$ is variance of the estimating equation / score.

Only under additional likelihood-model assumptions does this simplify to a pure
inverse-Fisher-information expression.

Therefore the repository's default per-qtype minimum is best understood as an
**engineering data floor**, not a theorem that 100 examples imply a particular
standard error.

---

## 21. Recommended uncertainty estimation: cluster bootstrap

If confidence intervals for $T$ are to be published, prefer a bootstrap that
respects known correlation structure.

A practical version is:

```text
choose clusters = suites or generator/template families

repeat B times:
    sample clusters with replacement
    collect their calibration examples
    fit T

report:
    median T
    2.5th percentile
    97.5th percentile
```

This is more defensible than pretending every templated row is independent.

For qtype-specific fits, bootstrap the relevant qtype examples while preserving
cluster membership.

---

## 22. Why calibration and evaluation must use different rows

Suppose the same data are used both to choose $T$ and to report improvement.

The parameter was explicitly selected to minimize loss on those data.
Consequently, its in-sample fitted loss is optimistically selected.

That is ordinary model-selection bias.

The exact amount of optimism depends on assumptions; it is not universally
$k/n$ outside regular correctly specified likelihood settings.

The benchmark therefore uses:

$$
\boxed{
\text{fit on calibrate}
\quad\rightarrow\quad
\text{report on test}
}
$$

This establishes row-level holdout:

> no test row directly contributes to the fitted temperature.

That is the claim we need. We do not need to claim that the partitions are
statistically independent in every latent sense.

---

## 23. Fixed-point composition theorem

Define:

$$
S_T(p)_i
=
\frac{p_i^{1/T}}
{\sum_jp_j^{1/T}}.
$$

Apply $S_T$, then another temperature $S_U$:

$$
S_U(S_T(p))_i
\propto
\left(p_i^{1/T}\right)^{1/U}
=
p_i^{1/(TU)}.
$$

Therefore:

$$
\boxed{
S_U\circ S_T
=
S_{TU}.
}
$$

**PROVED.**

Suppose $T^*$ is the interior optimum for a particular dataset and scalar
temperature family.

If the server applies $T^*$ exactly and the same dataset is fitted again,
then the residual optimum should be:

$$
U^*\approx1,
$$

subject to numerical tolerance and non-uniqueness/degeneracy.

---

## 24. What the fixed-point check proves — and what it does not

Running the same benchmark again after loading the artifact is useful for
detecting:

- calibration not being applied,
- calibration being applied twice,
- wrong qtype lookup,
- wrong artifact loaded,
- a changed server path,
- numerical implementation mismatch.

But if the second run uses the same benchmark rows, it is an **implementation
consistency check**, not a fresh statistical hypothesis test.

For stronger external evidence, perform the residual fit on genuinely new
labeled data from the same target distribution.


In the Bonsai-2-27B reference run, the already calibrated endpoint refitted to
`choice=0.99991057`, `noul=0.99959046`, and `score=0.99999629`. This is an
extremely close practical realization of the identity fixed point predicted by
the composition theorem.

---

## 25. Recommended before/after experiment

## Run A — raw endpoint

Run the selected benchmark without server-side calibration.

Collect raw predictions for:

```text
split=calibrate
split=test
```

Fit `calibration.json` from `calibrate` only.

Freeze that artifact.

## Offline held-out evaluation

Before changing the server, apply the frozen artifact offline to Run A's `test`
predictions.

This gives:

```text
raw test metrics
vs
offline-calibrated test metrics
```

Because `test` was not used to fit $T$, this is the cleanest direct evidence
of benchmark generalization.

## Run B — calibrated endpoint

Load exactly the frozen artifact into the server.

Keep constant, as far as practical:

- model weights,
- model file,
- quantization,
- server code,
- prompt template,
- adapters,
- inference settings.

Run the same benchmark under a distinct run name.

Then compare case by case:

$$
p^B
\stackrel{?}{\approx}
S_T(p^A).
$$

This checks that the production server implements the same transform the
benchmark fitted.

## Strong implementation invariant

For every test case:

```text
offline temperature_scale(raw_A, T)
≈
served probability_B
```

within a documented floating-point tolerance.

Also verify:

```text
argmax(raw_A) == argmax(probability_B)
```

unless an exact pre-existing tie is present.

---

## 26. What should be published for reproducibility?

Recommended layout:

```text
calibration/
├── calibration.json
├── before_stats.jsonl
├── before_cases.jsonl
├── after_stats.jsonl
├── after_cases.jsonl
├── before_calibration.json
├── after_calibration.json
└── provenance.json
```

Recommended provenance:

```json
{
  "model": "...",
  "model_sha256": "...",
  "quantization": "...",
  "server_commit": "...",
  "benchmark_commit": "...",
  "prompt_template_sha256": "...",
  "adapter_sha256": null,
  "calibration_sha256": "...",
  "before_run": "...",
  "after_run": "..."
}
```

A model name and endpoint URL alone do not uniquely identify quantization, model
bytes, prompt template, server implementation, or adapter state.

---

## 27. Correct inference-engine placement

Apply calibration to the **final candidate probability distribution**:

```text
prompt
  ↓
model forward pass
  ↓
candidate logits / scores
  ↓
construct final candidate distribution
  ↓
temperature scaling       ← HERE
  ↓
derive typed answer fields
```

This matters for engines that construct a large candidate set through several
conditional or multi-token readouts.

If the benchmark fitted a temperature to the final candidate probabilities,
applying temperature separately to intermediate conditional distributions is a
different operation.

> Compose the complete candidate distribution first, then apply its qtype
> temperature exactly once.

---

## 28. Recompute every derived answer field after calibration

## `choice`

Given calibrated probabilities $q_k$:

```text
probabilities = q
choice        = argmax(q)
```

The choice should equal the raw choice by the ordering proof.

Any API-specific derived confidence field should be recomputed from $q$, not
copied from the raw response.

## `score`

Given score levels $0,\ldots,K-1$:

```text
probabilities = q
score         = Σ k q_k
```

Temperature scaling preserves the most likely score level but can change the
expected numeric score because it changes mass assigned to non-winning levels.

Therefore `score` **must** be recomputed.

## `noul`

If the API stores only:

$$
p=P(\text{true}),
$$

reconstruct:

$$
(1-p,p),
$$

apply temperature scaling, then return the calibrated true probability.

---

## 29. Benchmark confidence vs API `confidence`

For calibration analysis, the benchmark derives top-label confidence from the
probability vector:

$$
c=\max_k p_k.
$$

That is the value used by top-label ECE.

An inference API may additionally expose a field named `confidence` with its own
legacy or compatibility semantics.

Those concepts should not be conflated.

The calibration artifact modifies the probability distribution. Any derived API
`confidence` field should subsequently be recomputed according to that API's
definition.

---

## 30. Artifact contract

The current artifact uses:

```json
{
  "schema_version": 2,
  "kind": "typed-decision-temperature-calibration",
  "method": "temperature_scaling",
  "scope": "per_qtype",
  "temperature": 1.0,
  "temperatures": {
    "choice": 1.0,
    "noul": 1.0,
    "score": 1.0
  }
}
```

Basic lookup:

```python
T = artifact.get("temperatures", {}).get(
    question_type,
    artifact["temperature"],
)
```

A loader should at minimum validate:

```text
schema_version == 2
kind == "typed-decision-temperature-calibration"
method == "temperature_scaling"
T is finite
T > 0
```

and validate deployment provenance when relevant identifiers are available.

---

## 31. What `deployable: true` means

`deployable` is a **mechanical eligibility flag**, not proof that the transform
improves every future dataset.

In the current implementation it primarily means:

- enough successful calibration examples were available,
- the run had a single model identity,
- the run had a single endpoint identity,
- a finite fitted temperature was produced.

The held-out `test` split does not control this flag.

That separation is intentional: test results should describe evidence, not feed
back into parameter selection or artifact eligibility.

A production policy may choose to be stricter, for example by rejecting
search-bound temperatures, missing exact model hashes, insufficient qtype data,
or unacceptable held-out degradation.

---

## 32. Search-bound temperatures

The implementation searches:

$$
T\in[0.05,20].
$$

If the optimum is returned at a boundary, the artifact records that fact.

A boundary result should be treated as a warning because:

1. the unconstrained optimum may lie outside the interval;
2. calibration data may not identify a useful interior optimum;
3. extreme temperatures can indicate a mismatch between the one-slope family
   and observed probabilities.

A rigorous deployment policy should inspect or reject at-bound fits.

---

## 33. Generalization under dataset shift

Calibration is a joint property of predictions and the distribution of outcomes
that materialize.

A temperature fitted for:

```text
model A
quantization Q
template P
traffic distribution D1
```

need not be optimal for:

```text
model A
quantization Q
template P
traffic distribution D2
```

Ovadia et al. provide large-scale evidence that uncertainty quality and post-hoc
calibration can degrade under increasing dataset shift:

https://papers.nips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html

The strongest defensible deployment statement is therefore:

> A temperature fitted on representative calibration data is expected to
> transfer best to future data drawn from a similar distribution and the same
> inference configuration.

That is a meaningful expectation, not a universal invariance claim.

---

## 34. What the Bonsai reference experiment now demonstrates

The reference experiment turns several previous expectations into observed
evidence for one exact deployment.

The frozen artifact was fitted on 5,499 calibration examples and evaluated on
22,001 disjoint test examples. Test NLL decreased from 0.438015 to 0.434996 and
Brier from 0.235513 to 0.233939 in the offline held-out evaluation.

The independently rerun calibrated server then produced test NLL 0.434984 and
Brier 0.233932 — within approximately $1.2\times10^{-5}$ and
$7.2\times10^{-6}$, respectively, of the offline prediction.

Finally, fitting another temperature layer to the already calibrated endpoint
returned all qtype temperatures within 0.041% of 1.

Therefore the following deployment-specific statement is supported:

> **On the held-out typed-decision-bench test distribution used in the
> 2026-09-22 Bonsai-2-27B reference experiment, qtype-stratified temperature
> scaling fitted only on the calibration split improved full-distribution NLL
> and Brier score, transferred to the independently rerun calibrated C++ server,
> and reached the expected near-identity residual-temperature fixed point.**

This is a substantial empirical result.

It still does not imply that the same temperatures are optimal for arbitrary
future production distributions or materially different inference
configurations.

## 35. Claim ladder

## PROVED

- $q_i\propto p_i^{1/T}$ equals logit temperature scaling for positive $p_i$.
- pairwise odds are raised to $1/T$.
- candidate ordering is preserved for every $T>0$.
- argmax is preserved.
- hard argmax accuracy is therefore unchanged.
- the cross-entropy objective is convex in $\beta=1/T$.
- there are no non-global local minima in $\beta$.
- temperature transforms compose as $S_U\circ S_T=S_{UT}$.

## EMPIRICALLY SUPPORTED BY PRIOR LITERATURE

- modern neural networks can be poorly calibrated;
- temperature scaling has historically been effective on many same-distribution
  classification benchmarks;
- richer calibrators can outperform simpler parametric families when the
  distortion shape is misspecified;
- calibration can deteriorate under dataset shift.

## EMPIRICALLY OBSERVED IN THE BONSAI-2-27B REFERENCE EXPERIMENT

For the exact reference deployment and benchmark distribution:

- temperatures fitted on 5,499 calibration rows improved held-out test NLL and
  Brier on 22,001 disjoint rows;
- `noul` showed the strongest transfer, including a large ECE reduction;
- `choice` was already close to calibrated and required only a mild correction;
- `score` improved NLL/Brier even though top-label ECE worsened, demonstrating
  why no single diagnostic should replace proper-score evaluation;
- the calibrated C++ run reproduced the offline predicted NLL/Brier to roughly
  $10^{-5}$ absolute error;
- a second fit on the already calibrated endpoint returned residual qtype
  temperatures essentially equal to 1.

These observations are deployment-specific empirical evidence, not algebraic
guarantees for every future model or traffic distribution.

## EXPECTED, THEN TESTED HERE

For a large calibration sample from the same typed-decision deployment:

- qtype temperatures are expected to generalize better than a highly flexible
  calibrator with the same data budget;
- held-out test NLL/Brier are likely to improve when dominant error is stable
  logit-scale distortion;
- a calibrated server should reproduce the benchmark's offline transform;
- a residual same-data fit should return near $1$ when the artifact is
  applied correctly and the optimum is well identified.

These are reasonable expectations, not algebraic guarantees.

## NOT CLAIMED

The method does not by itself establish that:

- every confidence bin is perfectly calibrated;
- top-label ECE captures full multiclass calibration;
- one qtype temperature is optimal for every domain or candidate count;
- calibration and test rows are statistically independent merely because they
  are disjoint;
- benchmark test performance is an unbiased estimate of arbitrary production traffic;
- fitted temperature remains optimal after distribution shift;
- temperature scaling improves model capability;
- temperature scaling repairs wrong discrete answers;
- a same-data fixed-point rerun is a formal statistical hypothesis test.

---

## 36. Relationship to established calibration methods

Temperature scaling is an established member of the post-hoc calibration family.

Guo et al. describe it as a one-parameter variant of Platt scaling.

Full Platt scaling fits a logistic map with slope and intercept.

Temperature scaling keeps only a positive scale:

$$
z\mapsto z/T.
$$

This restriction has useful consequences:

1. identity $T=1$ is naturally in the family;
2. candidate ordering is guaranteed unchanged.

The benchmark's contribution is therefore not invention of temperature scaling
itself.

The engineering contribution is the typed-decision protocol around it:

```text
explicit split contract
+ qtype stratification
+ portable calibration artifact
+ held-out evaluation
+ inference-engine loader contract
+ reproducibility/provenance workflow
```

Any stronger novelty claim should be based on a separate literature review.

---

## 37. References and further reading

## Primary scientific references

### Guo et al. — Temperature scaling for modern neural networks

Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger.
**On Calibration of Modern Neural Networks.**
ICML 2017, PMLR 70:1321–1330.

https://proceedings.mlr.press/v70/guo17a.html

Key relevance:

- documents miscalibration in modern neural networks;
- evaluates several post-hoc calibration methods;
- reports temperature scaling as surprisingly effective on most studied datasets;
- establishes temperature scaling as a standard practical baseline.

### Vaicenavicius et al. — What calibration evaluation actually means

Juozas Vaicenavicius, David Widmann, Carl Andersson, Fredrik Lindsten,
Jacob Roll, Thomas B. Schön.
**Evaluating model calibration in classification.**
AISTATS 2019, PMLR 89:3459–3467.

https://proceedings.mlr.press/v89/vaicenavicius19a.html

Key relevance:

- formalizes calibration evaluation;
- explains that multiclass calibration has multiple notions;
- motivates caution when interpreting confidence calibration alone.

### Ovadia et al. — Calibration under dataset shift

Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D. Sculley,
Sebastian Nowozin, Joshua V. Dillon, Balaji Lakshminarayanan, Jasper Snoek.
**Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty
Under Dataset Shift.**
NeurIPS 2019.

https://papers.nips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html

Key relevance:

- evaluates uncertainty under increasing dataset shift;
- shows why post-hoc calibration should not be assumed invariant under
  distribution change.

### Gneiting & Raftery — Proper scoring rules

Tilmann Gneiting, Adrian E. Raftery.
**Strictly Proper Scoring Rules, Prediction, and Estimation.**
Journal of the American Statistical Association, 2007.

https://doi.org/10.1198/016214506000001437

Key relevance:

- theoretical basis for evaluating probabilistic forecasts with proper scores;
- supports log score / NLL and Brier-type scores as principled probabilistic objectives.

### Kull, Silva Filho & Flach — Richer calibration maps

Meelis Kull, Telmo Silva Filho, Peter Flach.
**Beta calibration: a well-founded and easily implemented improvement on
logistic calibration for binary classifiers.**
AISTATS 2017, PMLR 54:623–631.

https://proceedings.mlr.press/v54/kull17a.html

Key relevance:

- demonstrates why simple sigmoid/logistic families can be misspecified;
- provides a richer alternative when one-slope calibration leaves systematic
  residual structure.

## Accessible engineering primers

- Calibration in statistics:
  https://en.wikipedia.org/wiki/Calibration_%28statistics%29
- Platt scaling:
  https://en.wikipedia.org/wiki/Platt_scaling
- Softmax:
  https://en.wikipedia.org/wiki/Softmax_function
- Cross-entropy:
  https://en.wikipedia.org/wiki/Cross-entropy
- Scoring rules:
  https://en.wikipedia.org/wiki/Scoring_rule
- Brier score:
  https://en.wikipedia.org/wiki/Brier_score

Wikipedia is listed as an accessible primer, not as the evidentiary basis for
the scientific claims above.

---

## 38. Practical interpretation

The most useful mental model is:

> Temperature scaling does not make a model smarter. It estimates how strongly
> to trust the probability contrast the model already produces.

When dominant calibration error is a stable scale distortion, a single
temperature can correct that distortion extremely efficiently.

Because the calibrator has very low capacity and historical experiments have
found temperature scaling effective across many same-distribution classification
settings, **good generalization from a large calibration split to a similar
held-out test distribution is a reasonable expectation**.

The benchmark is designed to measure whether that expectation is actually true
for each deployment.

For the Bonsai-2-27B reference deployment, the first complete before/after
experiment supports that expectation on the benchmark's held-out test
partition: proper scores improved, the served calibrated run reproduced the
offline prediction, and the residual temperature fit returned essentially
identity for all three qtypes.

The scientific claim should therefore be neither:

> "It is only a heuristic and we know almost nothing."

nor:

> "It guarantees honest probabilities everywhere."

The defensible middle is stronger and more useful:

> **Temperature scaling is a mathematically constrained, established post-hoc
> calibration method with strong historical empirical support. In this benchmark
> it is fitted only on explicit calibration rows and evaluated on held-out test
> rows. Because it learns only one scale parameter per sufficiently represented
> question type, it is expected to generalize well when calibration and deployment
> distributions are similar. That expectation is directly falsifiable through
> held-out NLL, Brier, top-label reliability, per-qtype analysis, and case-by-case
> verification of the deployed transform.**
