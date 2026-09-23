![typed-decision-bench logo](logo.png)

# 🧪 typed-decision-bench

> The eval bench (and novel `calibration.json` standard) that fixes the **overconfident decisions** problem in **any** open System One reproduction model - without touching a single model weight.

Looking for the current benchmark results? Check out [the published results](https://kyr0.github.io/typed-decision-bench/). They contain the base data in JSON and CSV as well, including links to every inference engine configuration for _intellectual honesty_ and _full trace provenance_.

**TL;DR:**
- 🎯 Evaluate **any** model and inference system that speaks the System One typed decisions API (`POST /v1/systemone`) — 275 capabilities, 27,598 held-out cases.
- 🌡️ Calibrate **any** model and inference System One inference system! **Calibration** changes what the system claims, never what it chooses. Any  **_confidently_ wrong model** becomes an **_honestly_ wrong model** via the new [**Qtype-Stratified Temperature Scaling method**](https://kyr0.github.io/typed-decision-bench/paper/) with **no re-training and no weight changes required**: Just run the `calibrate` split to fit a `calibration.json` (I propose this as a standard, see below); a compliant engine loading it provably states confidences that match its observed accuracy (see [`Bonsai-Llama-Jev`](https://github.com/kyr0/Bonsai-Llama-Jev) where I implemented this novel method).

## ✨ What's inside

- 📊 **275 suites / 27,598 cases** on an explicit `test`/`calibrate`/`train` split contract — details in the [Capability Matrix section](#capability-matrix-tested-under-this-benchmarks-regime) and [`METHODOLOGY.md`](METHODOLOGY.md)
- 🌡️ **Calibration standard** ([`CALIBRATION.md`](CALIBRATION.md)): run the bench, configure your inference engine to load the `calibration.json` produced by this bench. Internally, the inference engine must apply the temperature transform described below: `temperatures.get(question_type, artifact.temperature)` at serve time (that's extremely cheap!); and this fixes the **overconfident decisions** issues right away! It upgrades any model from "making typed decisions" to "making typed decisions and knowing how much to trust each decision". See my [`Bonsai-Llama-Jev`](https://github.com/kyr0/Bonsai-Llama-Jev) reference implementation. It's a just a few lines of code!
- 📝 **Line-aligned** `requests/<suite>.jsonl` ↔ `responses/<suite>.jsonl`; each line is a literal `SystemOneRequest` / `SystemOneResponse` with one atomic question; no gold leaks
- 🔤 **Question types:** __choice:__ 18,140 · __noul:__ 7,250 · __score:__ 2,208
- 🚫 **Zero-install tooling:** every script is self-contained Python 3.11+ with PEP 723 inline dependencies, run via `uv run`

## 📰 News 

2026-09-23 - vLLM's Semantic Router team [released interesting new Decision models - notably Lux-9B](https://huggingface.co/collections/llm-semantic-router/decision-10)! I'm on it.
2026-09-23 - Read my paper [on the math behind the Qtype-Stratified Temperature Scaling method for calibration](https://kyr0.github.io/typed-decision-bench/paper/) (works with every LM/BERT-backed model!).


## 💪 Other Benchmarks or "Why this bench exists"

I was working on `Bonsai-Llama-Jev`, `Spark-X2.5-Jev` and a small mmBERT Jev-like model, when I realized that I absolute cannot tell if the model generalizes well on a diverse set of decision tasks. Every single benchmark out there (that was released before 2026-09-22 and that I could find by deep research), focused on a narrow set of capabilities and had a low number of test cases (up to 20 per capability, and between 20 to 30 broad capabilities).

I'm not saying that these benchmarks are useless - they DO provide valuable first insights - but benchmarks with a narrow focus or a small number of test cases inside of a specific capability set, are not sufficient to evaluate generalization across a wide range of decision tasks. As many people in the 
community have already realized, Jev is such a great model and inference system, because it generalizes really well and because it is calibrated really well.
This is more valuable than some secret sauce or clever tricks in a model's architecture. 

I needed a benchmark that would evaluated my approaches _honestly_ and broadly. So I first deep researched capabilities (use-cases) and then created a synthetic dataset with GPT-6 Astra on Pro-level reasoning auto-regressively. I created one `JSONL` file with prepared request templates in `./requests/<capability>.jsonl` per capability. These requests directly match the OpenAPI specified System One API. Gold responses are frozen in `./responses/<capability>.jsonl`. Metadata is stored in `./metadata/<capability>.jsonl`. 

**Benchmark Alternatives:**

| Benchmark                                                                     | Actual evaluation data                                                                                                                                                                                                                                                                                                                                              |                                                          Scale | Best for                                                                                                                                |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------: | --------------------------------------------------------------------------------------------------------------------------------------- |
| **[kyr0/typed-decision-bench](https://github.com/kyr0/typed-decision-bench)** | [`requests/`](https://github.com/kyr0/typed-decision-bench/tree/main/requests) + [`responses/`](https://github.com/kyr0/typed-decision-bench/tree/main/responses) + [`metadata/`](https://github.com/kyr0/typed-decision-bench/tree/main/metadata) +  capability index |     **275 capabilities / 27,598 cases**, mostly 100/capability | **Broad applied generalization + task-specific capability lookup + calibration-aware Jev baseline comparison at a sane request budget** |
| **[Decision Index 0.1](https://github.com/apolinario/decision-index)**        | Frozen [`multimodalart/decision-index-suite`](https://huggingface.co/datasets/multimodalart/decision-index-suite), reconstructed from 37 upstream datasets                                                                                                                                                                                                          |                           **132,422 requests / 37 benchmarks** | **Broad standardized/academic capability measurement and externally sourced benchmark provenance**                                      |
| **[JevBench](https://github.com/fstandhartinger/jevbench)**                   | Public [`datasets/public/`](https://github.com/fstandhartinger/jevbench/tree/main/datasets/public) + sealed/private cases                                                                                                                                                                                                                                           |  **<1000 requests**; current benchmark includes sealed cases | **Public Jev ecosystem leaderboard: intelligence + calibration + latency + cost**                                                       |
| **[GaNotchVFX/jev-benchmarks](https://github.com/GaNotchVFX/jev-benchmarks)** | [Banking77](https://huggingface.co/datasets/mteb/banking77), [SNIPS](https://huggingface.co/datasets/benayas/snips), [Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full)                                                                                                                                                                      |                                                    3 workloads | **Concrete production-like routing/tagging scenarios with known external datasets**                                                     |
| **[AbdelStark/jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks)** | BTZSC variants of AG News, Banking77 and Emotion                                                                                                                                                                                                                                                                                                                    |                                      Small multi-dataset suite | **Probability quality, calibration/selective-risk style analysis on conventional classifiers**                                          |
| **[jev-eval-agent](https://github.com/vinilana/jev-eval-agent)**              | Six authored scenarios in [`evals/prompts.ts`](https://github.com/vinilana/jev-eval-agent/blob/main/evals/prompts.ts)                                                                                                                                                                                                                                               |                      6 multi-step tasks / 100-tool environment | **Agent tool selection/routing and measuring whether a typed-decision layer improves an agent**                                         |
| **[jev-measured](https://github.com/WallerChen/jev-measured)**                | Authored fixtures + small labelled support-ticket set                                                                                                                                                                                                                                                                                                               |                      8 application cases + 27 labelled tickets | **Operational characteristics: latency, cost, determinism, API behavior, multi-question calls**                                         |
| **[jev-phishing-bench](https://github.com/anisselbd/jev-phishing-bench)**     | [PhishNChips v5.2](https://huggingface.co/datasets/AreLit/PhishNChips)                                                                                                                                                                                                                                                                                              |                                               **2,000 emails** | **Deep domain-specific accuracy/calibration study with statistical controls**                                                           |



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



## Capability Matrix Tested Under This Benchmark's Regime

For every capability, 80 decisions are to be taken by the typed decision model. 20 more decisions are reserved to generate a probability distribution for Qtype-stratified calibration:

- **Code quality judgements:** (does documentation and code align? is a method too long given a specified max length? categorize cyclomatic complexity; does a test actually test the intended behavior? are important branches untested? what will the code return? will it produce the declared schema? does it call external functions? is it side-effect free? does it mutate state? arbitrary-code-execution potential? should this code be optimized or is optimization unlikely to matter? does it need explanatory comments? does it require a live datasource? browser/server/both? memory/CPU/GPU intensity; thread safety; blocking behavior; command destructiveness; estimated runtime class; programming-language identification; application-domain classification.)
- **Program-behavior prediction:** (given short code, predict output, exception, state mutation, filesystem/network effects, stdout/stderr, exit status, returned type, number of iterations, whether execution terminates; classic interview-style traps but also realistic application code.)
- **Code requirement satisfaction:** (given a natural-language task and candidate implementation, decide which requirements are actually implemented, partially implemented, contradicted, or absent; distinguish superficial keyword matches from behavioral compliance.)
- **Patch/diff correctness:** (given original code, requested change and patch, decide whether the patch actually solves the request; identify unrelated behavior changes, regressions, incomplete modifications, stale tests, obsolete branches.)
- **Test quality and adequacy:** (does the test exercise the behavior named by its title? can it pass despite a broken implementation? does it test implementation details rather than contract? missing boundary conditions? duplicated tests? property/invariant that should be tested? score semantic coverage independently from line coverage.)
- **API-contract compatibility:** (does client request match server contract? path/method/header/content-type mismatch; required vs optional fields; return shape; pagination contract; sync/async mismatch; REST/OpenAPI/GraphQL/RPC examples; backwards compatibility.)
- **Dependency/API existence judgements:** (does the referenced method plausibly belong to the named library/version? candidate calls from documentation vs fabricated APIs; deprecated methods; package-name confusion; stdlib vs third-party dependency; import-path compatibility.)
- **Version compatibility:** (given dependency/runtime versions and feature usage, decide compatible/incompatible/unknown; Python/Node/Rust/Java versions, CUDA/runtime pairs, database versions, browser features, serialization formats.)
- **Environment/portability judgements:** (browser-only, Node-only, POSIX-only, Windows-specific, GPU-required, CUDA-specific, root-required, interactive-terminal-dependent, filesystem-dependent, locale-dependent, architecture-dependent, isomorphic.)
- **Concurrency correctness:** (race conditions, shared mutable state, atomicity assumptions, lock ordering, possible deadlock, async misuse, reentrancy, thread safety, process safety, duplicate-job execution.)
- **Idempotency judgements:** (HTTP retries, payment processing, migration scripts, queue consumers, file transformations: can repeated execution alter the outcome incorrectly? distinguish read-only, naturally idempotent, explicitly idempotent and non-idempotent operations.)
- **Error-handling quality:** (exception swallowed? retry useful or harmful? retryable vs permanent failure; cleanup guaranteed? fallback masks corruption? error loses useful context? timeout required? failure leaves inconsistent state?)
- **Algorithm/data-structure appropriateness:** (list vs hash map vs tree vs heap; full sort vs top-k; linear scan vs indexed lookup; algorithm asymptotically inappropriate? optimization justified by expected input sizes?)
- **Complexity estimation:** (categorize time/memory complexity; distinguish worst/average/amortized; identify input variable controlling complexity; detect accidental quadratic behavior; score practical rather than merely asymptotic cost.)
- **Performance-bottleneck attribution:** (given code plus timings, identify CPU, I/O, memory bandwidth, network, lock contention, serialization, database, accelerator or startup bottleneck; decide whether proposed optimization affects the bottleneck.)
- **Parallelization suitability:** (is the task embarrassingly parallel? state synchronization needed? batchable? vectorizable? GPU-friendly? multiprocessing useful? parallelism likely slower because workload is tiny?)
- **Database-query reasoning:** (SQL result semantics, incorrect joins, duplicate rows, missing predicates, NULL semantics, aggregation correctness, N+1 queries, index usefulness, transaction isolation assumptions.)
- **Regex/pattern correctness:** (does regex accept/reject supplied examples? catastrophic-backtracking potential; anchored vs substring matching; Unicode assumptions; overly broad/overly strict validation.)
- **State-machine correctness:** (given states/events/transitions, is a transition legal? unreachable states? missing terminal state? event sequence impossible? transition introduces invariant violation?)
- **Serialization compatibility:** (JSON/JSONL/YAML/CSV/Protobuf-like representations; numeric/string coercion, enum incompatibilities, absent/null distinction, precision loss, ordering assumptions.)
- **Data-migration correctness:** (old/new schema plus transformation: is information lost? reversible? duplicate creation possible? migration safe to retry? values mapped to wrong enum/unit?)
- **Distributed-systems consistency:** (event order, stale replicas, at-least-once delivery, duplicate messages, leader changes, eventual consistency; decide whether observed states are possible and whether invariants survive.)
- **Cache correctness:** (stale cache, incorrect keying, tenant leakage, invalidation mistakes, TTL assumptions, cacheable vs non-cacheable responses, derived-data dependency changes.)
- **Observability adequacy:** (given system behavior and logs/metrics/traces, decide whether evidence can distinguish likely failure modes; identify missing correlation IDs, timings, error context, cardinality problems.)
- **Log/event-sequence diagnosis:** (given textual logs, decide most plausible execution path, component that failed first, causal vs downstream errors, whether retries recovered, whether timestamps violate expected ordering.)
- **Configuration correctness:** (`Dockerfile`, Compose, Kubernetes, CI, package config, compiler config, environment variables represented as text; incompatible options, wrong paths, missing required keys, overridden values.)
- **Command-line semantics:** (what does this pipeline actually do? destructive/non-destructive; local/network effects; blocking? requires stdin? can glob expand dangerously? expected output class; approximate runtime category.)
- **Open-source/license compatibility:** (given licenses and distribution arrangement, classify likely compatibility/obligation category from supplied license rules; notice/source-distribution requirements; distinguish linking, copying and mere tool usage.)
- **Accessibility semantics from source:** (HTML/ARIA/textual component representation: correct label association? keyboard reachable? heading hierarchy? interactive element role? duplicate labels? accessibility claim unsupported?)
- **Graph logic:** (given reasoning/causal graph, judge whether edges are supported, merely correlated, impossible, circular, redundant or hallucinated; determine whether conclusion follows from reachable evidence.)
- **Graph knowledge:** (given concept/entity graph, judge whether relation types are correct: `is-a`, `part-of`, `causes`, `located-in`, `authored-by`, `depends-on`; score semantic closeness of candidate edges.)
- **Ontology/taxonomy consistency:** (dog→mammal valid, mammal→dog too specific; instance-vs-class mistakes; disjoint categories; transitive relationships; parent category chosen at wrong abstraction level.)
- **Graph-path relevance:** (given query and a knowledge graph, rank possible paths by how directly they explain/connect the concepts; reject paths connected only through generic hubs.)
- **Temporal graph consistency:** (edge valid only before/after date X; employment, ownership, dependency/version, historical relationships; detect relations impossible simultaneously.)
- **Causal-chain validation:** (A preceded B but did not necessarily cause B; intervention evidence vs correlation; mediators/confounders; identify unsupported causal step in multi-hop explanation.)
- **Root-cause attribution:** (given symptoms/events and dependency graph, rank plausible root causes separately from downstream manifestations; include cases with insufficient evidence or multiple equally compatible causes.)
- **Evidence provenance:** (given claims and source snippets, map each claim to supporting evidence; unsupported, directly supported, inferred, contradicted, or sourced from the wrong document.)
- **Source reliability in-context:** (given explicit metadata such as primary record, repost, anonymous comment, official specification, outdated manual, decide which source is suitable for which claim without relying on source prestige alone.)
- **Semantic similarity and factual correctness:** (does summary preserve source meaning? contradiction vs paraphrase; over-generalization; omitted qualifier; invented detail; unsupported certainty; incorrect entity/date/quantity.)
- **Summary completeness:** (does summary cover all materially distinct topics? which topic is absent? does it spend disproportionate space on minor detail? distinguish concise-but-complete from lossy.)
- **Claim entailment:** (source says “may reduce latency”; candidate says “reduces latency”: unsupported certainty. Source says 8–12%; candidate says 10%: approximation vs fabrication. Multi-sentence entailment and contradiction.)
- **Unknown/known-state classification:** (given evidence set, distinguish true, false, likely, possible and genuinely unknown rather than guessing a binary answer.)
- **Answerability from provided context:** (question may have a real-world answer but cannot be answered from supplied state; distinguish “unknown to the model state” from “factually unknowable” and “explicitly contradicted.”)
- **Evidence sufficiency:** (is there enough evidence to make a requested decision at confidence threshold X? decide what additional evidence would discriminate between remaining alternatives.)
- **Fact-vs-inference classification:** (identify statements explicitly present, deductively derivable, probabilistically inferred, merely plausible, or unsupported.)
- **Hallucination detection:** (candidate response introduces person, date, dependency, number, cause, quotation, API, event or feature absent from source; include subtle composite hallucinations formed by combining true fragments.)
- **Topic/keyword fidelity:** (do keywords describe the source? overly broad/narrow terms; missing dominant topic; term present only incidentally; rank keywords by representativeness.)
- **Information-retrieval relevance:** (query plus 20–100 passages/items; rank relevance; distinguish exact lexical overlap from semantic relevance; IT/API/code/CSS/HTML/legal/scientific/general domains.)
- **Recommendation/reranking quality:** (query, explicit preferences/constraints, candidate products/documents/actions/items; order by fit; hard constraint violation must dominate superficial similarity; ties and insufficient-information cases.)
- **Entity resolution:** (are “International Business Machines”, “IBM Corp.” and “IBM” same entity in this context? distinguish same-name different entities using address/date/domain/account metadata.)
- **Record deduplication:** (decide exact duplicate, near duplicate, updated record, related-but-distinct, or ambiguous using noisy text fields.)
- **Cross-document reconciliation:** (invoice vs purchase order vs delivery note; resume vs application form; API docs vs implementation; identify matching, missing, contradictory and outdated facts.)
- **Document consistency:** (invoice subtotal/tax/total agreement; stated address vs country code; account owner vs invoice party; dates in plausible order; internal references resolve.)
- **Booking/accounting mapping:** (bank-statement descriptions → rent, payroll, cloud hosting, office equipment, tax, fee, refund, transfer, subscription etc.; debit/credit implications; recurring vs exceptional; business/private/unknown.)
- **Accounting reasonableness:** (is booking category compatible with description? VAT treatment stated in supplied rules? asset vs expense under supplied threshold? candidate mapping clearly impossible or merely uncertain?)
- **Invoice anomaly detection:** (implausible totals, duplicate positions, unusual quantity×price, inconsistent tax, dates, currencies, IBAN/country/address mismatch, changed payment destination.)
- **Fraud/scam likelihood:** (emails, invoices, letters, account-change requests, fake support messages; German phone but unrelated foreign banking coordinates, domain typo, impossible chronology, urgency combined with credential/payment request; score individual evidence rather than “odd = fraud.”)
- **Transaction anomaly detection:** (usual transaction pattern supplied in state; detect unusual amount, merchant, geography, frequency, timing or counterparty while tolerating legitimate one-off changes.)
- **Commercial-price plausibility:** (given supplied catalogue/reference prices, judge invoice/order positions as plausible, unusually high/low, incompatible unit or quantity, or not comparable.)
- **Budget consistency:** (proposed spend vs budget categories, recurring costs, commitments and reserve; identify oversubscription, double-counting or nominally cheap option causing downstream cost.)
- **Resource planning:** (staff, compute, storage, inventory, time, materials, advertising budget; what is required, optional, excessive or missing; quantities under explicit workload assumptions.)
- **Capacity planning:** (requests/s, workers, memory, storage growth, production capacity, hotel seats, warehouse space; choose sufficient capacity tier and detect impossible utilization assumptions.)
- **Inventory decisions:** (demand forecasts, lead times, spoilage, safety stock; reorder/hold/clearance decisions; distinguish shortage risk from overstock risk.)
- **Scheduling feasibility:** (jobs, dependencies, workers, durations, calendars; is schedule possible? earliest feasible slot? conflicting resources? missed prerequisite?)
- **Prioritization under constraints:** (rank tasks using explicit impact, urgency, dependency, cost and deadline; reject “urgent-looking” low-value work when criteria disagree.)
- **Pareto-optimal decision taking:** (multiple objectives such as latency/cost/accuracy/reliability; identify dominated options, Pareto frontier, and goal-compatible choices without pretending one global optimum exists.)
- **Expected-utility decisions:** (explicit outcomes, probabilities and utility/cost; select or rank choices; include loss aversion traps, tiny-probability/high-cost events and cases where expected value differs from expected utility.)
- **Bayesian evidence updating:** (prior plus diagnostic evidence/likelihood information; decide which hypothesis should gain/lose probability, rank posteriors, detect evidence that is actually non-informative.)
- **Game-theory judgements:** (dominant strategies, best response, coordination, prisoner’s dilemma, zero/non-zero-sum, repeated interaction, credible threats, signaling; probable outcome under explicitly stated player incentives.)
- **Adversarial-agent reasoning:** (another actor optimizes against you; identify exploitable deterministic strategy, deceptive signal, incentive incompatibility, commitment problem.)
- **Negotiation judgements:** (BATNA, reservation values, mutually beneficial ranges, concessions; distinguish agreement that dominates alternatives from one that only sounds balanced.)
- **Probability-support judgement:** (state gives probabilities/confidence/evidence; statement “it will probably be sunny in 3 days” must be evaluated against forecast values and uncertainty; distinguish 51%, 80%, 99% language.)
- **Probability calibration language:** (map “possible”, “likely”, “very likely”, “almost certain” to supplied quantitative conventions; identify text whose certainty overstates/understates evidence.)
- **Forecast consistency:** (multiple observations or forecasts through time; candidate prediction incompatible with current trajectory, confidence interval, seasonality or stated assumptions.)
- **Counterfactual plausibility:** (given causal rules/state, judge “if X had not happened, Y would still have happened”; distinguish counterfactual inference from simple temporal reversal.)
- **Base-rate awareness:** (rare event with noisy positive evidence; determine whether candidate conclusion ignores supplied prevalence/base rate.)
- **Sampling/statistical interpretation:** (does a conclusion generalize beyond described sample? selection bias, survivorship, confounding, tiny sample, inappropriate average, percentage-vs-percentage-point confusion.)
- **Unit understanding:** (compare quantities represented in different units; milliseconds vs seconds, MB vs GiB, km/h vs m/s, °C vs K when differences/absolute values matter, currencies with supplied conversion rates.)
- **Dimensional consistency:** (is an equation/result dimensionally possible? energy vs power, rate vs amount, area vs length, throughput vs latency; detect numerically plausible but unit-invalid answers.)
- **Order-of-magnitude reasoning:** (is result roughly 10, 1k, 1M or 1B? detect decimal/exponent/unit conversion mistakes without requiring exact arithmetic.)
- **Scale plausibility:** (is 200 GB plausible for a text file containing 500 records? 2 ms plausible for a transatlantic round trip? grounded in explicitly provided reference ranges where possible.)
- **Geometry and embodied navigation:** (textual 2D/3D spaces, orientation, coordinates, obstacles and goals; walk/drive/fly/jump/crawl constraints; choose feasible action and detect collision/dead end.)
- **Spatial-relation reasoning:** (left/right/front/behind/inside/intersecting/north/south; viewpoint transformations; object A left of B and observer rotates 180°; relative vs absolute orientation.)
- **Route feasibility:** (doors, one-way passages, bridge weight/height constraints, stairs vs wheelchair, vehicle turning radius, fuel/range; decide which route is actually traversable.)
- **Embodied affordances:** (can object be carried, climbed, entered, opened, driven through, used as support given stated dimensions/material/body capabilities?)
- **Physical commonsense:** (stack stability, containment, gravity, support, liquid flow, collision, friction, heating/cooling; plausible next physical state.)
- **Mechanical-system reasoning:** (gears, pulleys, valves, pumps, levers, simple electrical/mechanical state descriptions; direction/change propagation and failure-mode classification.)
- **Object permanence/state tracking:** (objects moved between locations/containers; after long distractor text identify where an object must, may or cannot be.)
- **Temporal ordering:** (events with explicit/relative dates, durations and dependencies; detect impossible chronology, calculate which events overlap, before/after relations.)
- **Calendar/time-zone understanding:** (appointments described across zones, daylight-saving offsets supplied or derivable from state, recurring schedules, “next Friday” relative to explicit reference date.)
- **Historical-fact verification:** (given date/country plus reference excerpts, decide whether stated ruler, border, institution, technology, currency or event is compatible with that time.)
- **Historical anachronism detection:** (objects, terminology, technologies, countries, institutions or social practices appearing before/after their described era.)
- **Geographical consistency:** (city-country-region relationships, distances/directions from supplied map-like facts, impossible travel sequence, address-country/currency/phone-code consistency.)
- **Weather-state interpretation:** (forecast table expressed textually; precipitation/cloud/wind/temperature state → supported natural-language judgement; distinguish “sunny”, “dry”, “warm”, “stormy”, “safe for activity X”.)
- **Scientific-hypothesis plausibility:** (observations vs hypotheses; which hypotheses are consistent, falsified or underdetermined? distinguish prediction from post-hoc explanation.)
- **Experimental-design judgement:** (does experiment test stated hypothesis? proper control? variable actually manipulated? confounder? measurement cannot distinguish candidate explanations?)
- **Measurement/uncertainty reasoning:** (significant figures, measurement interval, sensor error, uncertainty propagation qualitatively, whether claimed difference exceeds supplied uncertainty.)
- **Biological-system reasoning:** (text descriptions of cells, organisms, ecosystems, inheritance, physiology; identify plausible mechanisms, impossible category mixtures, cause/effect direction.)
- **Biological signal understanding:** (symptom/body-state narratives → pain location/type/intensity range, physical vs psychological vs mixed vs insufficient evidence; explicitly reward uncertainty rather than invented sensations.)
- **Medical differential-reasoning benchmarks:** (symptoms plus candidate ICD descriptions → score compatibility, contradiction and missing evidence; include superficially similar diagnoses, impossible combinations and insufficient-information cases; benchmark reasoning, not patient diagnosis.)
- **Medical chronology consistency:** (symptoms, medication, procedures and measurements over time; decide whether candidate explanation contradicts chronology or known state supplied in example.)
- **Translation correctness:** (source plus candidate translations; meaning preserved? polarity/tense/quantity/register changed? false friend? dropped clause? added information?)
- **Grammar judgement:** (locate erroneous token/span among \~20 words; agreement, tense, case, article, word order, punctuation; include grammatically valid but stylistically odd distractors.)
- **Glossary adherence:** (given N approved translations/technical terms, detect whether candidate translation uses required terms consistently; rank candidate glossary matches.)
- **Synonym/semantic-substitution scoring:** (which word/phrase can replace source expression in context without changing denotation, connotation, register or technical meaning?)
- **Word-sense disambiguation:** (“bank”, “port”, “model”, “class”, “cell”, “charge” etc.; determine intended sense from context and rank alternatives.)
- **Language/dialect identification:** (short/noisy/code-switched text; language family vs exact language; distinguish German/Dutch, Serbian/Croatian/Bosnian, Russian/Ukrainian etc.; permit uncertain/ambiguous.)
- **Code-switch understanding:** (sentence changes languages midstream; assign spans/language, preserve named entities/loan words, understand cross-language coreference.)
- **Pragmatic-intent understanding:** (“Could you open the window?” literal ability question vs request; indirect refusals, implied requests, soft commitments, rhetorical questions.)
- **Sarcasm and humor understanding:** (sarcasm, irony, understatement, absurdity, incongruity, callback, pun, garden-path joke; classify mechanism and whether literal interpretation differs from intended meaning.)
- **Humor explanation correctness:** (given joke plus candidate explanations, rank explanations; reject explanations inventing a pun or cultural reference absent from text.)
- **Mood/atmosphere understanding:** (cinematic/story descriptions → tense, calm, ominous, melancholic, joyful, uncanny etc.; distinguish environmental mood from character emotion.)
- **Tone/register classification:** (formal, intimate, hostile, deferential, clinical, promotional, passive-aggressive, uncertain; intensity score; detect mismatch with intended audience.)
- **Dialogue coherence:** (does response answer previous turn? wrong referent? repeats solved question? abrupt topic switch? ignores correction? assumes information never supplied?)
- **Conversation-state tracking:** (multi-turn text with changing choices/reservations/configuration; identify latest valid state rather than an obsolete earlier value.)
- **Persona consistency:** (persona explicitly specifies age, occupation, language, knowledge and experiences; candidate utterance contradicts biography, chronology or knowledge access.)
- **Theory-of-mind / false-belief reasoning:** (Alice saw object moved? Bob did not; what does each person believe? nested beliefs, deception, stale information, private communication.)
- **Embodied unknowns:** (what can this person/agent know given their location, sensors, messages, education and event exposure? reject omniscient answers.)
- **Embodied time-aware world views:** (persona, age, birthplace, biography, explicitly supplied beliefs, media exposure and historical time; judge which beliefs/reactions are plausible for that described individual without substituting stereotypes.)
- **Embodied emotions:** (baseline emotional/physical state plus new event → plausible next emotions; tooth pain + exhaustion + bad news vs rested state; allow multiple plausible reactions and reject deterministic overconfidence.)
- **Emotion-cause attribution:** (identify which described event most plausibly caused emotion vs merely co-occurred; distinguish disappointment, fear, embarrassment, anger, grief, relief, surprise.)
- **Social-norm reasoning:** (given explicit relationship/culture/context rules, classify action as expected, unusual, inappropriate or underdetermined; separate rule violation from personal dislike.)
- **Relationship/permission reasoning:** (manager/employee, parent/child, owner/guest, doctor/patient, service roles as explicitly defined: who can authorize/access/change what?)
- **Embodied animal understanding:** (species-specific state of dogs, cats, birds, fish, horses, rodents etc.; probable next action, stress/fear/feeding behavior, threat level, predator/prey relation, health concern vs normal behavior.)
- **Animal affordance/constraint reasoning:** (can this species climb/swim/fly/squeeze through/open/reach the described object? age/size/injury/environment alter feasible actions.)
- **Narrative next-event plausibility:** (given story state, rank candidate next events by causal continuity rather than genre cliché; impossible teleportation, forgotten injuries, unresolved physical constraints.)
- **Character-goal consistency:** (does proposed action advance the character’s explicitly stated goal? knowingly sacrifices one objective for another? action conflicts with known incentives?)
- **OCR/ngram-based text recognition:** (full document text plus focused sentence; tokenize into words/numbers and n-grams; identify customer number, invoice number, date, amount, reference, company/person/address from candidate spans.)
- **Noisy-OCR reconstruction:** (`I/1/l`, `O/0`, broken whitespace, hyphenation, duplicated characters; rank candidate reconstruction using surrounding textual constraints.)
- **Document-field association:** (multiple dates, IDs, addresses and amounts occur; determine which belongs to invoice, customer, supplier, delivery, payment or order rather than merely extracting nearest value.)
- **Document-type classification:** (invoice, quote, reminder, order confirmation, cancellation, contract, bank statement, medical report, support ticket, shipping notice, application, receipt.)
- **Document-workflow state:** (quote→order→delivery→invoice→payment→reminder; given documents/events, decide current state and which transitions have/have not happened.)
- **Table reasoning from linearized text:** (rows/columns represented textually; lookup, compare, aggregate, identify max/min, join related rows, respect headers/units and missing cells.)
- **Spreadsheet-formula semantics from text:** (formula plus cell values/ranges serialized textually; decide result/error/dependency; identify relative-reference mistakes without requiring spreadsheet modality.)
- **Form completeness:** (required/conditional fields based on provided rules; missing signature, inconsistent checkbox/value, field supplied in wrong section, conditional field unnecessarily demanded.)
- **Contract/clause consistency:** (given explicit contract clauses, decide whether proposed action/deadline/payment condition follows, conflicts, is unspecified or requires another clause; no free-form legal advice required.)
- **Requirement traceability:** (requirements → design → implementation → tests; determine which requirement has no implementation/test, which test maps to no requirement, which artifact contradicts another.)
- **Business-process conformance:** (defined workflow plus observed event sequence; legal transition, skipped approval, wrong role, duplicate action, expired prerequisite, valid exception path.)
- **Email-thread state understanding:** (who owes whom a response? latest agreed date/price/action? question already answered? quoted older text vs newest message; attachment mentioned but absent.)
- **Action-item extraction/verification:** (candidate action item actually promised/requested? correct owner/deadline? distinguish suggestion from commitment and condition from action.)
- **Product/catalog matching:** (natural-language request vs structured product descriptions serialized as text; rank correct SKU/model while respecting size/version/compatibility constraints.)
- **Compatibility matching:** (charger/device, software/plugin/runtime, replacement part/machine, datatype/API; distinguish physically similar from actually compatible.)
- **Taxonomy/category assignment:** (assign products/documents/issues/scientific concepts to hierarchical labels; nearest category vs overly general/overly specific class.)
- **Multi-label topic classification:** (text may genuinely belong to multiple domains; select all supported labels without spraying generic labels; include hierarchical and overlapping domains.)
- **Domain identification:** (biology, chemistry, finance, networking, compiler engineering, law, linguistics, medicine, music, statistics etc.; score domain mixture rather than forcing one class.)
- **Expertise-level estimation of text:** (introductory, practitioner, specialist/research-level based on concepts assumed and reasoning depth—not vocabulary complexity alone.)
- **Explanation-necessity judgement:** (is answer self-evident from supplied context, or does decision materially benefit from justification? distinguish safety/ambiguity/high-impact decisions from simple lookups.)
- **Explanation-quality judgement:** (given decision and rationale, does rationale actually support decision, merely restate it, introduce unrelated facts, reverse causality or use invalid evidence?)
- **Rationale-to-answer consistency:** (model selects A but its explanation logically supports B; numeric reasoning disagrees with final label; cited evidence contradicts final answer.)
- **Constraint-satisfaction judgement:** (candidate plan/output must obey many textual constraints: budget, size, dependency, date, allowed technologies, exclusions; identify exactly which constraints are violated.)
- **Soft-vs-hard constraint discrimination:** (“must”, “prefer”, “ideally”, “unless”, “at most”, “around”; rank options without treating preferences as requirements or vice versa.)
- **Preference learning from explicit examples:** (several accepted/rejected choices plus reasons; infer which candidate better matches demonstrated preference while distinguishing stable preference from incidental features.)
- **Multi-criteria scoring consistency:** (given explicit scoring rubric, does stated total/category follow component scores? candidate may manipulate weighting, normalize incorrectly or ignore veto criteria.)
- **Rubric-based quality assessment:** (article, support response, code review, product description, explanation; supplied rubric defines factuality/completeness/style/etc.; score dimensions separately rather than generic “good/bad.”)
- **Comparative judgement transitivity:** (A preferred to B, B to C under same criterion; detect inconsistent candidate ranking unless supplied context legitimately changes criterion.)
- **Pairwise preference prediction:** (two candidate outputs against explicit task/rubric; choose A/B/tie/insufficient evidence; useful primitive for reward/reranker training.)
- **Set coverage / diversity judgement:** (does selected set cover required topics/categories rather than containing five near-duplicates? rank candidate subsets by relevance plus non-redundancy.)
- **Redundancy detection:** (sentences, requirements, tests, retrieved documents, recommendations convey same semantic content despite different wording; exact vs partial redundancy.)
- **Information-density judgement:** (candidate contains useful distinct information vs verbosity/repetition; important for deciding compression/summarization without treating all length as bad.)
- **Data-quality validation:** (impossible ages/dates/ranges, inconsistent identifiers, duplicate IDs, unexpected missingness, categorical values outside domain, conflicting fields.)
- **Synthetic-data plausibility:** (does generated record obey supplied distributions, dependencies and business rules? identify impossible combinations and suspicious template repetition.)
- **Class-label leakage detection:** (training example contains target label or deterministic proxy in input; filename/category/template directly reveals gold decision.)
- **Dataset-example quality:** (ambiguous gold label, multiple correct answers, unsupported rationale, malformed state, trivial lexical shortcut, answer leaked in prompt, wrong difficulty.)
- **Benchmark-contamination clues:** (candidate task reproduces a supplied known example nearly verbatim vs legitimately same concept; n-gram/semantic overlap classifications from provided corpora.)
- **Adversarial-example detection:** (irrelevant distractors, conflicting metadata, misleading lexical overlap, injected false premise; determine which evidence is actually decision-relevant.)
- **Spurious-correlation resistance:** (training-style pattern says one label but causal/semantic evidence says another; names, formatting, length, position or irrelevant demographic-like attributes deliberately correlate with wrong answer.)
- **Invariant detection:** (given examples/transitions/code/data, decide which property remains invariant; proposed operation preserves or breaks it.)
- **Boundary-condition judgement:** (value exactly at min/max, zero, empty set, one-element collection, overflow threshold, inclusive/exclusive dates; determine which branch/rule applies.)
- **Failure-mode classification:** (timeout, validation failure, authorization failure, dependency outage, resource exhaustion, logic bug, malformed input, user cancellation; infer from textual observations.)
- **Recoverability judgement:** (retry, rollback, restart, compensate, request user input, permanent failure; classify based on supplied failure semantics.)
- **Reversibility/destructiveness:** (actions/code/commands/business operations: reversible, compensatable, partially reversible, destructive; estimate blast radius from explicitly supplied resources.)
- **Operational urgency:** (incident evidence → immediate intervention vs observe vs scheduled repair; based on explicit SLA/resource/error-rate thresholds rather than generic alarmism.)
- **Change-risk judgement:** (small local refactor vs schema migration vs auth-system rewrite; score probable blast radius using dependency/state/test evidence, independent of code length.)
- **Regression-likelihood judgement:** (changed module has many dependents, public contract change, no tests, state migration, or isolated internal refactor; rank patches by regression risk from provided evidence.)
- **Human-vs-machine task suitability:** (given workflow requirements, classify deterministic automation, LLM-compatible judgment, requires external measurement, requires human authority, or hybrid verification.)
- **Live-data requirement:** (can decision be made completely from supplied text/state, or does correctness require current database/network/sensor/time-dependent information?)
- **Information-acquisition value:** (when state is insufficient, rank possible next questions/measurements by expected ability to disambiguate hypotheses rather than merely asking for more data.)
- **Stopping/decision sufficiency:** (enough evidence already exists; gathering more data is unlikely to alter action vs decision should be deferred because critical uncertainty remains.)
- **Confidence-vs-evidence consistency:** (answer may be correct but confidence unjustifiably high/low; score whether confidence follows amount/quality/consistency of supplied evidence.)
- **Self-consistency across outputs:** (category says “safe”, score says 0.92 dangerous, rationale says destructive; detect contradictions among multiple typed outputs.)
- **Granularity selection:** (question supports exact value, interval, category, ranking or only unknown; choose the strongest answer type justified by evidence rather than inventing precision.)
- **Abstraction-level matching:** (question asks architecture-level issue but answer gives syntax detail; asks exact API but response gives generic concept; classify too abstract/appropriate/too specific.)
- **Analogy validity:** (two systems share relevant structure vs superficial similarity; determine which properties transfer and where analogy breaks.)
- **Category-error detection:** (treating latency as throughput, probability as confidence, class as instance, map as territory, correlation as cause, implementation as specification, price as value.)
- **Cognitive-bias/illogic detection:** (base-rate neglect, sunk cost, conjunction fallacy, framing, survivorship bias, anchoring, denominator neglect, availability bias; identify the reasoning error without assuming every surprising choice is biased.)
- **Contradiction-resolution:** (multiple claims disagree; determine whether genuinely contradictory, about different time periods/entities/scopes, or reconcilable through qualifiers.)
- **Scope reasoning:** (“all users” vs “paid users”; “Europe” vs “EU”; “may” vs “will”; “under load” vs universally; detect candidate conclusions that silently broaden scope.)
- **Quantifier reasoning in natural text:** (`all`, `none`, `some`, `at least one`, `exactly one`, `most`, `not every`; identify when paraphrase changes logical force without turning this into purely formal-logic tasks.)
- **Negation/polarity tracking:** (nested negation, “not uncommon”, “fails unless”, “cannot exclude”, “only if”; determine actual proposition and catch polarity-flipped summaries/translations.)
- **Reference/coreference resolution:** (which “it”, “they”, “former”, “latter”, “this version”, “that company” refers to; long technical/business documents with plausible distractors.)
- **Comparative-language reasoning:** (“A is 20% faster than B”, “B takes 20% longer”, “no worse than”, “second highest”; decide candidate ordering and detect non-equivalent paraphrases.)
- **Definition adherence:** (state explicitly defines a domain-specific term; later decision must use that local definition even if common usage differs.)
- **Local-rule-over-world-knowledge judgement:** (synthetic world says red tokens are valid only on Tuesdays; model must follow supplied world state rather than pretrained associations.)
- **World-model consistency:** (large textual state establishes entities/properties/rules; candidate event must respect all previous state, not just immediately preceding sentence.)
- **Long-context distractor resistance:** (relevant fact appears far from query among semantically similar but irrelevant material; identify correct evidence without positional bias.)
- **Needle-set retrieval:** (multiple separate facts across long state jointly determine answer; one fact alone suggests wrong answer; useful for testing distributed evidence integration.)
- **Cross-domain synthesis:** (decision requires combining e.g. unit conversion + API constraint + budget, or geography + invoice metadata + chronology; individual subtasks easy, composition determines gold.)
- **Minimal-sufficient-explanation selection:** (several explanations are true but some include unnecessary unsupported assumptions; select the explanation requiring the fewest additional assumptions while covering evidence.)
- **Competing-hypothesis discrimination:** (multiple plausible explanations; identify observations that favor one, observations shared by all, and what new evidence would separate them.)
- **Mechanism-vs-pattern distinction:** (data exhibits correlation/pattern, but proposed explanation claims a mechanism not established; score evidential support for mechanism separately from predictive usefulness.)
- **Decision robustness:** (small plausible changes to uncertain inputs should or should not change chosen action; classify robust vs knife-edge decisions.)
- **Sensitivity analysis:** (which parameter matters most to decision? e.g. cloud cost dominated by GPU hours rather than storage; ranking should change only when a threshold is crossed.)
- **Dominated-information detection:** (new evidence is redundant because stronger evidence already implies it; useful for deciding whether an additional test/query has information value.)
- **Consistency under paraphrase:** (same state/query expressed using alternate wording should yield same typed decision; detect lexical-position dependence.)
- **Consistency under irrelevant-context insertion:** (add true but decision-irrelevant facts; answer should remain unchanged.)
- **Consistency under option permutation:** (candidate options reordered or relabeled; output should follow semantics rather than positional preference.)
- **Calibration under ambiguity:** (clean cases deserve high confidence, near-boundary or conflicting cases lower confidence; intentionally include cases where correct output is tie/null rather than forced class.)
- **Action item assignment** (action_item_assignment): identify action items in meeting/conversation notes and who they are assigned to.
- **Ad policy violation** (ad_policy_violation): decide whether ad copy violates a supplied advertising policy.
- **Allergen present** (allergen_present): decide whether a food product contains a given allergen based on its ingredient text.
- **App review intent** (app_review_intent): classify the intent behind app-store reviews (bug report, feature request, praise, complaint).
- **Churn risk** (churn_risk): score customer churn risk from account and interaction signals.
- **City service request** (city_service_request): categorize and route municipal 311-style service requests.
- **Code review intent** (code_review_intent): classify the intent of code-review comments (defect, style, question, approval).
- **Commit intent** (commit_intent): categorize the intent of version-control commit messages (fix, feature, refactor, docs).
- **Contains pii** (contains_pii): decide whether a text snippet contains personally identifiable information.
- **Contains spoiler** (contains_spoiler): decide whether a review or comment reveals plot spoilers.
- **Content moderation** (content_moderation): classify rule-violating content under a supplied moderation policy.
- **Content type** (content_type): classify the type or format of a document or message.
- **Contract clause type** (contract_clause_type): identify clause types (termination, liability, confidentiality) in contract text.
- **Delivery exception** (delivery_exception): classify delivery and logistics exception events.
- **Dietary vegan** (dietary_vegan): decide whether a meal or product is vegan based on its description.
- **Document type** (document_type): classify documents into standard types (invoice, resume, contract, report).
- **Email intent** (email_intent): classify the intent of emails (request, complaint, notification, follow-up).
- **Email intent hard** (email_intent_hard): classify email intent under delimited distractor context and administrative noise.
- **Expense category** (expense_category): assign business expense categories to transaction descriptions.
- **Fair housing violation** (fair_housing_violation): detect fair-housing policy violations in listing or advertisement text.
- **Formality level** (formality_level): judge the formality register of a text.
- **Frustration level** (frustration_level): score customer frustration from support messages.
- **Frustration level hard** (frustration_level_hard): score customer frustration under delimited distractor context and administrative noise.
- **Gaming report type** (gaming_report_type): classify in-game user reports (cheating, abuse, bug, exploit).
- **Grammar issue** (grammar_issue): decide whether a sentence contains a grammatical error.
- **Hazmat shipping** (hazmat_shipping): decide hazardous-material shipping classification and restrictions.
- **Home service routing** (home_service_routing): route home-service requests to the correct trade and appointment outcome.
- **Incident severity** (incident_severity): classify the severity of incident reports.
- **Incident severity hard** (incident_severity_hard): classify incident severity under delimited distractor context and administrative noise.
- **Insurance claim priority** (insurance_claim_priority): prioritize insurance claims by urgency and impact.
- **Lead qualification** (lead_qualification): decide whether a sales lead qualifies under supplied criteria.
- **Meeting conflict** (meeting_conflict): detect scheduling conflicts between meetings.
- **News topic** (news_topic): classify news articles by topic.
- **Oncall route** (oncall_route): route on-call alerts to the correct responder or team.
- **Phishing email** (phishing_email): decide whether an email exhibits phishing indicators.
- **Question duplicate** (question_duplicate): decide whether two questions are duplicates.
- **Reading level** (reading_level): estimate the reading or grade level of a text.
- **Recipe cuisine** (recipe_cuisine): classify the cuisine of recipes from ingredients and instructions.
- **Refund eligible** (refund_eligible): decide refund eligibility under a supplied policy.
- **Refund eligible hard** (refund_eligible_hard): decide refund eligibility under delimited distractor context and administrative noise.
- **Return reason** (return_reason): classify the reason for a product return.
- **Review sentiment** (review_sentiment): classify the sentiment of reviews.
- **Review sentiment hard** (review_sentiment_hard): classify review sentiment under delimited distractor context and administrative noise.
- **Secret leak** (secret_leak): detect leaked secrets or credentials in text.
- **Secret leak hard** (secret_leak_hard): detect leaked secrets under delimited distractor context and administrative noise.
- **Sql injection risk** (sql_injection_risk): assess the SQL-injection risk of inputs and queries.
- **Support department** (support_department): route support tickets to the correct department.
- **Support department hard** (support_department_hard): route support tickets under delimited distractor context and administrative noise.
- **Suspicious transaction** (suspicious_transaction): flag suspicious financial transactions from descriptions and context.
- **Symptom triage** (symptom_triage): triage medical symptoms to an urgency and care level.
- **Travel policy violation** (travel_policy_violation): detect corporate travel-policy violations in booking requests.
- **Urgency** (urgency): assess the urgency of requests and messages.
- **Urgency hard** (urgency_hard): assess urgency under delimited distractor context and administrative noise.
- **Veterinary triage** (veterinary_triage): triage animal symptoms to an urgency and care level.
- **Voice assistant intent** (voice_assistant_intent): classify voice-assistant intents from utterances.
- **Warranty claim eligible** (warranty_claim_eligible): decide warranty claim eligibility under supplied terms.
- **Weather alert severity** (weather_alert_severity): classify the severity of weather alerts.
- **GPQA Diamond** (gpqa_diamond): Google-proof Q&A: graduate-level multiple-choice science questions written to resist internet lookup.

## 📖 CLI reference

### 🎛️ Make targets

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

## Technical Architecture

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