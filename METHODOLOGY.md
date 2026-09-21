# Synthetic Evaluation Dataset Generation Methodology

## Source

`CAPABILITIES.md` is the capability specification used to define the 275 added suites.

## Core construction

Each suite has 100 concrete System One requests. Generators are capability-family specific:

- software verification: code behavior, tests, API/version/portability, concurrency, SQL/regex, serialization/migration, distributed/cache/config/CLI;
- graph/evidence: reachability, ontology, temporal graph, causal chains, provenance, entailment, answerability, retrieval/reranking and reconciliation;
- business/decision: accounting, anomaly/fraud, resource/capacity/inventory/scheduling, Pareto/expected utility/Bayes/game theory, uncertainty and units;
- physical/scientific: navigation, spatial/route/affordance, mechanics, temporal/geographical/scientific/experimental/biological/medical chronology;
- language/social: translation, grammar, WSD, code-switching, pragmatics/humor/tone, dialogue/persona/ToM, social permissions and animal/world state;
- documents/products: OCR, field association, document workflow, tables/spreadsheets/forms/contracts, product/compatibility/taxonomy/domain;
- reliability/meta: explanation/constraint/preference/rubric quality, data/benchmark quality, operations/recoverability, epistemic sufficiency, long-context/needle-set, robustness/sensitivity/invariance.

## Gold derivation

Gold is deterministic by construction. Typical derivations include:

- arithmetic and threshold evaluation;
- explicit state transitions and dependency checks;
- set/graph reachability;
- supplied time intervals and chronology;
- supplied source/reference statements;
- expected utility and Bayes calculations;
- explicit authorization/policy rules;
- dimensional/unit conversions;
- explicit rubric scoring;
- synthetic program semantics;
- controlled semantic transformations.

Reference facts used for history/science/geography are embedded in the request state where needed, so those tasks evaluate reasoning from supplied evidence rather than external recall.

## Shortcut controls

- Request bodies are globally unique.
- Choice insertion order is varied.
- Gold distributions were audited to ensure at least two distinct outcomes.
- Gold and rationales live only in golden/metadata files.

## Important limitation

100 cases per suite improves paired-comparison resolution, but templated synthetic cases are correlated.