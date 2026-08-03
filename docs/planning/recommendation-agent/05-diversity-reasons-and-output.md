# Branch 05 - Diversity Selection, Grounded Reasons, and Agent Output

## Branch metadata

- **Proposed branch:** `feat/recommendation-result-shaping`
- **Base dependency:** Branch 04 merged
- **Policy dependencies:** accepted diversity policy and reason predicate table
  from Branch 01; U-11 resolved
- **Default explanation mode:** deterministic templates; no LLM

## Purpose

Implement the Recommendation Agent's only decision policy after inference:
preserve the strongest lead, choose intentionally different evidence-supported
alternatives, derive reason codes from frozen predicates, render constrained
observational text, and return a deterministic strict v2 success response.

## Related source-document sections

- Implementation design: **Result shaping**, **Grounded reason-code
  predicates**, Agent hard bounds, **Security, privacy, and safety**, and
  **Verification strategy - Intelligence**.
- Delivery plan: Phase 4 tasks 2, 4-6 and the complete Phase 4 exit gate.
- Inventory discrepancies: CF proxy correction and unsupported explanation
  evidence.

## Prerequisites and dependencies

- U-11 freezes all numeric thresholds, coefficients, tie band, omission/
  substitution behavior, and policy version.
- Inference response includes every named evidence signal needed by the reason
  table and no arbitrary prose.
- Backend v2 reviewer confirms it validates the same reason predicates against
  persisted evidence.
- The golden scenario has enough candidates to cover all types, plus fixtures
  where one or more types must be omitted.

## Scope

- immutable versioned diversity policy representation;
- stable Personal lead selection;
- Exploratory diversity penalty and bounded novelty bonus;
- Group-inspired eligibility from authorised group support;
- uniqueness, rank, type, and maximum-result invariants;
- allow-listed reason derivation from explicit predicates;
- deterministic reason priority and template composition;
- unsafe/unsupported-language defense in depth;
- property, mutation, golden, and Backend consumer contract fixtures.

### Explicit non-scope

- No hard-filter relaxation or candidate creation.
- No model probability adjustment/training or coefficient interpretation.
- No factual claim based on missing evidence.
- No guarantee of allergen safety, health, cleanliness, quality, or truth.
- No LLM. A later LLM renderer requires a separate disabled-by-default plan and
  Backend validation remains mandatory.

## Concrete files

```text
agent-service/app/agents/recommendation/src/recommendation_agent/policy/
  __init__.py
  diversity.py
  reason_predicates.py
  versions.py
agent-service/app/agents/recommendation/src/recommendation_agent/selection/
  __init__.py
  selector.py
  similarity.py
agent-service/app/agents/recommendation/src/recommendation_agent/reasons/
  __init__.py
  deriver.py
  models.py
  renderer.py
  templates.py
agent-service/app/agents/recommendation/src/recommendation_agent/domain/
  models.py
agent-service/app/agents/recommendation/src/recommendation_agent/workflow/
  context.py
  nodes.py
agent-service/app/agents/recommendation/src/recommendation_agent/main.py
agent-service/app/agents/recommendation/tests/unit/selection/
  test_selector.py
  test_similarity.py
agent-service/app/agents/recommendation/tests/unit/reasons/
  test_predicates.py
  test_deriver.py
  test_renderer.py
agent-service/app/agents/recommendation/tests/integration/
  test_result_shaping_golden.py
  test_result_shaping_omissions.py
  test_result_shaping_determinism.py
agent-service/app/agents/recommendation/tests/security/
  test_explanation_safety.py
agent-service/app/agents/recommendation/tests/contract/
  test_backend_agent_v2_fixtures.py
```

## Selection contract

### Common invariants

1. Start only from inference candidates whose IDs occur exactly once in the
   validated Agent request.
2. Use calibrated probability as the confidence value returned to Backend.
   Policy-adjusted selection utility is internal and is not mislabeled as model
   probability.
3. Sort every tie with the frozen stable order ending in request-scoped
   `candidateId`; never raw/model key or nondeterministic set order.
4. Rank 1 remains the highest-confidence lead under the accepted tie-band
   policy; novelty/group diversity cannot displace a materially higher score.
5. A candidate appears at most once. Output ranks are contiguous from 1 and at
   most three.
6. If evidence cannot support a result type, follow the frozen omit/substitute
   policy; never manufacture all three types.

### Lead/Personal

- Determine the maximum probability.
- Restrict any personal-evidence preference to candidates inside the accepted
  tie band of that maximum.
- Apply the frozen personal-evidence predicate (for example explicit
  preference, UserCF, ItemCF, or personal-history evidence as accepted).
- Select one deterministic lead and label it `PERSONAL` only according to the
  accepted type policy. If no personal evidence exists, follow the frozen type
  fallback; do not invent it.

### Exploratory

- Consider only unselected candidates eligible under the frozen exploratory
  evidence policy.
- Calculate selection utility from probability minus bounded similarity
  penalties plus bounded novelty bonus.
- Similarity may use only approved candidate facts already supplied: model meal
  or offering pseudonymous identity, cuisine, meal type, and/or place grouping
  defined by the policy. Do not derive identity from raw IDs.
- Penalty/bonus cannot exceed frozen caps or reorder rank 1.

### Group-inspired

- Consider only unselected candidates meeting the authorised group-support
  predicate and threshold in inference/request evidence.
- Group evidence remains separate from UserCF. It may support
  `GROUP_INSPIRED` and its reason but not `SIMILAR_USERS_LIKED`.
- If there is no qualifying candidate, omit/substitute exactly as frozen.

### Ordering of alternatives

The source design identifies types but does not mandate that Exploratory is
always rank 2 and Group-inspired rank 3 when their probabilities differ. U-11
must freeze one deterministic rule. The implementation must encode and test
that rule as `diversityPolicyVersion`; do not infer it from enum order.

## Reason contract

Use one central predicate table in Agent code generated or directly mirrored
from the accepted contract. Backend implements the same semantics independently
against persisted evidence.

| Concept | Minimum accepted predicate |
| --- | --- |
| Similar users liked it | UserCF available, finite score at/above threshold, and neighbor support at/above threshold. |
| Similar to liked meals | ItemCF available, finite score at/above threshold, and supporting-item count at/above threshold. |
| Preference match | Explicit candidate preference-match fact for named cuisine and/or meal type is true. Mere non-conflict is not a match. |
| Want to Try | Decision-time Want-to-Try fact is true. |
| Group-inspired | Authorised group support count/strength meets the frozen threshold. |
| Current context | At least one specific accepted budget, distance, meal-period, or requested-time availability match fact is present. |
| Cleanliness evidence | A current-enough recorded observation exists; template identifies it observationally and never says safe/clean/guaranteed. |

Probability, coefficient magnitude, candidate rank, missing allergen conflict,
group count, or personal count alone is never a reason predicate.

## Explanation contract

- Templates are code constants keyed by reason code/version and use only
  allow-listed, typed placeholders.
- Do not interpolate free text, model key, raw ID, note, arbitrary place fact,
  or numeric probability.
- Compose at most the frozen number of reasons in deterministic priority order.
- Bound UTF-8 bytes and characters; normalize whitespace; reject control/markup
  if not explicitly allowed by the contract.
- Required wording is observational, for example saved preference or recorded
  score. Forbidden claim families include `safe`, `allergen-free`,
  `allergy-safe`, `healthy`, `medical`, `guaranteed`, and equivalent frozen
  variants.
- If a template cannot be rendered from complete typed facts, fail the whole
  Agent response; do not drop the reason silently or emit generic praise.

## Ordered implementation tasks

1. Encode the accepted diversity and reason policies as immutable typed
   constants/data with explicit version strings and startup self-validation.
2. Add typed scored-candidate and selected-candidate domain models separating
   `probability` from internal selection utility.
3. Implement stable lead selection and personal tie-band preference. Add exact
   boundary tests at inside/equal/outside the band.
4. Implement approved candidate similarity as a pure function. Missing facts
   contribute only as frozen; they never imply diversity or similarity by
   default.
5. Implement bounded exploratory utility, cap all bonus/penalty contributions,
   and use deterministic candidate-ID tie-break.
6. Implement Group-inspired eligibility/selection without using UserCF proxy
   fields.
7. Implement accepted alternative ordering and omit/substitute behavior.
   Revalidate uniqueness/ranks/types after selection.
8. Implement central reason predicate functions. Each accepts only the typed
   candidate/evidence object and policy; no probability-as-explanation shortcut.
9. Implement deterministic reason priority/max-count and derive typed reason
   facts. If no approved reason supports a selected candidate, follow the
   accepted policy (omit candidate or fail result); never create a generic
   unsupported reason.
10. Implement versioned templates and strict renderer. Add a defense-in-depth
    unsafe-language validator over rendered output.
11. Wire selector/deriver/renderer into Plan 04 workflow context and replace
    production startup failure/doubles with real implementations.
12. In `build_success`, include policy versions and original probability;
    revalidate every output against input membership and canonical schema.
13. Add a golden full-set fixture and omission fixtures for new user, new meal,
    sparse group, no exploratory diversity, exact score ties, and fewer than
    three candidates.
14. Add one positive and multiple near-miss negative tests per reason threshold.
15. Add property tests: deterministic permutation handling according to stable
    tie rules, maximum three, unique IDs, contiguous ranks, probability range,
    lead preservation, no unsupported reason.
16. Add mutation tests or targeted guard tests proving that changing `>=`/`>`,
    removing availability, swapping support counters, or skipping uniqueness
    fails the suite.
17. Export Agent v2 success/failure fixtures for Backend consumer tests and
    record their checksum/version in the compatibility matrix.
18. Update README with policy versions and explicitly document why no LLM is in
    the ranking/explanation path.

## Test requirements

### Selection

- unique top score; exact ties; tie-band boundaries; personal evidence absent;
- novelty bonus/penalty boundaries and caps;
- same meal/cuisine/offering/place similarity cases from frozen policy;
- qualifying/non-qualifying group support;
- 1/2/3/100 candidates and every type omission/substitution case;
- repeated input and permitted input permutations produce frozen deterministic
  output; rank 1 confidence rule never violated.

### Reasons/templates

- each positive predicate and each missing flag/threshold near miss;
- UserCF reason fails with group support only;
- ItemCF reason fails with personal records only;
- preference reason fails for mere non-conflict;
- context reason names a specific fact;
- cleanliness wording remains observational and freshness-gated;
- duplicate/over-limit reason and unsupported code fail;
- injection/control/markup/unsafe-language canaries never reach output.

### Full workflow/Backend contract

- normal three-type golden output;
- cold-start LR output without CF reasons;
- sparse group omits group result/reason;
- identical request/inference/package/policy yields identical candidate order,
  types, reasons, templates, and versions;
- malicious selector/renderer output is rejected by terminal validation;
- Backend v2 fixture consumer accepts valid output and rejects all negative
  reason/ID/version examples.

## Acceptance criteria

- [ ] Highest-confidence lead is preserved under the frozen tie-band rule.
- [ ] At most three unique contiguous results are deterministic for identical
  input/package/policy.
- [ ] Missing type evidence produces only the accepted omission/substitution.
- [ ] Every reason has a passing predicate and Backend-corresponding fixture.
- [ ] Group/personal proxy semantics cannot pass CF reasons.
- [ ] Explanations are template-only, bounded, observational, and free of
  unsupported/unsafe claims.
- [ ] Production Agent startup uses real deterministic policies, not test doubles.

## Commit plan

1. `feat(recommendation): freeze deterministic diversity policy`
2. `feat(recommendation): select personal exploratory and group results`
3. `feat(recommendation): derive evidence-backed reason codes`
4. `feat(recommendation): render constrained explanation templates`
5. `test(recommendation): prove diversity grounding and determinism`
6. `test(contracts): publish Backend Agent v2 result fixtures`

## Verification

```powershell
Push-Location agent-service/app/agents/recommendation
uv run pytest tests/unit/selection tests/unit/reasons -v
uv run pytest tests/integration/test_result_shaping_golden.py `
  tests/integration/test_result_shaping_omissions.py `
  tests/integration/test_result_shaping_determinism.py -v
uv run pytest tests/security/test_explanation_safety.py -v
uv run pytest tests/contract/test_backend_agent_v2_fixtures.py -v
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
Pop-Location
```

Expected: all commands exit `0`; golden outputs/checksums match, property and
near-miss reason tests pass, and the unsafe-language canary matrix is absent
from every successful response/log.

## Pull Request hand-off

- **Title:** `feat(recommendation): shape diverse grounded recommendation results`
- Include policy/reason/template versions, numeric thresholds, golden fixture
  checksum, omission/type-order decision, exact commands, determinism/property
  evidence, and Backend consumer-test link.

## Rollback and unresolved items

The entire Agent v2 path can be disabled so Backend fallback serves requests.
A policy semantic change requires a new policy version and fixtures; never
silently change behavior under an existing version. U-11 must be resolved
before implementation.
