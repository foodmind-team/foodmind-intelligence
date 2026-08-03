# Recommendation Agent v2 contract notes

Status: local coordination baseline for implementation. External Backend and
inference owner approval remains a release gate; this document does not claim
approval that is not present in this repository.

## Ownership and compatibility tuple

- Agent contract: `recommendation-agent-v2`, owned by the FoodMind
  Intelligence Recommendation Agent maintainers.
- Feature contract: `recommendation-features-v2`, Backend-produced at
  `decisionAt`; the Agent consumes the allow-listed evidence catalog only.
- Inference contract: `recommendation-inference-v1`, owned by the FoodMind
  inference maintainers; this repository holds consumer fixtures, not an
  inference implementation.
- Model/key tuple: `hybrid-ranking-v1`, `recommendation-package-v1`, and <!-- gitleaks:allow -->
  `hmac-sha256-v1`.
- Policy tuple: `recommendation-diversity-v1`,
  `recommendation-reasons-v1`, and `recommendation-template-v1`.
- Internal route: `POST /internal/v1/recommendations/generate` carrying the
  strict v2 envelope; authentication is an exact constant-time Bearer secret
  comparison. The token never appears in a body, URL, or log.
- Backend v1 stays on its existing fallback path. There is no v1-to-v2
  payload coercion. The v2 window supports only the tuple above; incompatible
  changes require a new contract version.

The canonical Backend v1 fixtures remain owned by
`foodmind-backend@7ea2b90c1451d689c59d4ea37d337b4552220f44`. Their paths and
checksums are recorded without copying or reinterpreting the bytes in
`contracts/internal/agent/recommendation/v1/backend-fixture-manifest.json`.
They are v1 compatibility evidence only, never v2 acceptance evidence.

## Implemented design defaults

These defaults resolve implementation ambiguity inside this repository. The
external owner acceptance called out above is still required before staged v2
traffic.

- **U-02 feature ownership:** Backend supplies point-in-time preference,
  Want-to-Try, group, context, cleanliness, novelty, cuisine, and category
  facts. Inference alone supplies UserCF/ItemCF availability, scores, and
  support counts. The Agent never substitutes group counts for UserCF or
  personal counts for ItemCF.
- **U-03 compatibility:** the Agent is v2-only and does not expose a v1
  adapter. Backend v1 remains on its existing deterministic fallback path. V1
  retirement requires a separate owner-approved telemetry window; this module
  neither translates nor removes v1.
- **U-04 latency allocation:** Backend sends an absolute UTC `deadlineAt` (the
  synthetic contract baseline uses a two-second window). The Agent converts it
  once at admission to a monotonic budget, caps inference at 700 ms with
  100 ms connect and 50 ms pool bounds, reserves a 50 ms guard, rejects
  admission budgets below 100 ms, and makes at most one inference call. The
  local Agent-only P95 working bound is 100 ms; staged P95/P99 owner acceptance
  remains a release gate. No automatic retry is allowed.
- **U-09 operational routes:** repository convention is `GET /health/live`
  for process/event-loop liveness and `GET /health/ready` for configuration,
  inference-adapter, immutable-policy, workflow, and shutdown readiness. No
  `/live` or `/ready` aliases are exposed.
- **U-10 service binding:** local/test/CI use the exact constant-time Bearer
  binding. Non-local deployments require explicit strong credentials, private
  HTTPS inference, secret-manager rotation, private ingress, and an
  owner-selected TLS/network-policy binding.
- **Failure transport:** authentication failures use HTTP 401, request
  validation/size failures use 400/413, deadline failures use 408/504,
  inference availability/transport failures use 502/503/504, and internal
  selection/template failures use 500. When safe correlation fields can be
  recovered, the body is `failure.schema.json`; raw exception/upstream text is
  never returned.
- **U-11 policy constants:** the numeric diversity/reason/template values below
  are frozen for v1 policies.

## Diversity policy

- Lead tie band: `0.03` probability.
- Personal evidence: UserCF available with score at least `0.60` and neighbor
  support at least `3`, or preference match at least `0.70`.
- Novelty bonus: `0.08 * novelty`, capped at `0.08`.
- Repeated-category and repeated-cuisine penalties: `0.06` and `0.04`.
- Group eligibility: group rate at least `0.60` with at least `2` eligible
  members.
- Preferred slots are `PERSONAL`, `EXPLORATORY`, `GROUP_INSPIRED`. An
  ineligible type is omitted; remaining eligible candidates fill available
  slots without duplicating a candidate, up to three results.
- Stable ordering: policy-adjusted score descending, inference probability
  descending, model score descending, original request index ascending, then
  `candidateId` ascending.

## Reason and template policy

At most two reasons are emitted, in this priority order:

1. `USER_CF`: available, score >= 0.60, neighbor support >= 3.
2. `ITEM_CF`: available, score >= 0.60, supporting-item count >= 2.
3. `PREFERENCE_MATCH`: Backend point-in-time value >= 0.70.
4. `WANT_TO_TRY`: Backend point-in-time value is true.
5. `GROUP_POPULAR`: rate >= 0.60 and eligible-member count >= 2.
6. `CONTEXT_MATCH`: Backend point-in-time value >= 0.70.
7. `CLEANLINESS_OBSERVED`: Backend point-in-time value is true.

Each code maps to a fixed sentence fragment. Composition follows priority,
uses at most two fragments and 160 characters, and never uses free-form model
text. Forbidden claims include `guaranteed`, `best`, `healthiest`, `safest`,
`perfect`, clinical/nutritional outcomes, or claims about people not evidenced
by the request.

## Failure behavior

All failure codes are allow-listed by `failure.schema.json`. Retryable is true
only for transient connection, timeout, HTTP 5xx, unavailable, or response-size
conditions; the Agent itself still performs no retry. Schema/key/package/model
mismatches, candidate integrity failures, invalid evidence, and policy failures
are non-retryable and always trigger Backend fallback.

## Data minimisation

Correlation IDs and request-scoped candidate IDs are bounded safe identifiers.
Model user/meal/offering keys are accepted only where required and are never
logged or returned. Requests exclude database IDs, exact coordinates, free
text, credentials, comments, and arbitrary feature vectors. Fixtures are
synthetic and deliberately non-production.
