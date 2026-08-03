# Branch 03 - Inference Client, Compatibility, and Deadlines

## Branch metadata

- **Proposed branch:** `feat/recommendation-inference-client`
- **Base dependency:** Branch 02 merged
- **External dependency:** compatible `recommendation-inference-v1` endpoint and fixtures
- **Downstream attempts:** exactly one; no retry

## Purpose

Add the Agent's only external capability: a lifecycle-scoped, authenticated,
bounded inference client. Convert Backend's absolute deadline into a local
monotonic budget, send strict inference v1 requests, validate strict responses,
and map every transport/compatibility failure to an Agent typed failure.

This plan does not implement inference or any ML behavior.

## Related source-document sections

- Implementation design: **End-to-end online sequence**, **Inference service
  design** response, **Recommendation Agent design** one-call/deadline bounds,
  **Contract evolution**, and **Reliability**.
- Delivery plan: Phase 4 tasks 2-4 and failure exit gates.
- Local runtime architecture: tool boundary, inference-unavailable failure, and
  correlation.

## Prerequisites and dependencies

- Accepted inference v1 consumer fixtures/source revision from Plan 01.
- U-04 freezes Backend total, Agent guard, inference connect/pool/read/total,
  and P95/P99 budgets. Current Backend v1 defaults (800 ms read; two-second
  absolute deadline) are not silently reused for v2.
- U-10 provides inference credential binding and private network origin.
- Inference endpoint/path/readiness contract is owned externally.

## Scope

- strict inference v1 consumer Pydantic models;
- UTC/monotonic clock and deadline budget;
- lifecycle `httpx.AsyncClient` and `InferencePort` adapter;
- request mapping, auth/correlation/version headers, byte/time bounds;
- strict echoed-ID/candidate/version/probability/CF/evidence validation;
- safe typed transport/compatibility failure mapping;
- readiness dependency state and call metrics.

### Explicit non-scope

- No inference service code, model loader/package, features, CF, LR, ML,
  probability generation, or model release.
- No graph/selection/reason/template logic.
- No retry, circuit-selected alternative model, database, Backend tool, or web.

## Concrete files

```text
agent-service/app/agents/recommendation/src/recommendation_agent/
  clients/inference.py
  schemas/inference_v1.py
  time/__init__.py
  time/budget.py
  application/ports.py
  domain/errors.py
  domain/models.py
  config/settings.py
  api/health.py
  main.py
agent-service/app/agents/recommendation/tests/
  unit/test_deadline_budget.py
  unit/test_inference_client.py
  contract/test_inference_contract_v1.py
  integration/test_inference_transport.py
  integration/test_inference_readiness.py
  security/test_inference_client_privacy.py
```

## Interfaces and configuration

| Symbol | Required contract |
| --- | --- |
| `Clock` | Injected `utc_now()` and `monotonic()` for deterministic tests. |
| `DeadlineBudget#from_absolute` | Samples both clocks once, validates remaining time, and stores local monotonic expiry. |
| `DeadlineBudget#remaining` | Non-negative duration; caller subtracts accepted guard and never extends expiry. |
| `InferencePort#score` | Async transport-neutral command/result used only by Plan 04 `score_once`. |
| `RecommendationInferenceHttpClient#score` | One strict HTTP POST with Bearer/correlation/version headers, bounded time/body, strict response mapping, no retry. |

Add under `RECOMMENDATION_AGENT_`:

- `INFERENCE_BASE_URL`, `INFERENCE_ENDPOINT_PATH`;
- `INFERENCE_SERVICE_TOKEN`;
- `INFERENCE_CONNECT_TIMEOUT_MS`, `INFERENCE_POOL_TIMEOUT_MS`;
- `INFERENCE_MAX_RESPONSE_BYTES`;
- `DEADLINE_GUARD_MS`, `MIN_DEADLINE_BUDGET_MS`;
- accepted inference/feature/key versions or their source-manifest binding.

URL scheme/origin/path are configuration-only and allow-listed. A request cannot
choose host, route, token, package, or model version outside the frozen contract.

## Deadline semantics

1. Backend sends absolute UTC `deadlineAt`.
2. At Agent ingress, sample UTC and monotonic clocks once.
3. `remaining = deadlineAt - utc_now`; reject expired/below minimum.
4. `monotonic_expiry = monotonic_now + remaining`.
5. Before inference, total timeout is the lesser of configured downstream
   timeout and `remaining_now - DEADLINE_GUARD_MS`.
6. Forward original `deadlineAt`; never transmit process-local monotonic time.
7. Check remaining budget after the response and before Agent output.

## Ordered implementation tasks

1. Mirror accepted inference v1 request/success/failure models strictly and
   validate source fixtures/checksum in contract tests.
2. Define transport-neutral command/result/CF/evidence types. HTTP status/body
   never enters workflow/domain code.
3. Implement injected clock and deadline budget with timezone-aware UTC only.
4. Validate coherent settings: private HTTP(S) policy by environment, positive
   timeouts, guard smaller than total budget, response cap, secret strength.
5. Create one lifespan-scoped `httpx.AsyncClient` with bounded connections and
   no automatic retry hooks.
6. Map Agent request to inference request exactly: IDs, deadline, feature/key
   versions, model user/meal/offering keys, eligible candidates and accepted
   features only.
7. Add Bearer auth, safe correlation/version headers, JSON content type, and
   total/connect/pool/write/read bounds based on current remaining budget.
8. Perform exactly one HTTP request. Do not retry timeout, connection, 5xx,
   malformed, or ambiguous results.
9. Enforce response bytes before parsing; parse strict JSON/Pydantic.
10. Validate echoed request/trace, candidate one-to-one correlation, no unknown/
    duplicate/missing IDs except frozen recoverable status, exact compatibility
    versions, finite probability `[0,1]`, and CF availability/score/support.
11. Map connection, timeout, non-2xx, malformed, oversized, unavailable,
    schema, candidate, model/package/feature/key mismatch, and deadline exhaustion
    to stable Agent failure categories without raw downstream content.
12. Expose dependency readiness state based on configuration and a bounded
    compatibility/readiness check accepted by contract; do not score on every
    health probe.
13. Add safe metrics/logs for duration, candidate count, versions, and result
    category only; no key/feature/body/token/URL query.
14. Add fake transport/server tests for every failure and a spy asserting
    zero/one attempts.
15. Update Agent README with exact external dependency contract and explicitly
    state inference/ML ownership is outside this package.

## Test requirements

- expired/near deadline invokes inference zero times;
- valid request invokes exactly once; connection/timeout/5xx/malformed also
  never exceed one attempt;
- downstream timeout equals remaining-minus-guard and never exceeds config;
- correct auth/correlation/version headers and strict body;
- response cap before parse;
- wrong echo/version/key/model/package, unknown/duplicate/missing candidate,
  invalid probability/CF/status fail safely;
- cold-start unavailable CF signal is accepted when contract-valid;
- one shared client per lifespan and clean close;
- readiness does not leak origin/token/version details beyond safe contract;
- logs/errors contain no synthetic token/key/feature/body/signed-URL canary.

## Acceptance criteria

- [ ] Agent validates the canonical inference v1 fixture bytes/checksum.
- [ ] At most one downstream request occurs for every Agent request.
- [ ] Deadline tests prove Agent retains the accepted Backend guard.
- [ ] All invalid/unavailable inference results become typed Agent failures.
- [ ] No inference or ML implementation code/dependency is introduced.
- [ ] Health/readiness and logs remain safe.

## Commit plan

1. `feat(recommendation): add monotonic deadline budgets`
2. `feat(recommendation): add strict inference consumer models`
3. `feat(recommendation): call private inference exactly once`
4. `test(recommendation): cover inference failures compatibility and privacy`
5. `docs(recommendation): document inference dependency contract`

## Verification

```powershell
Push-Location agent-service/app/agents/recommendation
uv run pytest tests/unit/test_deadline_budget.py tests/unit/test_inference_client.py -v
uv run pytest tests/contract/test_inference_contract_v1.py -v
uv run pytest tests/integration/test_inference_transport.py `
  tests/integration/test_inference_readiness.py -v
uv run pytest tests/security/test_inference_client_privacy.py -v
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
Pop-Location
```

Expected: all commands exit `0`; evidence shows zero/one attempts, exact timeout
budget, fixture checksum, and zero sensitive canary leakage.

## Pull Request hand-off

- **Title:** `feat(recommendation): add bounded inference client`
- Include external inference schema revision/checksum, timeout allocation,
  call-count/failure matrix, config names, exact commands, and ownership note.

## Rollback and unresolved items

Rollback deploys the prior Agent image or disables Backend v2 routing. Backend
fallback remains the resilience path. U-04 and U-10 must be resolved before
non-local integration.
