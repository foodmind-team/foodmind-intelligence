# Cooking Plan Agent — Operations Runbook

> Handbook 12.10: operational steps for 8 failure scenarios.
> Version: 1.1 | Last updated: 2026-08-02

---

## Scenario 1: High FAILED Response Rate

**Symptom**: `/internal/v1/agents/cooking-plan/generate` returns FAILED status for > 5% of requests.

**Diagnosis**:

```bash
# Check logs for error codes (P2-03: public messages come from the central
# message catalog; diagnosis is keyed by error_code + correlation_id, not message text)
docker logs cooking-plan-agent 2>&1 | jq 'select(.level == "ERROR" or (.msg | contains("Workflow FAILED"))) | {code: .error_code, node: .node, correlation_id: .correlation_id}'

# Check structured metrics (if Prometheus is wired)
curl localhost:8000/health/ready
```

**Response**:

| Error Code | Action |
|-----------|--------|
| `SCHEDULE_VERIFICATION_FAILED` | **P1 alert** — indicates solver or verifier bug. Capture the failing request payload, solver status, and verifier issues. Roll back to previous image if rate > 1%. |
| `SCHEDULE_MODEL_INVALID` | **P1 alert** — a model-construction bug (contradictory constraints, invalid scheduling problem shape). Retryable per the error catalog (P3-05). Capture the request payload and task graph; roll back if rate > 0.5%. |
| `SCHEDULE_UNKNOWN` | **P2 alert** — solver hit its time limit before determining feasibility. Retryable per the error catalog (P3-05). Raised as a FAILED response, never as INFEASIBLE (P1-04). Review solver budget if rate climbs. |
| `EXTERNAL_PROVIDER_UNAVAILABLE` | Check LLM/Search provider status. Retryable per the error catalog (P3-05). If provider is down, service degrades to local inference — this is expected. |
| `INTERNAL_ERROR` | Check logs for unexpected exceptions. May indicate a code defect. |

> **P2-03 note**: client-facing `message` in FAILED responses always resolves through the public message catalog (`domain/errors.py` → `_PUBLIC_MESSAGES`), never from node exceptions. Missing catalog rows fail closed to `INTERNAL_ERROR`. To trace an incident, use `correlation_id` (present in response body and logs) — do not rely on the public message text.

**Rollback**: `docker run -d --name cpa-rollback <previous-image-digest>`

---

## Scenario 2: LLM Provider Unavailable

**Symptom**: Logs show `EXTERNAL_PROVIDER_UNAVAILABLE` for every recipe parsing request.

**Diagnosis**:

```bash
# Check provider reachability
curl -I https://api.anthropic.com/v1/messages

# Check Agent logs
docker logs cooking-plan-agent | grep "provider" | tail -10
```

**Response**:

1. **Degraded mode**: The Agent falls back to stub recipe parsing — plans will produce fewer details but the service remains available. No immediate action required.
2. **Extended outage (> 15 min)**: 检查 LLM API 状态页面。如果确认是 provider 侧异常，等待恢复即可。
3. **Network/routing issue**: 检查 Agent 容器的出网规则，确认没有防火墙阻挡。

**Does NOT require service restart** — the Agent handles provider failures gracefully at the request level.

---

## Scenario 3: Search Provider Unavailable

**Symptom**: Research gaps produce `needs_confirmation=True` instead of filled evidence.

**Diagnosis**:

```bash
docker logs cooking-plan-agent | grep "search" | tail -10
```

**Response**:

1. **Non-blocking**: Search failure is a soft degradation. The Agent returns `NEEDS_CONFIRMATION` with gaps flagged for user review. Service is still operational.
2. **Check timeout**: If logs show timeouts (not connection errors), adjust `research_timeout_seconds` via env var.

**Recovery**: Provider recovery is automatic on next request — no restart needed.

---

## Scenario 4: Solver Timeouts Increase

**Symptom**: CP-SAT solver returns `UNKNOWN` status (timed out before determining feasibility).

**Diagnosis**:

```bash
# Check solver wall-time metrics
docker logs cooking-plan-agent | jq 'select(.solver_status == "UNKNOWN") | {request_id, wall_time_seconds, task_count}'

# Check server load
docker stats cooking-plan-agent
```

**Response**:

1. **Increase solver budget**: Set `COOKING_PLAN_SOLVER_TIMEOUT_SECONDS=10` (default 5).
2. **Reduce task complexity**: Check that `max_task_count` is ≤ 100.
3. **Scale vertically**: Increase container CPU limit if consistently at 100%.
4. **Reduce concurrent requests**: Add rate limiting at the Spring Boot side.

**Note**: Solver timeouts are expected for very large plans (6 recipes × 100 tasks). Per P1-04 the result is a `FAILED` response with `SCHEDULE_UNKNOWN` — never `INFEASIBLE`, because the solver did not prove infeasibility; it simply ran out of budget.

---

## Scenario 5: Verifier Failures

**Symptom**: `SCHEDULE_VERIFICATION_FAILED` appears in logs.

**Diagnosis**:

```bash
# Capture detailed verifier issues
docker logs cooking-plan-agent | jq 'select(.error_code == "SCHEDULE_VERIFICATION_FAILED") | {request_id, issues}'
```

**Response**:

1. **P1 incident**: Verifier failures indicate a programming defect in CP-SAT model construction or schedule extraction. This is NOT a normal user outcome.
2. **Capture evidence**: Save the failing request payload, solver output, and verifier report.
3. **Rollback**: Immediately roll back to the previous immutable image.
4. **Investigate**: The verifier uses sweep-line checks independent of CP-SAT — a failure means the solver produced a result that violates constraints.

---

## Scenario 6: Spring-to-Agent Authentication Fails

**Symptom**: Spring Boot receives 401 from the Agent.

**Diagnosis**:

```bash
# Check Agent logs for auth failures
docker logs cooking-plan-agent | jq 'select(.code == "INVALID_INTERNAL_CREDENTIAL")'

# Verify token matches
docker exec cooking-plan-agent env | grep INTERNAL_SERVICE_TOKEN
```

**Response**:

1. **Token mismatch**: Verify `COOKING_PLAN_INTERNAL_SERVICE_TOKEN` is identical on both Spring Boot and Agent sides.
2. **Credential header**: The Spring Boot caller sends `Authorization: Bearer <token>` to the v1 compat endpoint (`/internal/v1/cooking-plans/generate`); the legacy native endpoint (`/internal/v1/agents/cooking-plan/generate`) still expects `X-Internal-Token`. Both credentials are compared against the same `internal_service_token` using constant-time comparison.
3. **Token strength (P0-08)**: In non-local environments the service token must be at least `COOKING_PLAN_MIN_SERVICE_TOKEN_LENGTH` (default 16) characters — shorter tokens are rejected with `INSUFFICIENT_CREDENTIAL_STRENGTH`.
4. **CORS (P0-08)**: Internal APIs do NOT enable CORS by default. If a browser caller needs cross-origin access, set `COOKING_PLAN_CORS_ALLOW_ORIGINS` to an explicit comma-separated allow-list. A wildcard (`*`) is rejected at startup.
5. **Network**: Verify Spring Boot can reach the Agent container on the internal network.

**Rollback**: Not required — fix configuration and restart.

---

## Scenario 7: Contract Version Mismatch

**Symptom**: Spring Boot receives 422 validation errors or unexpected response shapes.

**Diagnosis**:

```bash
# Check schema_version in requests
docker logs cooking-plan-agent | jq 'select(.schema_version) | {schema_version, request_id}'

# Compare OpenAPI schemas
curl http://agent:8000/openapi.json | jq '.components.schemas.GeneratePlanRequest.properties | keys'
```

**Response**:

1. **Schema version field**: If `schema_version` in requests does not match what the Agent expects, update Spring Boot to send the correct version (`"1.0"`).
2. **Compat contract**: The v1 compat endpoint (`/internal/v1/cooking-plans/generate`) requires `contractVersion: "cooking-agent-v1"`. Unsupported versions fail fast with `status: "FAILED"`. Its request/response shape mirrors the Java `AgentCookingRequest`/`AgentCookingResponse` records exactly — do not add extra fields (Java's `fail-on-unknown-properties` treats them as `MALFORMED_JSON`).
3. **Response shape change**: If a new Agent deployment changes the PlanResponse schema, deploy Spring Boot first (it validates Agent responses), then deploy the Agent.

**Rollback**: Roll back the Agent to the previous image. Database rollback belongs to Spring Boot.

---

## Scenario 8: Deployment Rollback

**Procedure**:

```bash
# 1. Identify the previous working image digest
docker images cooking-plan-agent --digests

# 2. Stop current container
docker stop cooking-plan-agent

# 3. Start from previous immutable image
docker run -d \
  --name cooking-plan-agent \
  --env-file /path/to/.env \
  -p 8000:8000 \
  <previous-image-digest>

# 4. Verify health
curl http://localhost:8000/health/ready

# 5. Check logs for errors
docker logs cooking-plan-agent --tail 20
```

**Important**:
- The Agent does not own any database migrations — database rollback belongs to Spring Boot.
- The Agent image is immutable (no volume mounts for code). Rollback is a simple container swap.

---

## Scenario 9: Overload / Backpressure (P1-03)

**Symptom**: `POST /internal/v1/agents/cooking-plan/generate` returns HTTP 503 with
`{"detail": {"code": "OVERLOADED", ...}}` and a `Retry-After` header.

**Capacity parameters** (process-level, single instance):

| Setting | Default | Meaning |
|---------|---------|---------|
| `COOKING_PLAN_MAX_ACTIVE_REQUESTS` | 20 | Max requests running concurrently |
| `COOKING_PLAN_MAX_QUEUED_REQUESTS` | 100 | Max requests waiting for a slot |
| `COOKING_PLAN_QUEUE_TIMEOUT_SECONDS` | 5.0 | Max wait before a queued request is rejected |

The limiter is **process-level**. Horizontal scaling limits are a separate
concern (P3-02).

**Diagnosis**:

```bash
# Watch active/queued/rejected metrics — the probe bypasses the limiter
curl -s localhost:8000/health/load | jq .
```

**Alert thresholds**:

- **Warn (P2)** when `active >= 0.8 * max_active` for > 30 s, or when
  `rejected_total` grows.
- **Page (P1)** when `queued` stays above `max_queued / 2` for > 60 s — the
  service is persistently saturated and callers are being turned away.

**Response**:

1. **Check provider health**: 503s usually mean upstream (LLM / solver) is
   slow; verify `llm_overall_timeout_seconds` and solver budget aren't
   exceeded for every request.
2. **Scale horizontally**: add another Agent instance — the limiter is
   per-process, so capacity grows with instances (dedupe/coordination via
   P3-02).
3. **Tune limits**: raise `max_active_requests` only with evidence of headroom
   in provider quota and CPU; never set `max_queued_requests` unbounded.

**Rollback**: reducing limits is a config change (`docker compose` env) — no
image rollback required.

---

## Scenario 10: Regional Safety Policy Unavailable (P3-04)

**Symptom**: requests FAIL with error code `SAFETY_POLICY_UNAVAILABLE`.

**Diagnosis**:

```bash
# Check which region/version triggered the rejection
docker logs cooking-plan-agent | jq 'select(.error_code == "SAFETY_POLICY_UNAVAILABLE") | {request_id, message}'

# Show registered policy packs
uv run python -c "from cooking_plan_agent.safety.policies import *; from cooking_plan_agent.safety.policy import supported_regions, latest_version; print('regions:', supported_regions()); print({r: latest_version(r) for r in supported_regions()})"
```

**Response** — the rejection is deliberate (D6: no silent fallback). Fix the
caller, never "relax" the policy:

| Trigger | Action |
|---------|--------|
| Unknown region in request `region` | Client must send a supported code (`US`/`SG`). Unknown regions never fall back to another pack. |
| Unknown `COOKING_PLAN_SAFETY_POLICY_VERSION` | Set the version to a registered one (or remove it to use the latest). |
| Policy not yet effective (`effective_at` in the future) | Either the pack is pre-release (do not use) or the environment clock is wrong. |
| Missing sources | A pack with no official sources is a packaging defect — it must not ship; fix in a code review, not at runtime. |

**Config reference**:

| Setting | Default | Meaning |
|---------|---------|---------|
| `COOKING_PLAN_SAFETY_POLICY_REGION` | `US` | Deployment-level default region; request `region` overrides it |
| `COOKING_PLAN_SAFETY_POLICY_VERSION` | (latest) | Explicit policy version; old versions stay registered for audit |

**Rollback / policy update**: a threshold change ships as a NEW policy version
(never mutating an existing one) so historical plans and checkpoints remain
auditable. Rolling back the service image must not delete any policy version
already used by historical plans.
