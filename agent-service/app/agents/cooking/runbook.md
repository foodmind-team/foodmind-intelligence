# Cooking Plan Agent — Operations Runbook

> Handbook 12.10: operational steps for 8 failure scenarios.
> Version: 1.0 | Last updated: 2026-07-31

---

## Scenario 1: High FAILED Response Rate

**Symptom**: `/internal/v1/agents/cooking-plan/generate` returns FAILED status for > 5% of requests.

**Diagnosis**:

```bash
# Check logs for error codes
docker logs cooking-plan-agent 2>&1 | jq 'select(.level == "ERROR") | {code: .error_code, node: .node, message: .message}'

# Check structured metrics (if Prometheus is wired)
curl localhost:8000/health/ready
```

**Response**:

| Error Code | Action |
|-----------|--------|
| `SCHEDULE_VERIFICATION_FAILED` | **P1 alert** — indicates solver or verifier bug. Capture the failing request payload, solver status, and verifier issues. Roll back to previous image if rate > 1%. |
| `EXTERNAL_PROVIDER_UNAVAILABLE` | Check LLM/Search provider status. If provider is down, service degrades to local inference — this is expected. |
| `INTERNAL_ERROR` | Check logs for unexpected exceptions. May indicate a code defect. |

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

**Note**: Solver timeouts are expected for very large plans (6 recipes × 100 tasks). The system falls back to `NEEDS_CONFIRMATION` or `INFEASIBLE`.

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
2. **Header missing**: Confirm Spring Boot sends `X-Internal-Token` header on every request.
3. **Network**: Verify Spring Boot can reach the Agent container on the internal network.

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
2. **Response shape change**: If a new Agent deployment changes the PlanResponse schema, deploy Spring Boot first (it validates Agent responses), then deploy the Agent.

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
