# Branch 06 - Recommendation Agent Hardening and Container

## Branch metadata

- **Proposed branch:** `chore/recommendation-agent-hardening`
- **Base dependency:** Branch 05 merged
- **Scope:** Recommendation Agent runtime only
- **Exit:** immutable private Agent image with required CI/security/operations gates

## Purpose

Harden the completed Agent workflow for private deployment: complete
observability, enforce capability/network/configuration limits, build a
non-root immutable container, prove secret/log privacy, wire root CI scans, and
document outage/rollback operations without adding inference or ML code.

## Related source-document sections

- Implementation design: **Security, privacy, and safety**, **Reliability and
  observability**, **Verification strategy - Intelligence**, and Agent portions
  of **Rollout and rollback**.
- Delivery plan: Phase 4 exit gate; Agent-specific Phase 5 security/latency/
  rollback tasks; staged-release observability requirements.
- Local runtime architecture: observability/deployment boundary.

## Prerequisites and dependencies

- Complete production-wired Agent workflow from Plan 05.
- U-04 latency/timeout/alert budgets accepted.
- U-09 health routes and U-10 production auth/TLS binding accepted.
- Deployment owner provides private network/secret injection conventions.
- Inference remains external; only its configured origin and health/contract
  behavior are in Agent scope.

## Scope

- structured logs and recursive redaction;
- low-cardinality metrics for Agent stages/dependency/failures/policies;
- health/readiness and graceful shutdown/backpressure behavior;
- private network and outbound-origin restrictions;
- dependency/import/capability tests;
- non-root minimal container and read-only-runtime checks;
- root CI: quality, contracts, security, secret/dependency/license, SBOM,
  container scan/build/smoke;
- Agent deployment/runbook/configuration/rollback documentation.

### Explicit non-scope

- No inference container/runtime, model/package registry, ML pipeline, training,
  dataset, model card, or model rollback.
- No public ingress/client API, Backend fallback implementation, database,
  cloud provider expansion, or new product feature.
- No LLM explanation.

## Concrete files

```text
agent-service/app/agents/recommendation/
  Dockerfile
  .dockerignore
  runbook.md
  src/recommendation_agent/
    main.py
    api/backpressure.py
    api/health.py
    observability/logging.py
    observability/metrics.py
    observability/redaction.py
    config/settings.py
  tests/
    security/test_app_security.py
    security/test_dependencies.py
    security/test_error_privacy.py
    security/test_log_redaction.py
    security/test_network_capabilities.py
    smoke/test_docker_smoke.py
    performance/test_agent_budget.py
    contract/test_openapi.py
.github/workflows/recommendation-agent-ci.yml
.github/dependabot.yml
deployment/docker/recommendation-agent.env.example
deployment/local/recommendation-agent.compose.yaml
docs/operations/recommendation-agent.md
```

Environment example files contain names/placeholders only, never usable secret
values. Provider-specific deployment manifests are `Proposed` only after a
provider is selected.

## Observability contract

### Correlation/log fields

Allowed structured fields:

- request/session/trace/agent-trace correlation IDs;
- Agent/inference/feature/key/diversity/reason/template versions when bounded;
- candidate/result counts;
- stage and total durations;
- success/failure category and readiness state.

Prohibited fields:

- service token/authorization header;
- raw user/group/place/meal/offering/database ID;
- model user/meal/offering key;
- full or partial feature vector/body/downstream body;
- preference/allergen/dietary details, free text, exception repr with request
  content, signed URL/query, environment values.

### Proposed metrics

Use a project-approved Prometheus/OpenTelemetry-compatible library with pinned
dependency. Exact metric names are **Proposed**:

- `foodmind_recommendation_agent_requests_total{result}`;
- `foodmind_recommendation_agent_request_duration_seconds`;
- `foodmind_recommendation_agent_stage_duration_seconds{stage}`;
- `foodmind_recommendation_agent_inference_calls_total{result}`;
- `foodmind_recommendation_agent_inference_duration_seconds`;
- `foodmind_recommendation_agent_failures_total{failure_code}`;
- `foodmind_recommendation_agent_input_candidates` and output result count;
- `foodmind_recommendation_agent_readiness`;
- active compatibility/policy version info with reviewed bounded labels.

Never label by request/session/trace/candidate/model key, explanation, URL, or
arbitrary version supplied by an untrusted request. Histograms/buckets follow
U-04 budgets.

## Container/runtime contract

- Python 3.13 slim image and locked production dependencies.
- Copy only `pyproject.toml`, `uv.lock`, required README/metadata, and `src/`;
  no tests, `.env`, Git metadata, caches, fixtures, local evidence, or secrets.
- Numeric/non-root service user with no home/shell dependency.
- One private port, bounded worker/concurrency choice documented.
- Healthcheck uses liveness; traffic routing uses readiness.
- Graceful termination stops accepting work, allows only accepted drain time,
  closes HTTP client, and does not extend request deadlines.
- Read-only root filesystem compatible; only explicit temp path writable if
  library/runtime requires it.
- No model mount, database credential, training volume, or Docker socket.
- Image labels/digest link source revision; deployment uses immutable digest.

## Ordered implementation tasks

1. Audit every log/exception/metric call across Agent code. Replace unsafe
   interpolation with allow-listed structured fields and recursive fail-closed
   redaction.
2. Add correlation context propagation across API, graph nodes, and inference
   client without storing full request/state in logging context.
3. Instrument total, graph-stage, inference, validation, selection, reason, and
   rendering timings plus stable result/failure counters.
4. Bound metric labels and add tests rejecting high-cardinality or request-
   controlled labels.
5. Finalize readiness: settings/auth/workflow compiled, policy versions loaded,
   and compatible inference dependency state. Liveness remains dependency-free.
6. Finalize backpressure/graceful shutdown. New generation requests receive a
   safe `503`/Retry-After while shutdown; health remains probeable.
7. Enforce configured inference origin only. Add a transport/network test that
   denies arbitrary hosts, redirects to unapproved origins, proxy-from-env if
   not accepted, and non-approved schemes.
8. Add architecture/dependency checks for prohibited DB, SQL driver, filesystem
   tool, browser/web-search, subprocess/code-execution, training/model-loader,
   Cooking/Chat/Search/Summary, and LLM imports.
9. Review configuration validation: non-local auth strength, HTTPS/private
   origin policy, timeout coherence, body/candidate/concurrency caps, no wildcard
   CORS, no debug/docs exposure if environment policy disables it.
10. Build minimal non-root Dockerfile from frozen lock. Add `.dockerignore` and
    image inspection tests for user, files, env, writable paths, ports, command,
    health, and no secret canary.
11. Add local compose with Agent plus an external-contract fake inference
    service only for development/test. The fake is fixture-driven and not an
    inference implementation or production image.
12. Extend root CI with independent format/lint/mypy/unit/contract/security/
    architecture/OpenAPI jobs plus dependency/license audit, secret scan, SBOM,
    image build/scan, non-root/read-only smoke, and merge gate.
13. Pin third-party CI actions/tools according to repository policy and use
    least-privilege workflow permissions; no deployment credentials on pull
    requests.
14. Add performance test for 1/10/100 candidates against fixture inference
    transport. Report Agent overhead separately from downstream delay and gate
    against U-04 accepted budget.
15. Write runbook: startup/readiness, config names, inference outage, contract
    mismatch, timeout spike, unsafe-reason spike, log privacy, image rollback,
    Backend fallback verification, and escalation owner placeholders.
16. Add deployment documentation for private ingress, TLS/service credential
    injection/rotation, egress allow-list, resource/time limits, dashboards,
    alerts, and retained evidence. Do not include secret values.

## Test requirements

### Observability/privacy

- canaries nested in request/features/model keys/token/exception/URL query never
  appear in logs, metrics, traces, error bodies, or health;
- allowed correlation/version/result fields remain traceable;
- failure at every graph/client stage increments one correct stable counter;
- labels remain from fixed enum/allow-list.

### Security/capability

- missing/wrong/weak auth; wildcard CORS; unsafe origin/redirect/proxy;
- oversized body/response, 101st candidate, queue saturation, shutdown request;
- no prohibited imports and no request-time filesystem/subprocess/DB/web/LLM;
- common dependency/secret/license/static/container scan gates run.

### Container/smoke

- image builds from clean checkout/frozen lock;
- runs non-root and under read-only root filesystem;
- contains no `.env`, tests, Git, cache, token canary, fixture bodies, or model;
- live/ready/generation contract behaves against fake inference;
- SIGTERM drains within accepted window and closes client;
- container uses only configured private inference origin.

### Performance

- Agent-only overhead and end-to-end fixture 1/10/100 candidate percentiles;
- no unbounded memory/task/client growth under accepted concurrency;
- deadline/fallback behavior remains correct under saturation.

## Acceptance criteria

- [ ] Structured logs/metrics are useful and contain no prohibited canary.
- [ ] All private/auth/origin/payload/concurrency/deadline limits are tested.
- [ ] Non-root read-only image passes health, contract, security, and smoke gates.
- [ ] SBOM and dependency/secret/container scans have no unresolved release-
  blocking finding.
- [ ] Agent overhead/total latency meets U-04 budget on recorded environment.
- [ ] Runbook demonstrates Agent outage still permits Backend fallback.
- [ ] No inference or ML source/runtime is added.

## Commit plan

1. `feat(recommendation): add safe Agent metrics and tracing`
2. `test(security): prove Recommendation Agent capability isolation`
3. `chore(container): package non-root Recommendation Agent`
4. `ci(recommendation): require security SBOM and container gates`
5. `test(recommendation): record bounded Agent performance`
6. `docs(operations): add Recommendation Agent runbook`

## Verification

```powershell
Push-Location agent-service/app/agents/recommendation
uv sync --frozen --dev
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest tests/unit tests/contract tests/integration tests/security -v
uv run pytest tests/performance/test_agent_budget.py -v
docker build --tag foodmind-recommendation-agent:ci .
docker run --rm --read-only --user 10001:10001 `
  foodmind-recommendation-agent:ci
Pop-Location
```

Also run the pinned dependency/license/secret/SBOM/container scanners from the
root workflow. Expected: zero failed tests, no release-blocking finding, non-
root/read-only smoke success, and recorded latency percentiles within U-04.

## Pull Request hand-off

- **Title:** `chore(recommendation): harden private Agent runtime`
- Include image digest, contract/policy versions, config names, exact commands,
  scan/SBOM summaries, redaction proof, performance environment/percentiles,
  readiness/outage demonstration, and open findings with owner/expiry.

## Rollback and unresolved items

Rollback deploys the prior Agent image/digest or disables Backend Agent v2.
There is no Agent persistence/migration. U-04/U-09/U-10 must be resolved before
release; provider-specific deployment remains outside until selected.
