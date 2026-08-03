# Branch 02 - Recommendation Agent Service Foundation

## Branch metadata

- **Proposed branch:** `feat/recommendation-agent-foundation`
- **Base dependency:** Branch 01 merged
- **Service root:** `agent-service/app/agents/recommendation/`
- **Packaging:** standalone Python 3.13 `uv` project mirroring Cooking Agent
- **Private API:** Backend-facing Agent v2 only, subject to U-03

## Purpose

Turn the empty Recommendation Agent placeholder into a strict private FastAPI
service shell with project configuration, Agent v2 models, service auth,
correlation, request bounds, safe errors, liveness/readiness, dependency
injection, tests, and root-level CI. No ranking or inference call lands here.

## Related source-document sections

- Inventory: **Intelligence runtime sources** and **Adjacent workflow boundary
  documents**.
- Implementation design: **Trust and ownership boundaries**,
  **Recommendation Agent design**, **Security, privacy, and safety**, and
  **Reliability and observability**.
- Delivery plan: Phase 4 tasks 1, 3-5 and Agent-specific exit gates.
- Local runtime architecture: Agent, graph state, tool boundary, failure modes,
  observability, and deployment boundary.

## Prerequisites and dependencies

- Agent v2 schemas, auth/route behavior, and error envelope are accepted.
- U-03 confirms v2-only versus an explicit v1 boundary adapter.
- U-07 confirms the standalone nested package layout.
- U-09 fixes health route names; this plan proposes repository convention
  `/health/live` and `/health/ready`.
- U-10 accepts Bearer auth for local/test and identifies production binding.

## Scope

- independent Python project/lockfile;
- strict Pydantic Agent v2 request/response/failure mirrors;
- internal Bearer auth and safe correlation propagation;
- body/candidate/text/concurrency/backpressure bounds;
- safe typed exception/error envelope;
- structured redacted logging foundation;
- app factory/lifecycle, liveness, and not-ready behavior;
- OpenAPI/contract/security/architecture tests and root CI.

### Explicit non-scope

- No inference client, Agent graph, diversity, reason, explanation, LLM, or
  success generation.
- No database/backend tool, public API, JWT, persistence, training, model
  package, Cooking, Chatbot, Search, Summary, or web route.
- No fake production recommendation output; readiness remains false until Plans
  03-05 wire the complete workflow.

## Concrete files

```text
agent-service/app/agents/recommendation/
  .dockerignore
  .gitignore
  .python-version
  README.md
  pyproject.toml
  uv.lock
  src/recommendation_agent/
    __init__.py
    main.py
    api/
      __init__.py
      dependencies.py
      errors.py
      health.py
      router.py
    application/
      __init__.py
      ports.py
      service.py
    config/
      __init__.py
      settings.py
    domain/
      __init__.py
      errors.py
      models.py
    observability/
      __init__.py
      logging.py
      redaction.py
    schemas/
      __init__.py
      agent_v2.py
  tests/
    conftest.py
    unit/test_settings.py
    unit/test_backpressure.py
    unit/test_redaction.py
    contract/test_agent_contract_v2.py
    contract/test_error_contract.py
    contract/test_openapi.py
    integration/test_app_lifecycle.py
    security/test_internal_auth.py
    security/test_payload_limits.py
    security/test_log_privacy.py
.github/workflows/recommendation-agent-ci.yml
```

Do not import `cooking_plan_agent`; adapt generic patterns so each service is
independently versioned and cannot route into another workflow.

## Interfaces and configuration

| Symbol | Required contract |
| --- | --- |
| `recommendation_agent.main:create_app` | Side-effect-free app factory; lifecycle installs settings/services and closes resources. |
| `Settings` | Strict `RECOMMENDATION_AGENT_` prefix; secrets use `SecretStr`; invalid non-local config fails startup. |
| `require_internal_service` | Frozen Bearer scheme, constant-time compare, stable safe errors, no credential logging. |
| `extract_correlation` | Accepts bounded log-safe request/trace header or generates safe local correlation. |
| `RecommendationAgentService#execute` | Accepts strict v2 request and delegates to an injected workflow port; not-ready until real workflow is installed. |
| `AgentWorkflow#run` | Async protocol defined now and implemented by Plan 04. |

### Proposed settings

Under `RECOMMENDATION_AGENT_`:

- `APP_ENV`, `LOG_LEVEL`;
- `INTERNAL_SERVICE_TOKEN`, `MIN_SERVICE_TOKEN_LENGTH`;
- `SUPPORTED_CONTRACT_VERSIONS`;
- `MAX_REQUEST_BYTES`, `MAX_RESPONSE_BYTES`, `MAX_CANDIDATES` (hard cap 100);
- `MAX_ACTIVE_REQUESTS`, `MAX_QUEUED_REQUESTS`, `QUEUE_TIMEOUT_SECONDS`;
- `CORS_ALLOW_ORIGINS` empty by default; wildcard rejected.

Inference URL/token/timeouts are added only in Plan 03. No LLM configuration.

### Routes

- `POST /internal/v1/recommendations/generate`: strict Agent v2 body, Bearer
  auth, canonical safe not-ready/failure response until workflow is installed.
- `GET /health/live`: process/event-loop only.
- `GET /health/ready`: `503` until the complete Agent workflow and compatible
  inference client are installed; safe check names only.

## Ordered implementation tasks

1. Create Python 3.13 `uv` project with FastAPI, Pydantic v2/settings and
   development gates aligned with Cooking: Ruff, strict mypy, pytest,
   pytest-asyncio/cov, Hypothesis.
2. Use `src/recommendation_agent` packaging and deterministic `uv.lock`; add
   ignores for `.env`, caches, virtualenv, logs, and generated OpenAPI.
3. Define strict immutable base models (`extra="forbid"`, strict scalars,
   whitespace/length/cardinality bounds) and Agent v2 Pydantic mirrors.
4. Validate canonical JSON Schema fixtures and Pydantic round-trip/camelCase
   output in contract tests.
5. Implement strict settings/environment validation and cached construction;
   secrets never appear in repr or validation responses.
6. Adapt constant-time Bearer auth from Cooking's verified pattern. Missing,
   wrong scheme, bad credential, and weak non-local token use accepted codes.
7. Validate correlation/header length/characters to prevent log injection and
   echo only safe correlation headers.
8. Enforce request bytes before JSON parsing, candidate cap in schema, response
   bytes before send, and bounded text/collections throughout.
9. Add active/queued request lease with timeout/503/Retry-After; health bypasses
   backpressure.
10. Implement canonical error handlers. Validation details may name safe JSON
    pointers but never rejected model keys/features/body/token.
11. Define `AgentWorkflow` port and not-ready application service. Production
    must never emit fixture candidates.
12. Build app factory/lifespan and register private generation plus health
    routes. CORS remains disabled unless explicit origins are accepted.
13. Implement deep redaction/structured JSON logging for secret/key/feature/
    URI/raw-exception canaries.
14. Add OpenAPI export/check and architecture tests prohibiting DB/training/
    model-package/web/LLM/Cooking/Chat imports.
15. Add root workflow jobs for frozen sync, format, lint, mypy, unit/contract/
    security tests, coverage, OpenAPI, architecture, and eventual Docker gate
    placeholder disabled until Plan 06.
16. Document exact commands, config names, private boundary, and expected
    readiness `503` before workflow completion.

## Test requirements

- every Branch 01 valid/negative Agent fixture;
- strict unknown fields/coercions, duplicate/over-100 candidates, oversized
  body/text, invalid IDs/versions/time/key shapes;
- missing/wrong auth, weak non-local secret, unsafe correlation;
- queue saturation/cancellation/release and health bypass;
- live `200`, ready `503`, generation canonical not-ready;
- import/app construction performs no network/file/model/DB call;
- captured logs/errors contain none of token/model-key/feature/body canaries;
- OpenAPI lists only Agent private route and health routes.

## Acceptance criteria

- [ ] `uv sync --frozen --dev` succeeds from the Agent project root.
- [ ] Agent v2 canonical fixtures parse/serialize exactly.
- [ ] Auth, body, candidate, response, and concurrency bounds are enforced.
- [ ] Service cannot return a recommendation without injected completed workflow.
- [ ] Live/ready semantics are safe and deterministic.
- [ ] Root CI executes the Agent foundation gates.
- [ ] No inference/ML/model-package implementation dependency is introduced.

## Commit plan

1. `build(recommendation): create dedicated Agent project`
2. `feat(recommendation): enforce Agent v2 boundary and auth`
3. `feat(recommendation): bound requests errors and health`
4. `test(recommendation): cover contracts security and lifecycle`
5. `ci(recommendation): require Agent service quality gates`
6. `docs(recommendation): document private service shell`

## Verification

```powershell
Push-Location agent-service/app/agents/recommendation
uv sync --frozen --dev
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest tests/unit tests/contract tests/security -v
uv run pytest tests/integration -v
uv run pytest --cov=recommendation_agent --cov-report=term-missing
Pop-Location
```

Expected: all commands exit `0`; generation is safely not-ready and no sensitive
canary appears in logs/errors.

## Pull Request hand-off

- **Title:** `feat(recommendation): establish private Agent service foundation`
- Include contract checksums, route/auth decision, config names, OpenAPI routes,
  exact results, redaction evidence, and readiness limitation.

## Rollback and unresolved items

Rollback is the previous image/commit; no persistent state exists. U-03/U-07/
U-09/U-10 must be resolved before their paths/contracts freeze.

