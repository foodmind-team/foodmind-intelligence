# FoodMind Recommendation Agent

Private, bounded FastAPI service for Recommendation Agent v2. The package is
independent of the Cooking Agent and contains no model training/loading,
database, web, filesystem, or arbitrary tool capability. DeepSeek is an
optional explanation renderer and is never a ranking provider.

## Local commands

```powershell
uv sync --frozen --dev
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest -v
uv run uvicorn recommendation_agent.main:app --host 127.0.0.1 --port 8004
```

The private generation boundary is
`POST /internal/v1/recommendations/generate` with
`Authorization: Bearer <service-token>`. The production app installs the
complete bounded workflow during lifespan startup. `/health/ready` is `200`
only while configuration, the inference client, immutable policies, workflow,
and shutdown state are ready; otherwise it is a safe `503`. `/health/live`
remains `200` and bypasses backpressure.

For local end-to-end migration testing only, setting
`RECOMMENDATION_AGENT_ENABLE_V1_COMPATIBILITY=true` exposes
`POST /internal/compat/v1/recommendations/generate`. That isolated route maps
the Backend's frozen v1 envelope into the canonical v2 workflow and maps the
validated result back to v1. Configuration rejects this route outside
local/test/CI, and the canonical v2 contract remains unchanged.

All configuration uses the `RECOMMENDATION_AGENT_` prefix:

Set `LLM_ENABLED=true`, `LLM_BASE_URL=https://api.deepseek.com`, and
`LLM_MODEL=deepseek-chat`. The service loads `DEEPSEEK_API_KEY` from the shared
`app/agents/.env`; `RECOMMENDATION_AGENT_LLM_API_KEY` is the explicit override.
Inference remains the only ranking provider. After deterministic selection and
reason derivation, DeepSeek may render one bounded explanation per selected
opaque candidate ID from allow-listed reason facts. Invalid or unavailable LLM
output falls back to deterministic wording without discarding ML results.

The canonical `/internal/v1/recommendations/generate` route accepts the
Backend's v1 envelope during migration and translates it into the same strict
v2 workflow. The `/internal/compat/...` alias remains local-only and is no
longer needed by the Backend configuration.

- `APP_ENV`, `LOG_LEVEL`, `INTERNAL_SERVICE_TOKEN`,
  `MIN_SERVICE_TOKEN_LENGTH`;
- `SUPPORTED_CONTRACT_VERSIONS`, `MAX_REQUEST_BYTES`,
  `MAX_RESPONSE_BYTES`, `MAX_CANDIDATES`;
- `MAX_ACTIVE_REQUESTS`, `MAX_QUEUED_REQUESTS`,
  `QUEUE_TIMEOUT_SECONDS`; and
- `CORS_ALLOW_ORIGINS` (empty by default; wildcard forbidden).

The only external capability is the separately owned inference runtime. Its
origin/path and Bearer credential are deployment settings
(`INFERENCE_BASE_URL`, `INFERENCE_ENDPOINT_PATH`,
`INFERENCE_SERVICE_TOKEN`). Connect/pool/total timeouts, response bytes,
connection count, deadline guard, and minimum ingress budget are bounded by
the corresponding `INFERENCE_*`, `DEADLINE_GUARD_MS`, and
`MIN_DEADLINE_BUDGET_MS` settings. The Agent sends exactly one
`recommendation-inference-v1` request and never retries. Inference/model/ML
implementation and model-package ownership remain outside this package.

Non-local environments require an explicit strong service token. Secrets,
model keys, feature values, request bodies, raw exceptions, and credentialed
URIs must never be logged or returned in errors.

Production result shaping is deterministic under
`recommendation-diversity-v1`, `recommendation-reasons-v1`, and
`recommendation-template-v1`. It preserves the confidence lead within the
frozen tie band, selects only evidence-supported Personal, Exploratory, and
Group-inspired candidates, and derives at most two allow-listed reasons.
No LLM participates in ranking, selection, or reason derivation. Explanation
rendering is constrained to approved reason vocabulary and deterministic
templates remain the failure path.

For the deterministic private-boundary smoke and release-evidence checksum
check, run:

```powershell
uv run python scripts/smoke_private_agent.py --mode fixture
uv run python scripts/verify_release_evidence.py
```

These are local fixture gates, not staged release or rollback approval. See
the root `docs/testing/recommendation-agent-evidence/README.md` index for the
external gates that remain required.
