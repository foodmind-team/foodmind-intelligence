# FoodMind Intelligence

FoodMind Intelligence contains FoodMind's private AI and recommendation runtime: Chatbot, Cooking Plan, Recommendation Agent, and model Inference services. It is not a public API. Web and Android must call only FoodMind Backend, which authenticates the user, supplies authorised context, validates results, and decides what can be returned.

## Live deployment

The user-facing FoodMind application is deployed at [https://13.229.2.154.sslip.io/](https://13.229.2.154.sslip.io/). Its AI features are reached through the authenticated Backend API; the private services in this repository do not have public endpoints.

## Services

| Service | Local port | Responsibility |
| --- | --- | --- |
| Chatbot | 8001 | Grounded, Backend-authorised conversation |
| Inference | 8002 | Validates a model package and scores authorised candidates |
| Cooking | 8003 | Cooking-plan and recipe-import workflow |
| Recommendation | 8004 | Bounded recommendation orchestration and explanations |

The host ports are for local diagnostics only. The integrated Infra Compose stack keeps these services private.

## Prerequisites

- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- Docker with Compose
- A sibling foodmind-ml checkout to build the local model package
- Optional: a DeepSeek-compatible provider key for enhanced generation

Without an LLM key, use the Infrastructure stack for deterministic fallback development. Do not commit real provider keys or service tokens.

## Quick start: local private services

From a workspace containing sibling foodmind-ml and foodmind-intelligence directories:

~~~bash
git clone https://github.com/foodmind-team/foodmind-ml.git
git clone https://github.com/foodmind-team/foodmind-intelligence.git
cd foodmind-ml
uv sync --frozen --dev
uv run python scripts/build_runtime_package.py --output .tmp/runtime/model-package
cd ../foodmind-intelligence/agent-service/app/agents
cp .env.example .env
docker compose up --build
~~~

Set DEEPSEEK_API_KEY in the ignored .env only when deliberately enabling the optional LLM paths. Then start the Backend with matching local service-token values from its .env.example. For the full authenticated product journey, prefer [FoodMind Infrastructure](https://github.com/foodmind-team/foodmind-infra), which composes Backend, PostgreSQL, the runtime, and the model package together.

## Local deployment

For a complete local FoodMind deployment, start
[FoodMind Infrastructure](https://github.com/foodmind-team/foodmind-infra).
It builds the model package and runs these services on a private Docker network
with the matching Backend service tokens. This repository's Compose file is for
private runtime diagnosis, not for exposing an API directly to Web or Android.

To run the private services independently on Windows PowerShell, first create
the model package in a sibling ML checkout, then start the agent Compose file:

```powershell
Set-Location ..\foodmind-ml
uv sync --frozen --dev
uv run python scripts/build_runtime_package.py --output .tmp/runtime/model-package
Set-Location ..\foodmind-intelligence\agent-service\app\agents
Copy-Item .env.example .env
# Add DEEPSEEK_API_KEY to .env only when the optional LLM paths are required.
docker compose -f docker-compose.yml up --build -d
Invoke-WebRequest http://localhost:8002/health/ready
```

The diagnostic listeners are Chatbot `8001`, Inference `8002`, Cooking `8003`,
and Recommendation `8004`. They are private service interfaces: a Backend must
use matching service tokens and clients must continue to use Backend
`/api/v1`. Inspect failures with `docker compose -f docker-compose.yml logs -f`
and stop the diagnostic stack with `docker compose -f docker-compose.yml down`.
Never commit `.env`, provider keys, or non-local service tokens.

## Run a component directly

Each service is an independent Python project. For example, start the private inference service after creating the model package:

~~~bash
cd inference-service
uv sync --frozen --dev
INFERENCE_MODEL_PACKAGE_DIR=../../foodmind-ml/.tmp/runtime/model-package INFERENCE_INTERNAL_SERVICE_TOKEN=local-inference-only uv run uvicorn app.main:app --host 127.0.0.1 --port 8002
~~~

Its liveness and readiness endpoints are /health/live and /health/ready. Its scoring endpoint is private and requires a bearer service token.

## Verify

Run checks from each affected component directory. The baseline pattern is:

~~~bash
uv sync --frozen --dev
uv run ruff format --check .
uv run ruff check .
uv run pytest
~~~

Cooking and Recommendation also have strict type checks configured; run the relevant uv run mypy command before a pull request. Tests must cover structured-schema validation, timeouts, deterministic fallbacks, private-service authentication, and permission-safe grounding.

## Repository layout

~~~text
agent-service/app/agents/  Chatbot, Cooking, and Recommendation services plus local Compose
inference-service/         FastAPI model-package validation and scoring service
contracts/                 Versioned private agent and inference contracts
deployment/                Container and environment templates
docs/                      Architecture, planning, test evidence, and runbooks
artifacts/                 Non-sensitive local artifacts and fixtures
~~~

## Security

Agents never access PostgreSQL directly and must not bypass Backend authorisation. Keep inputs and outputs schema-validated, restrict Backend tools to an allow-list, use short-lived service credentials, and avoid logging prompts, user data, tokens, or provider keys.

## License

No open-source license is currently included in this repository. Obtain permission from the maintainers before redistributing or reusing the code.
