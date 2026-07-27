# FoodMind Intelligence

FoodMind Intelligence contains FoodMind's private runtime AI components:

- A controlled Multi-Agent service
- A model-inference service that consumes versioned artifacts released by `foodmind-ml`

> **Current status:** directory framework only. No FastAPI application, Agent graph, inference endpoint, model loader, or deployment configuration has been implemented.

## Repository Role

This repository owns runtime intelligence:

- LangGraph state machines
- Five controlled Agent responsibilities
- Allow-listed tool invocation
- Pydantic request and response schemas
- Agent tracing and route decisions
- Runtime feature validation
- Loading a released model package
- UserCF/ItemCF/LR inference using the released package design
- Model-version and fallback metadata
- Private service health and readiness behaviour

This repository does not own:

- Offline model training or evaluation
- Raw or processed training data
- Public client APIs
- User authentication or resource authorisation
- PostgreSQL access
- Business persistence
- Android or Web code

Offline training belongs to `foodmind-ml`. The Spring Boot backend remains the system-facing authority.

## Runtime Boundary

```text
Android/Web
    │
    ▼
Spring Boot Backend
    │ authorised private request
    ▼
Agent Service ───────> Inference Service
    │                       │
    │ narrow tools          │ released model package
    ▼                       ▼
Spring Boot Backend     Model registry/storage
```

Neither runtime service is public. The Agent service may call only explicit backend tools and the private inference service. It must not connect to PostgreSQL.

## Relationship to `foodmind-ml`

```text
foodmind-ml
  → train and evaluate
  → package model + schemas + metadata + checksum
  → publish immutable release
  → foodmind-intelligence validates and loads release
  → inference response reports exact model version
```

Runtime code must never silently retrain or modify a released model artifact.

## Repository Structure

```text
foodmind-intelligence/
├── .github/workflows/
├── agent-service/
│   ├── app/
│   │   ├── api/
│   │   ├── agents/
│   │   │   ├── recommendation/
│   │   │   ├── cooking/
│   │   │   ├── chatbot/
│   │   │   ├── search/
│   │   │   └── summary/
│   │   ├── clients/
│   │   │   ├── backend/
│   │   │   └── inference/
│   │   ├── core/
│   │   ├── graphs/
│   │   ├── schemas/
│   │   └── tools/
│   └── tests/
│       ├── unit/
│       ├── integration/
│       └── contract/
├── inference-service/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── features/
│   │   ├── inference/
│   │   ├── model_registry/
│   │   └── schemas/
│   └── tests/
│       ├── unit/
│       ├── integration/
│       └── contract/
├── contracts/
│   ├── internal/
│   │   ├── agent/
│   │   ├── inference/
│   │   └── shared/
│   └── model-package/
│       ├── schema/
│       └── fixtures/
├── artifacts/
│   ├── manifests/
│   └── test-fixtures/
├── deployment/
│   ├── docker/
│   └── local/
└── docs/
    ├── architecture/
    └── operations/
```

## Five Agents

| Agent | Trigger | Allowed responsibility | Prohibited responsibility |
| --- | --- | --- | --- |
| Recommendation Agent | Dedicated recommendation action | Retrieve authorised context via tools, request inference, apply diversity, explain reason codes | Database access, public search, cooking |
| Cooking Planner Agent | Dedicated cooking action | Match controlled recipes to ingredients, budget, time, and diet | Recommendation scoring, arbitrary recipe invention |
| Chatbot Orchestrator | Chatbot message | Classify supported intent, maintain graph state, route to Chatbot specialist | Calling recommendation or cooking flows |
| Platform Search Agent | Search intent | Call authorised backend search and return source references | Public internet search, inaccessible content |
| Content Summary Agent | Shared-content summary/compare intent | Summarise resolved authorised references | Unshared data, unsupported facts |

## Agent Design Rules

Every graph must define:

- Typed input and output
- Explicit state fields
- Allowed transitions
- Tool schemas
- Maximum steps
- Timeout and retry limits
- Unsupported-intent behaviour
- Invalid-tool-output behaviour
- Trace and correlation fields

Natural-language generation must be grounded in tool evidence. An Agent may phrase verified reason codes but may not invent a restaurant fact, hygiene claim, model signal, or source.

## Inference Service

The inference service should:

- Validate the model-package manifest at startup
- Validate artifact checksums
- Validate feature-schema compatibility
- Load a specific immutable model version
- Reject incompatible requests
- Return scores, availability flags, and model metadata
- Expose readiness only after successful model loading
- Keep a safe deterministic fallback contract with the backend

Training, experiment selection, and metric production remain in `foodmind-ml`.

## Model Package Contract

A released model package is expected to include:

- Model version
- Training-code commit
- Dataset or snapshot identifier
- Feature-schema version
- Serialized model artifacts
- Preprocessing metadata
- Evaluation metrics
- Supported inference-contract version
- Creation timestamp
- SHA-256 checksums
- Limitations/model-card reference

The exact schema will live under `contracts/model-package/schema/` and must match the producer contract in `foodmind-ml`.

## Configuration

Planned runtime configuration:

| Variable | Purpose |
| --- | --- |
| `APP_ENV` | `local`, `test`, `staging`, or `production-demo` |
| `BACKEND_INTERNAL_BASE_URL` | Private Spring Boot internal-tool URL |
| `BACKEND_SERVICE_TOKEN` | Service credential for backend tools |
| `INFERENCE_SERVICE_BASE_URL` | Private inference-service URL used by Agent service |
| `INFERENCE_SERVICE_TOKEN` | Service credential |
| `MODEL_ARTIFACT_URI` | Immutable released model-package location |
| `MODEL_ARTIFACT_SHA256` | Expected package checksum |
| `MODEL_VERSION` | Version approved for runtime loading |
| `LLM_API_KEY` | Provider credential when an LLM integration is selected |

Variable bindings are not implemented yet. Real values belong in local secret storage or a managed secret service.

## API Policy

- All endpoints are private.
- Internal API is versioned under `/internal/v1`.
- Service authentication is mandatory.
- Requests include correlation and trace IDs.
- Responses are structured Pydantic models.
- Raw model objects and raw LLM responses are not returned.
- Android and Web never consume these endpoints.

## Testing Strategy

Agent service:

- Graph transition tests
- Tool allow-list tests
- Pydantic schema tests
- Unsupported-intent tests
- Timeout and retry tests
- Grounding and source-reference tests
- No-direct-database-access checks

Inference service:

- Manifest and checksum tests
- Feature-schema compatibility tests
- Deterministic toy-vector tests
- Cold-start and availability-flag tests
- Model-load and rollback tests
- Contract tests against `foodmind-ml` fixtures

## Security

- Services run on a private network.
- Service credentials are least privilege and rotated.
- Logs exclude prompts containing user content unless explicitly redacted.
- No arbitrary tool execution, URLs, SQL, or code evaluation.
- Tool inputs are bounded and schema-validated.
- Model artifacts are verified before loading.
- Unsupported and inaccessible claims are rejected.

## Contribution Workflow

1. Identify the owning runtime service.
2. Confirm the relevant internal-contract version.
3. Keep Agent and inference implementation logically separate.
4. Add schema, failure, and contract tests.
5. Record model-package compatibility for inference changes.
6. Run the complete service test suite.
7. Open a reviewed Pull Request.

## Further Reading

- [Runtime architecture](docs/architecture/runtime-architecture.md)
- [Model consumption and rollback](docs/operations/model-consumption.md)
