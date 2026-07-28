# Intelligence Runtime Architecture

## Design Objective

FoodMind Intelligence provides controlled reasoning and model inference without becoming a second business backend. Spring Boot continues to own identity, permissions, domain state, persistence, and client contracts.

## Service Separation

### Agent service

Owns:

- LangGraph workflows
- Agent state
- Tool selection
- Grounded response construction
- Agent-level tracing

### Inference service

Owns:

- Released model-package loading
- Feature-schema validation
- Runtime feature transformations defined by the package contract
- UserCF, ItemCF, and Logistic Regression inference
- Model-version metadata

The two services may share a deployment unit for MVP cost control, but their modules, APIs, tests, and contracts remain separate.

## Recommendation Flow

```text
1. Backend authenticates the user.
2. Backend retrieves authorised candidates and evidence.
3. Backend applies hard constraints.
4. Backend invokes Recommendation Agent with bounded context.
5. Agent requests inference for valid candidates.
6. Inference service loads the approved immutable model.
7. Inference service returns scores and model metadata.
8. Agent applies diversity policy, orders the candidates, and phrases verified reason codes.
9. Backend validates and persists the structured result.
10. Backend returns up to three ordered candidates to the client.
```

The inference score does not replace hard filtering and is not itself an explanation.

The first candidate is the lead result for the recommendation-first home
experience. Personal, Exploratory, and Group-inspired candidates remain in the
same structured response. Moving between those candidates is a client
presentation action and must not trigger a hidden inference call.

## Tool Boundary

Tools should be small and purpose-specific, for example:

- Get authorised recommendation context
- Resolve valid candidate evidence
- Execute or verify hard-filter results
- Search authorised platform content
- Resolve shared content references
- Retrieve controlled recipe records
- Score candidates through inference service

A tool must define:

- Typed input
- Typed output
- Authentication scope
- Maximum result size
- Timeout
- Safe error
- Trace metadata

Generic database, HTTP, filesystem, or code-execution tools are prohibited.

## Graph State

Graph state should contain only data needed for the active request:

- Request/session ID
- Trace ID
- User context supplied by Backend
- Route decision
- Candidate/reference IDs
- Tool results
- Validation status
- Model/fallback status
- Structured final output

Do not retain unrestricted user data or credentials in graph state.

## Failure Modes

### Backend tool timeout

- Stop the affected path.
- Return a structured unavailable status.
- Do not invent missing context.

### Inference unavailable

- Return explicit model-unavailable metadata.
- Allow Backend to use deterministic fallback.

### Invalid model package

- Fail readiness.
- Do not serve partially loaded inference.
- Retain the last approved version only through an explicit rollback procedure.

### Invalid Agent output

- Fail schema validation.
- Do not forward raw text.
- Allow Backend to execute its fallback.

### Unsupported Chatbot request

- Explain the supported FoodMind search/summary scope.
- Do not route to recommendation, cooking, or public internet tools.

### Explore request

- Explore does not invoke a new Agent workflow.
- Spring Boot returns only authorised group-visible or curated platform content.
- The Agent service must not reinterpret Explore as public internet search or a
  public/follower feed.

## Observability

Correlate:

- Backend request ID
- Agent trace ID
- Tool-call ID
- Inference request ID
- Model version
- Fallback status

Metrics should include:

- Request count and latency
- Tool failures
- Agent route distribution
- Schema-validation failures
- Inference latency
- Model-load status
- Fallback rate

Logs must not become a store for personal content.

## Contract Ownership

- Agent/inference internal schemas: `foodmind-intelligence`
- Model-package producer schema: coordinated with `foodmind-ml`
- Matching Backend DTO and contract tests: `foodmind-backend`
- Public API: `foodmind-backend`

Contract fixtures should be versioned and used by both producer and consumer tests.

## Deployment Boundary

Production-demo expectations:

- Private network only
- Health endpoint for process liveness
- Readiness endpoint dependent on required clients/model
- Managed secret injection
- Immutable container image
- Immutable model version
- Resource and timeout limits
- No local runtime training
