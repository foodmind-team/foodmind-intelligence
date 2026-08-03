# Branch 04 - Bounded Recommendation Agent Workflow

## Branch metadata

- **Proposed branch:** `feat/recommendation-agent-workflow`
- **Base dependency:** Branch 03 merged
- **Workflow engine:** LangGraph, matching repository-local Agent convention
- **Downstream calls:** exactly one inference call and no other tool call

## Purpose

Implement the finite structured state machine that validates the request,
invokes inference once, verifies compatibility, hands scored candidates to
deterministic result shaping, renders output, or terminates with a typed
failure. The graph must have no loop, retry, persistence, or open-ended tool
selection.

## Related source-document sections

- Implementation design: **Recommendation Agent design** flowchart and
  recommended hard bounds; **End-to-end online sequence**; **Reliability and
  observability**; **Verification strategy - Intelligence**.
- Delivery plan: Phase 4 task 2, task 4 bounds, task 5 prohibited-access tests,
  and Phase 4 exit gate.
- Local: runtime architecture **Graph state**, **Tool boundary**, and **Failure
  modes**.

## Prerequisites and dependencies

- Plan 03 supplies strict request models, deadline helper, and `InferencePort`.
- Plan 01 supplies accepted compatibility/failure taxonomies.
- Plan 05 will implement selectors/reasons/renderers behind ports. This branch
  uses deterministic test doubles and defines their interfaces.
- U-04 timeout allocations are fixed.

## Scope

- serializable typed `RecommendationState`;
- immutable workflow context with injected clients/policies/clocks;
- named nodes and fixed edges with one terminal success/failure;
- one-call enforcement and no retries/loops;
- deadline checks and compatibility validation;
- stable typed failure mapping;
- graph transition, mutation/property, and architecture tests.

### Explicit non-scope

- No final diversity coefficients, reason predicates, or templates; Plan 05.
- No checkpoint/database/task queue; synchronous bounded workflow only.
- No LLM, tool registry, dynamic routing, human confirmation, web, or Backend
  context retrieval.
- No fallback candidates inside Intelligence.

## Concrete files

```text
agent-service/app/agents/recommendation/src/recommendation_agent/workflow/
  __init__.py
  context.py
  graph.py
  nodes.py
  routing.py
  state.py
agent-service/app/agents/recommendation/src/recommendation_agent/application/
  ports.py
  service.py
agent-service/app/agents/recommendation/src/recommendation_agent/domain/
  errors.py
  models.py
agent-service/app/agents/recommendation/tests/unit/workflow/
  test_graph_shape.py
  test_nodes.py
  test_routing.py
  test_state.py
agent-service/app/agents/recommendation/tests/integration/
  test_workflow_success.py
  test_workflow_failures.py
  test_workflow_deadlines.py
agent-service/app/agents/recommendation/tests/security/
  test_workflow_capabilities.py
```

## Workflow contracts

### State

`RecommendationState` is a `TypedDict(total=False)` or the accepted LangGraph
serializable equivalent with only:

- immutable validated Agent request;
- safe `agentTraceId`;
- local deadline/budget value that is serializable without exposing a clock
  from another process;
- strict inference result;
- compatibility result;
- selected candidates;
- reason/evidence facts;
- final strict Agent response; and
- one typed workflow failure.

Do not store HTTP clients, tokens, settings with secrets, raw response bodies,
model artifacts/keys outside the request object, callbacks, or arbitrary
exceptions in graph state. Provider/client/policy objects belong in immutable
`WorkflowContext`.

### Ports for Plan 05

| Port | Contract |
| --- | --- |
| `ResultSelector#select` | Receives validated scored candidates and accepted policy; returns at most three unique typed selections deterministically. |
| `ReasonDeriver#derive` | Returns only allow-listed reason facts whose frozen predicates pass. |
| `ExplanationRenderer#render` | Creates bounded deterministic templates from approved facts; no I/O or LLM. |

### Graph shape

```text
START
  -> validate_envelope
  -> score_once
  -> validate_compatibility
  -> select_results
  -> derive_reasons
  -> render_explanations
  -> build_success
  -> END

Any node failure
  -> build_failure
  -> END
```

No conditional edge may return to an earlier node. `score_once` is the only
node permitted to use `InferencePort`.

## Ordered implementation tasks

1. Define serializable state and immutable context. Add a test that attempts to
   JSON/checkpoint-serialize state without secrets/clients, even though no
   runtime checkpoint is enabled.
2. Define typed workflow failure categories that map one-to-one to accepted
   Agent failure responses. Preserve safe inference failure category without
   raw body/exception.
3. Implement `validate_envelope`: verify contract/schema/key versions,
   request/session/trace/deadline, 1-100 unique candidates, strict candidate
   identifiers/features, and remaining budget.
4. Implement `score_once`: atomically mark/guard invocation in state/context,
   calculate remaining-minus-guard, call `InferencePort` once, and map any
   transport result to typed state.
5. Implement `validate_compatibility`: echoed request/trace/candidate IDs,
   candidate set/cardinality, model/package/feature/inference/key versions,
   status, probability/CF/evidence invariants, and remaining deadline.
6. Create selector/reason/renderer protocols and deterministic fixture doubles.
   Their invocation order and inputs are tested; Plan 05 replaces doubles in
   application wiring without changing graph edges.
7. Implement `select_results`, `derive_reasons`, and `render_explanations` nodes
   as pure port calls. They may write only their changed state subset.
8. Implement `build_success`: require no failure, compatible inference, unique
   ordered selections, reasons and explanations, then emit strict v2 response
   with exact version metadata.
9. Implement `build_failure`: emit accepted typed failure with echoed safe
   correlation/version fields; candidates empty; no partial selection/reason.
10. Define explicit routing after every error-capable node. A set failure always
    short-circuits directly to `build_failure`.
11. Compile graph once during lifespan and inject into application service.
    Invocation uses a fresh state/context per request and has an outer timeout
    no longer than remaining deadline.
12. Add a hard maximum step/node count assertion based on compiled graph shape.
    No dynamic `Command`, recursion, subgraph, retry policy, or interrupt.
13. Add architecture tests: only `score_once` imports/calls `InferencePort`; no
    module imports DB, SQL, browser, requests to arbitrary hosts, Cooking, Chat,
    Search, Summary, LLM, tool execution, or checkpoint packages.
14. Add trace events/metrics for node name, duration, result code, contract and
    safe model version. Do not store or log full state.
15. Wire the real graph into `RecommendationAgentService` and API. Until Plan 05
    selectors land, use test-only doubles; production wiring must fail startup
    rather than serve fixture selections.

## Test requirements

### Graph/state

- expected node/edge set, one START, two terminals, no cycles;
- state serialization contains no client/token/artifact/raw error;
- each node returns only documented changed fields;
- a failure at each node reaches `build_failure` and no later happy node;
- success visits every happy node once in exact order.

### Single call and deadlines

- inference spy call count exactly one on success and downstream failures;
- zero calls for invalid/expired/over-limit input;
- no retry on timeout/connection/5xx/malformed/incompatible response;
- deadline exhausted before/after inference returns typed failure and no
  partial output;
- slow selector/renderer is bounded by outer deadline.

### Compatibility/output

- wrong echoed IDs, missing/extra/duplicate candidate, wrong version/model/key,
  invalid probability/CF support/status, and over-limit response fail;
- success/failure strict response validates canonical schema;
- identical input/inference/policy produces byte-equivalent structured output
  aside from an explicitly generated `agentTraceId` whose determinism policy is
  frozen in Plan 01.

### Capability isolation

- monkeypatch/socket test permits only configured inference origin;
- no filesystem/package/database/web/LLM call during Agent request;
- cannot introduce candidate absent from input even with malicious selector
  double; success builder revalidates membership.

## Acceptance criteria

- [ ] Compiled workflow is acyclic, bounded, and has exactly one inference node.
- [ ] Every invalid/late/unavailable state terminates once with an accepted typed
  failure and empty results.
- [ ] Success revalidates all IDs/versions/bounds after policy ports.
- [ ] Architecture tests prove no prohibited capability or workflow import.
- [ ] Repeated identical fixtures follow identical node trace and structured
  output semantics.

## Commit plan

1. `feat(recommendation): define serializable bounded workflow state`
2. `feat(recommendation): call inference in one acyclic graph`
3. `feat(recommendation): validate compatibility and terminal output`
4. `test(recommendation): prove workflow bounds deadlines and isolation`
5. `feat(recommendation): wire bounded graph into private API`

## Verification

```powershell
Push-Location agent-service/app/agents/recommendation
uv run pytest tests/unit/workflow -v
uv run pytest tests/integration/test_workflow_success.py `
  tests/integration/test_workflow_failures.py `
  tests/integration/test_workflow_deadlines.py -v
uv run pytest tests/security/test_workflow_capabilities.py -v
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
Pop-Location
```

Expected: all commands exit `0`; graph inspection reports no cycle, call-count
evidence is zero/one as expected, and every injected node failure produces one
canonical typed failure.

## Pull Request hand-off

- **Title:** `feat(recommendation): add one-call bounded Agent workflow`
- Include graph node/edge listing, maximum steps, call-count and deadline matrix,
  exact commands, contract versions, prohibited-capability proof, and remaining
  Plan 05 production-wiring dependency.

## Rollback and unresolved items

Keep Backend v2 disabled until Plan 05 completes production selection/reasons.
Rollback deploys the prior Agent image or disables Agent routing; no state is
persisted in Intelligence. U-04 must be resolved before deadline gates merge.
