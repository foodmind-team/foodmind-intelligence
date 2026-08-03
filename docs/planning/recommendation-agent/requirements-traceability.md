# Recommendation Agent Requirements Traceability

- **Status:** Proposed implementation mapping
- **Target revision:** `foodmind-intelligence@87e1d7606e8bee6b91cf8cfc46f206ffeadabbfe`
- **Backend consumer revision:** `foodmind-backend@7ea2b90c1451d689c59d4ea37d337b4552220f44`
- **Scope:** Recommendation Agent only; inference and ML are external dependencies

## Purpose

Prove that Plans 01-07 cover the material Agent-owned requirements in the three
primary source documents and distinguish verified code from proposed paths.

## Verified current state

| Area | Verified state | Consequence |
| --- | --- | --- |
| Recommendation Agent | `agent-service/app/agents/recommendation/` has only a tracked `.gitkeep` at `HEAD`; it is already deleted in the working tree | No Agent code/schema/client/config/test exists. Do not restore or stage that pre-existing deletion. |
| Shared Agent shell | top-level `agent-service/app/{api,clients,core,graphs,schemas,tools}` and tests are `.gitkeep` placeholders | Do not assume an executable shared package. |
| Cooking Agent | standalone Python 3.13 `uv` project with FastAPI/Pydantic/LangGraph, `src` layout, strict gates, auth, correlation, errors, health, redaction, Docker | Reuse generic conventions only; do not import or route Cooking domain behavior. |
| Agent/inference/shared contracts | verified placeholder directories | Plan 01 adds Agent canonical files and inference/feature coordination manifests only. |
| Inference service | placeholder only in this repository | Explicitly outside this package; treated as private external contract dependency. |
| ML/model package | no Agent-owned implementation | Explicitly outside this package. Do not add training/model-loader/artifact plans. |
| Root CI/deployment | root workflow and deployment directories are placeholders | Plans 06-07 create Agent-specific gates/files only. |
| Root README | exists at `HEAD` but is already deleted in working tree | Do not restore in this task; Plan 07 updates it only after owner resolves deletion. |

## Verified Backend v1 consumer

- `POST /internal/v1/recommendations/generate`;
- `Authorization: Bearer`, `X-Correlation-ID`, contract header;
- `recommendation-agent-v1` / `recommendation-features-v1`;
- 250 ms connect, 800 ms read, 16,384-byte response defaults;
- Backend absolute deadline is currently two seconds;
- at most three ranks/types/scores/reasons/explanations;
- invalid transport/schema/ID/reason/version always falls back.

V1 currently sends raw `placeMealId` and uses invalid CF proxies. V2 must use
opaque candidate/model keys and explicit CF evidence. Keep v1 fixtures during
the accepted window, but never carry proxy semantics into Agent v2.

## Material requirement mapping

| Agent requirement | Plan(s) | Evidence |
| --- | --- | --- |
| Canonical `recommendation-agent-v2` and coordinated inference/feature consumer fixtures | 01 | Strict schemas, source manifests, checksums, compatibility matrix |
| Dedicated route; no Chat/Cooking/Search routing | 02, 04, 06 | OpenAPI and architecture/capability tests |
| Backend remains sole public/security/database/hard-filter authority | 01, 02, 06 | Minimal contract and prohibited-import/network tests |
| Max 100 in, max 3 unique ordered out | 01, 02, 04, 05 | Schema bounds and property/terminal validation |
| Exactly one inference call, no retry/other tool | 03, 04, 06 | Spy count and capability/network tests |
| Absolute deadline converted to local monotonic budget | 03, 04 | Clock/budget and expired/near-deadline tests |
| Strict inference response compatibility and typed failures | 01, 03, 04 | Consumer fixtures and malformed/version/ID matrix |
| Deterministic lead, exploratory diversity, group-inspired selection | 01, 05 | Frozen policy and golden/property tests |
| CF/group/preference/context reasons use explicit predicates | 01, 05 | Positive/near-miss fixtures; no proxy semantics |
| Template-only safe explanations | 05, 06 | Template snapshots and unsafe-language tests |
| Every private failure causes Backend fallback | 01, 03-05, 07 | Typed failure/Backend consumer integration matrix |
| Auth, private route, payload/concurrency bounds, redacted logs | 02, 03, 06 | Security and captured-log tests |
| Correlation, node/client latency, result/failure metrics | 02-06 | Structured observability assertions |
| Non-root immutable container, scans, SBOM, private local deployment | 06 | Container/CI/security evidence |
| Normal/cold-start/sparse-group/timeout/malformed/incompatible/unsafe/rollback paths | 07 | Agent+fake inference+Backend integration matrix |
| Shadow/staged release does not expose Agent directly to clients | 07 | Backend-owned flag dependency and release evidence |

## Fixed decisions

- Already-filtered candidates only; no candidate invention.
- One inference call; no database/web/tool/LLM ranking.
- UserCF reason requires explicit UserCF score/availability/support.
- ItemCF reason requires explicit ItemCF score/availability/support.
- Group evidence is not UserCF; personal count is not ItemCF.
- Model probability is not an explanation.
- Missing safety evidence is never phrased as safety.
- Templates are MVP; Backend independently revalidates.
- Inference/ML/model packages remain separate workstreams.

## Unresolved gates

Numbering retains IDs used across the package; omitted IDs concern inference/ML
implementation and are outside scope.

| ID | Status | Missing/conflict | Resolution | Blocks |
| --- | --- | --- | --- | --- |
| U-01 | **Unresolved** | Recommendation-v2 ADR/owners/final acceptance absent | Accept ADR and name Backend/inference reviewers | 01-07 |
| U-02 | **Unresolved** | Design does not fully distinguish Backend pre-inference features from inference-generated CF fields | Freeze Agent-visible request facts versus inference evidence with golden fixture | 01, 03-05, 07 |
| U-03 | **Unresolved** | Compatibility does not state whether Intelligence must expose v1 Agent adapter | Freeze v2-only versus explicit boundary adapter and retirement signal/window | 01-02, 07 |
| U-04 | **Unresolved** | V2 latency budget is open; current v1 800 ms read/two-second deadline conflict | Freeze Backend total, Agent guard, inference timeouts, P95/P99 | 01, 03-04, 06-07 |
| U-07 | **Proposed** | Shared Agent skeleton versus standalone Cooking convention | Adopt nested standalone Recommendation project or amend paths before Plan 02 | 02-07 |
| U-09 | **Unresolved** | Design says `/live`/`/ready`; Cooking uses `/health/live`/`/health/ready` | Accept one private operational route set | 01-02, 06 |
| U-10 | **Unresolved** | Current static Bearer versus production short-lived credential guidance | Preserve local/test Bearer; freeze production binding/rotation/TLS owner | 02-03, 06-07 |
| U-11 | **Unresolved** | Diversity coefficients/tie/reason thresholds/type omission not frozen | Approve versioned deterministic policy | 01, 05, 07 |
| U-13 | **Unresolved** | Root README is pre-deleted in working tree | Owner decides intent before Plan 07 documentation update | 07 docs only |

## External dependencies only

| Input | Owner | Agent use | Failure behavior |
| --- | --- | --- | --- |
| Accepted ADR/compatibility | `foodmind-docs` | Plan 01 gate | Do not freeze conflicting contract |
| Backend v2 request and validator | `foodmind-backend` | Producer/consumer integration | Keep Agent v2 disabled/fallback |
| Feature v2 facts/model-key version | Backend with inference/ML review | Agent request/inference forwarding/reasons | Reject mismatch |
| Inference v1 endpoint/fixtures | inference owner | One-call private dependency | Typed Agent failure; Backend fallback |
| CF/LR/model/package behavior | inference/ML owners | Opaque structured response only | Never implement in Agent |
| Shadow flags/fallback/UAT | `foodmind-backend` | Plan 07 coordinated evidence | No direct client traffic |

## Planning-package verification checklist

- [ ] Plans 01-07 each name sources, dependencies, files, ordered tasks,
  contracts, tests, acceptance, commands, rollback, and unresolved items.
- [ ] Every material Agent-owned requirement maps above.
- [ ] Inference/ML/model-package work appears only as external contract input.
- [ ] No task adds DB/public/client/training/model-loader/Chat/Cooking scope.
- [ ] New paths are marked `Proposed`; verified paths/symbols are accurate.
- [ ] Task order is small and dependency-safe.
- [ ] No application source code is part of this planning package.

