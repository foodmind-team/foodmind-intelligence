# Branch 01 - Recommendation Agent Contracts and Golden Fixtures

## Branch metadata

- **Proposed branch:** `feat/recommendation-agent-contracts-v2`
- **Base dependency:** current `main`
- **External gate:** accepted recommendation-v2 ADR and named Backend/inference reviewers
- **Primary deliverable:** Agent-owned v2 boundary plus coordination fixtures
- **Application behavior:** none

## Purpose

Freeze exactly what the Recommendation Agent receives from Backend, what it
receives from the external inference runtime, and what it returns to Backend.
This branch prevents the Agent implementation from guessing field names,
evidence semantics, deadlines, failure mapping, result policy, or compatibility.

## Related source-document sections

- Inventory: **Authority order**, **Reconciled decisions and discrepancies**,
  and **Documents to create or update during delivery**.
- Implementation design: **Feature contract v2**, **Pseudonymous model keys**,
  **Inference service design** response contract, **Recommendation Agent
  design**, **Result shaping**, **Grounded reason-code predicates**, and
  **Contract evolution and compatibility**.
- Delivery plan: Agent-relevant Phase 0 tasks and all of Phase 4.

## Prerequisites and dependencies

1. `foodmind-docs` accepts the recommendation-v2 ADR and compatibility policy.
2. Backend assigns a v2 request/response consumer reviewer.
3. The inference owner provides or approves `recommendation-inference-v1`
   schemas/fixtures. The Agent stores only a coordination copy or consumer
   fixture reference, not an inference implementation plan.
4. Backend/inference resolve U-02 feature ownership so the Agent can distinguish
   Backend point-in-time facts from inference-generated CF signals.
5. U-03, U-04, U-09, and U-11 are accepted.

## Scope

- canonical `recommendation-agent-v2` request, success, and typed-failure schemas;
- Agent consumer copy/reference for `recommendation-inference-v1`;
- Agent-visible subset/reference of `recommendation-features-v2`;
- exact deadline and single-inference-call semantics;
- compatibility, diversity, reason, template, and failure-policy versions;
- synthetic positive/negative/golden fixtures;
- compatibility matrix entries for Backend v1/v2 and inference v1.

### Explicit non-scope

- No inference service, model package, loader, CF/LR scoring, ML training,
  evaluation, dataset, model card, or release tooling.
- No Backend DTO/database/public API change.
- No Agent application code.
- No real identities, pseudonyms, features, feedback, or model output.

## Concrete files

All are **Proposed** beneath verified existing directories.

```text
contracts/internal/agent/recommendation/v2/
  request.schema.json
  response.schema.json
  failure.schema.json
  contract-notes.md
  fixtures/
    valid-normal.json
    valid-cold-start.json
    valid-sparse-group.json
    failure-inference-unavailable.json
    failure-deadline-exhausted.json
    invalid-unknown-candidate.json
    invalid-unsupported-reason.json
    invalid-model-key-version.json
    invalid-over-100-candidates.json
contracts/internal/inference/recommendation/v1/consumer-fixtures/
  valid-hybrid.json
  valid-cold-start.json
  failure-package-incompatible.json
  invalid-feature-schema.json
  source-manifest.json
contracts/internal/shared/recommendation-features/v2/
  agent-evidence-catalog.md
  source-manifest.json
artifacts/test-fixtures/recommendation/agent-golden-v2/
  request.json
  inference-response.json
  expected-agent-response.json
  checksums.sha256
docs/architecture/recommendation-agent-compatibility-matrix.md
```

`source-manifest.json` records canonical owner, repository, revision, path,
version, and SHA-256. It prevents a coordination copy from becoming an
untracked fork of inference/feature semantics.

Use JSON Schema Draft 2020-12 **Proposed** with
`additionalProperties: false`, explicit bounds/formats, and strict runtime
rejection of non-finite numbers.

## Agent v2 contract requirements

### Backend-to-Agent request

Freeze these semantics; exact names shown are **Proposed** where not already
named by the design:

| Field | Required contract |
| --- | --- |
| `contractVersion` | Constant `recommendation-agent-v2`. |
| `featureSchemaVersion` | Exact supported `recommendation-features-v2`. |
| `requestId`, `sessionId`, `traceId` | Bounded correlation fields echoed in every response. |
| `deadlineAt` | Absolute UTC time; converted once to a local monotonic budget. |
| `decisionAt` | Backend-persisted point-in-time decision timestamp. |
| `modelUserKey`, `modelKeyVersion` | HMAC-derived lookup key/version; never logged or returned. |
| `candidates` | 1-100 authorised, already hard-filtered candidates. |

Each candidate contains only:

- request-scoped opaque `candidateId`;
- model meal/offering keys required by inference;
- accepted point-in-time features/evidence required by inference, diversity,
  and reason validation; and
- no raw database/user/group/place/meal/offering ID, exact coordinates, free
  text, credentials, comments, or unrelated group records.

### Inference-to-Agent consumer contract

The Agent consumer fixtures must require:

- echoed request/trace and exact model/package/feature/inference/key versions;
- one response per accepted candidate or only the frozen recoverable candidate
  status behavior;
- probability in `[0,1]`;
- UserCF availability, score, and neighbor support;
- ItemCF availability, score, and supporting-item count;
- named non-sensitive evidence signals required by the reason table; and
- no prose, raw ID, model key echo, coefficient, or arbitrary feature vector.

Unknown user/meal support is cold-start/unavailable evidence. Contract/schema/
key/package incompatibility is a request-level Agent failure. The Agent does
not best-effort parse and does not retry inference.

### Agent-to-Backend response

Freeze:

- echoed safe identifiers and bounded `agentTraceId`;
- success or typed failure status;
- model/package/feature/inference/key/diversity/reason/template versions needed
  for Backend validation;
- zero to three unique input candidate IDs with contiguous ranks;
- accepted `PERSONAL`, `EXPLORATORY`, `GROUP_INSPIRED` vocabulary;
- original inference probability/model score;
- allow-listed reasons and bounded deterministic explanation; and
- no full feature echo unless Backend accepts a minimal named evidence subset.

### Failure taxonomy

Freeze stable Agent-facing categories for at least:

- invalid/oversized request and unsupported Agent/feature version;
- expired/insufficient/exhausted deadline;
- inference connection, timeout, non-2xx, malformed/oversized response, and
  unavailable status;
- inference contract/model/package/feature/key mismatch;
- unknown/duplicate/missing candidate or invalid probability/evidence;
- result-selection, unsupported-reason, and unsafe-template failure.

Record HTTP status versus structured-body behavior so Backend can map every
case to deterministic fallback without relying on raw text.

## Policy fixtures

Freeze the diversity/reason/template policies used by Plans 04-05:

- lead tie band and personal-evidence predicate;
- novelty bonus cap, similarity penalties, group threshold, alternative
  ordering, and type omission/substitution;
- stable tie order ending in request-scoped `candidateId`;
- reason thresholds, priority, maximum count, and policy version;
- constrained template composition and forbidden claim vocabulary.

Create positive and near-miss negative fixtures for UserCF, ItemCF,
preference-match, Want-to-Try, group, specific context, and cleanliness
observation. Group counts cannot satisfy UserCF; personal counts cannot satisfy
ItemCF.

## Ordered implementation tasks

1. Record accepted ADR revision, owners, supported versions, route/auth choice,
   latency allocation, and compatibility window in `contract-notes.md`.
2. Record current Backend v1 fixtures by canonical path/checksum; do not delete
   or restate v1 as v2 evidence.
3. Write strict Agent v2 request/response/failure schemas with all bounds,
   formats, version constants, and unknown-field rejection.
4. Import/reference the accepted inference v1 fixtures through a source
   manifest. Add only Agent-specific negative consumer fixtures.
5. Document the exact Agent-visible feature/evidence pointers, producer,
   nullability, units, decision cutoff, and log/reason permission. Resolve U-02.
6. Freeze typed failure and HTTP/body mapping; Backend reviewer confirms every
   result triggers safe fallback as intended.
7. Freeze diversity/reason/template policies and all U-11 numeric choices.
8. Build a 4-6-candidate synthetic golden scenario covering normal three-type,
   cold-start no-CF, sparse group, deterministic tie, and unsupported reason.
9. Add negative fixtures for unknown/duplicate IDs, wrong versions, missing CF
   availability/support, invalid probability, over-limit candidates/results,
   unsupported/unsafe reason text, and exhausted deadline.
10. Generate fixture checksums and compatibility matrix. Include canonical
    owner/revision for each coordination copy.
11. Validate every JSON against its schema with a pinned validator; Agent and
    Backend/inference consumer tests must use the same bytes/checksums.
12. Obtain Backend and inference reviewer approval before dependent branches.

## Validation requirements

- valid normal/cold-start/sparse/failure fixtures pass;
- unknown fields/coercions, invalid UUID/time/key/version, duplicate IDs,
  more than 100 candidates, more than three results, non-contiguous ranks,
  invalid probability, ambiguous CF availability, and unsafe text fail;
- no candidate outside the request can appear in success output;
- every reason has positive and threshold/missing-evidence negatives;
- no fixture contains real/persisted identity or model key;
- checksums reproduce exactly.

## Acceptance criteria

- [ ] Backend, Agent, and inference consumers validate identical fixture bytes.
- [ ] U-02/U-03/U-04/U-09/U-11 are resolved in contract notes.
- [ ] Every failure and reason has a machine-testable fixture.
- [ ] No field has ambiguous optionality, producer, unit, or evidence cutoff.
- [ ] Compatibility matrix names v1 window and exact v2/inference/feature/key/
  policy versions.
- [ ] No inference/ML/model-package implementation artifact is planned or added.

## Commit plan

1. `docs(contracts): freeze Recommendation Agent v2 envelope`
2. `test(contracts): add inference consumer and Agent golden fixtures`
3. `docs(contracts): freeze diversity reason and failure policies`
4. `docs(contracts): record Agent compatibility matrix`

## Verification

Use the pinned validator selected in this branch; record exact version:

```powershell
uvx --from check-jsonschema check-jsonschema `
  --schemafile contracts/internal/agent/recommendation/v2/request.schema.json `
  contracts/internal/agent/recommendation/v2/fixtures/valid-*.json
Get-FileHash artifacts/test-fixtures/recommendation/agent-golden-v2/*.json -Algorithm SHA256
```

Expected: valid fixtures pass, each negative fails its documented rule, and
recorded checksums match.

## Pull Request hand-off

- **Title:** `docs(contracts): freeze Recommendation Agent v2 contracts`
- Include ADR/reviewer links, compatibility tuple, policy thresholds, exact
  validation results, source-manifest revisions, and every resolved U-ID.

## Rollback and compatibility note

This is a new version. Rollback keeps Backend on v1/fallback; never mutate v1 or
reuse `recommendation-agent-v2` for incompatible semantics. Inference and ML
remain external workstreams.

