# Branch 07 - Recommendation Agent Integration and Release Evidence

## Branch metadata

- **Proposed branch:** `test/recommendation-agent-integration`
- **Base dependency:** Branch 06 merged
- **External dependencies:** Backend v2 adapter/fallback/shadow controls and
  compatible inference v1 environment
- **Exit:** Agent release candidate evidence; not inference/ML release approval

## Purpose

Prove the Agent module works at both private boundaries, remains invisible to
clients, produces Backend-valid grounded results, fails safely to deterministic
Backend fallback, supports shadow/staged routing, and can be rolled back by
Agent image/Backend flag without data or client changes.

## Related source-document sections

- Implementation design: **End-to-end online sequence**, **Contract evolution
  and compatibility**, **Verification strategy - End to end**, and **Rollout
  and rollback**.
- Delivery plan: Agent portions of Phases 5-7, acceptance checklist, PR/Git
  strategy, and evidence requirements.
- Inventory: Backend v1 fixtures/current behavior and required UAT paths.

## Prerequisites and dependencies

- Backend dual-contract v2 support, reason validator, deterministic fallback,
  and shadow flag exist in `foodmind-backend`.
- Compatible inference v1 service/fixture environment exists. The Agent test
  suite may use a fixture fake, but production evidence uses the owned service.
- U-01/U-03/U-04/U-10/U-11 accepted; compatibility matrix current.
- Agent release-candidate image from Plan 06 identified by digest.
- U-13 resolved before changing root `README.md`.

## Scope

- Agent producer/consumer contract checks against exact Backend/inference fixtures;
- local integration harness using fake inference for deterministic failures;
- staged environment integration with real private inference contract;
- normal/cold-start/sparse-group/timeout/malformed/incompatible/unsafe cases;
- Backend fallback and shadow-output non-exposure evidence;
- Agent metrics/latency/privacy/security checks;
- Agent image rollback and Backend-disable drills;
- compatibility/evidence/runbook/documentation finalization.

### Explicit non-scope

- No inference or ML implementation, package approval, dataset, training,
  evaluation, or model-card work.
- No Backend feature/filter/database/public API implementation.
- No Web/Android source or direct Agent access.
- No staged traffic enablement without Backend owner approval.
- No v1 removal; retirement is a later coordinated change after telemetry.

## Concrete files

```text
agent-service/app/agents/recommendation/tests/e2e/
  test_backend_agent_contract.py
  test_agent_inference_contract.py
  test_failure_to_fallback_matrix.py
  test_shadow_non_exposure.py
  test_release_scenarios.py
agent-service/app/agents/recommendation/tests/fixtures/
  fake_inference.py
  scenarios.py
agent-service/app/agents/recommendation/scripts/
  smoke_private_agent.py
  verify_release_evidence.py
deployment/local/recommendation-agent.compose.yaml
docs/testing/recommendation-agent-uat-matrix.md
docs/testing/recommendation-agent-evidence/README.md
docs/architecture/recommendation-agent-compatibility-matrix.md
docs/operations/recommendation-agent.md
README.md  # only after U-13 resolution
```

Do not commit raw runtime logs, tokens, real request/features/model keys, large
reports, screenshots with personal data, or inference/model artifacts. The
evidence index links approved redacted storage and records checksums/metadata.

## Integration matrix

| Scenario | Agent expectation | Backend/system expectation |
| --- | --- | --- |
| Normal personal/exploratory/group | deterministic <=3 grounded results | validates/persists Agent completion |
| New user/no CF | LR probability accepted, no CF reasons | valid cold-start result or fallback by policy |
| New meal/ItemCF unavailable | no ItemCF reason | other evidence may still rank |
| Sparse/no group | group type/reason omitted per policy | valid fewer-than-three response |
| No eligible candidate | Agent is not called | Backend completes no-candidate path |
| Past/near deadline | typed Agent failure, zero inference when applicable | fallback with safe code |
| Inference timeout/unavailable/non-2xx | one call, typed failure | deterministic fallback |
| Malformed/oversized inference | typed failure, no raw body | deterministic fallback |
| Contract/feature/key/model/package mismatch | typed incompatibility | deterministic fallback and mismatch metric |
| Unknown/duplicate/missing candidate | Agent rejects whole result | deterministic fallback |
| Unsupported reason/unsafe explanation | Agent must normally prevent it; malicious fixture rejected | Backend validator independently rejects/falls back |
| Identical replay | same structured candidates/order/reasons/policy | Backend idempotency returns same session/result |
| Try another/feedback/re-recommend | no new Agent call for client reveal | Backend/client-owned behavior remains stable |
| Agent outage | no direct client error from Agent | Backend public API remains available via fallback |

## Ordered implementation tasks

1. Pin exact Backend Agent v2 and inference v1 fixture revisions/checksums in
   the compatibility matrix and test data source manifests.
2. Add fixture-driven fake inference implementing only the accepted transport
   contract and scenario switches. It must not calculate CF/LR or ship in the
   production image.
3. Add Agent-to-inference contract tests covering success, cold start, timeout,
   unavailable, malformed, oversized, incompatible versions, invalid evidence,
   and deadline exhaustion.
4. Add Backend-to-Agent tests using exact Backend request/expected response
   fixtures; validate auth/correlation/version/deadline/body limits.
5. Run Backend v2 consumer tests against Agent output. Prove every candidate ID,
   rank, type, score, reason, explanation, and version passes the independent
   Backend validator.
6. Inject malicious Agent test fixtures at Backend boundary for unknown ID,
   duplicate rank/type, unsupported CF proxy reason, unsafe text, and mismatch;
   confirm Backend fallback. Do not weaken Agent tests merely because Backend
   revalidates.
7. Build the scenario matrix with synthetic data. Record exact preconditions,
   Agent/inference/Backend revisions, image digest, contract/policy versions,
   environment, UTC date, tester, expected/actual, and redacted evidence link.
8. Verify no-candidate, try-another, feedback, and re-recommend behavior as
   Backend-owned integration cases: Agent call count must be zero for no
   candidates and same-response try-another.
9. Enable Backend shadow mode only in approved synthetic/non-production/staged
   traffic. Assert shadow output never reaches clients or mutates the returned
   fallback result unexpectedly.
10. Compare Agent-side candidate/result counts, reason support, failure rates,
    latency, payload sizes, and version metrics against U-04/U-11 gates. Model
    quality/calibration evaluation remains outside Agent scope.
11. Run security checks for service auth, private routing, egress allow-list,
    payload limits, secret/log redaction, container scan, and direct client
    access denial.
12. Exercise rollback A: disable Backend Agent v2 and prove deterministic
    fallback without Agent/inference/client release.
13. Exercise rollback B: redeploy prior Agent image digest, wait for readiness,
    re-run smoke, confirm version/latency/failure metrics. Do not alter model or
    inference deployment as part of Agent rollback.
14. Record alert behavior for readiness failure, contract mismatch, timeout,
    invalid response/reason, fallback signal spike, and Agent latency. Expected
    single fallback does not page by itself.
15. Complete Agent runbook/evidence index/compatibility matrix. Update root
    README only after U-13 resolution and only with Agent status/commands.
16. Keep v1 fixtures and compatibility notes. V1 retirement requires telemetry
    showing no use for the accepted window and a separate reviewed plan.

## Test and evidence requirements

### Local/CI

- all Plan 01-06 tests plus e2e fixture matrix;
- Agent production image contains no fake inference/scenario test code;
- repeated full fixture suite deterministic;
- exact fixture checksums verified before tests.

### Staged private integration

- Backend -> Agent -> inference normal/cold-start/sparse-group;
- every timeout/unavailable/malformed/incompatible case;
- Agent/Backend independent unsupported-reason/unsafe-text rejection;
- private auth/TLS/egress and no direct Web/Android access;
- P95/P99 total and Agent-overhead budgets;
- logs/evidence redaction review.

### Rollback/release

- Backend-disable fallback drill timed and recorded;
- prior Agent image rollback timed and recorded;
- release identity: Agent Git SHA/tag/image digest, contracts/policies,
  Backend/inference revisions/environment/date/tester;
- no unresolved Critical/High or P0/P1 Agent security/safety defect.

## Acceptance criteria

- [ ] Exact Backend/Agent/inference contract fixtures pass at both boundaries.
- [ ] Normal/cold-start/sparse-group outputs are deterministic and grounded.
- [ ] Every private failure produces safe Backend fallback and no raw leakage.
- [ ] No-candidate and try-another cases prove Agent call-count expectations.
- [ ] Shadow output is not client-visible and Agent meets latency/error gates.
- [ ] Private auth/network/payload/privacy/security checks pass.
- [ ] Both Agent rollback paths succeed without data, inference/ML, or client change.
- [ ] Evidence ties exact revisions/image/contracts/policies/environment/date/tester.
- [ ] V1 remains until separate telemetry-based retirement.

## Commit plan

1. `test(contracts): verify Agent private producer and consumer boundaries`
2. `test(recommendation): cover fallback and release scenario matrix`
3. `test(security): prove private Agent staging boundary`
4. `test(recommendation): record shadow latency and rollback evidence`
5. `docs(recommendation): finalize Agent UAT and compatibility evidence`
6. `docs(operations): finalize Agent release and rollback runbook`

## Verification

```powershell
Push-Location agent-service/app/agents/recommendation
uv sync --frozen --dev
uv run pytest -v
uv run python scripts/smoke_private_agent.py --mode fixture
uv run python scripts/verify_release_evidence.py
Pop-Location
```

Run the exact Backend focused v2 consumer suite and staged commands documented
by the Backend owner. Record commands verbatim; do not claim an unrun
PostgreSQL/client/model-quality gate passed.

Expected: zero Agent test failures, exact fixture checksum match, staged matrix
pass, accepted latency/failure rates, redacted evidence, and both rollback
drills pass.

## Pull Request hand-off

- **Title:** `test(recommendation): prove Agent integration and rollback`
- Include Agent image/revision, Backend/inference revisions, contract/policy
  checksums, scenario table, exact commands/results, latency/failure metrics,
  security/redaction evidence, rollback timings, open items/owners, and explicit
  inference/ML non-scope.

## Rollback and blockers

Release is blocked by unresolved U-01/U-03/U-04/U-10/U-11, missing Backend v2
support, incompatible inference v1, failed privacy/security checks, or missing
rollback proof. Agent rollback never authorizes changing inference/model/ML
artifacts; use Backend fallback or prior Agent image.
