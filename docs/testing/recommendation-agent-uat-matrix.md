# Recommendation Agent UAT matrix

Evidence date: 2026-08-03. Local environment: Windows, Python 3.13, uv
0.12.1. Tester: Codex automated synthetic harness. Repository parent baseline:
`299125e77a79ee1dc1255c468a41ef31ce1fb431`; the exact Agent image source identity
is recorded in the release evidence index. No personal data, production
credentials, or real model artifacts were used.

## Local fixture results

| Scenario | Expected | Local evidence | Status |
| --- | --- | --- | --- |
| Normal personal/exploratory/group | At most three deterministic grounded results | Backend-style independent schema, membership, probability, and reason validator | PASS |
| Cold start/no CF | No CF-derived reason without support | Exact cold-start inference fixture and replay test | PASS |
| Sparse group | Unsupported Group result/reason omitted | One-member synthetic group fixture | PASS |
| Expired deadline | Typed failure and zero inference calls | Past absolute deadline test | PASS |
| Timeout/unavailable/non-2xx | One inference attempt and typed failure | Parameterized transport matrix | PASS |
| Malformed/oversized response | Typed failure, no raw body | Stream byte cap and malformed JSON scenarios | PASS |
| Feature/model/package/key mismatch | Exact mismatch code and fallback | Parameterized compatibility matrix | PASS |
| Unknown/duplicate/missing candidate | Reject whole inference result | Candidate membership matrix | PASS |
| Invalid probability/evidence | Reject whole inference result | Strict consumer fixture mutations | PASS |
| Backend fallback | Same safe object for every private failure | Independent Backend-style fallback harness | PASS |
| Unsafe/unsupported/malformed Agent output | Independent consumer rejection | Malicious output mutations | PASS |
| No candidates | Backend does not call Agent | Call-count harness | PASS |
| Try another/feedback | No additional Agent call | Backend-owned reveal/ack harness | PASS |
| Shadow mode | Agent result discarded; client sees fallback | Synthetic shadow wrapper | PASS |
| Private authentication | Missing credential rejected before inference | Agent boundary auth test | PASS |
| Agent-only latency | P95 below the local 100 ms working bound for 1/10/100 candidates | 20 synthetic runs per size; not a staged U-04 approval | PASS |
| Direct egress/network/privacy/payload controls | Fixed inference origin, no proxy/redirect, redacted errors/logs, bounded bytes | Plan 06 security suites | PASS |

These results establish local contract and orchestration behavior only. The fake
inference service returns accepted fixture bytes and contains no ranking, CF,
LR, or model logic. It is excluded from the production image.

## Required external release evidence

| Gate | Required identity/evidence | Status | Blocker/owner |
| --- | --- | --- | --- |
| Backend v1 retention suite | `foodmind-backend@7ea2b90c1451d689c59d4ea37d337b4552220f44`, exact focused Maven command | PASS | 11 tests; v1 only |
| Backend v2 consumer suite | Backend v2 adapter/validator/fixtures, exact command and owner approval | UNAVAILABLE | Pinned Backend revision contains only v1 |
| Staged Backend -> Agent -> inference | Private environment and approved synthetic routing | NOT RUN | Staged services unavailable |
| Private TLS/network policy | Platform policy and direct-client denial | NOT RUN | Deployment provider not selected |
| Staged P95/P99/failure gates | Accepted U-04/U-11 values and telemetry | NOT RUN | Owner acceptance and staged telemetry unavailable |
| Local image security identity | Digest-pinned bases, exact Agent digest, SBOM, clean CVE scan, non-root/read-only contract smoke | PASS (LOCAL) | Image is unpublished and unsigned; release registry identity remains external |
| Backend-disable rollback | Timed fallback drill | NOT RUN | Backend feature flag unavailable |
| Local Agent image rollback mechanism | Prior local candidate digest, timed readiness/smoke, current digest restore | PASS (LOCAL) | 19,112 ms rollback; 18,756 ms restore |
| Staged prior-image rollback | Prior published digest, timed redeploy/readiness/smoke | NOT RUN | No published image or staged environment |
| Fixture ownership | Backend/inference approvals of pinned bytes | PENDING | Source manifests deliberately set `approvalPending: true` |

Until every external row passes, this matrix is local evidence and must not be
used as production release approval. V1 remains supported; no retirement date
is inferred.
