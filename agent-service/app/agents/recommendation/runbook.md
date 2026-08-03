# Recommendation Agent runbook

## Startup and probes

Start from the locked image/digest with private ingress only. Liveness is
`GET /health/live`; traffic routing uses `GET /health/ready`. Readiness requires
validated configuration, the inference boundary, loaded v1 policies, the
compiled workflow, and a non-shutdown state. Health never returns origins,
tokens, model keys, candidate data, or feature values.

Required production secrets are injected through the platform secret store as
`RECOMMENDATION_AGENT_INTERNAL_SERVICE_TOKEN` and
`RECOMMENDATION_AGENT_INFERENCE_SERVICE_TOKEN`; rotate both by replacing the
deployment, never by logging or baking them into an image.

## Failure response

- Inference outage/5xx: confirm `INFERENCE_UNAVAILABLE`,
  `INFERENCE_CONNECTION_FAILED`, or `INFERENCE_HTTP_ERROR` counters, verify the
  private allow-listed origin from deployment configuration, and confirm
  Backend deterministic fallback is active. Do not retry inside the Agent.
- Contract/package/key mismatch: stop v2 traffic, compare the frozen
  compatibility tuple and source-manifest checksums, and escalate to the
  Backend/inference owner placeholders. Do not enable best-effort parsing.
- Timeout spike: compare Agent-stage and inference timing aggregates with the
  absolute deadline/guard. Reduce upstream traffic or restore the previous
  inference release; never extend a request deadline.
- Unsafe-reason/template spike: disable Backend v2 routing, retain only safe
  failure categories, and compare policy/template versions and fixtures.
- Saturation: `SERVICE_OVERLOADED` with `Retry-After` is expected. Check active
  and queued bounds; health remains probeable.

## Privacy check

Logs and metrics may contain bounded correlations, frozen versions, counts,
stages, durations, and stable result categories only. If any token, model key,
feature/body, raw identifier, exception body, or URL query appears, remove the
instance from traffic, restrict log access, rotate affected credentials, and
escalate to the security owner placeholder.

## Rollback

1. Disable Backend Recommendation Agent v2 routing and verify Backend fallback.
2. Roll back to the prior immutable Agent image digest; there is no Agent data
   migration or persistent state.
3. Confirm liveness/readiness and one-call inference compatibility.
4. Retain contract/policy versions, image digest, sanitized metrics, test/scan
   evidence, and incident timestamps. Never retain payloads or credentials.

Inference/model rollback is owned externally and is not performed by this
service.

## Release and drill recording

Before traffic enablement, record the Agent Git revision/tag and immutable
image digest, Backend and inference revisions, contract/policy tuple,
environment, UTC start/end, tester/reviewers, readiness and smoke results,
redacted latency/failure metrics, alerts observed, and outcome. Rollback A
disables the Backend v2 flag and times restoration of deterministic fallback.
Rollback B redeploys the recorded prior Agent digest, waits for readiness, and
re-runs the private smoke test. Neither drill changes inference/model artifacts
or clients.

Use `scripts/verify_release_evidence.py` for local checksum/index verification.
External staged and rollback results remain NOT RUN until their owning systems
are available; never represent the runbook procedure itself as drill evidence.
