# Recommendation Agent compatibility matrix

| Backend producer | Agent route | Agent contract | Feature schema | Inference contract | Model/package | Key | Policies | Support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Backend v1 | existing fallback | n/a | v1/legacy | n/a | n/a | n/a | n/a | Remains unchanged; no payload translation |
| Backend v2 | `/internal/v1/recommendations/generate` | `recommendation-agent-v2` | `recommendation-features-v2` | `recommendation-inference-v1` | `hybrid-ranking-v1` / `recommendation-package-v1` | `hmac-sha256-v1` | diversity/reasons/template v1 | Supported exact tuple |

The v1 compatibility window lasts until Backend v2 rollout is explicitly
accepted by the Backend owner. No end date or approval is asserted by this
repository. Any tuple mismatch produces typed fallback; no consumer performs
best-effort parsing. Coordination fixture ownership and revisions are recorded
in the adjacent source manifests.

Backend v1 evidence is pinned by reference—not copied—at
`foodmind-backend@7ea2b90c1451d689c59d4ea37d337b4552220f44` in
`contracts/internal/agent/recommendation/v1/backend-fixture-manifest.json`.
The Backend revision exposes only `recommendation-agent-v1` and
`recommendation-features-v1`; no v2 producer/validator is present there.

Backend consumer success fixture:
`artifacts/test-fixtures/recommendation/agent-golden-v2/expected-agent-response.json`
(`sha256:a05b3ef64eb5118c8fd720ec4fe457115e99139897a532f9c705a9864b5dc6e9`,
policy tuple diversity/reasons/template v1). Failure fixtures are the typed
Plan 01 bytes under `contracts/internal/agent/recommendation/v2/fixtures/`.

Pinned local fixture identity at repository baseline
`299125e77a79ee1dc1255c468a41ef31ce1fb431`:

- exact Backend-style golden request:
  `sha256:eca3fd02129ec0269649ff75be3a469052998a0212a79809a933ee3e9932fe0b`;
- exact hybrid inference response:
  `sha256:9326e1a7d782fdef20a2562be565e4e2d47dc5a15e357b136f791f7677fa4cd7`;
- exact expected Agent response:
  `sha256:a05b3ef64eb5118c8fd720ec4fe457115e99139897a532f9c705a9864b5dc6e9`;
- cold-start inference response:
  `sha256:a5e71162fea8ba36744d96b89f643225147dca6c1014d05499ef7d742746d6f0`.

The Agent and inference source manifests pin every coordination fixture used
by local end-to-end tests. `approvalPending: true` is intentional: these bytes
have not been confirmed against an external Backend repository or approved by
the Backend/inference owners. Local consumer tests are not a substitute for
that approval or for staged compatibility evidence.
