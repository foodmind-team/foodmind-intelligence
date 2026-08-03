# Recommendation Agent release evidence index

This index records redacted, reproducible evidence for the Agent release
candidate. It does not store runtime request bodies, tokens, model keys, real
identifiers, raw logs, screenshots, or model artifacts.

## Release identity

| Field | Value |
| --- | --- |
| Evidence date | 2026-08-03 |
| Tester | Codex automated synthetic harness |
| Local environment | Windows; Python 3.13; uv 0.12.1 |
| Repository baseline | Parent `299125e77a79ee1dc1255c468a41ef31ce1fb431`; Agent image source identity recorded below |
| Local Agent image digest | `sha256:9a9214256db7b710deb6ccd17b9db493bbd86277fb5d22305bd5d7f3329429ff` (unpublished local evidence image) |
| Image source-tree identity | 51 production-input files; `sha256:48489c62dffd7d68d7fc3aaa85326a1fb58c648787cf45a0a2ba79b3eff7f926` |
| Backend revision | `foodmind-backend@7ea2b90c1451d689c59d4ea37d337b4552220f44`; v1 present and focused suite green, v2 absent |
| Architecture-doc revision | `foodmind-docs@9503e0e34bda997f06100a08bdb2262eb5096b32`; v2 remains proposed |
| Inference/ML revision | Fixture manifest baseline `299125e77a79ee1dc1255c468a41ef31ce1fb431`; `foodmind-ml@91c18e7afcbd0ef8ce0af54385d83a69d24645dc` has no inference-v1 package; owner approval pending |
| Contract tuple | Agent v2; features v2; inference v1; hybrid/package/key v1 |
| Policy tuple | diversity v1; reasons v1; template v1 |

## Reproducible local evidence

- Golden SHA-256 file:
  `artifacts/test-fixtures/recommendation/agent-golden-v2/checksums.sha256`.
- Agent producer fixture manifest:
  `contracts/internal/agent/recommendation/v2/fixtures/source-manifest.json`.
- Canonical Backend v1 references:
  `contracts/internal/agent/recommendation/v1/backend-fixture-manifest.json`.
- Inference consumer fixture manifest:
  `contracts/internal/inference/recommendation/v1/consumer-fixtures/source-manifest.json`.
- Feature evidence manifest:
  `contracts/internal/shared/recommendation-features/v2/source-manifest.json`.
- Scenario expectations and actual local status:
  `docs/testing/recommendation-agent-uat-matrix.md`.
- Compatibility identity:
  `docs/architecture/recommendation-agent-compatibility-matrix.md`.
- Security/deployment boundary:
  `docs/operations/recommendation-agent.md` and the service `runbook.md`.

Run from `agent-service/app/agents/recommendation`:

```powershell
uv sync --frozen --dev
uv run pytest -v
uv run python scripts/smoke_private_agent.py --mode fixture
uv run python scripts/verify_release_evidence.py
```

The verifier fails closed for checksum drift or a missing evidence file. Its
external-gate line is informational and deliberately reports NOT RUN; it never
converts absent staged or rollback proof into a pass.

Local results recorded on 2026-08-03:

| Command/check | Result |
| --- | --- |
| `uv sync --frozen --dev` | PASS; 66 locked packages checked |
| `uv run ruff format --check .` | PASS; 105 files formatted |
| `uv run ruff check .` | PASS |
| `uv run mypy src tests` | PASS; 101 source/test files |
| `RECOMMENDATION_AGENT_RUN_DOCKER_SMOKE=1 uv run pytest -W error --cov=src/recommendation_agent --cov-fail-under=85 -q` | PASS; 167 passed, no skips or warnings, 89% branch-aware coverage |
| `uv run python scripts/smoke_private_agent.py --mode fixture` | PASS; ready 200, response 200, one inference call, three results |
| `uv run python scripts/verify_release_evidence.py` | PASS; 13 checksummed fixtures, 5 evidence files, and the 51-file image source identity |
| `check-jsonschema==0.33.3` meta/positive fixture checks | PASS |
| `pip-audit==2.9.0` against the locked environment | PASS; no known vulnerabilities |
| `docker compose ... config --quiet` | PASS |
| Digest-pinned Docker build | PASS; exact local digest `sha256:9a9214256db7b710deb6ccd17b9db493bbd86277fb5d22305bd5d7f3329429ff`, 77,899,995 bytes |
| Non-root/read-only/health/private contract smoke | PASS; UID/GID 10001, read-only root, healthy, one fixture inference call, three grounded results |
| CycloneDX SBOM | PASS; 132 components, 372,220 bytes, `sha256:5cfcf319fca23169f10c634f8fae7fe0460a3791edca93d558beb946b3e3d205` |
| Docker Scout CVE gate | PASS; 676 packages, 0 Critical/High/Medium/Low findings for the exact image digest |
| Trivy v0.56.1 image gate | PASS; Debian 13.6 image, 0 High/Critical findings |
| License policy gate | PASS; no GPL/AGPL dependency license; local package metadata excluded |
| Gitleaks v8.28.0 scoped scan | PASS; all 10 Recommendation Agent code/contract/deployment/evidence paths clean |
| Backend focused recommendation suite | PASS; exact owner command, 11 tests, v1 only, PostgreSQL Testcontainers included |
| Local image rollback/restore simulation | PASS; prior candidate ready+smoke 19,112 ms, current digest restore ready+smoke 18,756 ms |

The local image is not pushed or signed and therefore is not a production
registry identity.

## External evidence placeholders

Backend suite output, staged telemetry, platform network-policy proof, the
published/signed image identity, alert observations, and both rollback drill
records must be stored in an owner-approved redacted evidence system. Add only
links, checksums, tool versions, UTC timestamps, and reviewers here after those
artifacts exist. Production release remains blocked while these are absent.
The exact Backend command run from `foodmind-backend` was:

```powershell
.\mvnw.cmd test "-Dtest=*RecommendationAgent*,*AgentResultValidator*,*RecommendationTransaction*"
```

It passed 11 v1 tests. No Backend v2 adapter, DTO, validator, fixture, flag, or
v2-focused command exists at the pinned revision, so this result proves v1
retention only and cannot be represented as v2 acceptance.
