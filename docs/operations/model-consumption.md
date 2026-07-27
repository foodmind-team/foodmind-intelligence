# Model Consumption and Rollback

## Purpose

This runbook defines how FoodMind Intelligence consumes a model produced by `foodmind-ml`. No release process is implemented yet; this document establishes the required boundary.

## Release Inputs

An approved release should provide:

- Immutable model-package URI
- Model version
- Package SHA-256
- Manifest
- Feature-schema version
- Inference-contract version
- Evaluation summary
- Model-card/limitations reference
- Producer Git commit

Do not consume a mutable `latest` artifact in staging or production-demo.

## Validation Before Promotion

1. Download the package from the approved registry.
2. Verify package checksum.
3. Validate manifest schema.
4. Confirm inference-contract compatibility.
5. Confirm feature-schema compatibility.
6. Load the package in an isolated validation process.
7. Run known input/output fixtures.
8. Run cold-start fixtures.
9. Confirm model metadata is returned by inference.
10. Record the approved version in deployment configuration.

Failure at any step blocks promotion.

## Startup Behaviour

The inference service should:

1. Read the configured immutable URI and expected checksum.
2. Fetch or mount the package.
3. Verify it before deserialisation.
4. Validate supported artifact types.
5. Load preprocessing metadata and model artifacts.
6. Execute a local smoke fixture.
7. Mark readiness successful.

Liveness may be healthy before readiness. Traffic must not reach a process that has not loaded a valid model.

## Runtime Response Metadata

Every inference response should include:

- `modelVersion`
- `featureSchemaVersion`
- `inferenceContractVersion`
- `modelStatus`
- Availability flags for collaborative features
- Trace ID

Do not expose filesystem paths, registry credentials, or internal serialized objects.

## Rollback

Rollback is configuration-driven:

1. Select the last approved immutable package.
2. Update the deployment's model URI, version, and checksum together.
3. Start new instances.
4. Wait for readiness and smoke checks.
5. Shift traffic.
6. Confirm version and fallback metrics.
7. Retain incident evidence.

Do not overwrite the failed artifact under the same version.

## Local Development

Local inference tests should use small, non-sensitive fixtures under `artifacts/test-fixtures/`. Production or large training artifacts should not be committed.

Suggested local configuration:

```text
APP_ENV=local
MODEL_ARTIFACT_URI=<local-or-test-package>
MODEL_ARTIFACT_SHA256=<expected-checksum>
MODEL_VERSION=<test-version>
```

Exact commands will be documented after the Python project files exist.

## Compatibility Matrix

Maintain a release record containing:

| Runtime commit | Model version | Feature schema | Inference contract | Status |
| --- | --- | --- | --- | --- |
| To be recorded | To be recorded | To be recorded | To be recorded | Planned |

## Security

- Use an allow-listed registry location.
- Verify checksums before deserialisation.
- Restrict artifact write permissions to the training/release process.
- Restrict runtime credentials to read-only access.
- Never load user-supplied model files.
- Do not log package credentials or signed URLs.
