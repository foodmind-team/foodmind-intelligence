"""Verify checksummed local Recommendation Agent release evidence."""

import hashlib
import json
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
GOLDEN = REPOSITORY_ROOT / "artifacts/test-fixtures/recommendation/agent-golden-v2"
INFERENCE_FIXTURES = REPOSITORY_ROOT / "contracts/internal/inference/recommendation/v1/consumer-fixtures"
AGENT_FIXTURES = REPOSITORY_ROOT / "contracts/internal/agent/recommendation/v2/fixtures"
FEATURE_CONTRACT = REPOSITORY_ROOT / "contracts/internal/shared/recommendation-features/v2"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IMAGE_SOURCE_SHA256 = "0e0f2e74d4de939461cc44287f82ef32ca50087bc9ded8f06ac40642b7bfaee0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _verify_golden() -> int:
    verified = 0
    for line in (GOLDEN / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        checksum, relative = line.split(maxsplit=1)
        if _sha256(GOLDEN / relative) != checksum:
            raise RuntimeError(f"golden checksum mismatch: {relative}")
        verified += 1
    return verified


def _verify_file_manifest(root: Path) -> int:
    manifest = _load(root / "source-manifest.json")
    verified = 0
    for entry in manifest["files"]:
        if _sha256(root / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"fixture checksum mismatch: {entry['path']}")
        verified += 1
    return verified


def _verify_feature_manifest() -> int:
    manifest = _load(FEATURE_CONTRACT / "source-manifest.json")
    target = REPOSITORY_ROOT / manifest["path"]
    if _sha256(target) != manifest["sha256"]:
        raise RuntimeError("feature evidence catalog checksum mismatch")
    return 1


def _verify_required_evidence_files() -> int:
    relative_paths = (
        "docs/architecture/recommendation-agent-compatibility-matrix.md",
        "docs/testing/recommendation-agent-uat-matrix.md",
        "docs/testing/recommendation-agent-evidence/README.md",
        "docs/operations/recommendation-agent.md",
        "agent-service/app/agents/recommendation/runbook.md",
    )
    missing = [relative for relative in relative_paths if not (REPOSITORY_ROOT / relative).is_file()]
    if missing:
        raise RuntimeError(f"missing evidence files: {', '.join(missing)}")
    return len(relative_paths)


def _verify_image_source_tree() -> int:
    paths = [
        PROJECT_ROOT / ".dockerignore",
        PROJECT_ROOT / "Dockerfile",
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "uv.lock",
        PROJECT_ROOT / "README.md",
        *sorted((PROJECT_ROOT / "src").rglob("*.py")),
    ]
    entries = [
        f"{path.relative_to(PROJECT_ROOT).as_posix()} {_sha256(path)}"
        for path in sorted(paths, key=lambda value: value.relative_to(PROJECT_ROOT).as_posix())
    ]
    digest = hashlib.sha256("\n".join(entries).encode()).hexdigest()
    if digest != EXPECTED_IMAGE_SOURCE_SHA256:
        raise RuntimeError("image source tree checksum mismatch")
    return len(paths)


def main() -> None:
    count = (
        _verify_golden()
        + _verify_file_manifest(AGENT_FIXTURES)
        + _verify_file_manifest(INFERENCE_FIXTURES)
        + _verify_feature_manifest()
    )
    evidence_count = _verify_required_evidence_files()
    source_count = _verify_image_source_tree()
    print(f"PASS local checksums: verified={count}")
    print(f"PASS local evidence index: files={evidence_count}")
    print(f"PASS image source tree: files={source_count} sha256={EXPECTED_IMAGE_SOURCE_SHA256}")
    print(
        "NOT RUN external gates: Backend v2 suite unavailable, staged private integration, "
        "published image signing, staged rollback drills"
    )


if __name__ == "__main__":
    main()
