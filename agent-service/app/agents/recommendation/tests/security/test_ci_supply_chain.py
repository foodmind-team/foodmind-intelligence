"""Supply-chain configuration is immutable and release-gated."""

import re
from pathlib import Path

from conftest import REPOSITORY_ROOT

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_github_actions_are_pinned_to_full_commit_shas() -> None:
    workflow = (REPOSITORY_ROOT / ".github/workflows/recommendation-agent-ci.yml").read_text(encoding="utf-8")
    action_refs = re.findall(r"^\s*-?\s*uses:\s*([^\s#]+)", workflow, flags=re.MULTILINE)
    assert action_refs
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", reference) for reference in action_refs)
    assert "pip-audit==2.9.0" in workflow
    assert "pip-licenses==5.5.5" in workflow
    assert re.search(r"zricethezav/gitleaks:v8\.28\.0@sha256:[0-9a-f]{64}", workflow)
    assert re.search(r"aquasec/trivy:0\.56\.1@sha256:[0-9a-f]{64}", workflow)
    assert '--fail-on "GPL-2.0-only;' in workflow


def test_docker_sources_are_digest_pinned() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.startswith("# syntax=docker/dockerfile:1.7@sha256:")
    base_images = re.findall(r"^FROM\s+([^\s]+)", dockerfile, flags=re.MULTILINE)
    assert len(base_images) == 2
    assert all(re.fullmatch(r"[^@]+@sha256:[0-9a-f]{64}", image) for image in base_images)
