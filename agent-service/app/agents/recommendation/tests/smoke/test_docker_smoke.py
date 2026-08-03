import os
import shutil
import subprocess
from pathlib import Path

import pytest


def test_dockerfile_is_non_root_minimal_and_health_bounded() -> None:
    project = Path(__file__).resolve().parents[2]
    dockerfile = (project / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.13-slim" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert "COPY src ./src" in dockerfile
    assert "COPY tests" not in dockerfile
    assert "HEALTHCHECK" in dockerfile and "/health/live" in dockerfile
    assert ".env" not in dockerfile


@pytest.mark.skipif(
    shutil.which("docker") is None or os.getenv("RECOMMENDATION_AGENT_RUN_DOCKER_SMOKE") != "1",
    reason="set RECOMMENDATION_AGENT_RUN_DOCKER_SMOKE=1 with Docker available",
)
def test_prebuilt_image_runs_non_root_read_only() -> None:
    docker_path = shutil.which("docker")
    assert docker_path is not None
    result = subprocess.run(  # noqa: S603
        [
            docker_path,
            "run",
            "--rm",
            "--read-only",
            "--user",
            "10001:10001",
            "--entrypoint",
            "python",
            "foodmind-recommendation-agent:ci",
            "-c",
            "import os; assert os.getuid() == 10001; import recommendation_agent",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
