"""Shared Recommendation Agent tests."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from recommendation_agent.config.settings import Settings
from recommendation_agent.main import create_app

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
AGENT_FIXTURES = REPOSITORY_ROOT / "contracts/internal/agent/recommendation/v2/fixtures"
GOLDEN_FIXTURES = REPOSITORY_ROOT / "artifacts/test-fixtures/recommendation/agent-golden-v2"


@pytest.fixture(autouse=True)
def disable_live_llm_in_offline_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let the developer's shared DeepSeek key escape into offline tests."""

    monkeypatch.setenv("RECOMMENDATION_AGENT_LLM_ENABLED", "false")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture
def settings() -> Settings:
    return Settings(internal_service_token=SecretStr("test-internal-token"), app_env="test")


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings=settings, install_default_workflow=False)) as test_client:
        yield test_client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-internal-token"}
