"""P5-3: schedule repair loop 配置。"""

from cooking_plan_agent.config.settings import get_settings


def test_repair_defaults(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.delenv("COOKING_PLAN_SCHEDULE_REPAIR_MAX_ATTEMPTS", raising=False)
    monkeypatch.delenv("COOKING_PLAN_SCHEDULE_REPAIR_LLM_ENABLED", raising=False)
    settings = get_settings()
    assert settings.schedule_repair_max_attempts == 2
    assert settings.schedule_repair_llm_enabled is False


def test_repair_env_override(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("COOKING_PLAN_SCHEDULE_REPAIR_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("COOKING_PLAN_SCHEDULE_REPAIR_LLM_ENABLED", "true")
    settings = get_settings()
    assert settings.schedule_repair_max_attempts == 4
    assert settings.schedule_repair_llm_enabled is True
