from cooking_plan_agent.config.settings import Settings


def test_shared_agent_env_ignores_unrelated_keys_and_redacts_secrets(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.delenv("COOKING_PLAN_INTERNAL_SERVICE_TOKEN", raising=False)
    monkeypatch.delenv("COOKING_PLAN_LLM_API_KEY", raising=False)
    unrelated_sentinel = "shared-deepseek-sentinel-must-not-leak"
    cooking_sentinel = "cooking-secret-sentinel-must-not-leak"
    shared_env = tmp_path / ".env"
    shared_env.write_text(
        "\n".join(
            (
                f"DEEPSEEK_API_KEY={unrelated_sentinel}",
                "CHAT_AGENT_ENABLED=true",
                "COOKING_PLAN_INTERNAL_SERVICE_TOKEN=test-token",
                f"COOKING_PLAN_LLM_API_KEY={cooking_sentinel}",
            )
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=shared_env)

    assert settings.internal_service_token == "test-token"
    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == cooking_sentinel
    rendered = repr(settings)
    captured = capsys.readouterr()
    assert unrelated_sentinel not in rendered + captured.out + captured.err
    assert cooking_sentinel not in rendered + captured.out + captured.err


def test_shared_deepseek_key_is_used_when_no_cooking_override(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("COOKING_PLAN_LLM_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    shared_env = tmp_path / ".env"
    shared_env.write_text("DEEPSEEK_API_KEY=shared-cooking-test-key\n", encoding="utf-8")

    settings = Settings(_env_file=shared_env)

    assert settings.llm_api_key is not None
    assert settings.llm_api_key.get_secret_value() == "shared-cooking-test-key"
