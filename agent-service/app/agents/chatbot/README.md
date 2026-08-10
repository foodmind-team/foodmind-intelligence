# FoodMind Chat Agent

Private FastAPI service for `POST /internal/v1/chat/generate`. It mirrors the
Spring Boot `chat-agent-v1` contract, authenticates `Authorization: Bearer`,
and uses an OpenAI-compatible chat-completions provider such as DeepSeek.

Copy `.env.example` to `.env`, set `CHAT_AGENT_LLM_API_KEY` (or the shared
`DEEPSEEK_API_KEY`), then run:

```shell
uv sync
uv run uvicorn chat_agent.main:app --host 0.0.0.0 --port 8004
```

For a repeatable local service, run this from the `foodmind-intelligence`
repository root instead. Its `CHAT_AGENT_INTERNAL_SERVICE_TOKEN` must match the
Backend `CHAT_AGENT_SERVICE_TOKEN`.

```powershell
$env:CHAT_AGENT_INTERNAL_SERVICE_TOKEN = "local-chat-token"
docker compose -f deployment/local/chat-agent.compose.yaml up --build
```

Set `CHAT_AGENT_LLM_ENABLED=true` and provide `CHAT_AGENT_LLM_API_KEY` or
`DEEPSEEK_API_KEY` only when you want provider-backed answers. The service
starts without a provider key and returns its deterministic grounded/navigation
fallback.

If the provider is unavailable, the service returns a bounded grounded or
navigation fallback so an otherwise valid frontend message is not lost.
