# FoodMind Chat Agent

Private FastAPI service for `POST /internal/v1/chat/generate`. It mirrors the
Spring Boot `chat-agent-v1` contract, authenticates `Authorization: Bearer`,
and uses an OpenAI-compatible chat-completions provider such as DeepSeek.

Copy `.env.example` to `.env`, set `CHAT_AGENT_LLM_API_KEY`, then run:

```shell
uv sync
uv run uvicorn chat_agent.main:app --host 0.0.0.0 --port 8004
```

If the provider is unavailable, the service returns a bounded grounded or
navigation fallback so an otherwise valid frontend message is not lost.

