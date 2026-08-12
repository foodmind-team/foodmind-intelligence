FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN uv sync --no-dev

FROM python:3.13-slim-bookworm AS runtime
RUN groupadd --system foodmind && useradd --system --gid foodmind --home-dir /app foodmind
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
USER foodmind
EXPOSE 8001
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health/ready', timeout=2)"]
CMD ["uvicorn", "chat_agent.main:app", "--host", "0.0.0.0", "--port", "8001"]
