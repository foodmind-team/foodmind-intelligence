# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
FROM ghcr.io/astral-sh/uv:0.12.1@sha256:cf4eedcaa81655197f625739489effcbe71b61ceb1506f332c3facae5deceded AS uv
FROM python:3.14-slim@sha256:cea0e6040540fb2b965b6e7fb5ffa00871e632eef63719f0ea54bca189ce14a6

ARG SOURCE_REVISION=unknown
LABEL org.opencontainers.image.title="FoodMind Recommendation Agent" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.source="foodmind-intelligence"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PATH="/app/.venv/bin:${PATH}" \
    HOME=/nonexistent \
    TMPDIR=/tmp

COPY --from=uv /uv /usr/local/bin/uv

RUN groupadd --gid 10001 recommendation \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin recommendation

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev \
    && find /app -type d -name __pycache__ -prune -exec rm -rf {} +

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=2s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=1).read()"]

CMD ["uvicorn", "recommendation_agent.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "40", "--timeout-graceful-shutdown", "5", "--no-access-log"]
