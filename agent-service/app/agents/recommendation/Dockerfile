# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
FROM ghcr.io/astral-sh/uv:0.12.5@sha256:e85be844203885286c60ffad8a858d48afb6c5a5c237ca0e67f12e74b8f174b1 AS uv
FROM python:3.13-slim@sha256:6771159cd4fa5d9bba1258caf0b82e6b73458c694d178ad97c5e925c2d0e1a91

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

RUN apt-get update \
    && apt-get upgrade --yes \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip uninstall --yes setuptools \
    && groupadd --gid 10001 recommendation \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin recommendation

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev \
    && find /app -type d -name __pycache__ -prune -exec rm -rf {} +

USER 10001:10001
EXPOSE 8004

HEALTHCHECK --interval=10s --timeout=2s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8004/health/live', timeout=1).read()"]

CMD ["uvicorn", "recommendation_agent.main:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "1", "--limit-concurrency", "40", "--timeout-graceful-shutdown", "5", "--no-access-log"]
