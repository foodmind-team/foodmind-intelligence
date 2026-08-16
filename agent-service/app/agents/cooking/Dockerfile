# ============================================================================
# Cooking Plan Agent — production Docker image (Handbook 12.2)
# ============================================================================
# Features per Handbook 12.2:
#   - Exact Python 3.13 (locked by .python-version and pyproject.toml)
#   - Installs from uv.lock for reproducible builds
#   - Copies only src/ + pyproject.toml + uv.lock (no tests, no .env)
#   - Runs as non-root user "agent"
#   - Production ASGI server (uvicorn with single worker — CP-SAT is CPU-bound)
#   - Exposes only canonical Cooking Agent port 8003
# ============================================================================

FROM ghcr.io/astral-sh/uv:0.12.1@sha256:cf4eedcaa81655197f625739489effcbe71b61ceb1506f332c3facae5deceded AS uv
FROM python:3.13-slim@sha256:6771159cd4fa5d9bba1258caf0b82e6b73458c694d178ad97c5e925c2d0e1a91

# Install uv for deterministic package installation.
COPY --from=uv /uv /usr/local/bin/uv

# Create non-root user (handbook 12.2).
RUN apt-get update \
    && apt-get upgrade --yes \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip uninstall --yes setuptools \
    && groupadd --system agent \
    && useradd --system --no-create-home --gid agent agent

WORKDIR /app

# uv settings for production install.
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_SYSTEM_PYTHON=1

# Copy project files and install in one layer.
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install production dependencies (no dev deps: pytest/ruff/mypy/hypothesis).
RUN uv sync --frozen --no-dev

# Ensure the agent user owns the app directory.
RUN chown -R agent:agent /app

# Switch to non-root user.
USER agent

# Expose only the internal service port.
EXPOSE 8003

# Health check: verify the event loop is alive without external calls.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD /app/.venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://localhost:8003/health/live')" || exit 1

# Production ASGI server. Single worker: CP-SAT is CPU-bound per worker.
# Use the venv Python that uv created to ensure all deps are available.
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "cooking_plan_agent.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8003", \
     "--workers", "1", \
     "--timeout-keep-alive", "30", \
     "--log-level", "info"]
