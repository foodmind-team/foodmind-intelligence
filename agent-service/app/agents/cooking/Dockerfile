# ============================================================================
# Cooking Plan Agent — production Docker image (Handbook 12.2)
# ============================================================================
# Features per Handbook 12.2:
#   - Exact Python 3.13 (locked by .python-version and pyproject.toml)
#   - Installs from uv.lock for reproducible builds
#   - Copies only src/ + pyproject.toml + uv.lock (no tests, no .env)
#   - Runs as non-root user "agent"
#   - Production ASGI server (uvicorn with single worker — CP-SAT is CPU-bound)
#   - Exposes only internal port 8000
# ============================================================================

FROM python:3.13-slim

# Install uv for deterministic package installation.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create non-root user (handbook 12.2).
RUN groupadd --system agent && useradd --system --no-create-home --gid agent agent

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

# The async task store uses /app/data. Create it in the image so the service
# also starts correctly when the container is run without Compose volumes.
RUN mkdir -p /app/data && chown -R agent:agent /app

# Switch to non-root user.
USER agent

# Expose only the internal service port.
EXPOSE 8000

# Health check: only accept traffic once required local services are ready.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD /app/.venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/ready')" || exit 1

# Production ASGI server. Single worker: CP-SAT is CPU-bound per worker.
# Use the venv Python that uv created to ensure all deps are available.
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "cooking_plan_agent.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "30", \
     "--log-level", "info"]
