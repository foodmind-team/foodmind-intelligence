FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1

RUN pip install --no-cache-dir uv==0.12.1 \
    && groupadd --gid 10001 foodmind \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin foodmind

WORKDIR /service
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY app ./app

USER 10001:10001
EXPOSE 8002
CMD ["uv", "run", "--frozen", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
