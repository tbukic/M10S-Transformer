ARG PYTHON_VERSION="3.13"
ARG UV_VERSION="0.9.21"

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv_image
FROM python:${PYTHON_VERSION}-slim AS runtime

COPY --from=uv_image /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /data

COPY pyproject.toml README.md ./
COPY src ./src
RUN uv sync --no-dev

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "minimal10digittransformer"]
