FROM python:3.12-slim AS base

WORKDIR /app

RUN pip install --no-cache-dir uv

# Copy source + config first (needed by hatchling build)
COPY pyproject.toml .
COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/

RUN uv pip install --system .

CMD ["python", "-m", "pangu.main"]
