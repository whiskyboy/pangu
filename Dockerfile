FROM python:3.12-slim AS base

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
RUN uv pip install --system .

COPY src/ src/
COPY config/ config/

CMD ["python", "-m", "trading_agent.main"]
