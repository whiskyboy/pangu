FROM python:3.12-slim AS builder

WORKDIR /app
RUN pip install --no-cache-dir uv

COPY pyproject.toml .
COPY src/ src/
COPY config/ config/

RUN uv pip install --system .

# ---------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/pangu /usr/local/bin/pangu
COPY config/ config/

VOLUME ["/app/data", "/app/config"]

CMD ["pangu", "run", "start"]
