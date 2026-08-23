# ── Stage 1: base image ───────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# Install system utilities needed by Poetry and requests
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# ── Stage 2: dependencies ─────────────────────────────────────────────────────
# Copy dependency files first so Docker caches this layer.
# The layer only re-runs when pyproject.toml or poetry.lock changes.
COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --without dev --no-interaction --no-ansi

# ── Stage 3: application ──────────────────────────────────────────────────────
COPY app/ ./app/

# Expose FastAPI (8000) and Phoenix (6006)
EXPOSE 8000 6006

# Run the FastAPI server
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
