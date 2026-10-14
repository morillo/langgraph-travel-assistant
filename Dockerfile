# ── Stage 1: base image ───────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# uv (static binary) — https://docs.astral.sh/uv/guides/integration/docker/
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

WORKDIR /app

# ── Stage 2: dependencies ─────────────────────────────────────────────────────
# Copy dependency files first so Docker caches this layer.
# The layer only re-runs when pyproject.toml or uv.lock changes.
COPY pyproject.toml uv.lock ./

# Runtime deps only (no dev / evals groups), straight from the lockfile.
ENV UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1
RUN uv sync --frozen --no-default-groups --no-install-project

# ── Stage 3: application ──────────────────────────────────────────────────────
COPY app/ ./app/

# Expose FastAPI (8000) and Phoenix (6006)
EXPOSE 8000 6006

# Run the FastAPI server from the project venv
ENV PATH="/app/.venv/bin:$PATH"
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
