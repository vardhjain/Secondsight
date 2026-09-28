# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Dockerfile for Secondsight.
#
# Builds a CPU-only image that launches the interactive Gradio Re-ID demo.
# Dependencies come from the committed uv.lock, with CPU torch wheels selected
# by the "cpu" extra. The build context is kept small by .dockerignore (data,
# outputs, weights, and notebooks are excluded and mounted at runtime instead).
#
# Build:
#   docker build -t secondsight .
#
# Run the demo (CPU), reachable at http://localhost:7860:
#   docker run --rm -p 127.0.0.1:7860:7860 \
#       -v "$(pwd)/outputs:/app/outputs:ro" \
#       secondsight
# ---------------------------------------------------------------------------

# uv as a named stage so Dependabot's docker updater can track its version.
FROM ghcr.io/astral-sh/uv:0.12.19 AS uv

FROM python:3.11-slim-trixie AS base

# Quiet, unbuffered Python. uv copies packages (no hardlinks across the cache
# mount), precompiles bytecode, and installs into /app/.venv.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}"

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /app

# Install the locked dependencies first (without project code) for better layer
# caching. The uv cache lives in a BuildKit cache mount, so it never ends up in
# an image layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --extra cpu --extra demo --no-install-project

# Copy the project and install it (non-editable) into the environment.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra cpu --extra demo --no-editable

# Run as an unprivileged user. Everything under /app is only read at runtime,
# so it can stay owned by root; caches (torch, matplotlib, Gradio) go to HOME.
RUN useradd --create-home --uid 1000 appuser
ENV HOME=/home/appuser \
    TORCH_HOME=/home/appuser/.cache/torch \
    REID_WEIGHTS=/app/outputs/best.pth
USER appuser

# Gradio default port.
EXPOSE 7860

# Bind to all interfaces so the published port reaches the demo, and turn off
# Gradio's usage analytics.
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False

# Liveness probe: the Gradio server should answer on the exposed port. The app
# builds the model and indexes the gallery images before the server starts
# listening, so the start period is long enough to cover that work on a CPU.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/').read()" || exit 1

# Launch the interactive probe-vs-gallery demo by default. Weights are read
# from REID_WEIGHTS; the app falls back to a random-init model if absent.
CMD ["python", "app/gradio_app.py"]
