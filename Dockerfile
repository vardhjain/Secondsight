# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Dockerfile for person-reid-market1501.
#
# Builds a lean image that launches the interactive Gradio Re-ID demo. The
# build context is kept small by .dockerignore (data, outputs, weights, and
# notebooks are excluded and mounted at runtime instead).
#
# Build:
#   docker build -t person-reid-market1501 .
#
# Run the demo (CPU):
#   docker run --rm -p 7860:7860 \
#       -v "$(pwd)/outputs:/app/outputs" \
#       person-reid-market1501
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

# Faster, quieter, reproducible Python; let uv manage the environment in-place.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_TORCH_BACKEND=cpu \
    REID_WEIGHTS=/app/outputs/best.pth \
    PATH="/app/.venv/bin:${PATH}"

# Minimal system libraries required by OpenCV / Pillow at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv from its official, statically-linked image, pinned to an exact
# version for reproducible builds.
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first (without project code) for better layer caching.
COPY pyproject.toml README.md ./
COPY src/reid/__init__.py ./src/reid/__init__.py
RUN uv sync --no-dev --no-install-project

# Copy the rest of the project and install it into the environment.
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY app ./app
RUN uv sync --no-dev

# Run as an unprivileged user. The runtime only reads from the mounted
# /app/outputs (weights) and the venv, so a recursive chown of /app suffices;
# HOME is set so libraries that need a writable home (caches) don't fail.
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
ENV HOME=/home/appuser
USER appuser

# Gradio default port.
EXPOSE 7860

# Bind to all interfaces so the demo is reachable from outside the container.
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# Liveness probe: the Gradio server should answer on the exposed port. The
# start period is generous because the first request constructs the model.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/').read()" || exit 1

# Launch the interactive probe-vs-gallery demo by default. Weights are read
# from REID_WEIGHTS; the app falls back to a random-init model if absent.
CMD ["python", "app/gradio_app.py"]
