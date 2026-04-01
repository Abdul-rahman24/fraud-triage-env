# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_MODE=in-repo
ARG ENV_NAME=fraud_triage_env

COPY . /app/env
WORKDIR /app/env

RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi
    
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# ==========================================
# Final runtime stage (Hugging Face Compliant)
# ==========================================
FROM ${BASE_IMAGE}

# 1. HUGGING FACE REQUIREMENT: Create and use a non-root user (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# 2. Copy files from builder and grant ownership to the new non-root 'user'
COPY --from=builder --chown=user /app/env/.venv $HOME/app/.venv
COPY --from=builder --chown=user /app/env $HOME/app/env

ENV PATH="$HOME/app/.venv/bin:$PATH"
ENV PYTHONPATH="$HOME/app/env:$PYTHONPATH"

# 3. HUGGING FACE REQUIREMENT: Change Healthcheck port to 7860
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# 4. HUGGING FACE REQUIREMENT: Start Uvicorn on port 7860
CMD ["sh", "-c", "cd $HOME/app/env && uvicorn server.app:app --host 0.0.0.0 --port 7860"]