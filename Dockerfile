# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

# Keep working directory consistent
WORKDIR /app/env

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

COPY . /app/env

RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi
    
RUN uv sync --no-install-project --no-editable
RUN uv sync --no-editable

# ==========================================
# Final runtime stage (Hugging Face Compliant)
# ==========================================
FROM ${BASE_IMAGE}

# 1. Create the non-root user (UID 1000)
RUN useradd -m -u 1000 user
USER user

# 2. Use the exact same path as the builder so the .venv doesn't break
WORKDIR /app/env

# 3. Copy the built environment and grant ownership to the user
COPY --from=builder --chown=user /app/env /app/env

# 4. Add the virtual environment to the system path
ENV PATH="/app/env/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# 5. Hugging Face specific port settings
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]