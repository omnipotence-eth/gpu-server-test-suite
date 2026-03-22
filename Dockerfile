# ─── GPU Server Diagnostic Test Suite ────────────────────────────
# Multi-stage build with NVIDIA CUDA runtime base image.
# Requires: NVIDIA Container Toolkit (nvidia-docker2)
#
# Build:  docker build -t gpu-diag .
# Run:    docker run --gpus all gpu-diag diag --level medium
# Shell:  docker run --gpus all -it gpu-diag bash
# ─────────────────────────────────────────────────────────────────

# Stage 1: Build dependencies
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip3 install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

LABEL maintainer="Tremayne Timms <ttimmsinternational@gmail.com>"
LABEL description="GPU Server Diagnostic Test Suite"
LABEL version="1.0.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    pciutils \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
WORKDIR /app
COPY src/ src/
COPY config/ config/
COPY pyproject.toml .
COPY requirements.txt .

# Create non-root user for running diagnostics
RUN groupadd -r gpudiag && useradd -r -g gpudiag -d /app gpudiag \
    && mkdir -p /app/reports \
    && chown -R gpudiag:gpudiag /app

# Expose Prometheus metrics port
EXPOSE 9835

# Health check via metrics endpoint
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:9835/health')" || exit 1

# Default: run medium diagnostic level with metrics server
ENTRYPOINT ["python3", "-m", "src.main"]
CMD ["diag", "--level", "medium", "--metrics-port", "9835"]
