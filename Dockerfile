FROM python:3.12-slim AS base

WORKDIR /app

# System deps: Pillow, pyvips C lib, and gcc toolchain to compile pyvips wheel
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libjpeg62-turbo-dev \
        zlib1g-dev \
        libvips-dev \
        build-essential \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (saves ~1.5 GB vs full torch)
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# Install remaining deps (torch already satisfied, pip will skip it)
RUN pip install --no-cache-dir -r requirements.txt

# Remove build toolchain after wheel is compiled to shrink the final image
RUN apt-get purge -y --auto-remove build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY app/ app/
COPY run.py .

# Pre-download model weights at build time so first request is instant
ARG PRELOAD_MODEL=false
RUN if [ "$PRELOAD_MODEL" = "true" ]; then \
    python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('vikhyatk/moondream2', revision='2025-01-09'); \
    AutoModelForCausalLM.from_pretrained('vikhyatk/moondream2', revision='2025-01-09', trust_remote_code=True)"; \
    fi

ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV OMP_NUM_THREADS=4
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

CMD ["python", "run.py"]
