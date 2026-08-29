# finrag — retrieval-augmented question answering over SEC 10-K filings.
#
# Two stages so the runtime image carries no compiler and no build metadata,
# and one deliberate choice in each.
#
# Build:  docker build -t finrag .
# Run:    docker compose up

# ----------------------------------------------------------------- builder
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch, installed first so the dependency resolver treats it as
# satisfied. The default wheel bundles CUDA -- roughly two gigabytes of it --
# which a container with no GPU can do exactly nothing with.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

WORKDIR /src
# Only the metadata first: this layer is cached unless the dependencies
# themselves change, so editing a source file does not reinstall torch.
COPY pyproject.toml README.md ./
COPY src ./src
# Serving extras only. [eval] would add mlflow, ragas and datasets, none of
# which a request touches, and [semantic] would add unstructured and spaCy --
# needed to *build* an index, never to answer from one.
RUN pip install ".[local,api,app,groq,vertex]"

# ----------------------------------------------------------------- runtime
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/models \
    # gRPC logs one line per file descriptor after a fork, and the embedding
    # model forks on load.
    GRPC_VERBOSITY=ERROR

# curl is for the healthcheck below; nothing else needs it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Bake both models into the image. Without this the first request downloads
# ~175MB from HuggingFace, which makes a cold container slow and, worse, makes
# answering depend on HuggingFace being reachable at that moment.
RUN finrag warmup

WORKDIR /app
COPY app.py ./

# An unprivileged user owning only what it must write to.
RUN useradd --create-home --uid 10001 finrag \
    && chown -R finrag:finrag /opt/models /app
USER finrag

EXPOSE 8000 8501

# /ready rather than /health: a container with no index is alive and useless,
# and an orchestrator should know the difference.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/ready || exit 1

CMD ["finrag", "serve", "--host", "0.0.0.0", "--port", "8000"]
