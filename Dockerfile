# syntax=docker/dockerfile:1.7
#
# Production image for the LLM Evaluation & Guardrails Framework.
#
# Multi-stage: dependencies are compiled into a virtualenv in the builder and
# only the finished venv is copied into the runtime stage, so no compiler
# toolchain, package index cache, or build metadata ships to production.
#
# Build:
#   docker build -t llm-eval-guardrails:0.1.0 .
# Run (credentials come from the task/pod role, never from the image):
#   docker run --rm -e LEG_AWS__REGION=us-east-1 llm-eval-guardrails:0.1.0 \
#       llm-eval --dataset s3://bucket/eval.jsonl --output s3://bucket/reports

# --------------------------------------------------------------------------- #
# Stage 1: build the virtualenv
# --------------------------------------------------------------------------- #
FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Dependency metadata is copied before the source so that edits to application
# code reuse the cached dependency layer.
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install .

# --------------------------------------------------------------------------- #
# Stage 2: runtime
# --------------------------------------------------------------------------- #
FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="llm-eval-guardrails" \
      org.opencontainers.image.description="LLM evaluation and guardrails framework for RAG" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.licenses="Proprietary"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    # Force single-line JSON logs even when a TTY is attached, so CloudWatch
    # always receives parseable records.
    LEG_JSON_LOGS=true \
    SERVICE_NAME=llm-eval-guardrails \
    SERVICE_VERSION=0.1.0

# Patch the base image, then drop the package index to keep the layer small.
# ca-certificates is required for TLS to Bedrock, S3 and OpenAI.
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install --no-install-recommends -y ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user with no login shell.
RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --no-create-home --shell /usr/sbin/nologin app

COPY --from=builder --chown=root:root /opt/venv /opt/venv

WORKDIR /app
USER app

# Verifying the import at build time turns a broken dependency set into a
# failed build rather than a crash on the first production invocation.
RUN python -c "import llm_eval_guardrails; print(llm_eval_guardrails.__version__)"

# Confirms the interpreter and package are importable; it makes no network
# call, so it stays valid for batch jobs that expose no HTTP port.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import llm_eval_guardrails, sys; sys.exit(0)"]

ENTRYPOINT ["llm-eval"]
CMD ["--help"]
