FROM python:3.10-slim-buster AS base
LABEL maintainer="GordeevAS"

# Set up a non-root user
RUN useradd -m -U appuser

# Install essential packages and cleanup
USER root
RUN apt-get update && apt-get install -y \
    wget libssl-dev libffi-dev cmake \
    libnuma-dev pkgconf libbz2-dev && \
    apt-get clean libturbojpeg

# Switch to the non-root user
USER appuser

WORKDIR /app

# Copy source code
COPY --chown=appuser:appuser src/ src/

# Build the production image
FROM base AS prod

# Install requirements
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=appuser:appuser .env .
COPY --chown=appuser:appuser README.md .
COPY --chown=appuser:appuser setup.py .
COPY --chown=appuser:appuser configs/ configs/
RUN pip install --no-cache-dir .

ARG PORT=8000
ENV PORT=$PORT
CMD uvicorn --host 0.0.0.0 --port $PORT src.app:kafka_app

# tests & lint image
FROM base AS test

COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY --chown=appuser:appuser pyproject.toml .
COPY --chown=appuser:appuser setup.cfg .
