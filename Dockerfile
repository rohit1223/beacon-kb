# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1: builder
# Installs the package and the server extra into a separate layer so that
# only the installed site-packages are copied to the runtime stage.
# Python 3.13-slim is chosen because Docling (>=2.5) and qdrant-client (>=1.12)
# publish pre-built wheels for this version, so no compiler is required.
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder

WORKDIR /build

# Install build tooling only (not present in slim base).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy package metadata and all source files.
# setuptools needs README.md, LICENSE, and src/beacon_kb/version.py
# (via version = {attr = "beacon_kb.version.__version__"}) at build time.
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Install the package with the server extra into the prefix.
# --root-user-action=ignore silences the pip warning about running as root
# inside Docker.
RUN pip install --no-cache-dir --root-user-action=ignore \
        ".[server]"

# ---------------------------------------------------------------------------
# Stage 2: runtime
# Minimal image with only the installed packages.
# No build tools, no package manager caches.
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

# Create a non-root user so the container does not run as root.
RUN groupadd --system beacon && useradd --system --gid beacon --no-create-home beacon

WORKDIR /app

# Copy installed packages from the builder.
COPY --from=builder /usr/local/lib/python3.13/site-packages/ \
                     /usr/local/lib/python3.13/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Create data directories and give ownership to the non-root user.
RUN mkdir -p /data/qdrant && chown -R beacon:beacon /data /app

USER beacon

# Data volumes:
# /data/beacon.db - SQLite state database
# /data/qdrant    - Qdrant embedded storage
VOLUME ["/data"]

EXPOSE 8000

# Environment variable defaults.
# Override BEACON_QDRANT__PATH and BEACON_STATE__DB_PATH via env or compose.
ENV BEACON_STATE__DB_PATH=/data/beacon.db \
    BEACON_QDRANT__PATH=/data/qdrant \
    BEACON_SERVER__HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# The app factory entrypoint.
# --factory flag tells uvicorn that the import path returns a factory callable.
CMD ["uvicorn", "beacon.server.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
