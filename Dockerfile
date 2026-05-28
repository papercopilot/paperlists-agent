# --- Stage 1: build sqlite FTS5 index from the JSON corpus ---
FROM python:3.11-slim AS builder

WORKDIR /build
ARG PAPERLISTS_DATA_REF=main
RUN pip install --no-cache-dir orjson

COPY query-api/paperlists_api ./paperlists_api

# Fetch the upstream paperlists JSON corpus inside the builder to avoid
# committing or uploading hundreds of MB of JSON through the deployment path.
RUN python - <<'PY' "$PAPERLISTS_DATA_REF"
import sys
import tarfile
import urllib.request
from pathlib import Path

ref = sys.argv[1]
archive = Path("/tmp/paperlists.tar.gz")
url = f"https://github.com/papercopilot/paperlists/archive/refs/heads/{ref}.tar.gz"
urllib.request.urlretrieve(url, archive)
with tarfile.open(archive, "r:gz") as tar:
    tar.extractall("/tmp")
src = next(Path("/tmp").glob("paperlists-*"))
for child in src.iterdir():
    if child.is_dir() and not child.name.startswith(".") and child.name != "tools":
        target = Path("/data") / child.name
        target.parent.mkdir(parents=True, exist_ok=True)
        child.rename(target)
PY

RUN python -m paperlists_api.indexer /data /build/papers.db && \
    python -c "import sqlite3; c=sqlite3.connect('/build/papers.db'); c.execute('VACUUM'); c.close()" && \
    ls -lh /build/papers.db

# --- Stage 2: lean runtime ---
FROM python:3.11-slim

WORKDIR /app
ARG RAILWAY_GIT_COMMIT_SHA
ARG RAILWAY_GIT_BRANCH
ARG RAILWAY_SNAPSHOT_ID
ARG RAILWAY_ENVIRONMENT
ARG RAILWAY_ENVIRONMENT_NAME
ENV PYTHONUNBUFFERED=1 \
    PAPERLISTS_DB=/app/papers.db \
    PAPERLISTS_DB_IMMUTABLE=1 \
    PAPERLISTS_DB_CACHE_KIB=65536 \
    PAPERLISTS_DB_MMAP_BYTES=536870912 \
    PAPERLISTS_GIT_SHA=${RAILWAY_GIT_COMMIT_SHA} \
    PAPERLISTS_GIT_BRANCH=${RAILWAY_GIT_BRANCH} \
    PAPERLISTS_DEPLOYMENT_ID=${RAILWAY_SNAPSHOT_ID} \
    PAPERLISTS_ENVIRONMENT=${RAILWAY_ENVIRONMENT_NAME} \
    WEB_CONCURRENCY=4

COPY query-api/pyproject.toml ./pyproject.toml
COPY query-api/paperlists_api ./paperlists_api
RUN pip install --no-cache-dir .

COPY --from=builder /build/papers.db /app/papers.db

EXPOSE 8000
CMD ["uvicorn", "paperlists_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
