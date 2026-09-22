# Lumen API image — synthetic demo / cloud plane.
#
# Contains code and Python dependencies only. No data, no model weights, no
# secrets: the build context is an allowlist (.dockerignore), retrieval models
# are fetched into the /models volume at pinned revisions
# (scripts/fetch_models.py), and the database is a separate service.
#
#   docker compose -f docker-compose.demo.yml build api

FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Non-root runtime user; /models is the mount point for the model-weights volume
# (a new named volume inherits this ownership).
RUN useradd --create-home --uid 10001 lumen \
    && mkdir -p /models \
    && chown lumen:lumen /models

COPY --chown=lumen:lumen src/ src/
COPY --chown=lumen:lumen configs/models.json configs/models.json
COPY --chown=lumen:lumen scripts/load_synthetic_demo.py scripts/fetch_models.py scripts/demo_smoke_test.py scripts/

USER lumen

# Demo plane by default. HF_HUB_OFFLINE: the API only ever loads weights from
# /models; scripts/fetch_models.py is run with HF_HUB_OFFLINE=0 to populate it.
ENV LUMEN_DATA_PLANE=demo \
    LUMEN_MODELS_DIR=/models \
    HF_HOME=/models/.hf-cache \
    HF_HUB_OFFLINE=1 \
    LUMEN_TRACING=0

EXPOSE 8000

# One worker: models are process-local and a second worker would load a second copy.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
