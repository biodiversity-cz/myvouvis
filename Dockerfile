# Herbarium DwC databot + optional HTTP API (e-infra / K8s).
FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app/vendor/lm2

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.8.4

COPY pyproject.toml /app/
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main

COPY src/ /app/src/
COPY config/ /app/config/
COPY prompts/ /app/prompts/
# vendor/lm2: component_detector + weights/best.pt (LM2 acd, release v-2-3)
COPY vendor/lm2/ /app/vendor/lm2/

RUN groupadd -g 1000 appgroup \
    && useradd -u 1000 -g appgroup -M -s /usr/sbin/nologin appuser \
    && chown -R appuser:appgroup /app
USER appuser

EXPOSE 8080
# Default: HTTP API (K8s probes on :8080). Batch: override command to
# ["python", "src/main.py", "herbarium-dwc"]
ENTRYPOINT ["uvicorn"]
CMD ["api.app:app", "--host", "0.0.0.0", "--port", "8080"]
