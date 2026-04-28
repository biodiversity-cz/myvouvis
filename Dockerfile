# Synchronous VoucherVision API; GPU optional (CPU is slower for LM2, see VENDOR.md).
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/VoucherVision:/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY VoucherVision/requirements.txt /app/VoucherVision/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN mkdir -p /app/data/runs
COPY vv_api/ /app/vv_api/
COPY VoucherVision/ /app/VoucherVision/

# Non-root user (Kubernetes Pod Security restricted / runAsNonRoot)
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -M -s /usr/sbin/nologin appuser && \
    mkdir -p /app/.cache && \
    chown -R appuser:appgroup /app
ENV HOME=/app

USER appuser

# One worker: one long request; --timeout should align with ingress (~180s).
EXPOSE 8080
CMD ["gunicorn", "vv_api.main:app", "-k", "uvicorn.workers.UvicornWorker", "-w", "1", "-b", "0.0.0.0:8080", "--timeout", "180", "--graceful-timeout", "30", "--access-logfile", "-", "--error-logfile", "-"]
