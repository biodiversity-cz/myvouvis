# myvouvis — herbarijní DwC databot

Tenký databot: celý arch → LM2 detekce štítku → vision LLM (e-infra) → Darwin Core JSON do `databots.databot_results`.

Vstup: JPEG/PNG/WebP přímo; pyramid TIFF (`.tif`) — pipeline vybere největší úroveň pyramidy.

## Lokálně

Requires **Python 3.13** (same as coco-bbox-detector; 3.11 was only a temporary local fallback).

```bash
cd myvouvis
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# pip install -e . ne — aplikace, ne knihovna (package-mode = false v pyproject.toml)
export PYTHONPATH=src
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://llm.ai.e-infra.cz/v1/
git lfs pull   # vendor/lm2/weights/best.pt (~268 MB)
python -m cli test_image/test.png
```

Alternativa Poetry:

```bash
poetry env use python3.13 && poetry install
export PYTHONPATH=src
poetry run python -m cli test_image/test.png
```

## Batch databot (jako coco)

```bash
export DB_HOST=... DB_DATABASE=... DB_USER=... DB_PASSWORD=...
export S3_ENDPOINT_URL=... S3_BUCKET=... S3_ACCESS_KEY=... S3_SECRET_KEY=...
PYTHONPATH=src poetry run python src/main.py herbarium-dwc
```

## HTTP (test)

```bash
PYTHONPATH=src poetry run uvicorn api.app:app --host 0.0.0.0 --port 8080
curl -F "file=@test_image/test.png" http://localhost:8080/v1/transcribe
```

## K8s

Váhy LM2 jsou v Docker image — bez PVC. Viz [INSTALL-K8S.md](../INSTALL-K8S.md).

- Deployment: `kubectl apply -f k8s/deployment.yaml`
- Secret `vouvis-creds`: `OPENAI_API_KEY`, `OPENAI_LLM_PRESET` (volitelně `DB_*`, `S3_*` pro batch jinde)

## Výstup v DB

Darwin Core + LM2 bounding boxy (pixely, xyxy):

```json
{
  "dwc": { "scientificName": "...", "recordedBy": "..." },
  "validation": { "ok": true, "missing": [] },
  "detections": [
    { "bbox": [3120, 45, 3890, 210], "category": "barcode", "confidence": 0.94 },
    { "bbox": [2850, 420, 3920, 1180], "category": "label", "confidence": 0.88 }
  ],
  "primary_label": {
    "bbox": [2850, 420, 3920, 1180],
    "category": "label",
    "confidence": 0.88
  }
}
```

## LM2 váhy

```bash
git lfs pull   # vendor/lm2/weights/best.pt
./scripts/vendorize_lm2.sh   # ověří vendor/lm2/component_detector
```
