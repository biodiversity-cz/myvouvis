# myvouvis

Herbarijní **DwC databot**: celý arch → LM2 detekce primárního štítku → vision LLM (e-infra) → Darwin Core JSON do `databots.databot_results` (stejný kontrakt jako [coco-bbox-detector](https://github.com/biodiversity-cz/coco-bbox-detector)).

## Rychlý start

Python **3.13** recommended (see coco-bbox-detector). **`pip install -e .` nefunguje** — projekt je aplikace (`package-mode = false`), ne pip balíček. Použijte:

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

Or Poetry:

```bash
poetry env use python3.13
poetry install
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://llm.ai.e-infra.cz/v1/
# LM2 váhy jsou v vendor/lm2/weights/best.pt (git lfs pull)
PYTHONPATH=src poetry run python -m cli test_image/test.png
```

Viz [docs/getting-started.md](docs/getting-started.md).

## Struktura

| Cesta | Účel |
|-------|------|
| `src/pipeline/sheet.py` | Jádro `process_sheet()` |
| `src/bots/implementations/herbarium_dwc_databot.py` | Batch databot adapter |
| `src/api/app.py` | HTTP `/v1/transcribe` (test) |
| `src/vendor/lm2.py` | Jediný most k LM2 YOLOv5 |
| `vendor/lm2/` | `component_detector/` + `weights/best.pt` (Git LFS) |

**Výstup do DB:** jen `result_data.dwc` + `result_data.validation` — žádné bboxy.

Po `vendorize_lm2.sh` a nastavení vah lze smazat `VoucherVision/` a `vv_api/`.
