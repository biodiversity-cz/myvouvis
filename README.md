# myvouvis

Herbarijní **DwC databot**: celý arch → LM2 detekce primárního štítku → vision LLM (e-infra) → Darwin Core JSON do `databots.databot_results` (stejný kontrakt jako [coco-bbox-detector](https://github.com/biodiversity-cz/coco-bbox-detector)).

## Rychlý start

Python **3.13** recommended: 
```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
```

nebo Poetry:

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
| `src/api/app.py` | HTTP `/v1/transcribe-full`, `/v1/transcribe-dwc`, `/v1/transcribe-bbox` |
| `src/vendor/lm2.py` | Jediný most k LM2 YOLOv5 |
| `vendor/lm2/` | `component_detector/` + `weights/best.pt` (Git LFS) |

**Výstup do DB:** `result_data.dwc` + `result_data.validation` + LM2 bboxy (`detections`, `primary_label`).

LM2 detektor je vendored v `vendor/lm2/` (viz [NOTICE.md](NOTICE.md)).

[//]: # (obligatory branding for EOSC.CZ)
 
<hr style="margin-top: 100px; margin-bottom: 20px">
<p style="text-align: left"> <img src="https://webcentrum.muni.cz/media/3831863/seda_eosc.png" alt="EOSC CZ Logo" height="90"> </p>
This project output was developed with financial contributions from the EOSC CZ initiative throught the project National Repository Platform for Research Data (CZ.02.01.01/00/23_014/0008787) funded by Programme Johannes Amos Comenius (P JAC) of the Ministry of Education, Youth and Sports of the Czech Republic (MEYS).

<p style="text-align: left"> <img src="https://webcentrum.muni.cz/media/3832168/seda_eu-msmt_eng.png" alt="EU and MŠMT Logos" height="90"> </p>
