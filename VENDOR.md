# Vendored VoucherVision

Upstream: [Gene-Weaver/VoucherVision](https://github.com/Gene-Weaver/VoucherVision)

- **Pinned commit:** `a579b7c0c931f55ad103a355f9f5fa32a1b368a1` (same as `main` at import time, message: “NBC prompt #5”)
- **Vendored path:** `VoucherVision/` (working tree; `.git` excluded from the snapshot)
- **Submodules** (LLaVA, SLTP, quick-diff) are included in the tree as checked out in that commit.

To refresh intentionally: clone the upstream repo, check out a chosen tag or commit, run `git submodule update --init --recursive`, sync into `VoucherVision/`, and update this file.

The FastAPI service adds `VoucherVision` to `PYTHONPATH` and calls `voucher_vision` from `vouchervision.vouchervision_main` (no Streamlit), with `is_real_run=False` and `progress_report=None` so the CLI path does not require a GUI `progress_report`.

LLM to e-INFRA: set `OPENAI_BASE_URL` (e.g. `https://llm.ai.e-infra.cz/v1/`), `OPENAI_API_KEY`, and optionally `OPENAI_LLM_MODEL` to the real model id served by the endpoint. `OPENAI_LLM_MODEL` overrides the API model name while keeping a supported `LLM_version` in VoucherVision (e.g. `GPT 4o mini 2024-07-18`).

OCR: Google Cloud Vision (e.g. `OCR_option: [normal]`) needs `GOOGLE_PROJECT_ID`, `GOOGLE_LOCATION`, and `GOOGLE_APPLICATION_CREDENTIALS` (JSON string of the service account, same convention as VoucherVision `is_hf` mode).

## Web UI a Kubernetes

- **Nahrát / Darwin Core tabulka:** otevřete `/app` (GET). Výsledky běhů a náhledy ořezů jsou v `data/runs/{run_id}/` (lokální cache; v `.gitignore`).
- **Kubernetes:** manifesty jsou v `k8s/`. Aplikace: `kubectl apply -f k8s/`.
