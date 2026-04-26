from __future__ import annotations

import pathlib
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from vv_api.config import get_settings
from vv_api.pipeline import transcribe_to_dwc
from vv_api import web as web_routes

_STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="myVoucherVision API",
    version="0.1.0",
    description="myVoucherVision: web UI and API over VoucherVision; LLM via OpenAI-compatible API (e.g. e-INFRA), DwC mapping; UI at /app.",
)

app.include_router(web_routes.router)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/healthz")
def healthz() -> dict:
    v = get_settings().vendor_root
    ok = (v / "vouchervision" / "vouchervision_main.py").is_file()
    return {"status": "ok" if ok else "degraded", "vendor_root": str(v)}


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/app", status_code=302)


@app.post("/v1/transcribe")
def transcribe(
    file: UploadFile | None = File(default=None),
    include_vv_raw: bool = Form(default=False),
) -> JSONResponse:
    """
    Accept one herbarium / label image; run VoucherVision and return DwC-shaped JSON
    plus optional raw `last_JSON_response` from the pipeline.
    """
    settings = get_settings()
    if not settings.openai_api_key.strip():
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not set (e-INFRA or other OpenAI-compatible key).",
        )
    if not settings.google_project_id or not settings.google_application_credentials:
        raise HTTPException(
            status_code=503,
            detail="Google Vision requires GOOGLE_PROJECT_ID and GOOGLE_APPLICATION_CREDENTIALS.",
        )

    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded (multipart field: file)")

    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    suffix = pathlib.Path(file.filename).suffix.lower() or ".jpg"
    if suffix not in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"):
        raise HTTPException(
            status_code=400, detail="Unsupported file type; use a common image format."
        )

    result, img_err, proc_err, _last_json = transcribe_to_dwc(
        raw,
        suffix,
        settings,
        use_rgb_mode=1,
        run_name=None,
        include_vv_raw=include_vv_raw,
        artifact_dir=None,
    )

    if img_err is not None:
        return JSONResponse(
            status_code=img_err.status,
            content={"detail": str(img_err), "code": "image_validation"},
        )
    if proc_err is not None:
        if isinstance(proc_err, TimeoutError):
            raise HTTPException(status_code=504, detail=str(proc_err)) from proc_err
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"VoucherVision failed: {proc_err!s}",
                "type": proc_err.__class__.__name__,
            },
        )

    return JSONResponse(status_code=200, content=result)
