from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline.sheet import process_sheet  # noqa: E402
from pipeline.types import OutputMode  # noqa: E402

app = FastAPI(title="myvouvis", version="0.2.0")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


async def _transcribe_upload(
    file: UploadFile,
    output_mode: OutputMode,
) -> JSONResponse:
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        result = process_sheet(tmp_path, output_mode=output_mode)
        return JSONResponse(result.as_score(output_mode))
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/v1/transcribe-full")
async def transcribe_full(
    file: Annotated[UploadFile, File(description="Herbarium sheet image")],
) -> JSONResponse:
    return await _transcribe_upload(file, OutputMode.full)


@app.post("/v1/transcribe-dwc")
async def transcribe_dwc(
    file: Annotated[UploadFile, File(description="Herbarium sheet image")],
) -> JSONResponse:
    return await _transcribe_upload(file, OutputMode.dwc)


@app.post("/v1/transcribe-bbox")
async def transcribe_bbox(
    file: Annotated[UploadFile, File(description="Herbarium sheet image")],
) -> JSONResponse:
    return await _transcribe_upload(file, OutputMode.bbox)
