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

app = FastAPI(title="myvouvis", version="0.2.0")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/transcribe")
async def transcribe(
    file: Annotated[UploadFile, File(description="Herbarium sheet image")],
) -> JSONResponse:
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        result = process_sheet(tmp_path)
        return JSONResponse(result.as_score())
    finally:
        tmp_path.unlink(missing_ok=True)
