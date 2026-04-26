from __future__ import annotations

import json
import secrets
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from vv_api.config import get_settings
from vv_api.pipeline import (
    RGB_WHOLE_SHEET,
    collect_artifact_relpaths,
    transcribe_to_dwc,
)
from vv_api.run_store import STORE, RunPayload

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

router = APIRouter(tags=["web"])


def _artifact_hrefs(run_id: str, rels: list[str]) -> list[str]:
    base = f"/app/artifacts/{run_id}/"
    return [base + quote(r, safe="/") for r in rels]


@router.get("/app", response_class=HTMLResponse)
def app_upload(request: Request) -> HTMLResponse:
    s = get_settings()
    return templates.TemplateResponse(
        request,
        "upload.html",
        {
            "error": None,
            "max_mb": round(s.max_upload_bytes / (1024 * 1024)),
            "max_edge": s.max_image_edge_px,
            "resize_edge": s.resize_max_edge_px,
        },
    )


@router.post("/app/process", response_model=None)
def app_process(
    request: Request,
    file: UploadFile | None = File(default=None),
) -> RedirectResponse | HTMLResponse:
    s = get_settings()
    ctx = {
        "error": None,
        "max_mb": round(s.max_upload_bytes / (1024 * 1024)),
        "max_edge": s.max_image_edge_px,
        "resize_edge": s.resize_max_edge_px,
    }

    if not s.openai_api_key.strip():
        ctx["error"] = "Není nastaven OPENAI_API_KEY (e-INFRA / OpenAI-kompatibilní klíč)."
        return templates.TemplateResponse(request, "upload.html", ctx, status_code=503)
    if not s.google_project_id or not s.google_application_credentials:
        ctx["error"] = (
            "Chybí GOOGLE_PROJECT_ID nebo GOOGLE_APPLICATION_CREDENTIALS (OCR Google Vision)."
        )
        return templates.TemplateResponse(request, "upload.html", ctx, status_code=503)

    if file is None or not file.filename:
        ctx["error"] = "Vyberte prosím obrázek."
        return templates.TemplateResponse(request, "upload.html", ctx, status_code=400)

    raw = file.file.read()
    if not raw:
        ctx["error"] = "Soubor je prázdný."
        return templates.TemplateResponse(request, "upload.html", ctx, status_code=400)

    suffix = Path(file.filename).suffix.lower() or ".jpg"
    if suffix not in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"):
        ctx["error"] = "Nepodporovaný formát. Použijte JPEG, PNG, TIFF nebo WebP."
        return templates.TemplateResponse(request, "upload.html", ctx, status_code=400)

    run_id = secrets.token_urlsafe(16).replace(".", "_")
    s.data_runs_path.mkdir(parents=True, exist_ok=True)
    artifact_dest = s.data_runs_path / run_id

    result, img_err, proc_err, last_json = transcribe_to_dwc(
        raw,
        suffix,
        s,
        use_rgb_mode=RGB_WHOLE_SHEET,
        run_name=run_id,
        include_vv_raw=False,
        artifact_dir=artifact_dest,
    )

    if img_err is not None:
        ctx["error"] = str(img_err)
        return templates.TemplateResponse(
            request, "upload.html", ctx, status_code=img_err.status
        )

    if proc_err is not None:
        if isinstance(proc_err, TimeoutError):
            ctx["error"] = str(proc_err)
            return templates.TemplateResponse(request, "upload.html", ctx, status_code=504)
        ctx["error"] = f"Zpracování selhalo: {proc_err!s}"
        return templates.TemplateResponse(request, "upload.html", ctx, status_code=500)

    rels = collect_artifact_relpaths(artifact_dest)
    dwc = result.get("darwin_core") or {}
    vv = result.get("voucher_vision") or {}

    STORE.set(
        run_id,
        RunPayload(
            darwin_core=dwc,
            voucher_vision=vv,
            last_json=last_json,
            artifact_relpaths=rels,
        ),
    )
    return RedirectResponse(
        f"/app/result/{run_id}",
        status_code=303,
    )


@router.get("/app/result/{run_id}", response_class=HTMLResponse)
def app_result(request: Request, run_id: str) -> HTMLResponse:
    p = STORE.get(run_id)
    if p is None:
        raise HTTPException(
            status_code=404,
            detail="Tento výsledek již není k dispozici (vypršel nebo neexistuje).",
        )
    dwc_map = p.darwin_core.get("dwc") or {}
    # Seřadit: stabilní pořadí pro tabulku
    rows = sorted(dwc_map.items(), key=lambda x: str(x[0]).lower())
    s = get_settings()
    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "run_id": run_id,
            "rows": rows,
            "validation": p.darwin_core.get("validation"),
            "voucher_vision": p.voucher_vision,
            "meta_json": json.dumps(
                p.voucher_vision, ensure_ascii=False, indent=2, default=str
            ),
            "artifact_hrefs": _artifact_hrefs(run_id, p.artifact_relpaths),
            "has_artifacts": bool(p.artifact_relpaths),
        },
    )


def _is_under(parent: Path, p: Path) -> bool:
    try:
        p.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


@router.get("/app/artifacts/{run_id}/{filepath:path}")
def app_artifact_file(run_id: str, filepath: str) -> FileResponse:
    if STORE.get(run_id) is None:
        raise HTTPException(status_code=404, detail="Neplatný nebo expirovaný běh.")
    s = get_settings()
    base = (s.data_runs_path / run_id).resolve()
    target = (base / filepath).resolve()
    if not _is_under(base, target) or not target.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path=target, filename=target.name)
