from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import yaml
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Dict, List, Optional

from vv_api.config import Settings, get_settings

_LOCK = threading.Lock()
_CONFIGURED = False


def _ensure_python_path(vendor_root: Path) -> None:
    root = str(vendor_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def _install_openai_model_override() -> None:
    from vouchervision.model_maps import ModelMaps

    if not hasattr(ModelMaps, "_vv_orig_get_api_name"):
        ModelMaps._vv_orig_get_api_name = ModelMaps.get_api_name.__func__  # type: ignore[assignment]

    o = os.environ.get("OPENAI_LLM_MODEL")
    if not o:
        # Restore stock binding if we had previously overridden
        if getattr(ModelMaps, "_vv_override_applied", False):
            ModelMaps.get_API_name = classmethod(  # type: ignore[assignment]
                ModelMaps._vv_orig_get_api_name
            )
            ModelMaps._vv_override_applied = False  # type: ignore[attr-defined]
        return
    if getattr(ModelMaps, "_vv_override_applied", False) and (
        getattr(ModelMaps, "_openai_name_override", None) == o
    ):
        return

    o_local = o

    @classmethod
    def _get_api_name(cls, key):  # noqa: N805
        k = str(key)
        if o_local and k.startswith("GPT_") and "AZURE" not in k:
            return o_local
        return ModelMaps._vv_orig_get_api_name(cls, key)  # type: ignore[misc]

    ModelMaps.get_API_name = _get_api_name  # type: ignore[assignment]
    ModelMaps._vv_override_applied = True  # type: ignore[attr-defined]
    ModelMaps._openai_name_override = o_local  # type: ignore[attr-defined]


def _build_cfg(
    settings: Settings,
    run_name: str,
    images_dir: Path,
    out_dir: Path,
    *,
    use_rgb_label_images: int = 1,
    prompt_version: Optional[str] = None,
) -> dict:
    base = settings.vendor_root / "custom_VV_config.yaml"
    with open(base, encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    ocr_opt = [s.strip() for s in settings.vv_ocr_options.split(",") if s.strip()]
    if not ocr_opt:
        ocr_opt = ["normal"]
    cfg["leafmachine"]["LLM_version"] = settings.vv_llm_version
    cfg["leafmachine"]["use_RGB_label_images"] = use_rgb_label_images
    cfg["leafmachine"]["project"]["dir_images_local"] = str(images_dir)
    cfg["leafmachine"]["project"]["dir_output"] = str(out_dir)
    cfg["leafmachine"]["project"]["run_name"] = run_name
    cfg["leafmachine"]["project"]["prompt_version"] = (
        prompt_version or settings.prompt_version
    )
    cfg["leafmachine"]["project"]["OCR_option"] = ocr_opt
    cfg["leafmachine"]["project"]["num_workers"] = 1
    return cfg


def _apply_env(settings: Settings) -> None:
    if settings.openai_base_url:
        os.environ["OPENAI_BASE_URL"] = settings.openai_base_url
    if settings.openai_api_key:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    if settings.google_project_id:
        os.environ["GOOGLE_PROJECT_ID"] = settings.google_project_id
    if settings.google_location:
        os.environ["GOOGLE_LOCATION"] = settings.google_location
    if settings.google_application_credentials:
        g = settings.google_application_credentials
        if not g.strip().startswith("{"):
            p = Path(g)
            if p.is_file():
                g = p.read_text(encoding="utf-8")
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = g
    if settings.openai_llm_model:
        os.environ["OPENAI_LLM_MODEL"] = settings.openai_llm_model


def run_voucher_vision_in_dirs(
    settings: Settings,
    images_dir: Path,
    out_dir: Path,
    run_name: str,
    config_path: Path,
    *,
    path_custom_prompts: Optional[str] = None,
) -> Dict[str, Any]:
    global _CONFIGURED
    with _LOCK:
        if not _CONFIGURED:
            _apply_env(settings)
        _ensure_python_path(settings.vendor_root)
        _install_openai_model_override()
        _CONFIGURED = True

    if path_custom_prompts:
        path_custom = path_custom_prompts
    else:
        with open(config_path, encoding="utf-8") as f:
            c = yaml.safe_load(f)
        pv = c["leafmachine"]["project"]["prompt_version"]
        path_custom = str(settings.vendor_root / "custom_prompts" / pv)
    dir_home = str(settings.vendor_root)
    path_api = str(settings.vendor_root / "api_cost" / "api_cost.yaml")

    from vouchervision.vouchervision_main import voucher_vision

    return voucher_vision(
        str(config_path.resolve()),
        dir_home=dir_home,
        path_custom_prompts=path_custom,
        cfg_test=None,
        progress_report=None,
        json_report=False,
        path_api_cost=path_api,
        test_ind=None,
        is_hf=True,
        is_real_run=False,
    )


def _copy_visual_artifacts(project_dir: Path, dest: Path) -> List[str]:
    """
    Copy PNG/JPG previews from a VV run directory (Cropped_Images, overlays, …).
    Returns relative URL paths (posix) under dest.
    """
    dest.mkdir(parents=True, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
    relpaths: List[str] = []
    if not project_dir.is_dir():
        return relpaths
    for p in sorted(project_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        try:
            if p.stat().st_size > 15 * 1024 * 1024:
                continue
        except OSError:
            continue
        rel = p.relative_to(project_dir)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        relpaths.append(str(rel).replace("\\", "/"))
        if len(relpaths) >= 48:
            break
    return relpaths


def run_in_temp(
    image_bytes: bytes,
    suffix: str,
    *,
    settings: Optional[Settings] = None,
    run_name: Optional[str] = None,
    use_rgb_mode: int = 1,
    prompt_version: Optional[str] = None,
    artifact_dest: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run VV on one image; temp workspace is removed after the call.
    If artifact_dest is set, copy image previews from the run output there.
    """
    s = settings or get_settings()
    name = run_name or s.run_name
    with tempfile.TemporaryDirectory(prefix="vv_api_") as tmp:
        base = Path(tmp)
        in_dir = base / "in"
        out_dir = base / "out"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        (in_dir / f"upload{suffix}").write_bytes(image_bytes)

        cfg = _build_cfg(
            s,
            name,
            in_dir,
            out_dir,
            use_rgb_label_images=use_rgb_mode,
            prompt_version=prompt_version,
        )
        cfg_path = base / "vv_run.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                cfg,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        path_custom_arg = str(
            s.vendor_root / "custom_prompts" / cfg["leafmachine"]["project"]["prompt_version"]
        )

        def _call() -> Dict[str, Any]:
            return run_voucher_vision_in_dirs(
                s,
                in_dir,
                out_dir,
                name,
                cfg_path,
                path_custom_prompts=path_custom_arg,
            )

        if s.request_timeout_s and s.request_timeout_s > 0:
            with ThreadPoolExecutor(1) as ex:
                fut = ex.submit(_call)
                try:
                    result = fut.result(timeout=s.request_timeout_s)
                except FuturesTimeout as e:
                    raise TimeoutError(
                        f"VoucherVision exceeded {s.request_timeout_s} seconds"
                    ) from e
        else:
            result = _call()

        if artifact_dest is not None:
            project = out_dir / name
            _copy_visual_artifacts(project, artifact_dest)

        return result
