from __future__ import annotations

import pathlib
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# e-INFRA (OpenAI-compatible) model ids — switch via OPENAI_LLM_PRESET (env).
# OPENAI_LLM_MODEL still overrides both when set.
OpenaiLlmPreset = Literal["gpt-oss-120b", "deepseek-v3.2-thinking", "qwen3.5-122b"]

OPENAI_LLM_PRESET_TO_MODEL_ID: dict[OpenaiLlmPreset, str] = {
    "gpt-oss-120b": "gpt-oss-120b",
    "deepseek-v3.2-thinking": "deepseek-v3.2-thinking",
    "qwen3.5-122b": "qwen3.5-122b",
}


def _default_vendor_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent / "VoucherVision"


def _default_dwc_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent / "dwc_map.yaml"


def _default_data_runs() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent / "data" / "runs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    vendor_root: pathlib.Path = Field(default_factory=_default_vendor_root)

    # LLM: OpenAI-compatible (e-INFRA CZ) — env:
    #   OPENAI_API_KEY, OPENAI_BASE_URL,
    #   OPENAI_LLM_PRESET (gpt-oss-120b | deepseek-v3.2-thinking | qwen3.5-122b),
    #   or OPENAI_LLM_MODEL when you need an arbitrary model id (overrides preset).
    openai_api_key: str = ""
    openai_base_url: str = "https://llm.ai.e-infra.cz/v1/"

    openai_llm_preset: OpenaiLlmPreset = "gpt-oss-120b"
    openai_llm_model: Optional[str] = None

    vv_llm_version: str = "GPT 4o mini 2024-07-18"
    # Comma-separated OCR options, e.g. "normal" or "hand" (Google Cloud Vision)
    vv_ocr_options: str = "normal"

    google_project_id: str = ""
    google_location: str = "us-central1"
    google_application_credentials: str = ""

    prompt_version: str = "SLTPvM_default.yaml"
    run_name: str = "api_run"

    max_upload_bytes: int = 20 * 1024 * 1024
    max_image_edge_px: int = 8000
    resize_max_edge_px: int = 4096
    jpeg_quality: int = 92

    dwc_map_path: pathlib.Path = Field(default_factory=_default_dwc_path)

    # Per-run Web UI: persisted previews under data/runs/{run_id}/
    data_runs_path: pathlib.Path = Field(default_factory=_default_data_runs)
    # In-memory result TTL (seconds) for the results page
    run_result_ttl_s: int = 3600

    request_timeout_s: int = 180


@lru_cache
def get_settings() -> Settings:
    return Settings()
