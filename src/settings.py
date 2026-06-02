from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

OpenaiLlmPreset = Literal["gpt-oss-120b", "deepseek-v3.2-thinking", "qwen3.5-122b"]

PRESET_TO_MODEL: dict[OpenaiLlmPreset, str] = {
    "gpt-oss-120b": "gpt-oss-120b",
    "deepseek-v3.2-thinking": "deepseek-v3.2-thinking",
    "qwen3.5-122b": "qwen3.5-122b",
}

_REPO = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LM2 YOLOv5 runtime under vendor/lm2 (override with LM2_ROOT).
    lm2_root: Path | None = None
    dwc_map_path: Path = Field(default=_REPO / "config" / "dwc_map.yaml")
    prompt_path: Path = Field(default=_REPO / "prompts" / "SLTPvM_vision.yaml")

    openai_api_key: str = ""
    openai_base_url: str = "https://llm.ai.e-infra.cz/v1/"
    openai_llm_preset: OpenaiLlmPreset = "gpt-oss-120b"
    openai_llm_model: str | None = None

    lm2_conf_threshold: float = 0.5
    lm2_weights_path: Path | None = None
    lm2_device: str = ""

    label_category: str = "label"

    request_timeout_s: int = 180

    def resolved_llm_model(self) -> str:
        if self.openai_llm_model:
            return self.openai_llm_model
        return PRESET_TO_MODEL[self.openai_llm_preset]

    def resolved_lm2_root(self) -> Path:
        if self.lm2_root is not None:
            return self.lm2_root
        slim = _REPO / "vendor" / "lm2"
        if (slim / "component_detector" / "detect.py").is_file():
            return slim
        raise FileNotFoundError(
            f"LM2 vendor tree missing at {slim}. Clone with git lfs pull or set LM2_ROOT."
        )

    def default_lm2_weights(self) -> Path:
        if self.lm2_weights_path:
            return self.lm2_weights_path
        return _REPO / "vendor" / "lm2" / "weights" / "best.pt"


@lru_cache
def get_settings() -> Settings:
    return Settings()
