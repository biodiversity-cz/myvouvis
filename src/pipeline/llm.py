from __future__ import annotations

import base64
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

from pipeline.exceptions import LlmParseError
from settings import Settings


def _build_system_prompt(prompt_path: Path) -> str:
    with prompt_path.open(encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    rules = spec.get("rules") or {}
    rules_block = "\n".join(f'  "{k}": "{v}"' for k, v in rules.items())
    template = "{\n" + rules_block + "\n}"
    return (
        f"{spec.get('instructions', '').strip()}\n\n"
        f"{spec.get('json_formatting_instructions', '').strip()}\n\n"
        f"JSON template (all keys required, empty string if unknown):\n{template}"
    )


@lru_cache
def _cached_system_prompt(prompt_path: str) -> str:
    return _build_system_prompt(Path(prompt_path))


@lru_cache
def _openai_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, timeout=180.0)


def extract_label_json(image_bytes: bytes, mime: str, settings: Settings) -> dict[str, Any]:
    if not settings.openai_api_key:
        raise LlmParseError("OPENAI_API_KEY is not set")

    client = _openai_client(settings.openai_api_key, settings.openai_base_url)
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    system = _cached_system_prompt(str(settings.prompt_path.resolve()))

    response = client.chat.completions.create(
        model=settings.resolved_llm_model(),
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract label fields from this herbarium specimen label image.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
    )
    text = (response.choices[0].message.content or "").strip()
    return _parse_json(text)


def _parse_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise LlmParseError("LLM did not return JSON") from None
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise LlmParseError("LLM JSON root must be an object")
    return parsed
