import os
from pathlib import Path

import yaml

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


class Config:
    def __init__(self, path: str | Path | None = None):
        cfg_path = Path(path) if path else _DEFAULT_CONFIG
        with cfg_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.connection = data.get("connection", {})
        self.bots = data.get("bots", {})
        self.application = data.get("application", {})
        self.s3 = data.get("s3", {})

    def get_bot_config(self, bot_name):
        return self.bots.get(bot_name, {})

    def get_database_config(self, key, default=None):
        if key in self.connection:
            return self.connection[key]
        return os.getenv(f"DB_{key.upper()}", default)

    def get_application_config(self, key, default=None):
        value = self.application.get(key, default)
        if value is None:
            return default
        return value

    def get_s3_config(self, key: str, default=None):
        if key in self.s3:
            return self.s3[key]
        return os.getenv(f"S3_{key.upper()}", default)
