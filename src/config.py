from pathlib import Path

import yaml
from pydantic import BaseModel


class Pathsconfig(BaseModel):
    data_path: str


class AppConfig(BaseModel):
    paths: Pathsconfig


def load_config(path: str = "config.yaml") -> AppConfig:
    with Path(path).open() as f:
        return AppConfig(**yaml.safe_load(f))
