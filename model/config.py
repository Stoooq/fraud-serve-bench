from pathlib import Path

import yaml
from pydantic import BaseModel


class PathsConfig(BaseModel):
    data_path: str
    artifacts_dir: str


class TrainingConfig(BaseModel):
    test_size: float
    max_depth: int
    n_estimators: int
    random_state: int


class AppConfig(BaseModel):
    paths: PathsConfig
    training: TrainingConfig


def load_config(path: Path = Path(__file__).parent / "config.yaml") -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open() as f:
        cfg = AppConfig(**yaml.safe_load(f))

    base = config_path.parent
    cfg.paths.data_path = (base / cfg.paths.data_path).resolve()
    cfg.paths.artifacts_dir = (base / cfg.paths.artifacts_dir).resolve()
    return cfg
