"""
api/config.py
API configuration management using YAML and environment variables.
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml


class APIConfig:
    """Loads and caches API settings."""

    def __init__(self, config_path: str = "configs/api.yaml"):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.config_path = self.root_dir / config_path
        self.cfg = self._load_defaults()

        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    file_cfg = yaml.safe_load(f) or {}
                    self._deep_update(self.cfg, file_cfg)
            except Exception as e:
                print(f"[Config] Error reading {self.config_path}: {e}")

        # Environment variable overrides
        if "DATA_DIR" in os.environ:
            self.cfg["paths"]["data_dir"] = os.environ["DATA_DIR"]
        if "EVAL_PARTITION" in os.environ:
            self.cfg["paths"]["eval_partition"] = os.environ["EVAL_PARTITION"]
        if "MODEL_PATH" in os.environ:
            self.cfg["paths"]["model_path"] = os.environ["MODEL_PATH"]
        if "API_PORT" in os.environ:
            self.cfg["server"]["port"] = int(os.environ["API_PORT"])
        if "API_HOST" in os.environ:
            self.cfg["server"]["host"] = os.environ["API_HOST"]

    def _load_defaults(self) -> Dict[str, Any]:
        return {
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "title": "PathSnap Map-Matching & Route Reconstruction API",
                "version": "1.0.0"
            },
            "paths": {
                "data_dir": "Student folder",
                "eval_partition": "student_test",
                "train_partition": "train",
                "model_path": "models/lightgbm_tuned_v1.joblib",
                "scaler_path": "models/scaler.joblib",
                "hmm_config": "configs/hmm.yaml",
                "log_file": "logs/api.log"
            },
            "pipeline": {
                "max_candidates": 12,
                "spatial_search_k": 35,
                "network_buffer_m": 150.0,
                "max_network_segments": 25000,
                "gps_gap_threshold_s": 5.0,
                "use_stub": False
            }
        }

    def _deep_update(self, d: Dict, u: Dict):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v

    @property
    def host(self) -> str:
        return self.cfg["server"]["host"]

    @property
    def port(self) -> int:
        return self.cfg["server"]["port"]

    @property
    def data_dir(self) -> Path:
        p = Path(self.cfg["paths"]["data_dir"])
        return p if p.is_absolute() else self.root_dir / p

    @property
    def eval_partition(self) -> str:
        return self.cfg["paths"]["eval_partition"]

    @property
    def model_path(self) -> Path:
        p = Path(self.cfg["paths"]["model_path"])
        return p if p.is_absolute() else self.root_dir / p

    @property
    def scaler_path(self) -> Path:
        p = Path(self.cfg["paths"]["scaler_path"])
        return p if p.is_absolute() else self.root_dir / p

    @property
    def hmm_config_path(self) -> Path:
        p = Path(self.cfg["paths"]["hmm_config"])
        return p if p.is_absolute() else self.root_dir / p

    @property
    def log_file(self) -> Path:
        p = Path(self.cfg["paths"]["log_file"])
        return p if p.is_absolute() else self.root_dir / p

    @property
    def pipeline(self) -> Dict[str, Any]:
        return self.cfg["pipeline"]


config = APIConfig()
