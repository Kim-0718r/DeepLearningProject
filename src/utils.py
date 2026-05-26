import json
import random
from pathlib import Path

import numpy as np
import torch

try:
    from .config import Config
except ImportError:
    from config import Config


def set_seed(seed=Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_output_dirs(cfg=Config):
    ensure_dir(cfg.CHECKPOINT_DIR)
    ensure_dir(cfg.FIGURE_DIR)
    ensure_dir(cfg.METRIC_DIR)


def save_json(data, path):
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
