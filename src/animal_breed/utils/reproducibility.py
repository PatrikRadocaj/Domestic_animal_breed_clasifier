"""Reproducibility helpers for deterministic runs and run metadata."""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def git_commit(repo_root: str | Path = ".") -> str:
    """Return the current git commit hash, or a sentinel when unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"
    return result.stdout.strip()


def git_status_short(repo_root: str | Path = ".") -> str:
    """Return short git status for run metadata."""
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"
    return result.stdout.strip()


def run_metadata(cfg: DictConfig, repo_root: str | Path = ".") -> dict[str, Any]:
    """Collect metadata that should be logged for every reproducible run."""
    return {
        "seed": int(cfg.get("seed", 42)),
        "git_commit": git_commit(repo_root),
        "git_status_short": git_status_short(repo_root),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }

