"""Central paths, device detection, and YAML config loading.

Everything resolves relative to the repo root so the same code runs on the
Windows host, inside WSL2, and on the HPC without edits. Override the data/output
roots with the CAC_DATA_DIR / CAC_OUTPUTS_DIR env vars if you relocate them
(e.g. to native WSL fs for faster IO on the full run).
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("CAC_DATA_DIR", REPO_ROOT / "data"))
OUTPUTS_DIR = Path(os.environ.get("CAC_OUTPUTS_DIR", REPO_ROOT / "outputs"))
RAW_DIR = OUTPUTS_DIR / "raw"           # per-model inference JSONL (REAL outputs only)
MOCK_RAW_DIR = RAW_DIR / "mocks"        # fixture outputs, isolated to keep loaders clean
FIG_DIR = OUTPUTS_DIR / "figures"
CONFIG_DIR = REPO_ROOT / "config"

# Cached artifacts (paths only; producers live in the relevant modules).
CIFAR10H_PROBS = DATA_DIR / "cifar10h-probs.npy"      # (10000, 10) human label dists
CIFAR10_IMAGES = DATA_DIR / "cifar10_test_images.npy"  # (10000, 32, 32, 3) uint8
CIFAR10_LABELS = DATA_DIR / "cifar10_test_labels.npy"  # (10000,) int  (alignment check)
DINO_EMB = OUTPUTS_DIR / "cifar10h_dino_emb.npy"        # (10000, 384) float32
MTA_SCORES = OUTPUTS_DIR / "cifar10h_mta.npy"           # (10000,) float in [0,1]

for _d in (DATA_DIR, OUTPUTS_DIR, RAW_DIR, MOCK_RAW_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def device() -> str:
    """Return 'cuda' if a GPU is visible, else 'cpu'. Used by DINOv2 / cross-encoder."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_yaml(name: str) -> dict:
    """Load a YAML file from config/ by filename (with or without .yaml suffix)."""
    p = CONFIG_DIR / name
    if p.suffix == "":
        p = p.with_suffix(".yaml")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
