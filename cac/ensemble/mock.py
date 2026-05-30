"""Fixture VLM outputs in the real JSONL format (NO GPU / NO vLLM required).

Purpose: exercise the full real-data pipeline (loaders -> JSD -> MTA -> calibration
-> figure) before the WSL/vLLM stack exists, and to power unit tests. These are
NOT real model outputs — distributions are Dirichlet draws around the human labels
(à la v4 `sim_model`) and rationales are class-templated. Anything written here is
clearly a stand-in; real results come from `run.run_vision_inference`.
"""

from __future__ import annotations

import json

import numpy as np

from cac import config
from cac.data.labels import CIFAR10_CLASSES, IDX_TO_CLASS, LETTERS

# (model_key, family_noise) — mirrors v4's per-family noise scale.
DEFAULT_MODELS = [
    ("mock-llama", 0.08),
    ("mock-qwen", 0.15),
    ("mock-mistral", 0.22),
]

# Per-class visual cues, so agreeing models produce similar rationale text (MTA signal).
_CLASS_FEATURES = {
    "airplane": ["wings", "fuselage", "sky background"],
    "automobile": ["four wheels", "metal body", "windshield"],
    "bird": ["beak", "feathers", "small body"],
    "cat": ["pointed ears", "whiskers", "fur"],
    "deer": ["antlers", "slender legs", "brown coat"],
    "dog": ["snout", "floppy ears", "fur"],
    "frog": ["green skin", "webbed feet", "smooth body"],
    "horse": ["mane", "long legs", "hooves"],
    "ship": ["hull", "water around it", "deck"],
    "truck": ["large cargo bed", "six wheels", "boxy cab"],
}


def _confidence_qualifier(p_max: float) -> str:
    return "high" if p_max > 0.7 else "medium" if p_max > 0.4 else "low"


def _mock_rationale(cls: str, p_max: float) -> dict:
    feats = _CLASS_FEATURES.get(cls, ["distinctive shape"])
    return {
        "key_features": feats,
        "reasoning": f"The visible {feats[0]} and {feats[1]} are characteristic of a {cls}.",
        "conclusion": f"The main object is a {cls}.",
        "confidence_qualifier": _confidence_qualifier(p_max),
    }


def _sim_dist(human_probs: np.ndarray, noise: float, rng) -> np.ndarray:
    out = np.array([rng.dirichlet(p / noise + 1e-3) for p in human_probs])
    out = np.clip(out, 1e-12, None)
    return out / out.sum(axis=1, keepdims=True)


def write_mock_outputs(human_probs: np.ndarray, n: int | None = None,
                       models=DEFAULT_MODELS, seed: int = 42) -> list[str]:
    """Write one JSONL per mock model. Returns the list of paths."""
    rng = np.random.default_rng(seed)
    hp = human_probs if n is None else human_probs[:n]
    paths = []
    for key, noise in models:
        dists = _sim_dist(hp, noise, rng)
        path = config.MOCK_RAW_DIR / f"{key}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for i, d in enumerate(dists):
                ci = int(d.argmax())
                cls = IDX_TO_CLASS[ci]
                rationale = _mock_rationale(cls, float(d[ci]))
                rec = {
                    "idx": i,
                    "dist": [round(float(x), 6) for x in d],
                    "letter": LETTERS[ci],
                    "label_dist": cls,
                    "label_json": cls,
                    "rationale": rationale,
                    "rationale_raw": "",
                    "parse_status": "ok",
                }
                f.write(json.dumps(rec) + "\n")
        paths.append(str(path))
        print(f"[mock] wrote {len(dists)} records -> {path}")
    return paths
