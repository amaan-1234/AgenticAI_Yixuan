"""v4 simulation functions, preserved verbatim for the real-vs-sim gap analysis.

Phase 3.1 ('compare real-data results to simulation predictions ... this gap
analysis is itself a contribution') needs the original simulated signals. These are
the unchanged v4 simulators: Dirichlet model outputs, entropy-derived MTA, and
synthetic CLIP embeddings, plus the families x models ablation config. Used only by
`run_pipeline.py --simulate`; the real pipeline never imports them.
"""

from __future__ import annotations

import numpy as np

# Per-family Dirichlet noise scale (smaller = better-calibrated model).
FAMILY_NOISE = {
    "Meta-Llama": [0.08, 0.12],
    "Alibaba-Qwen": [0.15, 0.19],
    "Mistral": [0.22, 0.26],
    "Google-Gemma": [0.10, 0.14],
    "Microsoft-Phi": [0.18, 0.23],
}

ABLATION_CONFIGS = {
    "2F×1M (Llama+Qwen)": [("Meta-Llama", 0), ("Alibaba-Qwen", 0)],
    "3F×1M (Llama+Qwen+Mistral)": [("Meta-Llama", 0), ("Alibaba-Qwen", 0), ("Mistral", 0)],
    "3F×2M (6 models total)": [("Meta-Llama", 0), ("Meta-Llama", 1),
                               ("Alibaba-Qwen", 0), ("Alibaba-Qwen", 1),
                               ("Mistral", 0), ("Mistral", 1)],
    "5F×1M (5 families)": [("Meta-Llama", 0), ("Alibaba-Qwen", 0), ("Mistral", 0),
                           ("Google-Gemma", 0), ("Microsoft-Phi", 0)],
}

PRIMARY = "3F×1M (Llama+Qwen+Mistral)"


def sim_model(human_p: np.ndarray, noise_scale: float, rng) -> np.ndarray:
    out = [rng.dirichlet(p / noise_scale + 1e-3) for p in human_p]
    arr = np.clip(np.array(out), 1e-12, None)
    return arr / arr.sum(axis=1, keepdims=True)


def simulate_mta(human_ent: np.ndarray, rng, noise: float = 0.15) -> np.ndarray:
    ent_norm = (human_ent - human_ent.min()) / (human_ent.max() - human_ent.min() + 1e-12)
    base_mta = 1.0 - ent_norm
    return np.clip(base_mta + rng.normal(0, noise, size=len(human_ent)), 0, 1)


def simulate_clip_embeddings(human_ent: np.ndarray, dim: int = 64, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng(42)
    n = len(human_ent)
    ent_norm = (human_ent - human_ent.min()) / (human_ent.max() - human_ent.min() + 1e-12)
    emb = rng.standard_normal((n, dim))
    for d in range(8):
        emb[:, d] += ent_norm * rng.uniform(1.5, 3.0)
    return emb


def ensemble_distributions(human_probs, config_name, rng):
    """Build the list of simulated model distributions for an ablation config."""
    return [sim_model(human_probs, FAMILY_NOISE[fam][idx], rng)
            for fam, idx in ABLATION_CONFIGS[config_name]]
