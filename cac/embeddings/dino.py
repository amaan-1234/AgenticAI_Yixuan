"""Real DINOv2-ViT-S/14 embeddings for the pre-router (Phase 1.4).

Replaces the 64-dim synthetic `simulate_clip_embeddings` from v4 with real frozen
backbone features. DINOv2-small is ~22M params and runs comfortably on the 8 GB
laptop GPU (or CPU) at full 10k scale. Output: (N, 384) float32.

The pre-router (`cac/pipeline/prerouter.py`) trains a logistic head on these to
predict 'trivially easy' images with an asymmetric loss.
"""

from __future__ import annotations

import numpy as np

from cac import config

MODEL_ID = "facebook/dinov2-small"  # ViT-S/14, 384-dim CLS embedding


def extract(images: np.ndarray, batch_size: int = 256, model_id: str = MODEL_ID,
            device: str | None = None) -> np.ndarray:
    """Extract pooled DINOv2 embeddings for images [N,32,32,3] uint8 -> [N,384] float32."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    device = device or config.device()
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()

    embs = []
    n = len(images)
    for start in range(0, n, batch_size):
        batch = images[start:start + batch_size]
        pil = [Image.fromarray(img).convert("RGB") for img in batch]
        inputs = processor(images=pil, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        # pooler_output is the layernormed CLS token (B, 384).
        emb = out.pooler_output.float().cpu().numpy()
        embs.append(emb)
        print(f"[dino] {min(start + batch_size, n)}/{n}", end="\r")
    print()
    return np.concatenate(embs, axis=0).astype(np.float32)


def extract_and_cache(images: np.ndarray, out_path=config.DINO_EMB, **kw) -> np.ndarray:
    """Extract embeddings and cache to .npy (skips work if cache exists)."""
    if out_path.exists():
        emb = np.load(out_path)
        print(f"[dino] loaded cached embeddings {emb.shape} from {out_path}")
        return emb
    emb = extract(images, **kw)
    np.save(out_path, emb)
    print(f"[dino] saved {emb.shape} -> {out_path}")
    return emb
