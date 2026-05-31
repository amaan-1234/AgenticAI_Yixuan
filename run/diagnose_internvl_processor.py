"""Diagnose InternVL2 image preprocessing (no GPU, no vLLM).

    python -m run.diagnose_internvl_processor
    python -m run.diagnose_internvl_processor --n 8
    python -m run.diagnose_internvl_processor --no-sweep        # config dump only
    python -m run.diagnose_internvl_processor --no-config       # JSON sweep only
    python -m run.diagnose_internvl_processor --json-out PATH

Targets the open question from HPC: InternVL2-8B confidence is ~0.69 / entropy 1.03
on CIFAR-10 vs Qwen2-VL-7B's ~0.93 / 0.24. fp16 vs AWQ moved the needle only
marginally (0.71->0.75 acc), so the gap is upstream of quantization. This script
loads ONLY the AutoProcessor (no LLM, no GPU) and:

1. Prints each processor's full image-preprocessing config (named attrs +
   image_processor.to_dict() leftovers).
2. Sweeps a fixed set of preprocessing variants per image and writes one JSON
   record per image to outputs/diagnostics/internvl_processor_sweep.jsonl. Each
   variant tests a different (resize target, resample method, input format)
   combination, and the JSON captures pixel_values shape + dtype + per-channel
   means + effective-upscale ratios. Compare the records between InternVL2 and
   Qwen2-VL for the same image to isolate where InternVL diverges.

Runs in seconds on an HPC login node.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from cac import config
from cac.data import cifar10h
from cac.data.labels import CIFAR10_CLASSES
from cac.models.vllm_runner import VLMRunner, _build_user_messages, _family

# Attributes worth surfacing in the human-readable config dump. hasattr filters
# per family (InternVL has min/max_dynamic_patch; Qwen has size/patch_size;
# anything missing is skipped). The to_dict() leftovers catch anything we missed.
INTERESTING_ATTRS = (
    "image_size",
    "size",
    "crop_size",
    "patch_size",
    "min_dynamic_patch",
    "max_dynamic_patch",
    "use_thumbnail",
    "dynamic_image_size",
    "do_resize",
    "do_rescale",
    "do_normalize",
    "do_center_crop",
    "image_mean",
    "image_std",
    "rescale_factor",
)

# Variant sweep: (name, resize_target, resample_method, input_format).
# resize_target == 0 means "feed at native 32x32" (no resize).
# input_format ∈ {'pil', 'np'} — tests whether the processor handles numpy
# input identically to PIL.
RESIZE_VARIANTS = (
    ("resize_224_bicubic_pil",  224, "BICUBIC",  "pil"),
    ("resize_336_bicubic_pil",  336, "BICUBIC",  "pil"),
    ("resize_448_bicubic_pil",  448, "BICUBIC",  "pil"),
    ("resize_224_bilinear_pil", 224, "BILINEAR", "pil"),
    ("resize_224_bicubic_np",   224, "BICUBIC",  "np"),
    ("native_32_no_resize",       0, "BICUBIC",  "pil"),
)


# --- human-readable config dump -----------------------------------------------

def _print_processor_config(proc, model_id: str):
    print(f"\n========== {model_id}  (family={_family(model_id)}) ==========")
    print(f"  processor class: {type(proc).__name__}")

    ip = getattr(proc, "image_processor", None)
    target = ip if ip is not None else proc
    if ip is None:
        print("  [warn] processor.image_processor is None; printing processor attrs:")
    else:
        print(f"  image_processor class: {type(ip).__name__}")

    print("  --- named attributes (only those present) ---")
    seen = set()
    for k in INTERESTING_ATTRS:
        if hasattr(target, k):
            print(f"    {k:<22} = {getattr(target, k)!r}")
            seen.add(k)

    if ip is not None and hasattr(ip, "to_dict"):
        full = ip.to_dict()
        leftovers = {k: v for k, v in full.items()
                     if k not in seen and not k.startswith("_")}
        if leftovers:
            print("  --- additional image_processor.to_dict() entries ---")
            for k, v in leftovers.items():
                vs = repr(v)
                if len(vs) > 200:
                    vs = f"{type(v).__name__}(...truncated, len={len(v) if hasattr(v, '__len__') else '?'})"
                print(f"    {k:<22} = {vs}")

    tok = getattr(proc, "tokenizer", None)
    if tok is not None:
        print(f"  tokenizer: {type(tok).__name__}  is_fast={tok.is_fast}")


def _print_runtime_shape(proc, model_id: str, image_np):
    """Feed one image through the runner's exact _prep_image and report shapes."""
    pil = VLMRunner._prep_image(image_np, model_id)
    print(f"  [runtime] PIL after VLMRunner._prep_image: size={pil.size} mode={pil.mode}")

    text = "What is in this image? Answer with a single letter A-J."
    messages = _build_user_messages(model_id, text)
    print(f"  [runtime] messages[0]['content'] type = "
          f"{type(messages[0]['content']).__name__}")

    try:
        if _family(model_id) == "phi3v":
            chat_text = proc.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            chat_text = proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        snippet = chat_text[:160].replace("\n", "\\n")
        print(f"  [runtime] rendered prompt (first 160 chars): {snippet!r}")
    except Exception as e:
        print(f"  [runtime] apply_chat_template failed: {type(e).__name__}: {e}")

    ip = getattr(proc, "image_processor", None)
    if ip is None:
        print("  [runtime] no image_processor; cannot report tensor shape")
        return
    try:
        out = ip(images=pil, return_tensors="pt")
        pv = out.get("pixel_values")
        if pv is not None:
            print(f"  [runtime] pixel_values.shape = {tuple(pv.shape)} dtype={pv.dtype}")
            print(f"  [runtime] pixel_values stats: "
                  f"min={float(pv.min()):.3f} max={float(pv.max()):.3f}")
    except Exception as e:
        print(f"  [runtime] image_processor failed: {type(e).__name__}: {e}")


# --- JSON variant sweep -------------------------------------------------------

def _tensor_stats(pv) -> dict:
    """Compute JSON-safe stats over a pixel_values tensor (CHW, NCHW, or NNCHW)."""
    pv_f = pv.float()
    stats = {
        "shape": list(pv.shape),
        "dtype": str(pv.dtype),
        "min": round(float(pv_f.min()), 4),
        "max": round(float(pv_f.max()), 4),
        "mean": round(float(pv_f.mean()), 4),
        "std": round(float(pv_f.std()), 4),
    }
    # Locate the channel dim (RGB == 3) and report per-channel means. Most
    # processors emit (N, C, H, W) or (C, H, W); a 5-D (B, N_tiles, C, H, W)
    # output from some InternVL processors is also handled.
    ch_axis = None
    for i, d in enumerate(pv.shape):
        if d == 3:
            ch_axis = i
            break
    if ch_axis is not None:
        reduce_dims = tuple(i for i in range(pv.ndim) if i != ch_axis)
        ch_means = pv_f.mean(dim=reduce_dims)
        stats["mean_per_channel"] = [round(float(c), 4) for c in ch_means]
    return stats


def _build_input_for_variant(image_np: np.ndarray, variant):
    """Apply (resize, resample, input_format) and return the object handed to the processor."""
    _, size, resample_name, input_format = variant
    if size:
        pil = Image.fromarray(image_np).convert("RGB").resize(
            (size, size), getattr(Image, resample_name)
        )
        source_size = size
    else:
        pil = Image.fromarray(image_np).convert("RGB")
        source_size = pil.size[0]
    if input_format == "np":
        return np.array(pil), source_size, list(pil.size)
    return pil, source_size, list(pil.size)


def _sweep_one_variant(proc, image_np: np.ndarray, variant) -> dict:
    name = variant[0]
    record = {"variant": name, "resize_to": variant[1],
              "resample": variant[2], "input_format": variant[3]}
    ip = getattr(proc, "image_processor", None)
    if ip is None:
        record["error"] = "no image_processor on this proc"
        return record
    try:
        inp, source_size, pil_size = _build_input_for_variant(image_np, variant)
        record["input_size"] = pil_size
        out = ip(images=inp, return_tensors="pt")
        pv = out.get("pixel_values")
        if pv is None:
            record["error"] = "no pixel_values in processor output"
            record["output_keys"] = list(out.keys())
            return record
        record["pixel_values"] = _tensor_stats(pv)
        tile_h = pv.shape[-2] if pv.ndim >= 2 else None
        if tile_h:
            record["tile_size"] = [pv.shape[-2], pv.shape[-1]]
            record["effective_upscale_from_input"] = round(tile_h / source_size, 3)
            record["effective_upscale_from_cifar"] = round(tile_h / 32, 3)
            if pv.ndim == 4 and pv.shape[0] > 1:
                record["n_tiles"] = pv.shape[0]
            elif pv.ndim == 5:
                record["n_tiles"] = pv.shape[1]
            else:
                record["n_tiles"] = 1
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
    return record


def _processor_summary(proc) -> dict:
    """Compact per-model snapshot for the JSON record header."""
    ip = getattr(proc, "image_processor", None)
    if ip is None:
        return {"image_processor": None}
    d = ip.to_dict() if hasattr(ip, "to_dict") else {}
    return {
        "image_processor_class": type(ip).__name__,
        "image_mean": d.get("image_mean"),
        "image_std": d.get("image_std"),
        "size": d.get("size") or d.get("image_size"),
        "crop_size": d.get("crop_size"),
        "do_resize": d.get("do_resize"),
        "do_normalize": d.get("do_normalize"),
        "min_dynamic_patch": d.get("min_dynamic_patch"),
        "max_dynamic_patch": d.get("max_dynamic_patch"),
        "dynamic_image_size": d.get("dynamic_image_size"),
        "use_thumbnail": d.get("use_thumbnail"),
        "rescale_factor": d.get("rescale_factor"),
    }


# --- main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="OpenGVLab/InternVL2-8B-AWQ",
                    help="primary model id to diagnose")
    ap.add_argument("--compare", default="Qwen/Qwen2-VL-7B-Instruct-AWQ",
                    help="comparison model id; pass empty string to skip")
    ap.add_argument("--n", type=int, default=4,
                    help="number of CIFAR images to sweep (default 4)")
    ap.add_argument("--no-config", action="store_true",
                    help="skip the human-readable config dump")
    ap.add_argument("--no-sweep", action="store_true",
                    help="skip the JSON variant sweep")
    default_out = config.OUTPUTS_DIR / "diagnostics" / "internvl_processor_sweep.jsonl"
    ap.add_argument("--json-out", default=str(default_out),
                    help=f"path to write the JSON sweep records (default: {default_out})")
    args = ap.parse_args()

    model_ids = [args.model] + ([args.compare] if args.compare else [])

    from transformers import AutoProcessor
    procs = {}
    for mid in model_ids:
        print(f"[load] AutoProcessor: {mid}")
        procs[mid] = AutoProcessor.from_pretrained(mid, trust_remote_code=True)

    images, _, labels_idx = cifar10h.prepare()
    n = min(args.n, len(images))

    if not args.no_config:
        for mid in model_ids:
            _print_processor_config(procs[mid], mid)
            _print_runtime_shape(procs[mid], mid, images[0])

    if not args.no_sweep:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n_variants = len(RESIZE_VARIANTS)
        print(f"\n[sweep] {n} images x {n_variants} variants x {len(model_ids)} models "
              f"-> {out_path}")

        proc_summaries = {mid: _processor_summary(procs[mid]) for mid in model_ids}
        with out_path.open("w", encoding="utf-8") as f:
            for i in range(n):
                img = images[i]
                label = CIFAR10_CLASSES[int(labels_idx[i])]
                record = {
                    "image_idx": i,
                    "cifar_shape": list(img.shape),
                    "human_label": label,
                    "processor_summaries": proc_summaries,
                    "models": [],
                }
                for mid in model_ids:
                    record["models"].append({
                        "model_id": mid,
                        "family": _family(mid),
                        "variants": [_sweep_one_variant(procs[mid], img, v)
                                     for v in RESIZE_VARIANTS],
                    })
                f.write(json.dumps(record) + "\n")

        print(f"[ok] wrote {n} records, {n * n_variants * len(model_ids)} variant rows")
        print("     read back with: import json; "
              f"[json.loads(l) for l in open('{out_path}')]")

    print("\n========== HOW TO READ THE OUTPUT ==========")
    print("  1. effective_upscale_from_cifar > 6 means the encoder sees a heavily")
    print("     upscaled CIFAR (likely high-frequency artifact territory).")
    print("  2. n_tiles > 1 on a uniform CIFAR upscale means dynamic tiling fired —")
    print("     the model is averaging features over redundant tiles. Force single")
    print("     tile via max_dynamic_patch=1 / dynamic_image_size=False.")
    print("  3. processor_summaries.image_mean / image_std should match what the")
    print("     model was trained on. CLIP-style: [0.485, 0.456, 0.406] /")
    print("     [0.229, 0.224, 0.225]. A mismatch silently flattens softmax.")
    print("  4. pixel_values.mean_per_channel should be near 0 if normalization is")
    print("     working. Values >>0 or <<0 indicate normalization drift.")
    print("  5. resize_224 vs resize_448 vs native_32: the resize that yields the")
    print("     LOWEST effective_upscale_from_cifar at the same final tile size is")
    print("     the one to wire into VLMRunner._prep_image for the family.")


if __name__ == "__main__":
    main()
