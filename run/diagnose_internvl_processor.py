"""Diagnose InternVL2 image preprocessing (no GPU, no vLLM).

    python -m run.diagnose_internvl_processor
    python -m run.diagnose_internvl_processor --model OpenGVLab/InternVL2-8B
    python -m run.diagnose_internvl_processor --compare Qwen/Qwen2-VL-7B-Instruct-AWQ
    python -m run.diagnose_internvl_processor --compare ""    # solo

Hypothesis being tested: InternVL2-8B-AWQ shows degraded soft labels (argmax_acc
0.71, mean_max_prob 0.597, entropy 1.233) vs Qwen2-VL-7B-AWQ (0.94/0.93/0.24) on
identical CIFAR-10 inputs, and fp16 only nudges it to 0.75/0.69/1.03 — so the gap
is upstream of quantization. InternVL2's processor uses dynamic high-res tiling
(`min_dynamic_patch` / `max_dynamic_patch`) and a 448x448 native tile, while
`VLMRunner._prep_image` upscales CIFAR's 32x32 to 224x224. The processor may be
(a) further upscaling 224 -> 448 (14x over the original — high-frequency artifacts
dominate the encoder), or (b) emitting an unexpected number of tiles, or (c)
applying an image_mean / image_std mismatched to what the AWQ checkpoint expects.

This script loads ONLY the processor (no LLM, no GPU), feeds one CIFAR image
through the runner's actual `_prep_image` path, and prints the resulting
pixel_values tensor shape + every processor config attribute that could affect
the soft-label quality. Compare InternVL2's output to Qwen2-VL's to isolate the
divergence. Runs in seconds on an HPC login node.
"""

from __future__ import annotations

import argparse

from cac.data import cifar10h
from cac.models.vllm_runner import VLMRunner, _build_user_messages, _family

# Attributes worth surfacing across processor families. `hasattr` filters per model
# (InternVL has min/max_dynamic_patch, Qwen has size/patch_size, etc.), so the same
# list works for both — anything missing is just skipped.
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


def _print_processor_config(model_id: str):
    """Load AutoProcessor and print every attribute relevant to image preprocessing."""
    from transformers import AutoProcessor

    print(f"\n========== {model_id}  (family={_family(model_id)}) ==========")
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print(f"  processor class: {type(proc).__name__}")

    ip = getattr(proc, "image_processor", None)
    if ip is None:
        print("  [warn] processor.image_processor is None; printing processor attrs:")
        target = proc
    else:
        print(f"  image_processor class: {type(ip).__name__}")
        target = ip

    print("  --- named attributes (only those present) ---")
    seen = set()
    for k in INTERESTING_ATTRS:
        if hasattr(target, k):
            print(f"    {k:<22} = {getattr(target, k)!r}")
            seen.add(k)

    # Full dict dump — catches anything we didn't anticipate (e.g. dynamic_high_resolution,
    # max_num, min_num, force_image_size, downsample_ratio for InternVL custom processors).
    if ip is not None and hasattr(ip, "to_dict"):
        full = ip.to_dict()
        leftovers = {k: v for k, v in full.items() if k not in seen and not k.startswith("_")}
        if leftovers:
            print("  --- additional image_processor.to_dict() entries ---")
            for k, v in leftovers.items():
                # Compact long lists/dicts (e.g. dict of size buckets); show type+len if huge.
                vs = repr(v)
                if len(vs) > 200:
                    vs = f"{type(v).__name__}(len={len(v) if hasattr(v, '__len__') else '?'})"
                print(f"    {k:<22} = {vs}")

    tok = getattr(proc, "tokenizer", None)
    if tok is not None:
        print(f"  tokenizer: {type(tok).__name__}  is_fast={tok.is_fast}")

    return proc


def _print_runtime_shape(proc, model_id: str, image_np):
    """Feed one CIFAR image through the runner's exact pre-processing path and report shapes."""
    pil = VLMRunner._prep_image(image_np)  # the runner's upscale-to-224 BICUBIC step
    print(f"  [runtime] PIL after VLMRunner._prep_image: size={pil.size} mode={pil.mode}")

    text = "What is in this image? Answer with a single letter A-J."
    messages = _build_user_messages(model_id, text)
    content_type = type(messages[0]["content"]).__name__
    print(f"  [runtime] messages[0]['content'] type = {content_type}")

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
        print("  [runtime] no image_processor on proc; cannot report tensor shape")
        return

    try:
        out = ip(images=pil, return_tensors="pt")
        pv = out.get("pixel_values")
        if pv is None:
            print(f"  [runtime] image_processor output keys: {list(out.keys())}")
            return
        print(f"  [runtime] pixel_values.shape = {tuple(pv.shape)} dtype={pv.dtype}")
        print(f"  [runtime] pixel_values stats: min={float(pv.min()):.3f} "
              f"max={float(pv.max()):.3f} mean={float(pv.mean()):.3f}")

        if pv.ndim == 4 and pv.shape[0] > 1:
            print(f"  [INFO] dynamic tiling produced {pv.shape[0]} tiles of "
                  f"{pv.shape[-2]}x{pv.shape[-1]} from a {pil.size} input.")
        elif pv.ndim == 4 and pv.shape[0] == 1:
            print(f"  [INFO] single tile of {pv.shape[-2]}x{pv.shape[-1]} from a "
                  f"{pil.size} input (effective upscale: "
                  f"{pv.shape[-2] / pil.size[0]:.1f}x).")
        elif pv.ndim == 5:
            # (B, N_tiles, C, H, W) shape used by some InternVL processors
            print(f"  [INFO] 5-D output (B={pv.shape[0]}, N_tiles={pv.shape[1]}, "
                  f"tile={pv.shape[-2]}x{pv.shape[-1]}).")
    except Exception as e:
        print(f"  [runtime] image_processor call failed: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="OpenGVLab/InternVL2-8B-AWQ",
                    help="primary model id to diagnose")
    ap.add_argument("--compare", default="Qwen/Qwen2-VL-7B-Instruct-AWQ",
                    help="comparison model id; pass empty string to skip")
    args = ap.parse_args()

    images, _, _ = cifar10h.prepare()
    img = images[0]
    print(f"[data] CIFAR image idx=0 shape={img.shape} dtype={img.dtype}")

    proc = _print_processor_config(args.model)
    _print_runtime_shape(proc, args.model, img)

    if args.compare:
        proc2 = _print_processor_config(args.compare)
        _print_runtime_shape(proc2, args.compare, img)

    print("\n========== INTERPRETATION ==========")
    print("  Diagnostic targets:")
    print("    1. pixel_values.shape — if InternVL emits a single 448x448 tile from a")
    print("       224x224 input, the encoder sees a 14x upscale of the original 32x32")
    print("       CIFAR. High-frequency artifacts flatten the logits -> high entropy.")
    print("       Fix: upscale CIFAR to image_processor.image_size in _prep_image, but")
    print("       conditionally per family so Qwen's variable-resolution path is left")
    print("       alone.")
    print("    2. number of tiles — if dynamic tiling fires (N>1) on a near-uniform")
    print("       CIFAR upscale, the model is averaging features over redundant tiles")
    print("       of the same content. Set max_dynamic_patch=1 (or dynamic_image_size=")
    print("       False) for CIFAR.")
    print("    3. image_mean / image_std — must match training. A normalization shift")
    print("       silently flattens softmax. Compare AWQ vs fp16 vs 2B values.")
    print("    4. tokenizer.is_fast — informational; slow tokenizer alone won't move")
    print("       logprob mass like this, but worth noting for the writeup.")


if __name__ == "__main__":
    main()
