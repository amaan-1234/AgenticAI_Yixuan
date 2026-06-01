"""Entry point: run the VLM ensemble over CIFAR-10H images (Phase 1.1).

    # local 100-image verification batch (WSL2 + vLLM):
    python -m run.run_vision_inference --n 100

    # full run, all active models (HPC, sequential):
    python -m run.run_vision_inference --profile hpc --n 10000

    # ONE model of a profile (SLURM array task; --model-index = SLURM_ARRAY_TASK_ID):
    python -m run.run_vision_inference --profile hpc --model-index 0 --n 10000

    # fixture outputs for pipeline testing (NO GPU; Windows-friendly):
    python -m run.run_vision_inference --mock --n 200

Models are read from config/models_vision.yaml. `--profile` selects `models:` (active)
or a `profiles.<name>` list. After a run, validates that every record has a parseable
label and a 10-way distribution summing to 1 (the 1.1 acceptance check). When
--model-index is given, only that model runs and a per-model validator gates success.
"""

import argparse

import numpy as np

from cac import config
from cac.data import cifar10h
from cac.data.labels import CIFAR10_CLASSES
from cac.ensemble import inference


def resolve_models(cfg: dict, profile: str) -> list[dict]:
    """Return the model list for the requested profile ('active' -> cfg['models'])."""
    if profile in (None, "active"):
        return cfg["models"]
    profiles = cfg.get("profiles", {})
    if profile not in profiles:
        raise SystemExit(f"unknown profile '{profile}'; have {list(profiles)} + 'active'")
    return profiles[profile]


def validate(model_keys, n, mock: bool = False):
    dists, keys = inference.load_distributions(model_keys, mock=mock)
    assert dists.shape[0] == len(model_keys) and dists.shape[1] == n, dists.shape
    assert dists.shape[2] == len(CIFAR10_CLASSES)
    sums = dists.sum(axis=2)
    assert np.allclose(sums, 1.0, atol=1e-4), "distributions do not sum to 1"
    rats = inference.load_rationales(model_keys, mock=mock)
    n_empty = sum(1 for item in rats for r in item if not r.strip())
    print(f"[validate] {len(keys)} models x {n} images: distributions OK (sum=1).")
    print(f"[validate] empty rationales: {n_empty} (lower is better)")
    print("[ok] 1.1 acceptance: valid structured outputs + 10-way distributions.")


def _select_index(items: list, model_index: int | None):
    if model_index is None:
        return items
    if not 0 <= model_index < len(items):
        raise SystemExit(f"--model-index {model_index} out of range (0..{len(items) - 1})")
    return [items[model_index]]


def _skip_existing(key: str, n: int, mock: bool = False) -> bool:
    """True iff the model's JSONL already has at least `n` records."""
    base = config.MOCK_RAW_DIR if mock else config.RAW_DIR
    p = base / f"{key}.jsonl"
    if not p.exists():
        return False
    with open(p, encoding="utf-8") as f:
        return sum(1 for _ in f) >= n


# Known LLM kwargs that may appear at the TOP LEVEL of a model entry in
# models_vision.yaml (alongside `key` / `id`). Top-level wins over both the
# nested `runner:` block and the global `runner:` defaults.
_LIFTABLE_KWARGS = ("quantization", "max_model_len", "dtype")


def _resolve_runner_kwargs(global_runner: dict, model_entry: dict) -> dict:
    """Merge global runner defaults, the model's nested `runner:` block, and any
    top-level overrides in _LIFTABLE_KWARGS. Top-level wins. Pure / unit-testable."""
    kw = dict(global_runner or {})
    kw.update(model_entry.get("runner", {}) or {})
    for k in _LIFTABLE_KWARGS:
        if k in model_entry:
            kw[k] = model_entry[k]
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", "--n-images", type=int, dest="n", default=100,
                    help="number of images (from index 0)")
    ap.add_argument("--mock", action="store_true", help="write fixture outputs (no GPU)")
    ap.add_argument("--profile", default="active",
                    help="model set: 'active' (cfg models:) or a profiles.<name>")
    ap.add_argument("--model-index", type=int, default=None,
                    help="run only the Nth model of the profile (SLURM array task)")
    ap.add_argument("--skip-if-exists", action="store_true",
                    help="skip a model whose outputs/raw/<key>.jsonl already has >= --n records")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    images, human_probs, _ = cifar10h.prepare()
    sub = images[: args.n]

    if args.mock:
        from cac.ensemble.mock import DEFAULT_MODELS, write_mock_outputs

        models = _select_index(DEFAULT_MODELS, args.model_index)
        keys = [k for k, _ in models]
        to_write = []
        for m in models:
            if args.skip_if_exists and _skip_existing(m[0], args.n, mock=True):
                print(f"[skip] {m[0]}: already has >= {args.n} records")
            else:
                to_write.append(m)
        if to_write:
            write_mock_outputs(human_probs, n=args.n, models=to_write)
        if args.model_index is not None:
            inference.validate_model_output(keys[0], args.n, mock=True)
        else:
            validate(keys, args.n, mock=True)
        return

    cfg = config.load_yaml("models_vision")
    global_kw = cfg.get("runner", {})
    model_list = _select_index(resolve_models(cfg, args.profile), args.model_index)
    for m in model_list:
        if args.skip_if_exists and _skip_existing(m["key"], args.n):
            print(f"[skip] {m['key']}: already has >= {args.n} records")
            if args.model_index is not None:
                inference.validate_model_output(m["key"], args.n)
            continue
        kw = _resolve_runner_kwargs(global_kw, m)
        print(f"\n=== {m['key']} ({m['id']}) ===")
        inference.run_model(m["key"], m["id"], sub, batch_size=args.batch_size,
                            runner_kwargs=kw)
        if args.model_index is not None:
            inference.validate_model_output(m["key"], args.n)

    if args.model_index is None:
        validate([m["key"] for m in model_list], args.n)


if __name__ == "__main__":
    main()
