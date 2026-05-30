"""Entry point: download CIFAR-10H + CIFAR-10 test images and run the alignment gate.

    python -m run.prepare_data
    python -m run.prepare_data --download-models --profile hpc      # + pre-stage weights
    python -m run.prepare_data --download-models --profile hpc --dry-run   # sizes only

Safe to re-run; downloads are cached under data/. The --download-models flag also
pre-fetches the VLM ensemble + cross-encoder + DINOv2 weights into the HF hub cache for
offline HPC use (compute nodes usually have no internet).
"""

import argparse
import os

from cac import config
from cac.data import cifar10h


def _hub_cache(cache_dir: str | None) -> str:
    """Resolve the HF hub cache dir, matching what run_*.slurm set HF_HOME to."""
    if cache_dir:
        return cache_dir
    hf_home = os.environ.get("HF_HOME", str(config.REPO_ROOT / ".hf_cache"))
    return os.path.join(hf_home, "hub")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-models", action="store_true",
                    help="also pre-fetch model weights (VLMs + cross-encoder + DINOv2)")
    ap.add_argument("--profile", default="hpc",
                    help="VLM set to stage: 'hpc' (default), 'active', or a profiles.<name>")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --download-models: print sizes only, do not download")
    ap.add_argument("--yes", action="store_true",
                    help="with --download-models: skip the confirmation prompt")
    ap.add_argument("--cache-dir", default=None,
                    help="override HF hub cache dir (default: $HF_HOME/hub or repo .hf_cache/hub)")
    args = ap.parse_args()

    images, human_probs, labels = cifar10h.prepare()
    print(f"[ok] images {images.shape} {images.dtype}, probs {human_probs.shape}")
    print("[ok] data prep complete - alignment gate passed.")

    if args.download_models:
        from cac.models import staging

        hub_cache = _hub_cache(args.cache_dir)
        repo_ids = staging.all_repo_ids(args.profile)
        staging.download_all(repo_ids, hub_cache, yes=args.yes, dry_run=args.dry_run)
        hf_home = os.path.dirname(hub_cache)
        print(f"\n[stage] to use on HPC (no internet on compute nodes):")
        print(f"  rsync -av {hf_home}/  <HPC>:/scratch/$USER/cac/.hf_cache/")
        print(f"  rsync -av {config.DATA_DIR}/  <HPC>:/scratch/$USER/cac/data/")
        print(f"  run_*.slurm already set HF_HOME=$PWD/.hf_cache (point it at the copy).")


if __name__ == "__main__":
    main()
