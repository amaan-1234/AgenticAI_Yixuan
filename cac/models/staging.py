"""Pre-stage model weights for offline HPC runs.

HPC compute nodes usually have no internet, so all weights must be downloaded on an
internet-connected machine and rsynced to scratch. This module resolves exactly the
repos the pipeline loads (VLM ensemble + cross-encoder for MTA + DINOv2 for the
pre-router), estimates their disk footprint, and fetches them into the HF hub cache
that `run_*.slurm` point `HF_HOME` at.
"""

from __future__ import annotations

from cac import config


def vlm_repo_ids(profile: str = "hpc") -> list[str]:
    """HF ids for the VLM set ('active' = config models:, else profiles.<profile>)."""
    cfg = config.load_yaml("models_vision")
    if profile in (None, "active"):
        models = cfg["models"]
    else:
        profiles = cfg.get("profiles", {})
        if profile not in profiles:
            raise SystemExit(f"unknown profile '{profile}'; have {list(profiles)} + 'active'")
        models = profiles[profile]
    return [m["id"] for m in models]


def all_repo_ids(profile: str = "hpc") -> list[str]:
    """VLM ids + the cross-encoder (MTA) + DINOv2 (pre-router), de-duplicated.

    Imports the model constants so the staged set always matches what the pipeline
    actually loads at runtime.
    """
    from cac.embeddings.dino import MODEL_ID as DINO_ID
    from cac.mta.cross_encoder import MODEL_ID as CE_ID

    ids = vlm_repo_ids(profile) + [CE_ID, DINO_ID]
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def estimate_size_gb(repo_id: str) -> float | None:
    """Approx download size in GB via the HF API, or None if unavailable."""
    from huggingface_hub import HfApi

    try:
        info = HfApi().model_info(repo_id, files_metadata=True)
        total = sum((s.size or 0) for s in (info.siblings or []))
        return total / 1e9 if total else None
    except Exception as e:  # gated/no-metadata/offline — report as unknown
        print(f"  [warn] size unknown for {repo_id}: {type(e).__name__}")
        return None


def download_all(repo_ids: list[str], hub_cache: str, yes: bool = False,
                 dry_run: bool = False) -> None:
    """Print a size table then snapshot_download each repo into `hub_cache`."""
    print(f"\nModels to stage into: {hub_cache}")
    print(f"  {'repo':<48} {'size':>10}")
    print("  " + "-" * 60)
    sizes = {}
    for rid in repo_ids:
        gb = estimate_size_gb(rid)
        sizes[rid] = gb
        print(f"  {rid:<48} {(f'{gb:.2f} GB' if gb else 'unknown'):>10}")
    known = sum(v for v in sizes.values() if v)
    n_unknown = sum(1 for v in sizes.values() if not v)
    print("  " + "-" * 60)
    print(f"  TOTAL (known): {known:.2f} GB"
          + (f"  (+{n_unknown} repos of unknown size)" if n_unknown else ""))

    if dry_run:
        print("[dry-run] no files downloaded.")
        return

    if not yes:
        resp = input("\nProceed with download? [y/N] ").strip().lower()
        if resp not in ("y", "yes"):
            print("Aborted.")
            return

    from huggingface_hub import snapshot_download

    for rid in repo_ids:
        print(f"\n[download] {rid} ...")
        snapshot_download(repo_id=rid, cache_dir=hub_cache)
    print(f"\n[ok] staged {len(repo_ids)} repos into {hub_cache}")
