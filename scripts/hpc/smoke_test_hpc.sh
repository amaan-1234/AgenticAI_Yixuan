#!/usr/bin/env bash
# No-GPU HPC smoke test: validate the data-pipeline WIRING on the cluster filesystem
# BEFORE spending GPU hours. Catches the usual HPC breaks (paths, venv, module versions,
# missing pre-staged data) on a cheap 10-minute CPU job.
#
# Run directly:   bash scripts/hpc/smoke_test_hpc.sh
# Or as a job:    sbatch scripts/hpc/smoke_test_hpc.sh   (edit FIXME_* first)
#
#SBATCH --job-name=cac-smoke
#SBATCH --partition=FIXME_PARTITION    # FIXME (CPU partition is fine — no GPU needed)
#SBATCH --account=FIXME_ACCOUNT        # FIXME
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=FIXME_EMAIL
#SBATCH --output=logs/%x-%j.out

# NOTE: intentionally no `set -e` — we catch each step and report PASS/FAIL with a reason.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"

fail() { echo "FAIL: $1"; exit 1; }

echo "== [1/6] activate venv =="
# shellcheck disable=SC1091
source .venv-gpu/bin/activate || fail "step 1 (venv) — .venv-gpu missing? see scripts/setup_wsl.md"

echo "== [2/6] import check =="
python -c "import cac, numpy, torch, sklearn" || fail "step 2 (imports) — deps not installed in venv"

echo "== [3/6] data presence =="
python -c "import os,sys; from cac import config; \
sys.exit(0 if (config.CIFAR10H_PROBS.exists() and config.CIFAR10_IMAGES.exists()) else 1)" \
  || fail "step 3 (data) — run 'python -m run.prepare_data' on the LOGIN node (compute nodes often lack internet)"

echo "== [4/6] per-model mock inference (50 images x 3, mirrors the array) =="
for i in 0 1 2; do
  python -m run.run_vision_inference --mock --model-index "$i" --n-images 50 --skip-if-exists \
    || fail "step 4 (mock inference / validate_model_output) for model-index $i"
done

echo "== [5/6] audit outputs (schema + JSD) =="
python -m run.audit_outputs || fail "step 5 (audit_outputs)"

echo "== [6/6] model-cache pre-stage check (soft) =="
python - <<'PY'
from pathlib import Path
hub = Path.home() / ".cache" / "huggingface" / "hub"
have = {p.name for p in hub.glob("models--*")} if hub.exists() else set()
need = ["models--facebook--dinov2-small", "models--cross-encoder--ms-marco-MiniLM-L-6-v2"]
missing = [n for n in need if n not in have]
if missing:
    print(f"NOTE: HF cache missing {missing} — pre-stage on the LOGIN node before run_aggregate:")
    print("      python -m run.extract_embeddings && python -m run.compute_mta --sanity")
else:
    print("model cache OK (DINOv2 + cross-encoder present)")
PY

echo "PASS: HPC wiring OK (venv + imports + data + mock 50x3 + audit)."
