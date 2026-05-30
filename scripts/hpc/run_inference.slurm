#!/bin/bash
# CIFAR-10H VLM inference as a SLURM JOB ARRAY (Phase 2.1) — one task per model.
# Each array task runs ONE model from `profiles.hpc` in config/models_vision.yaml and
# self-validates its output (run_vision_inference -> validate_model_output), so a bad
# task is marked FAILED and the afterok dependency for aggregation will not fire.
#
# ### EDIT (or set via scripts/hpc/submit.sh): partition, account, walltime, email ###
# Recommended: use submit.sh, which sets these once and passes them as sbatch flags
# (overriding the FIXME_* placeholders below) and chains the aggregate job.
#
# Manual run order (if not using submit.sh):
#   1) ONCE on the LOGIN node (compute nodes often lack internet):
#        python -m run.prepare_data
#   2) mkdir -p logs
#   3) AID=$(sbatch --parsable scripts/hpc/run_inference.slurm)
#   4) sbatch --dependency=afterok:$AID scripts/hpc/run_aggregate.slurm
#
#SBATCH --job-name=cac-vlm-infer
#SBATCH --partition=FIXME_PARTITION         # FIXME: e.g. gpu (check `sinfo`)
#SBATCH --account=FIXME_ACCOUNT             # FIXME: (check `sacctmgr show user $USER`)
#SBATCH --time=FIXME_WALLTIME               # FIXME: e.g. 08:00:00
#SBATCH --array=0-2                          # 3 tasks = 3 VLMs in profiles.hpc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=FIXME_EMAIL             # FIXME: alerts only on job failure
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

# --- environment (confirm exact module names with `module avail`) -------------
module load cuda/12.1
module load python/3.11
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv-gpu/bin/activate
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
# Optional fast scratch:
# export CAC_DATA_DIR=/scratch/$USER/cac/data
# export CAC_OUTPUTS_DIR=/scratch/$USER/cac/outputs

# --pilot => first 100 images only (reality check before the full 10k GPU spend).
#   bash scripts/hpc/submit.sh --pilot
N_IMAGES=10000
if [ "${1:-}" = "--pilot" ]; then N_IMAGES=100; fi

echo "== array task ${SLURM_ARRAY_TASK_ID}: one VLM from profiles.hpc, N=${N_IMAGES} =="
python -m run.run_vision_inference \
    --profile hpc \
    --model-index "${SLURM_ARRAY_TASK_ID}" \
    --n "${N_IMAGES}" \
    --batch-size 64

echo "array task ${SLURM_ARRAY_TASK_ID} complete and validated."
