#!/bin/bash
# Aggregation after the inference array (Phase 2.1/2.3). Runs once, AFTER all 3 model
# tasks succeed. Light models (DINOv2, cross-encoder) — a small GPU or CPU partition is fine.
#
# ### EDIT (or set via scripts/hpc/submit.sh): partition, account, walltime, email ###
# Submit gated on the array succeeding (submit.sh does this for you):
#   AID=$(sbatch --parsable scripts/hpc/run_inference.slurm)
#   sbatch --dependency=afterok:$AID scripts/hpc/run_aggregate.slurm
#
#SBATCH --job-name=cac-aggregate
#SBATCH --partition=FIXME_PARTITION         # FIXME (a CPU partition also works)
#SBATCH --account=FIXME_ACCOUNT             # FIXME
#SBATCH --time=FIXME_WALLTIME               # FIXME: e.g. 02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=FIXME_EMAIL
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load cuda/12.1
module load python/3.11
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv-gpu/bin/activate
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
# export CAC_DATA_DIR=/scratch/$USER/cac/data
# export CAC_OUTPUTS_DIR=/scratch/$USER/cac/outputs

python -m run.extract_embeddings     # idempotent (cached) — ensures DINOv2 embeddings exist
python -m run.compute_mta            # real MTA from the array's rationales
python -m run.build_results          # consolidated outputs/results.jsonl
python -m run.run_pipeline --real    # figure + tracking matrix from real signals
python -m run.report                 # reality-check report vs the v4 simulation benchmarks

echo "aggregation complete -> outputs/results.jsonl, outputs/figures/pipeline_real.png"
