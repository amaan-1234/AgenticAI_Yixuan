#!/usr/bin/env bash
# One-place submitter for the HPC pipeline. Set the 5 variables below, then:
#   bash scripts/hpc/submit.sh            # full 10k run
#   bash scripts/hpc/submit.sh --pilot    # 100-image pilot (reality check first)
#
# It submits the inference ARRAY, then the AGGREGATE job gated on all array tasks
# succeeding (--dependency=afterok). The sbatch CLI flags below OVERRIDE the FIXME_*
# placeholders inside the .slurm files, so you only edit values here.
#
# Prereq (run ONCE on the login node — compute nodes often lack internet):
#   python -m run.prepare_data
set -euo pipefail
cd "$(dirname "$0")/../.."

# ===== EDIT THESE =====
PARTITION="FIXME_PARTITION"     # e.g. gpu        (check: sinfo)
ACCOUNT="FIXME_ACCOUNT"         # e.g. proj1234   (check: sacctmgr show user $USER)
WALLTIME="08:00:00"             # inference array walltime
AGG_WALLTIME="02:00:00"         # aggregation walltime
EMAIL="FIXME_EMAIL"             # for --mail-type=FAIL alerts
# ======================

mkdir -p logs
common=(--partition="$PARTITION" --account="$ACCOUNT" --mail-user="$EMAIL")

echo "Submitting inference array (args: ${*:-none})..."
AID=$(sbatch --parsable "${common[@]}" --time="$WALLTIME" \
      scripts/hpc/run_inference.slurm "$@")
echo "  inference array job id: $AID"

echo "Submitting aggregation (afterok:$AID)..."
GID=$(sbatch --parsable "${common[@]}" --time="$AGG_WALLTIME" \
      --dependency=afterok:"$AID" scripts/hpc/run_aggregate.slurm)
echo "  aggregate job id: $GID"

echo "Done. Watch: squeue -u \$USER ; tail -f logs/cac-vlm-infer-*.out"
