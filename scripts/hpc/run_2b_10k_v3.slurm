#!/bin/bash
#SBATCH --job-name=cac-2b-10k
#SBATCH --partition=public
#SBATCH --account=grp_yixuanh2
#SBATCH --qos=public
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=amaanmohamed55@gmail.com
#SBATCH --output=logs/%x-%j.out
set -uo pipefail

PY=/home/akalemul/.conda/envs/cac/bin/python
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME=/scratch/$USER/cac/.hf_cache
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH

echo "== env check =="
$PY -c "import sys,numpy,torch,vllm; print(sys.executable); print('numpy',numpy.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available(),'vllm',vllm.__version__)" || { echo "ENV BROKEN"; exit 1; }

echo "== 2B ensemble (active profile), N=10000 =="
$PY -m run.run_vision_inference --profile active --n 10000 --batch-size 64 --skip-if-exists

echo "== aggregate =="
$PY -m run.compute_mta
$PY -m run.build_results
$PY -m run.run_pipeline --real
$PY -m run.report
echo "DONE -> outputs/results.jsonl, outputs/figures/pipeline_real.png"
