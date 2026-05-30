#!/usr/bin/env bash
# Phase 1.1 local acceptance: 100 images x 3 VLMs end-to-end with real signals.
# Run inside WSL2 Ubuntu with the GPU venv active (see scripts/setup_wsl.md).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1.0 data prep + alignment gate =="
python -m run.prepare_data

echo "== 1.4 DINOv2 embeddings (full 10k, cached) =="
python -m run.extract_embeddings

echo "== 1.1 VLM ensemble on 100 images (real models) =="
python -m run.run_vision_inference --n 100 --skip-if-exists

echo "== 1.3 MTA from real rationales =="
python -m run.compute_mta

echo "== pipeline with REAL signals =="
python -m run.run_pipeline --real

echo "== SMOKE TEST COMPLETE — see outputs/figures/pipeline_real.png =="
