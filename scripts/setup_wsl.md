# Local GPU setup (WSL2 Ubuntu + vLLM) — Phase 1.1

The RTX 4060 Laptop (8 GB) can run **one 7-8B VLM at a time** (sequential swap). Use
this only for the **100-image verification batch**; the full 10k×3 run goes to the HPC.

vLLM is Linux-only, so we run it inside WSL2 Ubuntu. The lightweight steps
(`prepare_data`, `extract_embeddings`, `compute_mta`, `run_pipeline`) also work in
native Windows Python — only `run_vision_inference` (real models) needs WSL.

## 1. Install an Ubuntu WSL distro
Currently only the `docker-desktop` distro exists (`wsl -l -v`). Install Ubuntu:

```powershell
wsl --install -d Ubuntu          # in an elevated PowerShell; reboot if prompted
```

Verify the GPU is visible inside WSL (the Windows driver 555.97 supports WSL CUDA):

```bash
nvidia-smi                       # should list the RTX 4060
```

## 2. Python env + vLLM
vLLM is stable on Python 3.10–3.12 (NOT 3.13). Inside Ubuntu:

```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv git
cd /mnt/c/Yixuan_AgenticProject          # repo is visible from WSL here
python3.11 -m venv .venv-gpu
source .venv-gpu/bin/activate
pip install -U pip
pip install -r requirements-gpu.txt
```

Verify CUDA + vLLM:

```bash
python -c "import vllm, torch; print('cuda', torch.cuda.is_available())"   # -> cuda True
```

## 3. (gated models) HuggingFace login
Some checkpoints require accepting a license:

```bash
pip install huggingface_hub
huggingface-cli login            # paste an HF token with access to the gated repos
```

## 4. Run the smoke test (1.1 acceptance)
```bash
bash scripts/smoke_test_100.sh
```

## 8 GB tips
- `config/models_vision.yaml` defaults to **small** VLMs that fit 8 GB. The full
  7-8B AWQ ensemble lives under `profiles.hpc`.
- If a model OOMs: lower `max_model_len`, set `max_num_seqs: 1`, keep `enforce_eager: true`,
  or switch to a smaller `id`. The local run only validates the *pipeline*; final
  numbers come from the HPC run.
- IO note: `/mnt/c/...` is slower than the native WSL filesystem. Fine for 100 images;
  for the full run, consider cloning into `~/` inside WSL and pointing
  `CAC_DATA_DIR` / `CAC_OUTPUTS_DIR` there.
