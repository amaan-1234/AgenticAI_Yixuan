# Cost-Aware Calibration via Heterogeneous Ensemble Disagreement

Multi-agent annotation-calibration system. The ensemble of small open-weight models
produces soft-label distributions + rationales; inter-model disagreement
(**JSD** primary, **MTA** secondary) drives calibrated escalation to a frontier
model only on genuinely ambiguous items. Evaluated against human disagreement on
**CIFAR-10H** (vision) and **ChaosNLI** (text, cross-domain).

**Phase 1 (this repo)** replaces the v4 *simulation* (`files (1)/test_disagreement_v4.py`)
with **real model infrastructure**: real VLM soft labels via vLLM+logprobs, real
cross-encoder MTA, real DINOv2 pre-router embeddings — fed into the existing
calibration pipeline. See `files (1)/literature_review.md` for the full background.

## Status (Phase 1, CIFAR-10H / vision-first)

| Step | What | State |
|---|---|---|
| 1.0 | Data prep + ordering alignment gate | **done & verified** (agreement 0.9921) |
| 1.4 | Real DINOv2-ViT-S/14 embeddings (10k×384) | **done & verified** (pre-router AUC 0.767) |
| 1.3 | Real cross-encoder MTA (`ms-marco-MiniLM-L-6-v2`) | **done & verified** (sanity test passes) |
| —   | Pipeline refactor + `--simulate`/`--real` | **done** (sim reproduces v4: dual r≈0.79) |
| 1.1 | VLM ensemble via vLLM+Outlines (logprob soft labels) | **code complete**; needs WSL+vLLM to run |
| 1.2 | ChaosNLI text path | scaffolded (`cac/data/chaosnli.py`, `config/models_text.yaml`) |
| —   | Pre-router false-easy at N=100 WSL2 = 22.2% | **investigate at N=10k HPC** — see `cac/pipeline/prerouter.py` |

Everything except **1.1** runs and is verified on the Windows host (CUDA). 1.1 needs
the Linux+CUDA vLLM stack (WSL2 locally for the 100-image smoke test; HPC for the
full 10k run). A fixture (`run_vision_inference --mock`) exercises the full `--real`
pipeline without a GPU.

## Layout
```
cac/                 # package
  config.py          # paths, device, YAML loader
  targets.py         # human entropy + 'truly hard' mask
  data/              # cifar10h (download+align), labels, chaosnli (stub)
  models/            # vllm_runner, prompts, schema  (Linux+CUDA only)
  ensemble/          # soft_labels (logprobs->dist), jsd, inference, mock
  mta/cross_encoder  # real MTA
  embeddings/dino    # real DINOv2
  pipeline/          # weights, calibration, prerouter, metrics, figures (v4 §5-12)
  sim/legacy_v4      # original simulators (for --simulate + Phase-3.1 gap analysis)
run/                 # prepare_data, run_vision_inference, compute_mta,
                     #   extract_embeddings, run_pipeline
config/              # cifar10h.yaml, models_vision.yaml, models_text.yaml
scripts/             # setup_wsl.md, smoke_test_100.sh, hpc/run_inference.slurm
tests/               # jsd, soft_labels, data alignment
```

## Quickstart

### On the Windows host (no vLLM) — verifies 1.0 / 1.3 / 1.4 + pipeline
```powershell
pip install -r requirements-light.txt
python -m run.prepare_data            # download + alignment gate
python -m run.extract_embeddings      # real DINOv2 (1.4)
python -m run.compute_mta --sanity    # cross-encoder acceptance (1.3)
python -m run.run_vision_inference --mock --n 200   # fixture ensemble outputs
python -m run.run_pipeline --simulate # reproduce v4
python -m run.run_pipeline --real     # real DINOv2 + real MTA + fixture distributions
python -m pytest tests/ -q
```

### Real VLM inference (1.1) — WSL2 Ubuntu or HPC
See `scripts/setup_wsl.md`, then:
```bash
bash scripts/smoke_test_100.sh        # 100 images x 3 VLMs, end-to-end
```
Full 10k run on the cluster: `sbatch scripts/hpc/run_inference.slurm`
(set `models_vision.yaml` `models:` to the `profiles.hpc` 7-8B ensemble first).

## Key design choices
- **Soft labels = token logprobs.** The answer is constrained to a single class
  letter (A..J); the position-0 logprobs over those letters are softmaxed into a
  genuine distribution — not a self-reported confidence number.
- **One model at a time on 8 GB.** Sequential load→infer→free; full ensemble on HPC.
- **`--simulate` is preserved**, not deleted: it powers the Phase-3.1 real-vs-sim gap
  analysis (a stated contribution).
- **Alignment gate first.** A silent CIFAR-10 / CIFAR-10H ordering mismatch would
  invalidate every metric, so it is asserted before any inference.

## Notes / risks
- vLLM ⇄ model-version support changes fast; `requirements-gpu.txt` pins a version —
  re-run the smoke test after any bump.
- `ms-marco-MiniLM-L-6-v2` is a *relevance* cross-encoder; it cleanly separates
  agreeing vs contradicting rationales in the sanity test but is asymmetric. An NLI
  cross-encoder is a candidate if MTA signal quality is weak on real rationales
  (a Discussion-section limitation).
