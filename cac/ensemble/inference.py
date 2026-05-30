"""Sequential ensemble inference + output loaders (Phase 1.1).

On 8 GB we run one model at a time (load -> infer -> save JSONL -> free VRAM ->
next). The full 10k x 3 run targets the HPC; locally we run the 100-image
verification batch. The loaders are pure-IO (no vLLM import) so the pipeline can
read results on the Windows host.

Output: outputs/raw/<model_key>.jsonl, one record per image:
  {"idx", "dist":[10], "letter", "label_dist", "label_json", "rationale"}
"""

from __future__ import annotations

import json

import numpy as np

from cac import config
from cac.data.labels import IDX_TO_CLASS, LETTER_TO_IDX
from cac.ensemble.soft_labels import logprobs_to_dist


def _raw_path(model_key: str, mock: bool = False):
    base = config.MOCK_RAW_DIR if mock else config.RAW_DIR
    return base / f"{model_key}.jsonl"


def model_keys_from_outputs(mock: bool = False) -> list[str]:
    """Model keys for which an output JSONL exists, sorted for stable ensemble order.

    Default scans `outputs/raw/` non-recursively, which skips `outputs/raw/mocks/`
    automatically — real loaders stay clean even if mock fixtures exist.
    """
    base = config.MOCK_RAW_DIR if mock else config.RAW_DIR
    return sorted(p.stem for p in base.glob("*.jsonl"))


# --------------------------------------------------------------------------- run
def run_model(model_key: str, model_id: str, images: np.ndarray, *,
              batch_size: int = 16, runner_kwargs: dict | None = None) -> str:
    """Run one VLM over `images`, write its JSONL, free VRAM. Returns the output path."""
    from cac.models.vllm_runner import VLMRunner

    runner = VLMRunner(model_id, **(runner_kwargs or {}))
    out_path = _raw_path(model_key)
    n = len(images)
    with open(out_path, "w", encoding="utf-8") as f:
        for start in range(0, n, batch_size):
            chunk = list(images[start:start + batch_size])
            letter_lps = runner.distribution_call(chunk)
            rats = runner.rationale_call(chunk)
            for k, (llp, rj) in enumerate(zip(letter_lps, rats)):
                dist = logprobs_to_dist(llp)
                letter = max(llp, key=llp.get) if llp else ""
                rec = {
                    "idx": start + k,
                    "dist": [round(float(x), 6) for x in dist],
                    "letter": letter,
                    "label_dist": IDX_TO_CLASS.get(LETTER_TO_IDX.get(letter, -1), ""),
                    "label_json": rj.get("label", ""),
                    "rationale": rj.get("rationale", {}),
                    "rationale_raw": rj.get("_raw", ""),
                    "parse_status": rj.get("parse_status", ""),
                }
                f.write(json.dumps(rec) + "\n")
            print(f"[infer:{model_key}] {min(start + batch_size, n)}/{n}", end="\r")
    print()
    runner.close()
    print(f"[infer:{model_key}] wrote {out_path}")
    return str(out_path)


def run_ensemble(images: np.ndarray, models: list[tuple[str, str]], **kw):
    """models: list of (model_key, model_id). Runs each sequentially."""
    for key, mid in models:
        print(f"\n=== {key} ({mid}) ===")
        run_model(key, mid, images, **kw)


def validate_model_output(model_key: str, expected_n: int, rationale_tol: float = 0.02,
                          mock: bool = False):
    """Validate one model's JSONL: record count, dist sums to 1, structured rationale.

    Raises on any invalid distribution (critical) or if the fraction of records with a
    malformed rationale exceeds `rationale_tol`. Designed to be the gate a SLURM array
    task runs before it is allowed to succeed.
    """
    from cac.models.schema import RATIONALE_KEYS

    path = _raw_path(model_key, mock=mock)
    if not path.exists():
        raise FileNotFoundError(f"missing output for '{model_key}': {path}")
    recs = _read_jsonl(path)
    if len(recs) != expected_n:
        raise AssertionError(f"{model_key}: {len(recs)} records != expected {expected_n}")

    bad_rationale = 0
    for r in recs:
        dist = r.get("dist")
        if not dist or abs(sum(dist) - 1.0) > 1e-4:
            raise AssertionError(
                f"{model_key} idx {r.get('idx')}: distribution invalid "
                f"(sum={sum(dist) if dist else None})"
            )
        rat = r.get("rationale", {})
        if not (isinstance(rat, dict) and all(k in rat for k in RATIONALE_KEYS)):
            bad_rationale += 1

    frac = bad_rationale / max(len(recs), 1)
    if frac > rationale_tol:
        raise AssertionError(
            f"{model_key}: {bad_rationale}/{len(recs)} ({frac:.1%}) malformed rationales "
            f"> tol {rationale_tol:.1%}"
        )
    note = f" ({bad_rationale} malformed rationales within tol)" if bad_rationale else ""
    print(f"[validate:{model_key}] {len(recs)} records OK: dist sums to 1, "
          f"rationale structured{note}.")
    return True


# ------------------------------------------------------------------------ loaders
def _read_jsonl(path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_distributions(model_keys: list[str] | None = None, mock: bool = False
                       ) -> tuple[np.ndarray, list[str]]:
    """Return (dists (M,N,K), model_keys). Records assumed aligned by file order/idx."""
    keys = model_keys or model_keys_from_outputs(mock=mock)
    if not keys:
        base = config.MOCK_RAW_DIR if mock else config.RAW_DIR
        raise FileNotFoundError(f"no inference outputs in {base}")
    mats = []
    for key in keys:
        recs = sorted(_read_jsonl(_raw_path(key, mock=mock)), key=lambda r: r["idx"])
        mats.append(np.array([r["dist"] for r in recs], dtype=np.float64))
    return np.stack(mats, axis=0), keys


def load_rationales(model_keys: list[str] | None = None, mock: bool = False
                    ) -> list[list[str]]:
    """Return [N][M] rationale TEXT for MTA (structured objects flattened to strings)."""
    from cac.models.schema import rationale_to_text

    keys = model_keys or model_keys_from_outputs(mock=mock)
    per_model = []
    for key in keys:
        recs = sorted(_read_jsonl(_raw_path(key, mock=mock)), key=lambda r: r["idx"])
        per_model.append([rationale_to_text(r.get("rationale", {})) for r in recs])
    n = len(per_model[0])
    return [[per_model[m][i] for m in range(len(keys))] for i in range(n)]


def load_rationale_objects(model_keys: list[str] | None = None, mock: bool = False
                           ) -> list[list[dict]]:
    """Return [N][M] structured rationale objects (for the results DB / analysis)."""
    keys = model_keys or model_keys_from_outputs(mock=mock)
    per_model = []
    for key in keys:
        recs = sorted(_read_jsonl(_raw_path(key, mock=mock)), key=lambda r: r["idx"])
        per_model.append([r.get("rationale", {}) for r in recs])
    n = len(per_model[0])
    return [[per_model[m][i] for m in range(len(keys))] for i in range(n)]


def load_argmax_labels(model_keys: list[str] | None = None, mock: bool = False
                       ) -> np.ndarray:
    """Return (M, N) predicted class indices (argmax of each model's distribution)."""
    dists, _ = load_distributions(model_keys, mock=mock)
    return dists.argmax(axis=2)
