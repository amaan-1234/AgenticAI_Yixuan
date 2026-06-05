"""Full calibration pipeline (refactor of v4 sections 5-12).

    python -m run.run_pipeline --simulate     # reproduce v4 with simulated signals
    python -m run.run_pipeline --real         # use real JSD / MTA / DINOv2 embeddings

--real reads ensemble distributions from outputs/raw/*.jsonl (real VLM outputs, or
the fixtures from `run_vision_inference --mock`), computes real JSD, real MTA from
rationales (cross-encoder), and loads real DINOv2 embeddings. --simulate keeps the
legacy simulators so the real-vs-sim gap (Phase 3.1) can be measured side by side.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.stats import mode as scipy_mode
from sklearn.model_selection import train_test_split

from cac import config
from cac.data import target_source
from cac.data import cifar10h
from cac.ensemble.jsd import mean_pairwise_jsd
from cac.pipeline import calibration, figures, metrics, prerouter, weights
from cac.targets import BUDGET, hard_mask

FRONTIER_ACC = 0.95  # GPT-4o/Claude-level CIFAR-10 ceiling (Phase 2.5 will measure)


def _split(n):
    return train_test_split(np.arange(n), test_size=0.3, random_state=42)


def _gather_real(dataset='cifar10h'):
    """Return (dists (M,N,K), mta (N,), embeddings (N,384), human_probs (N,10), keys)."""
    from cac.ensemble import inference
    from cac.mta.cross_encoder import MTAScorer

    dists, keys = inference.load_distributions()
    n = dists.shape[1]
    human_probs = target_source.human_probs(dataset)[:n]

    rationales = inference.load_rationales(keys)
    mta = MTAScorer().mta_for_rationales(rationales)

    emb = target_source.embeddings(dataset, n)
    assert emb.shape[0] == n, f"embedding count {emb.shape[0]} != N {n}"
    print(f"[real] N={n}, models={keys}")
    return dists, mta, emb, human_probs, keys


def _gather_sim():
    """Return (dists, mta, embeddings, human_probs, keys, ablation) for --simulate."""
    from cac.sim import legacy_v4 as sim

    _, human_probs, _ = cifar10h.prepare()
    n = len(human_probs)
    he = hard_mask(human_probs)[1]
    rng = np.random.default_rng(42)

    ablation = {}
    for name in sim.ABLATION_CONFIGS:
        models = sim.ensemble_distributions(human_probs, name, rng)
        jsd = mean_pairwise_jsd(np.stack(models))
        r, rho = metrics.correlations(jsd, he)
        ablation[name] = {"r": r, "rho": rho, "n": len(models)}

    primary = sim.ensemble_distributions(human_probs, sim.PRIMARY, rng)
    dists = np.stack(primary)
    mta = sim.simulate_mta(he, rng)
    emb = sim.simulate_clip_embeddings(he)
    keys = [f"{fam}[{i}]" for fam, i in sim.ABLATION_CONFIGS[sim.PRIMARY]]
    return dists, mta, emb, human_probs, keys, ablation


def run(mode: str, dataset: str = 'cifar10h'):
    ablation = None
    if mode == "real":
        dists, mta, emb, human_probs, keys = _gather_real(dataset)
    else:
        dists, mta, emb, human_probs, keys, ablation = _gather_sim()

    truly_hard, he, cutoff = hard_mask(human_probs)
    n = len(human_probs)
    idx_tr, idx_va = _split(n)

    # --- signals -----------------------------------------------------------
    ensemble_jsd = mean_pairwise_jsd(dists)
    jsd_norm = metrics.normalise_01(ensemble_jsd)
    one_minus_mta = 1.0 - mta

    w_mi = weights.mi_weights(jsd_norm, one_minus_mta, he)
    w_lr = weights.logistic_weights(jsd_norm[idx_tr], one_minus_mta[idx_tr],
                                    truly_hard[idx_tr].astype(float))
    alpha, beta = w_mi["alpha"], w_mi["beta"]
    escalation = alpha * jsd_norm + beta * one_minus_mta

    r_jsd, rho_jsd = metrics.correlations(ensemble_jsd, he)
    r_dual, rho_dual = metrics.correlations(escalation, he)

    # --- calibration + pre-router -----------------------------------------
    cal = calibration.fit_calibrators(escalation, truly_hard, idx_tr, idx_va, BUDGET)
    best = cal["best"]
    pr = prerouter.train_prerouter(emb, truly_hard, idx_tr, idx_va)
    cost = metrics.cost_stages(pr["hard"], best["p_val"], best["tau"], len(idx_va))

    # --- tracking matrix ---------------------------------------------------
    human_argmax = human_probs.argmax(1)
    model_argmax = dists.argmax(2)                       # (M, N)
    single_acc = max(float((model_argmax[m][idx_va] == human_argmax[idx_va]).mean())
                     for m in range(dists.shape[0]))
    majority = scipy_mode(model_argmax, axis=0, keepdims=False).mode  # (N,)
    escalated_va = best["p_val"] >= best["tau"]
    casc = metrics.cascade_accuracy(majority[idx_va], human_argmax[idx_va],
                                    escalated_va, FRONTIER_ACC)

    _print_summary(mode, keys, r_jsd, rho_jsd, r_dual, rho_dual, w_mi, w_lr,
                   best, cost, single_acc, casc, pr, len(idx_va))

    # --- figure ------------------------------------------------------------
    R = {
        "title": f"Cost-Aware Calibration — {'REAL' if mode == 'real' else 'SIMULATED'} "
                 f"signals — CIFAR-10H (N={n})",
        "human_entropy": he, "jsd_norm": jsd_norm, "escalation_signal": escalation,
        "r_jsd": r_jsd, "rho_jsd": rho_jsd, "r_dual": r_dual, "rho_dual": rho_dual,
        "alpha": alpha, "beta": beta,
        "w_mi": (w_mi["alpha"], w_mi["beta"]), "w_lr": (w_lr["alpha"], w_lr["beta"]),
        "sig_val": escalation[idx_va], "y_val": cal["y_val"],
        "p_iso": cal["iso"]["p_val"], "p_platt": cal["platt"]["p_val"],
        "ece_iso": cal["iso"]["ece"], "ece_platt": cal["platt"]["ece"],
        "bin_acc_iso": cal["iso"]["bin_acc"], "bin_acc_platt": cal["platt"]["bin_acc"],
        "p_cal": best["p_val"], "tau": best["tau"], "cal_name": best["name"],
        "cost": cost, "n_val": len(idx_va), "budget": BUDGET, "ablation": ablation,
    }
    out = config.FIG_DIR / f"pipeline_{mode}.png"
    figures.plot_summary(R, out)
    print(f"\n[done] figure -> {out}")


def _print_summary(mode, keys, r_jsd, rho_jsd, r_dual, rho_dual, w_mi, w_lr, best,
                   cost, single_acc, casc, pr, n_val):
    bar = "=" * 70
    print(f"\n{bar}\nPIPELINE SUMMARY -- {mode.upper()} signals\n{bar}")
    print(f"  Ensemble models:    {keys}")
    print(f"  JSD:   r={r_jsd:.4f}  rho={rho_jsd:.4f}")
    print(f"  Dual:  r={r_dual:.4f}  rho={rho_dual:.4f}  (dr={r_dual - r_jsd:+.4f})")
    print(f"  alpha/beta (MI):       a={w_mi['alpha']:.4f}  b={w_mi['beta']:.4f}")
    print(f"  alpha/beta (logistic): a={w_lr['alpha']:.4f}  b={w_lr['beta']:.4f}")
    print(f"  Calibrator:     {best['name']}  ECE={best['ece']:.4f}  "
          f"prec={best['prec']:.3f}  rec={best['rec']:.3f}  tau={best['tau']:.3f}")
    print(f"  Pre-router:     skip={pr['skip_rate']*100:.1f}%  "
          f"false-easy={pr['false_easy_rate']*100:.1f}%")
    print(f"  Cost stages:    skip={cost['stage_1']}  consensus={cost['stage_2']}  "
          f"frontier={cost['stage_3']} ({cost['frontier_pct']*100:.1f}%)")
    print(f"  Cost vs frontier-only: {cost['cost_ratio']:.1f}x cheaper")
    print(f"  Single-VLM acc: {single_acc:.3f}   Cascade acc: {casc['cascade_acc']:.3f}")
    print(bar)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--simulate", action="store_const", const="sim", dest="mode")
    g.add_argument("--real", action="store_const", const="real", dest="mode")
    ap.add_argument("--dataset", default="cifar10h", choices=["cifar10h","chaosnli"])
    args = ap.parse_args()
    run("real" if args.mode == "real" else "sim", args.dataset)


if __name__ == "__main__":
    main()
