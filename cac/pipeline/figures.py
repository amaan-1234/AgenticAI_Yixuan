"""8-panel summary figure (v4 section 11), driven by a results dict.

Works in both --simulate and --real modes. The ablation panel is shown when an
ablation dict is provided (simulate, or Phase-2.4 real ablation); otherwise it
shows the single ensemble's correlation.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_summary(R: dict, out_path) -> str:
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(R.get("title", "Cost-Aware Calibration — Summary"),
                 fontsize=13, fontweight="bold", y=1.0)

    he = R["human_entropy"]

    # (a) ablation or single-ensemble correlation
    ax = axes[0, 0]
    abl = R.get("ablation")
    if abl:
        names = list(abl.keys())
        x = np.arange(len(names))
        rv = [abl[n]["r"] for n in names]; pv = [abl[n]["rho"] for n in names]
        ax.bar(x - 0.18, rv, 0.35, label="Pearson r", color="#3A7FC1")
        ax.bar(x + 0.18, pv, 0.35, label="Spearman ρ", color="#E07B39")
        ax.set_xticks(x); ax.set_xticklabels([n.split("(")[0].strip() for n in names], fontsize=8)
        ax.set_title("Families × Models Ablation")
        ax.legend(fontsize=8)
    else:
        ax.bar([0, 1], [R["r_jsd"], R["r_dual"]], color=["#3A7FC1", "#E07B39"])
        ax.set_xticks([0, 1]); ax.set_xticklabels(["JSD-only", "Dual"])
        ax.set_title("Ensemble Correlation (Pearson r)")
    ax.set_ylim(0, 1); ax.set_ylabel("Correlation")

    # (b) JSD scatter
    ax = axes[0, 1]
    ax.scatter(he, R["jsd_norm"], c=he, cmap="RdYlGn_r", alpha=0.2, s=4, edgecolors="none")
    ax.set_xlabel("Human Label Entropy"); ax.set_ylabel("Normalised JSD")
    ax.set_title(f"JSD (r={R['r_jsd']:.3f}, ρ={R['rho_jsd']:.3f})")

    # (c) dual signal scatter
    ax = axes[0, 2]
    ax.scatter(he, R["escalation_signal"], c=he, cmap="RdYlGn_r", alpha=0.2, s=4, edgecolors="none")
    ax.set_xlabel("Human Label Entropy")
    ax.set_ylabel(f"Dual (α={R['alpha']:.2f}, β={R['beta']:.2f})")
    ax.set_title(f"Dual Signal (r={R['r_dual']:.3f}, ρ={R['rho_dual']:.3f})")

    # (d) alpha/beta methods
    ax = axes[0, 3]
    methods = ["MI-Derived", "Logistic", "Hardcoded\n(v3)"]
    alphas = [R["w_mi"][0], R["w_lr"][0], 0.6]; betas = [R["w_mi"][1], R["w_lr"][1], 0.4]
    xab = np.arange(3)
    ax.bar(xab - 0.15, alphas, 0.3, label="α (JSD)", color="#3A7FC1")
    ax.bar(xab + 0.15, betas, 0.3, label="β (MTA)", color="#E07B39")
    ax.set_xticks(xab); ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(0, 1); ax.set_title("α/β Tuning Methods"); ax.legend(fontsize=8)

    sig_val = R["sig_val"]; yv = R["y_val"]; order = np.argsort(sig_val)

    # (e) calibration curves
    ax = axes[1, 0]
    ax.scatter(sig_val, yv, alpha=0.03, s=3, color="#888")
    ax.plot(sig_val[order], R["p_iso"][order], color="#E07B39", lw=2,
            label=f"Isotonic (ECE={R['ece_iso']:.4f})")
    ax.plot(sig_val[order], R["p_platt"][order], color="#3A7FC1", lw=2,
            label=f"Platt (ECE={R['ece_platt']:.4f})")
    ax.set_xlabel("Raw Dual Signal"); ax.set_ylabel("P(truly hard)")
    ax.set_title("Calibration Curves"); ax.legend(fontsize=8)

    # (f) reliability diagram
    ax = axes[1, 1]
    bc = np.linspace(0.05, 0.95, 10)
    ax.bar(bc - 0.04, R["bin_acc_platt"], width=0.07, alpha=0.7, color="#3A7FC1", label="Platt")
    ax.bar(bc + 0.04, R["bin_acc_iso"], width=0.07, alpha=0.7, color="#E07B39", label="Isotonic")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("Predicted P(hard)"); ax.set_ylabel("Observed fraction")
    ax.set_title("Reliability Diagram"); ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # (g) escalation sweep
    ax = axes[1, 2]
    taus = np.linspace(0, 1, 300)
    pcal = R["p_cal"]
    precs, recs, costs = [], [], []
    for tau in taus:
        esc = pcal >= tau
        tp = np.sum(esc & (yv == 1)); fp = np.sum(esc & (yv == 0)); fn = np.sum(~esc & (yv == 1))
        precs.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        recs.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        costs.append(esc.mean())
    ax.plot(taus, precs, label="Precision", color="#E07B39", lw=2)
    ax.plot(taus, recs, label="Recall", color="#3A7FC1", lw=2)
    ax.plot(taus, costs, label="Frontier %", color="#B55E99", lw=2, ls="--")
    ax.axvline(R["tau"], color="grey", ls=":", alpha=0.7, label=f"τ={R['tau']:.2f}")
    ax.axhline(R["budget"], color="#B55E99", ls=":", alpha=0.4)
    ax.set_xlabel(f"Threshold τ ({R['cal_name']})"); ax.set_ylabel("Rate")
    ax.set_title("Escalation Sweep"); ax.legend(fontsize=8); ax.set_ylim(0, 1.05)

    # (h) pipeline cost breakdown
    ax = axes[1, 3]
    labels = ["Pre-router\nskip", "Ensemble\nconsensus", "Frontier\nescalation"]
    n_val = R["n_val"]
    sizes = [R["cost"]["stage_1"], R["cost"]["stage_2"], R["cost"]["stage_3"]]
    pcts = [s / n_val * 100 for s in sizes]
    bars = ax.barh(labels, pcts, color=["#55A868", "#3A7FC1", "#E07B39"], alpha=0.85,
                   edgecolor="white")
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("% of images"); ax.set_title("Pipeline Cost Breakdown"); ax.set_xlim(0, 100)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)
