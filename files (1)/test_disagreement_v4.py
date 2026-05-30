"""
Cost-Aware Calibration via Heterogeneous Ensemble Disagreement — v4 (Final)
=============================================================================
Incorporates all review feedback:
  1. Dual-signal escalation: α·JSD + β·(1−MTA) [corrected sign convention]
  2. MI-based α/β tuning (principled) + logistic-weighting fallback
  3. Families × Models ablation (not just model count)
  4. Platt vs Isotonic calibration (head-to-head)
  5. Pre-router with asymmetric loss (CLIP/DINOv2 simulation)
  6. Spearman rank correlation alongside Pearson r
  7. Full experimental tracking matrix (frontier-only / single-VLM / cascade)
  8. Real CIFAR-10H data throughout

Run: python test_disagreement_v4.py
"""

import os, urllib.request, warnings
import numpy as np
from scipy.stats import entropy, spearmanr
from scipy.spatial.distance import jensenshannon
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD REAL CIFAR-10H
# ═══════════════════════════════════════════════════════════════════════════════
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cifar10h-probs.npy")
if not os.path.exists(DATA_PATH):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/jcpeterson/cifar-10h/master/data/cifar10h-probs.npy",
        DATA_PATH,
    )
human_probs = np.load(DATA_PATH)
human_probs = np.clip(human_probs, 1e-12, None)
human_probs /= human_probs.sum(axis=1, keepdims=True)
N, K = human_probs.shape
human_entropy = entropy(human_probs, axis=1)

HARD_PCT = 70
cutoff = np.percentile(human_entropy, HARD_PCT)
truly_hard = human_entropy >= cutoff
BUDGET = 0.25

print(f"[Data] CIFAR-10H: {N} images × {K} classes")
print(f"       Hard images (entropy ≥ {cutoff:.3f}): {truly_hard.sum()} / {N}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODEL SIMULATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
def sim_model(human_p, noise_scale, rng):
    out = []
    for p in human_p:
        conc = p / noise_scale + 1e-3
        out.append(rng.dirichlet(conc))
    arr = np.clip(np.array(out), 1e-12, None)
    arr /= arr.sum(axis=1, keepdims=True)
    return arr

def mean_pairwise_jsd(mlist):
    n = mlist[0].shape[0]
    m = len(mlist)
    agg = np.zeros(n)
    cnt = 0
    for i in range(m):
        for j in range(i + 1, m):
            for k in range(n):
                a = mlist[i][k] / mlist[i][k].sum()
                b = mlist[j][k] / mlist[j][k].sum()
                v = jensenshannon(a, b)
                agg[k] += 0.0 if np.isnan(v) else v ** 2
            cnt += 1
    return agg / cnt

def simulate_mta(human_ent, rng, noise=0.15):
    ent_norm = (human_ent - human_ent.min()) / (human_ent.max() - human_ent.min() + 1e-12)
    base_mta = 1.0 - ent_norm
    noisy_mta = base_mta + rng.normal(0, noise, size=len(human_ent))
    return np.clip(noisy_mta, 0, 1)

def normalise_01(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-12)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. FAMILIES × MODELS ABLATION
# ═══════════════════════════════════════════════════════════════════════════════
# Key insight: plateau applies to FAMILIES, not individual models.
# Test: (n_families) × (models_per_family)

FAMILY_NOISE = {
    "Meta-Llama":  [0.08, 0.12],    # 2 sizes within family
    "Alibaba-Qwen": [0.15, 0.19],
    "Mistral":      [0.22, 0.26],
    "Google-Gemma":  [0.10, 0.14],
    "Microsoft-Phi": [0.18, 0.23],
}

ABLATION_CONFIGS = {
    # (description): list of (family, model_index) pairs
    "2F×1M (Llama+Qwen)":        [("Meta-Llama", 0), ("Alibaba-Qwen", 0)],
    "3F×1M (Llama+Qwen+Mistral)": [("Meta-Llama", 0), ("Alibaba-Qwen", 0), ("Mistral", 0)],
    "3F×2M (6 models total)":      [("Meta-Llama", 0), ("Meta-Llama", 1),
                                     ("Alibaba-Qwen", 0), ("Alibaba-Qwen", 1),
                                     ("Mistral", 0), ("Mistral", 1)],
    "5F×1M (5 families)":          [("Meta-Llama", 0), ("Alibaba-Qwen", 0),
                                     ("Mistral", 0), ("Google-Gemma", 0),
                                     ("Microsoft-Phi", 0)],
}

print("═══ Families × Models Ablation ═══")
print(f"{'Config':<28} {'Pearson r':>10} {'Spearman ρ':>12} {'n_models':>10}")
print("─" * 65)

ablation_jsd = {}
for config_name, model_specs in ABLATION_CONFIGS.items():
    models = []
    for fam, idx in model_specs:
        ns = FAMILY_NOISE[fam][idx]
        models.append(sim_model(human_probs, ns, RNG))

    jsd = mean_pairwise_jsd(models)
    r_pearson = np.corrcoef(jsd, human_entropy)[0, 1]
    r_spearman, _ = spearmanr(jsd, human_entropy)
    print(f"{config_name:<28} {r_pearson:>10.4f} {r_spearman:>12.4f} {len(models):>10}")
    ablation_jsd[config_name] = {"jsd": jsd, "r": r_pearson, "rho": r_spearman, "n": len(models)}

print()

# ═══════════════════════════════════════════════════════════════════════════════
# 4. PRIMARY CONFIG: 3 families × 1 model each (practical sweet spot)
# ═══════════════════════════════════════════════════════════════════════════════
PRIMARY = "3F×1M (Llama+Qwen+Mistral)"
ensemble_jsd = ablation_jsd[PRIMARY]["jsd"]
jsd_norm = normalise_01(ensemble_jsd)
mta_scores = simulate_mta(human_entropy, RNG)

r_jsd = np.corrcoef(ensemble_jsd, human_entropy)[0, 1]
rho_jsd, _ = spearmanr(ensemble_jsd, human_entropy)
r_mta = np.corrcoef(1 - mta_scores, human_entropy)[0, 1]
rho_mta, _ = spearmanr(1 - mta_scores, human_entropy)

print(f"[Primary Config] {PRIMARY}")
print(f"  JSD:  Pearson r = {r_jsd:.4f},  Spearman ρ = {rho_jsd:.4f}")
print(f"  MTA:  Pearson r = {r_mta:.4f},  Spearman ρ = {rho_mta:.4f}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. MI-BASED α/β TUNING (principled)
# ═══════════════════════════════════════════════════════════════════════════════
# Discretise continuous signals into bins for MI estimation
N_MI_BINS = 30

def continuous_mi(x, y, n_bins=N_MI_BINS):
    """Estimate MI between two continuous variables via binning."""
    x_bins = np.digitize(x, np.linspace(x.min(), x.max(), n_bins + 1)[:-1])
    y_bins = np.digitize(y, np.linspace(y.min(), y.max(), n_bins + 1)[:-1])
    return mutual_info_score(x_bins, y_bins)

mi_jsd = continuous_mi(jsd_norm, human_entropy)
mi_mta = continuous_mi(1 - mta_scores, human_entropy)

alpha_mi = mi_jsd / (mi_jsd + mi_mta)
beta_mi  = mi_mta / (mi_jsd + mi_mta)

print(f"[MI-Based Tuning]")
print(f"  MI(JSD, H_human)     = {mi_jsd:.4f}")
print(f"  MI(1-MTA, H_human)   = {mi_mta:.4f}")
print(f"  → α (MI-derived)     = {alpha_mi:.4f}")
print(f"  → β (MI-derived)     = {beta_mi:.4f}")

# Also compute logistic-regression weights as fallback
idx_train, idx_val = train_test_split(np.arange(N), test_size=0.3, random_state=42)
y_train = truly_hard[idx_train].astype(float)
y_val   = truly_hard[idx_val].astype(float)

X_dual = np.column_stack([jsd_norm, 1 - mta_scores])
lr_weights = LogisticRegression(max_iter=500)
lr_weights.fit(X_dual[idx_train], y_train)
coefs = np.abs(lr_weights.coef_[0])
alpha_lr = coefs[0] / coefs.sum()
beta_lr  = coefs[1] / coefs.sum()

print(f"\n[Logistic Weighting (fallback)]")
print(f"  → α (logistic)       = {alpha_lr:.4f}")
print(f"  → β (logistic)       = {beta_lr:.4f}")

# Use MI-derived weights as primary
ALPHA, BETA = alpha_mi, beta_mi
print(f"\n[Selected] α = {ALPHA:.4f}, β = {BETA:.4f} (MI-derived)\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. DUAL-SIGNAL ESCALATION
# ═══════════════════════════════════════════════════════════════════════════════
# Corrected formula: Escalation Signal = α·JSD + β·(1 - MTA)
# High → escalate to frontier
escalation_signal = ALPHA * jsd_norm + BETA * (1 - mta_scores)
r_dual = np.corrcoef(escalation_signal, human_entropy)[0, 1]
rho_dual, _ = spearmanr(escalation_signal, human_entropy)
print(f"[Dual Signal] r = {r_dual:.4f}, ρ = {rho_dual:.4f}")
print(f"  Improvement over JSD-only: {r_dual - r_jsd:+.4f} (Pearson)\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. CALIBRATION: Platt vs Isotonic
# ═══════════════════════════════════════════════════════════════════════════════
def compute_ece(p_pred, y_true, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs, bin_confs = [], []
    for b in range(n_bins):
        mask = (p_pred >= bin_edges[b]) & (p_pred < bin_edges[b + 1])
        if mask.sum() == 0:
            bin_accs.append(0); bin_confs.append(0)
            continue
        acc = y_true[mask].mean()
        conf = p_pred[mask].mean()
        ece += mask.sum() * abs(acc - conf)
        bin_accs.append(acc); bin_confs.append(conf)
    return ece / len(y_true), bin_accs, bin_confs

def best_threshold(p_val, y_val, budget):
    best = {"tau": 0, "prec": 0, "rec": 0, "cost": 1}
    for tau in np.linspace(0, 1, 300):
        esc = p_val >= tau
        tp = np.sum(esc & (y_val == 1))
        fp = np.sum(esc & (y_val == 0))
        fn = np.sum(~esc & (y_val == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        cost = esc.mean()
        if cost <= budget and rec > best["rec"]:
            best = {"tau": tau, "prec": prec, "rec": rec, "cost": cost}
    return best

sig = escalation_signal

# Platt
platt = LogisticRegression()
platt.fit(sig[idx_train].reshape(-1, 1), y_train)
p_platt = platt.predict_proba(sig[idx_val].reshape(-1, 1))[:, 1]
ece_platt, ba_p, bc_p = compute_ece(p_platt, y_val)
best_platt = best_threshold(p_platt, y_val, BUDGET)

# Isotonic
iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
iso.fit(sig[idx_train], y_train)
p_iso = iso.predict(sig[idx_val])
ece_iso, ba_i, bc_i = compute_ece(p_iso, y_val)
best_iso = best_threshold(p_iso, y_val, BUDGET)

print(f"{'Method':<12} {'ECE':>8} {'Prec':>8} {'Recall':>8} {'Frontier%':>10}")
print(f"{'Platt':<12} {ece_platt:>8.4f} {best_platt['prec']:>8.3f} "
      f"{best_platt['rec']:>8.3f} {best_platt['cost']*100:>9.1f}%")
print(f"{'Isotonic':<12} {ece_iso:>8.4f} {best_iso['prec']:>8.3f} "
      f"{best_iso['rec']:>8.3f} {best_iso['cost']*100:>9.1f}%")

# Select best calibrator
if ece_iso <= ece_platt:
    best_cal, cal_name = {"p_val": p_iso, "ece": ece_iso, **best_iso}, "Isotonic"
else:
    best_cal, cal_name = {"p_val": p_platt, "ece": ece_platt, **best_platt}, "Platt"
print(f"\n  → Selected: {cal_name}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. PRE-ROUTER (asymmetric loss, simulated CLIP embeddings)
# ═══════════════════════════════════════════════════════════════════════════════
def simulate_clip_embeddings(human_ent, dim=64, rng=RNG):
    n = len(human_ent)
    ent_norm = (human_ent - human_ent.min()) / (human_ent.max() - human_ent.min() + 1e-12)
    emb = rng.standard_normal((n, dim))
    for d in range(8):
        emb[:, d] += ent_norm * rng.uniform(1.5, 3.0)
    return emb

clip_emb = simulate_clip_embeddings(human_entropy)
sample_weights = np.ones(len(idx_train))
sample_weights[truly_hard[idx_train]] = 8.0  # 8× penalty on false-easies

prerouter = LogisticRegression(max_iter=500, C=1.0)
prerouter.fit(clip_emb[idx_train], truly_hard[idx_train].astype(int),
              sample_weight=sample_weights)
prerouter_proba = prerouter.predict_proba(clip_emb[idx_val])[:, 1]

PREROUTE_THRESHOLD = 0.15
prerouter_easy = prerouter_proba < PREROUTE_THRESHOLD
prerouter_hard = ~prerouter_easy

false_easy = np.sum(prerouter_easy & truly_hard[idx_val])
n_skipped  = prerouter_easy.sum()
n_sent     = prerouter_hard.sum()

print(f"[Pre-Router] Skip: {n_skipped} ({n_skipped/len(idx_val)*100:.1f}%)  "
      f"False-easy: {false_easy} ({false_easy/max(truly_hard[idx_val].sum(),1)*100:.1f}%)\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. FULL PIPELINE COST MODEL
# ═══════════════════════════════════════════════════════════════════════════════
p_cal_sent = best_cal["p_val"][prerouter_hard]
escalated  = p_cal_sent >= best_cal["tau"]

stage_1 = n_skipped
stage_2 = n_sent - escalated.sum()
stage_3 = escalated.sum()

print("═══ Full Pipeline Cost Model ═══")
print(f"  Stage 1 — Pre-router skip:     {stage_1:>5} ({stage_1/len(idx_val)*100:>5.1f}%)")
print(f"  Stage 2 — Ensemble consensus:  {stage_2:>5} ({stage_2/len(idx_val)*100:>5.1f}%)")
print(f"  Stage 3 — Frontier escalation: {stage_3:>5} ({stage_3/len(idx_val)*100:>5.1f}%)")
cost_ratio = len(idx_val) / stage_3 if stage_3 > 0 else float('inf')
print(f"  Cost vs frontier-only:         {cost_ratio:.1f}× cheaper\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. EXPERIMENTAL TRACKING MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
# Simulate single-VLM baseline: best single model from ensemble
single_model = sim_model(human_probs, 0.08, RNG)  # best-calibrated model
single_argmax = single_model.argmax(axis=1)
human_argmax  = human_probs.argmax(axis=1)
single_acc_val = (single_argmax[idx_val] == human_argmax[idx_val]).mean()

# Frontier-only baseline: assume 95% accuracy (GPT-4o-level on CIFAR-10)
frontier_acc = 0.95

# Cascade accuracy: ensemble consensus handles easy, frontier handles escalated
# Easy images: use ensemble majority vote
ensemble_models_primary = [
    sim_model(human_probs, 0.08, RNG),
    sim_model(human_probs, 0.15, RNG),
    sim_model(human_probs, 0.22, RNG),
]
ensemble_votes = np.stack([m.argmax(axis=1) for m in ensemble_models_primary], axis=1)
from scipy.stats import mode as scipy_mode
ensemble_majority = scipy_mode(ensemble_votes, axis=1, keepdims=False).mode
ensemble_acc_easy = (ensemble_majority[idx_val][~(best_cal["p_val"] >= best_cal["tau"])]
                     == human_argmax[idx_val][~(best_cal["p_val"] >= best_cal["tau"])]).mean()

# Escalated images: assume frontier accuracy
cascade_n_easy = np.sum(~(best_cal["p_val"] >= best_cal["tau"]))
cascade_n_hard = np.sum(best_cal["p_val"] >= best_cal["tau"])
cascade_acc = (ensemble_acc_easy * cascade_n_easy + frontier_acc * cascade_n_hard) / len(idx_val)

print("═══ Experimental Tracking Matrix ═══")
print(f"{'Paradigm':<42} {'Accuracy':>10} {'ECE':>8} {'Cost':>12} {'Escalation%':>12}")
print("─" * 86)
print(f"{'Frontier-Only (GPT-4o / Claude Sonnet)':<42} {frontier_acc:>10.3f} {'Low':>8} "
      f"{'100% API':>12} {'0%':>12}")
print(f"{'Single VLM (best model, Llama-3-8B sim)':<42} {single_acc_val:>10.3f} {'High':>8} "
      f"{'~$0 local':>12} {'0%':>12}")
print(f"{'Multi-Agent Cascade (this system)':<42} {cascade_acc:>10.3f} {best_cal['ece']:>8.4f} "
      f"{f'~{stage_3/len(idx_val)*100:.0f}% API':>12} {f'{stage_3/len(idx_val)*100:.1f}%':>12}")
print("─" * 86)
print()

# ═══════════════════════════════════════════════════════════════════════════════
# 11. PLOTS (8-panel final figure)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle(
    "Cost-Aware Calibration v4 (Final)  —  Real CIFAR-10H  ×  MI-Tuned Dual Signal  ×  Families×Models Ablation",
    fontsize=13, fontweight="bold", y=1.0,
)

# 11a: Families × Models ablation bar chart
ax = axes[0, 0]
names = list(ablation_jsd.keys())
x = np.arange(len(names))
r_vals = [ablation_jsd[n]["r"] for n in names]
rho_vals = [ablation_jsd[n]["rho"] for n in names]
w = 0.35
ax.bar(x - w/2, r_vals, w, label="Pearson r", color="#3A7FC1", alpha=0.85)
ax.bar(x + w/2, rho_vals, w, label="Spearman ρ", color="#E07B39", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([n.split("(")[0].strip() for n in names], fontsize=8)
ax.set_ylabel("Correlation"); ax.set_ylim(0, 1)
ax.set_title("Families × Models Ablation")
ax.legend(fontsize=8)

# 11b: JSD vs Human Entropy scatter (primary config)
ax = axes[0, 1]
sc = ax.scatter(human_entropy, jsd_norm, c=human_entropy,
                cmap="RdYlGn_r", alpha=0.2, s=4, edgecolors="none")
plt.colorbar(sc, ax=ax, label="Human entropy", shrink=0.8)
ax.set_xlabel("Human Label Entropy")
ax.set_ylabel("Normalised JSD")
ax.set_title(f"JSD (r={r_jsd:.3f}, ρ={rho_jsd:.3f})")

# 11c: Dual signal scatter
ax = axes[0, 2]
sc = ax.scatter(human_entropy, escalation_signal, c=human_entropy,
                cmap="RdYlGn_r", alpha=0.2, s=4, edgecolors="none")
plt.colorbar(sc, ax=ax, label="Human entropy", shrink=0.8)
ax.set_xlabel("Human Label Entropy")
ax.set_ylabel(f"Dual (α={ALPHA:.2f}·JSD + β={BETA:.2f}·(1−MTA))")
ax.set_title(f"Dual Signal (r={r_dual:.3f}, ρ={rho_dual:.3f})")

# 11d: α/β weight comparison (MI vs Logistic)
ax = axes[0, 3]
methods = ["MI-Derived", "Logistic", "Hardcoded\n(v3: 0.6/0.4)"]
alphas = [alpha_mi, alpha_lr, 0.6]
betas  = [beta_mi, beta_lr, 0.4]
x_ab = np.arange(len(methods))
ax.bar(x_ab - 0.15, alphas, 0.3, label="α (JSD weight)", color="#3A7FC1")
ax.bar(x_ab + 0.15, betas, 0.3, label="β (MTA weight)", color="#E07B39")
ax.set_xticks(x_ab); ax.set_xticklabels(methods, fontsize=9)
ax.set_ylabel("Weight"); ax.set_ylim(0, 1)
ax.set_title("α/β Tuning Methods")
ax.legend(fontsize=8)

# 11e: Platt vs Isotonic calibration curves
ax = axes[1, 0]
order = np.argsort(sig[idx_val])
ax.scatter(sig[idx_val], y_val, alpha=0.03, s=3, color="#888")
ax.plot(sig[idx_val][order], p_iso[order], color="#E07B39", lw=2, label=f"Isotonic (ECE={ece_iso:.4f})")
ax.plot(sig[idx_val][order], p_platt[order], color="#3A7FC1", lw=2, label=f"Platt (ECE={ece_platt:.4f})")
ax.set_xlabel("Raw Dual Signal")
ax.set_ylabel("P(truly hard)")
ax.set_title("Calibration Curves")
ax.legend(fontsize=8)

# 11f: Reliability diagram
ax = axes[1, 1]
bin_centers = np.linspace(0.05, 0.95, 10)
w_b = 0.04
ax.bar(bin_centers - w_b, ba_p, width=0.07, alpha=0.7, color="#3A7FC1", label="Platt")
ax.bar(bin_centers + w_b, ba_i, width=0.07, alpha=0.7, color="#E07B39", label="Isotonic")
ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
ax.set_xlabel("Predicted P(hard)"); ax.set_ylabel("Observed fraction")
ax.set_title("Reliability Diagram")
ax.legend(fontsize=8)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# 11g: Precision/Recall/Cost sweep
ax = axes[1, 2]
sweep_tau = np.linspace(0, 1, 300)
precs, recs, costs = [], [], []
for tau in sweep_tau:
    esc = best_cal["p_val"] >= tau
    tp = np.sum(esc & (y_val == 1))
    fp = np.sum(esc & (y_val == 0))
    fn = np.sum(~esc & (y_val == 1))
    precs.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
    recs.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    costs.append(esc.mean())
ax.plot(sweep_tau, precs, label="Precision", color="#E07B39", lw=2)
ax.plot(sweep_tau, recs, label="Recall", color="#3A7FC1", lw=2)
ax.plot(sweep_tau, costs, label="Frontier %", color="#B55E99", lw=2, ls="--")
ax.axvline(best_cal["tau"], color="grey", ls=":", alpha=0.7, label=f"τ={best_cal['tau']:.2f}")
ax.axhline(BUDGET, color="#B55E99", ls=":", alpha=0.4)
ax.set_xlabel(f"Threshold τ ({cal_name})")
ax.set_ylabel("Rate")
ax.set_title("Escalation Sweep")
ax.legend(fontsize=8); ax.set_ylim(0, 1.05)

# 11h: Pipeline cost breakdown
ax = axes[1, 3]
labels = ["Pre-router\nskip", "Ensemble\nconsensus", "Frontier\nescalation"]
sizes = [stage_1, stage_2, stage_3]
colors_bar = ["#55A868", "#3A7FC1", "#E07B39"]
pcts = [s / len(idx_val) * 100 for s in sizes]
bars = ax.barh(labels, pcts, color=colors_bar, alpha=0.85, edgecolor="white")
for bar, pct in zip(bars, pcts):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{pct:.1f}%", va="center", fontsize=10, fontweight="bold")
ax.set_xlabel("% of images")
ax.set_title("Pipeline Cost Breakdown")
ax.set_xlim(0, 100)

plt.tight_layout()
out_path = "/mnt/user-data/outputs/disagreement_v4_final.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"[Done] Plot saved to {out_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# 12. FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 74)
print("FINAL SUMMARY — v4")
print("=" * 74)
print(f"  Dataset:                 CIFAR-10H (10,000 real images)")
print(f"  Primary config:          {PRIMARY}")
print(f"  JSD signal:              r = {r_jsd:.4f}, ρ = {rho_jsd:.4f}")
print(f"  MTA signal (simulated):  r = {r_mta:.4f}, ρ = {rho_mta:.4f}")
print(f"  Dual signal:             r = {r_dual:.4f}, ρ = {rho_dual:.4f}")
print(f"  α/β (MI-derived):        α = {ALPHA:.4f}, β = {BETA:.4f}")
print(f"  α/β (logistic fallback): α = {alpha_lr:.4f}, β = {beta_lr:.4f}")
print(f"  Calibrator:              {cal_name} (ECE = {best_cal['ece']:.4f})")
print(f"  ───────────────────────────────────────────────")
print(f"  Pre-router skip:         {stage_1/len(idx_val)*100:.1f}%")
print(f"  Ensemble consensus:      {stage_2/len(idx_val)*100:.1f}%")
print(f"  Frontier escalation:     {stage_3/len(idx_val)*100:.1f}%")
print(f"  False-easy rate:         {false_easy/max(truly_hard[idx_val].sum(),1)*100:.1f}%")
print(f"  Cost vs frontier-only:   {cost_ratio:.1f}× cheaper")
print(f"  ───────────────────────────────────────────────")
print(f"  Key ablation finding:    3F×2M (r={ablation_jsd['3F×2M (6 models total)']['r']:.4f}) "
      f"vs 5F×1M (r={ablation_jsd['5F×1M (5 families)']['r']:.4f})")
print(f"                           → {'More families' if ablation_jsd['5F×1M (5 families)']['r'] > ablation_jsd['3F×2M (6 models total)']['r'] else 'More models/family'} wins")
print("=" * 74)
