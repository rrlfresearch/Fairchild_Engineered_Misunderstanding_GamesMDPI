"""
fig_S1_sensitivity.py

Supplementary Figure S1: sensitivity of the intervention ranking to the
stylized parameter set. Re-runs the four-intervention comparison at a fixed
20% dose across six (p, c_R, u_R) combinations, each evaluated at its own
analytical threshold pi*, and plots the long-run action rate. Addresses the
reviewer request to show the main intervention result does not depend on the
specific values (p=0.5, c_R=0.35, u_R=1.0).
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from felt_understanding_abm import Params, run, analytical_threshold

FIG_DIR = "/home/RF/work/abm/figures"
os.makedirs(FIG_DIR, exist_ok=True)

N_SEEDS = 3
DOSE = 0.20
N_SIDE = 250
MEANDEG = 8
T = 120
INTERVENTIONS = ["none", "bridge_seeding", "cost_reduction", "channel_hardening"]
LABELS = {"none": "baseline", "bridge_seeding": "bridge seeding",
          "cost_reduction": "cost reduction", "channel_hardening": "channel hardening"}
COLORS = {"none": "gray", "bridge_seeding": "tab:green",
          "cost_reduction": "tab:orange", "channel_hardening": "tab:purple"}


def long_run_rate(res):
    half = len(res.action_rate_A) // 2
    return (res.action_rate_A[half:].mean() + res.action_rate_B[half:].mean()) / 2


def beta_for_p(p_target, total=4.0):
    return p_target * total, (1.0 - p_target) * total


COMBOS = [
    (0.50, 0.35, 1.0, "p=.50\ncR=.35\nuR=1.0"),
    (0.40, 0.35, 1.0, "p=.40"),
    (0.60, 0.35, 1.0, "p=.60"),
    (0.50, 0.25, 1.0, "cR=.25"),
    (0.50, 0.45, 1.0, "cR=.45"),
    (0.50, 0.35, 1.3, "uR=1.3"),
]

results = {iv: [] for iv in INTERVENTIONS}
pi_stars = []
for p_prior, cR, uR, label in COMBOS:
    a, b = beta_for_p(p_prior)
    pi_star = analytical_threshold(Params(beta_alpha=a, beta_beta=b, c_R_min=cR, u_R=uR))
    pi_stars.append(pi_star)
    for iv in INTERVENTIONS:
        vals = []
        for s in range(N_SEEDS):
            pr = Params(T=T, pi=pi_star, beta_alpha=a, beta_beta=b, c_R_min=cR, u_R=uR,
                        intervention=iv, intervention_dose=DOSE if iv != "none" else 0.0,
                        N_A=N_SIDE, N_B=N_SIDE, mean_degree=MEANDEG, seed=s * 17 + 3)
            vals.append(long_run_rate(run(pr)))
        results[iv].append(float(np.mean(vals)))
    print(f"{label.splitlines()[0]:>10}  pi*={pi_star:.2f}  " +
          "  ".join(f"{LABELS[iv].split()[0]}={results[iv][-1]:.3f}" for iv in INTERVENTIONS))

x = np.arange(len(COMBOS))
w = 0.2
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
for k, iv in enumerate(INTERVENTIONS):
    ax.bar(x + (k - 1.5) * w, results[iv], w, label=LABELS[iv], color=COLORS[iv])
ax.set_xticks(x)
ax.set_xticklabels([c[3] for c in COMBOS], fontsize=9)
ax.set_ylabel("Long-run action rate (at each combination's \u03c0*)", fontsize=11)
ax.set_xlabel("Parameter combination (deviation from stylized baseline)", fontsize=11)
ax.set_title("", fontsize=12)
ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
ax.grid(True, axis="y", alpha=0.3, linestyle=":")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figS1_sensitivity.png"), bbox_inches="tight", dpi=150)
plt.close()

with open(os.path.join(FIG_DIR, "figS1_sensitivity_data.json"), "w") as f:
    json.dump({"combos": [c[:3] for c in COMBOS], "pi_stars": pi_stars,
               "results": results}, f, indent=2)
print("saved figS1_sensitivity.png")
