"""
fig_S2_initial_belief.py

Supplementary Figure S2: robustness to the initial-belief assumption.
Re-runs the no-intervention model from three starting beliefs mu(0) in
{0.9, 0.7, 0.5} across the perturbation range and plots the long-run action
rate. Demonstrates that the system is bistable: from a low start the
population settles into the collapsed basin even at pi=0, which is why the
main analyses fix mu(0)=0.9 to study sustainability of the cooperative basin
(Section 4.2) rather than spontaneous emergence.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from felt_understanding_abm import Params, run, analytical_threshold

FIG_DIR = "/home/RF/work/abm/figures"
os.makedirs(FIG_DIR, exist_ok=True)

N_SEEDS = 4
N_SIDE = 250
MEANDEG = 8
T = 200

def long_run_rate(res):
    half = len(res.action_rate_A) // 2
    return (res.action_rate_A[half:].mean() + res.action_rate_B[half:].mean()) / 2

PIS = np.array([0.0, 0.075, 0.15, 0.225, 0.30])
MU0S = [0.9, 0.7, 0.5]
COLORS = {0.9: "tab:blue", 0.7: "tab:purple", 0.5: "tab:red"}
pi_star = analytical_threshold(Params())

results = {m: [] for m in MU0S}
results_se = {m: [] for m in MU0S}
for mu0 in MU0S:
    for pi in PIS:
        vals = []
        for s in range(N_SEEDS):
            pr = Params(T=T, pi=float(pi), initial_mu=mu0,
                        N_A=N_SIDE, N_B=N_SIDE, mean_degree=MEANDEG, seed=s * 23 + 5)
            vals.append(long_run_rate(run(pr)))
        results[mu0].append(float(np.mean(vals)))
        results_se[mu0].append(float(np.std(vals, ddof=1) / np.sqrt(N_SEEDS)))
    print(f"mu0={mu0}: " + "  ".join(f"pi={p:.2f}:{r:.3f}" for p, r in zip(PIS, results[mu0])))

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
for mu0 in MU0S:
    ax.errorbar(PIS, results[mu0], yerr=results_se[mu0], fmt="o-",
                color=COLORS[mu0], linewidth=2.2, markersize=7, capsize=4,
                label=f"\u03bc(0) = {mu0}")
ax.axvline(pi_star, color="black", linestyle="--", linewidth=1, alpha=0.5,
           label=f"\u03c0* = {pi_star:.2f}")
ax.set_xlabel("Adversarial perturbation, \u03c0", fontsize=12)
ax.set_ylabel("Long-run action rate", fontsize=12)
ax.set_title("",
             fontsize=12)
ax.grid(True, alpha=0.3, linestyle=":")
ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "figS2_initial_belief.png"), bbox_inches="tight", dpi=150)
plt.close()

with open(os.path.join(FIG_DIR, "figS2_initial_belief_data.json"), "w") as f:
    json.dump({"pis": [float(p) for p in PIS], "n_seeds": N_SEEDS,
               "results_mean": results, "results_se": results_se}, f, indent=2)
print("saved figS2_initial_belief.png")
