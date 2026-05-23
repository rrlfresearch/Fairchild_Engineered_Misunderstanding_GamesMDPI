import os, time, numpy as np
import matplotlib.pyplot as plt
from felt_understanding_abm import Params, run

FIG_DIR = "/home/c/work/abm/figures"


def mean_action_rate(res):
    half = len(res.action_rate_A) // 2
    return (res.action_rate_A[half:].mean() + res.action_rate_B[half:].mean()) / 2


print("Figure 5: targeting policies...")
t0 = time.time()
gammas = np.linspace(0.0, 0.4, 6)
policies = ["uniform", "central", "bridge"]
colors = {"uniform": "tab:gray", "central": "tab:red", "bridge": "tab:purple"}
N_SEEDS = 5
results = {pol: [] for pol in policies}
results_se = {pol: [] for pol in policies}  # standard error across seeds
for gam in gammas:
    for pol in policies:
        rates = []
        for s in range(N_SEEDS):
            p = Params(T=150, pi=0.15, gamma=gam, targeting=pol,
                       N_A=150, N_B=150, mean_degree=6,
                       seed=s * 31 + 11)
            rates.append(mean_action_rate(run(p)))
        results[pol].append(float(np.mean(rates)))
        results_se[pol].append(float(np.std(rates, ddof=1) / np.sqrt(N_SEEDS)))
    print(f"  gamma={gam:.2f} done ({time.time()-t0:.1f}s)")

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
for pol in policies:
    ax.errorbar(gammas, results[pol], yerr=results_se[pol], fmt="o-",
                label=f"{pol.capitalize()} targeting", color=colors[pol],
                linewidth=2.2, markersize=7, capsize=4, capthick=1.5,
                elinewidth=1.2)
ax.set_xlabel("Adversarial injection rate, \u03b3", fontsize=12)
ax.set_ylabel("Long-run action rate (mean \u00b1 SE across 5 seeds)", fontsize=12)
ax.set_title("Figure 5. Adversary efficiency by targeting policy (\u03c0 = 0.15 baseline)",
             fontsize=13)
ax.grid(True, alpha=0.3, linestyle=":")
ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig5_targeting.png"), bbox_inches="tight", dpi=150)
plt.close()
print(f"Figure 5 done ({time.time()-t0:.1f}s)")

import json
with open(os.path.join(FIG_DIR, "fig5_data.json"), "w") as f:
    json.dump({
        "gammas": list(float(g) for g in gammas),
        "n_seeds": N_SEEDS,
        "results_mean": {k: list(v) for k, v in results.items()},
        "results_se":   {k: list(v) for k, v in results_se.items()},
    }, f, indent=2)
