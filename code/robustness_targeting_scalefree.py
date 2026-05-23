"""
robustness_targeting_scalefree.py

Robustness check for Figure 5's targeting-policy result. Re-runs the
targeting comparison on a large scale-free network to test whether
the "uniform-as-good-as-targeted" finding survives or whether bridge
targeting shows its theoretical advantage at scale.

This is the "Middle Path": one focused check to
isolate whether Figure 5's result is a small-N artifact.
"""
import os, time, json
import numpy as np
import matplotlib.pyplot as plt
from felt_understanding_abm import Params, run

FIG_DIR = "/home/claude/work/abm/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def mean_action_rate(res):
    half = len(res.action_rate_A) // 2
    return (res.action_rate_A[half:].mean() + res.action_rate_B[half:].mean()) / 2


print("Robustness check: targeting policies on scale-free N=5000")
print("=" * 60)
t0 = time.time()

# Same gamma range as original Figure 5; 5 seeds per combo to give meaningful
# standard errors. Scale-free runs at N=5000 are expensive (~10 s per run),
# so 5 seeds x 3 policies x 5 gamma levels x ~10 s = ~12 minutes total.
gammas = np.linspace(0.0, 0.4, 5)
policies = ["uniform", "central", "bridge"]
N_SEEDS = 5
results = {pol: [] for pol in policies}
results_se = {pol: [] for pol in policies}

for gam in gammas:
    for pol in policies:
        rates = []
        for s in range(N_SEEDS):
            tstart = time.time()
            p = Params(T=100, pi=0.15, gamma=gam, targeting=pol,
                       N_A=2500, N_B=2500, mean_degree=8,
                       network_type="scale_free",
                       seed=s * 31 + 11)
            rates.append(mean_action_rate(run(p)))
            print(f"  gamma={gam:.2f} pol={pol:8s} seed={s} -> {rates[-1]:.3f} "
                  f"({time.time()-tstart:.1f}s, total {time.time()-t0:.1f}s)")
        results[pol].append(float(np.mean(rates)))
        results_se[pol].append(float(np.std(rates, ddof=1) / np.sqrt(N_SEEDS)))
    print(f"  gamma={gam:.2f} summary: " +
          ", ".join(f"{pol}={results[pol][-1]:.3f}\u00b1{results_se[pol][-1]:.4f}"
                    for pol in policies))

print(f"\nTotal time: {time.time()-t0:.1f}s\n")
print("FINAL RESULTS (mean \u00b1 SE):")
for pol in policies:
    print(f"  {pol:8s}: " + ", ".join(
        f"{m:.4f}\u00b1{se:.4f}"
        for m, se in zip(results[pol], results_se[pol])))

# Plot with error bars
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
colors = {"uniform": "tab:gray", "central": "tab:red", "bridge": "tab:purple"}
for pol in policies:
    ax.errorbar(gammas, results[pol], yerr=results_se[pol], fmt="o-",
                label=f"{pol.capitalize()} targeting", color=colors[pol],
                linewidth=2.2, markersize=7, capsize=4, capthick=1.5,
                elinewidth=1.2)
ax.set_xlabel("Adversarial injection rate, \u03b3", fontsize=12)
ax.set_ylabel("Long-run action rate (mean \u00b1 SE across 5 seeds)", fontsize=12)
ax.set_title("Figure 5R. Targeting policies on scale-free network "
             "(N=5000, \u03c0=0.15)", fontsize=13)
ax.grid(True, alpha=0.3, linestyle=":")
ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig5R_targeting_scalefree.png"),
            bbox_inches="tight", dpi=150)
plt.close()

with open(os.path.join(FIG_DIR, "fig5R_data.json"), "w") as f:
    json.dump({
        "gammas": list(float(g) for g in gammas),
        "n_seeds": N_SEEDS,
        "results_mean": {k: list(v) for k, v in results.items()},
        "results_se":   {k: list(v) for k, v in results_se.items()},
    }, f, indent=2)

print(f"\nFigure saved to {FIG_DIR}/fig5R_targeting_scalefree.png")
