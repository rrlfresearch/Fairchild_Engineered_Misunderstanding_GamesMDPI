"""
fig6_bridge_dose_response.py

Supplementary figure: dose-response curve for bridge seeding, the most
effective intervention from Figure 4. Tests whether the effectiveness
of bridge seeding scales smoothly with dose or whether there are
diminishing returns / threshold effects.
"""
import os, time
import numpy as np
import matplotlib.pyplot as plt
from felt_understanding_abm import Params, run, analytical_threshold

FIG_DIR = "/home/RF/work/abm/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def mean_action_rate(res):
    half = len(res.action_rate_A) // 2
    return (res.action_rate_A[half:].mean() + res.action_rate_B[half:].mean()) / 2


print("Figure 6: bridge seeding dose-response...")
t0 = time.time()

doses = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40])
pis = [0.15, 0.25, 0.35]   # below, near, above threshold
colors = {0.15: "tab:blue", 0.25: "tab:orange", 0.35: "tab:red"}
labels = {0.15: "\u03c0 = 0.15 (below \u03c0*)",
          0.25: "\u03c0 = 0.25 (approaching \u03c0*)",
          0.35: "\u03c0 = 0.35 (above \u03c0*)"}

results = {pi_val: [] for pi_val in pis}
for pi_val in pis:
    for dose in doses:
        rates = []
        for s in range(3):
            iv = "bridge_seeding" if dose > 0 else "none"
            p = Params(T=150, pi=pi_val,
                       intervention=iv, intervention_dose=dose,
                       N_A=300, N_B=300, mean_degree=8,
                       seed=s * 37 + 19)
            rates.append(mean_action_rate(run(p)))
        results[pi_val].append(np.mean(rates))
    print(f"  pi={pi_val:.2f} done ({time.time()-t0:.1f}s)")

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
for pi_val in pis:
    ax.plot(doses, results[pi_val], "o-", color=colors[pi_val],
            label=labels[pi_val], linewidth=2.2, markersize=7)
ax.set_xlabel("Bridge seeding dose (fraction of cross-group edges seeded)",
              fontsize=12)
ax.set_ylabel("Long-run action rate", fontsize=12)
ax.set_title("",
             fontsize=13)
ax.grid(True, alpha=0.3, linestyle=":")
ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig6_bridge_dose.png"),
            bbox_inches="tight", dpi=150)
plt.close()
print(f"  done ({time.time()-t0:.1f}s)")
