import os, time, numpy as np
import matplotlib.pyplot as plt
from felt_understanding_abm import Params, run, layer_a_threshold

FIG_DIR = "/home/c/work/abm/figures"


def mean_action_rate(res):
    half = len(res.action_rate_A) // 2
    return (res.action_rate_A[half:].mean() + res.action_rate_B[half:].mean()) / 2


print("Figure 4: interventions...")
t0 = time.time()
pis = np.linspace(0.05, 0.40, 6)
interventions = ["none", "bridge_seeding", "cost_reduction", "channel_hardening"]
labels = {
    "none": "No intervention (baseline)",
    "bridge_seeding": "Bridge seeding (recognition advocates)",
    "cost_reduction": "Cost reduction (dialogue platforms)",
    "channel_hardening": "Channel hardening (platform moderation)",
}
colors = {"none": "gray", "bridge_seeding": "tab:green",
          "cost_reduction": "tab:orange", "channel_hardening": "tab:purple"}
dose = 0.20
results = {iv: [] for iv in interventions}
for pi_val in pis:
    for iv in interventions:
        rates = []
        for s in range(2):
            p = Params(T=150, pi=pi_val, intervention=iv,
                       intervention_dose=dose if iv != "none" else 0.0,
                       N_A=300, N_B=300, mean_degree=8,
                       seed=s * 29 + 7)
            rates.append(mean_action_rate(run(p)))
        results[iv].append(np.mean(rates))
    print(f"  pi={pi_val:.2f} done ({time.time()-t0:.1f}s)")

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
for iv in interventions:
    ax.plot(pis, results[iv], "o-", label=labels[iv],
            color=colors[iv], linewidth=2, markersize=7)
pi_star = layer_a_threshold(Params())
ax.axvline(pi_star, color="black", linestyle="--", linewidth=1, alpha=0.5,
           label=f"\u03c0* = {pi_star:.2f}")
ax.set_xlabel("Adversarial perturbation, \u03c0", fontsize=12)
ax.set_ylabel("Long-run action rate", fontsize=12)
ax.set_title(f"Figure 4. Intervention effectiveness (dose = {dose:.0%} of agents/edges)",
             fontsize=13)
ax.grid(True, alpha=0.3, linestyle=":")
ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_interventions.png"), bbox_inches="tight", dpi=150)
plt.close()
print(f"Figure 4 done ({time.time()-t0:.1f}s)")

# Save results for the report
import json
with open(os.path.join(FIG_DIR, "fig4_data.json"), "w") as f:
    json.dump({k: list(v) for k, v in results.items()}, f, indent=2)
