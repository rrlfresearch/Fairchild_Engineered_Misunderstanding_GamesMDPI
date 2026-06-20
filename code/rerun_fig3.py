import os, time, numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from felt_understanding_abm import Params, run, analytical_threshold

FIG_DIR = "/home/RF/work/abm/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def mean_action_rate(res):
    half = len(res.action_rate_A) // 2
    return (res.action_rate_A[half:].mean() + res.action_rate_B[half:].mean()) / 2


print("Figure 3: phase diagram (pi x h)...")
t0 = time.time()
pis = np.linspace(0.0, 0.45, 7)
hs = np.linspace(0.5, 0.95, 6)
grid = np.zeros((len(hs), len(pis)))
for i, h_val in enumerate(hs):
    for j, pi_val in enumerate(pis):
        rates = []
        for s in range(2):
            p = Params(T=100, pi=pi_val, h=h_val,
                       N_A=200, N_B=200, mean_degree=8,
                       seed=s * 23 + 5)
            rates.append(mean_action_rate(run(p)))
        grid[i, j] = np.mean(rates)
    print(f"  h={h_val:.2f} done ({time.time()-t0:.1f}s)")

fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
cmap = LinearSegmentedColormap.from_list("coop", ["#922b21", "#e8d5c4", "#1f4e79"])
im = ax.imshow(grid, origin="lower", aspect="auto",
               extent=[pis[0], pis[-1], hs[0], hs[-1]],
               cmap=cmap, vmin=0, vmax=grid.max())
pi_star = analytical_threshold(Params())
ax.axvline(pi_star, color="white", linestyle="--", linewidth=2,
           label=f"Analytical threshold \u03c0* = {pi_star:.2f}")
ax.set_xlabel("Adversarial perturbation, \u03c0", fontsize=12)
ax.set_ylabel("Network homophily, h", fontsize=12)
ax.set_title("", fontsize=13)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Mean action rate (fraction playing a_R)", fontsize=11)
ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_phase_diagram.png"), bbox_inches="tight", dpi=150)
plt.close()
print(f"Figure 3 done ({time.time()-t0:.1f}s)")
