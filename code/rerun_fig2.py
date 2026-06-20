import os, time, numpy as np
import matplotlib.pyplot as plt
from felt_understanding_abm import Params, run, analytical_threshold

FIG_DIR = "/home/RF/work/abm/figures"
os.makedirs(FIG_DIR, exist_ok=True)


def smooth(arr, window=10):
    """Centered rolling mean without edge artifacts."""
    n = len(arr)
    out = np.zeros(n)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = arr[lo:hi].mean()
    return out


def figure_2_trajectories():
    print("Figure 2 (final)...")
    t0 = time.time()
    pi_star = analytical_threshold(Params())
    # Use three pi values in the meaningful range (below threshold)
    # to show the gradient from cooperative to collapsed
    scenarios = [
        ("\u03c0 = 0.05 (well below \u03c0*)", 0.05, "tab:blue"),
        ("\u03c0 = 0.15 (approaching \u03c0*)", 0.15, "tab:purple"),
        (f"\u03c0 = 0.30 (at \u03c0*)", 0.30, "tab:red"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150, sharey=True)
    n_seeds = 3
    for label, pi_val, color in scenarios:
        ar_A = None; ar_B = None
        for s in range(n_seeds):
            p = Params(T=200, pi=pi_val, seed=s * 7 + 100)
            res = run(p)
            if ar_A is None:
                ar_A = res.action_rate_A.copy()
                ar_B = res.action_rate_B.copy()
            else:
                ar_A += res.action_rate_A
                ar_B += res.action_rate_B
        ar_A /= n_seeds; ar_B /= n_seeds
        ar_A_smooth = smooth(ar_A, 10)
        ar_B_smooth = smooth(ar_B, 10)
        t = np.arange(len(ar_A))
        axes[0].plot(t, ar_A_smooth, color=color, linewidth=2.2, label=label)
        axes[1].plot(t, ar_B_smooth, color=color, linewidth=2.2, label=label)

    for ax, title in zip(axes, ["Group A: environmentalists",
                                "Group B: pro-nuclear / technocratic"]):
        ax.set_xlabel("Time step", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, 0.5)
        ax.grid(True, alpha=0.3, linestyle=":")
    axes[0].set_ylabel("Observable felt understanding\n(fraction of group sending authentic signals)",
                       fontsize=11)
    axes[0].legend(loc="upper right", fontsize=10, framealpha=0.95)
    fig.suptitle("",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_trajectories.png"),
                bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  done ({time.time()-t0:.1f}s)")


figure_2_trajectories()
