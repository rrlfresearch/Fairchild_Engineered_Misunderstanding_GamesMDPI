"""
Robustness to the initial-belief assumption (Reviewer 2, minor).

The manuscript starts all agents at mu(0)=0.9 to place the system in the
cooperative basin and then test sustainability under perturbation. Reviewer 2
worries 0.9 is unrealistically high. This script re-runs with mu(0)=0.5 (the
prior / a less idealized start) and mu(0)=0.7, across perturbation regimes,
with NO intervention, and reports the long-run action rate so we can see
whether the steady state is governed by pi rather than by the start point.
"""
import numpy as np
from felt_understanding_abm import Params, run, analytical_threshold

N_SEEDS = 4
N_SIDE = 250
MEANDEG = 8
T = 200

def long_run_rate(res):
    half = len(res.action_rate_A) // 2
    return (res.action_rate_A[half:].mean() + res.action_rate_B[half:].mean()) / 2

pi_star = analytical_threshold(Params())
PIS = [0.0, 0.15, 0.30]          # below, mid, at threshold
MU0S = [0.9, 0.7, 0.5]

print(f"Initial-belief robustness (pi* = {pi_star:.2f}); long-run action rate, mean over {N_SEEDS} seeds")
print("=" * 70)
header = "  pi    " + "".join(f"mu0={m:<4}" for m in MU0S) + "  max gap"
print(header)
maxgap_overall = 0.0
for pi in PIS:
    row = {}
    for mu0 in MU0S:
        vals = []
        for s in range(N_SEEDS):
            pr = Params(T=T, pi=pi, initial_mu=mu0,
                        N_A=N_SIDE, N_B=N_SIDE, mean_degree=MEANDEG,
                        seed=s * 23 + 5)
            vals.append(long_run_rate(run(pr)))
        row[mu0] = float(np.mean(vals))
    gap = max(row.values()) - min(row.values())
    maxgap_overall = max(maxgap_overall, gap)
    cells = "".join(f"{row[m]:<8.3f}" for m in MU0S)
    print(f"  {pi:.2f}  {cells}  {gap:.3f}")

print("=" * 70)
print(f"Largest steady-state gap across initial beliefs: {maxgap_overall:.3f}")
print("Interpretation: if gaps are small, the long-run state is governed by pi,")
print("not by the starting belief, so mu(0)=0.9 is an innocuous convenience.")
