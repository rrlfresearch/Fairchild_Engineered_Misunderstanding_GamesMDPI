"""
Sensitivity analysis for Reviewer 2 (major #2): does the intervention
ranking (bridge seeding > cost reduction > channel hardening > baseline)
depend on the stylized parameter set (p=0.5, c_R=0.35, u_R=1.0)?

For each (p, c_R, u_R) combination we evaluate all four interventions at a
fixed 20% dose, at that combination's own analytical threshold pi* and at a
below-threshold point (pi* - 0.10), and check whether the ordering holds.
"""
import numpy as np
from felt_understanding_abm import Params, run, analytical_threshold

N_SEEDS = 3
DOSE = 0.20
N_SIDE = 250
MEANDEG = 8
T = 120

INTERVENTIONS = ["none", "bridge_seeding", "cost_reduction", "channel_hardening"]
SHORT = {"none": "baseline", "bridge_seeding": "bridge",
         "cost_reduction": "cost", "channel_hardening": "channel"}


def long_run_rate(res):
    half = len(res.action_rate_A) // 2
    return (res.action_rate_A[half:].mean() + res.action_rate_B[half:].mean()) / 2


def beta_for_p(p_target, total=4.0):
    return p_target * total, (1.0 - p_target) * total


# (prior p, c_R, u_R, label)
COMBOS = [
    (0.50, 0.35, 1.0, "BASELINE p=.50 cR=.35 uR=1.0"),
    (0.40, 0.35, 1.0, "lower prior   p=.40"),
    (0.60, 0.35, 1.0, "higher prior  p=.60"),
    (0.50, 0.25, 1.0, "lower cost    cR=.25"),
    (0.50, 0.45, 1.0, "higher cost   cR=.45"),
    (0.50, 0.35, 1.3, "higher payoff uR=1.3"),
]

print("Sensitivity of intervention ranking to (p, c_R, u_R)")
print("=" * 78)
all_ok = True
for p_prior, cR, uR, label in COMBOS:
    a, b = beta_for_p(p_prior)
    base = Params(beta_alpha=a, beta_beta=b, c_R_min=cR, u_R=uR)
    pi_star = analytical_threshold(base)
    pis = [round(max(0.05, pi_star - 0.10), 3), round(pi_star, 3)]
    print(f"\n{label}   (pi* = {pi_star:.3f})")
    for pi in pis:
        rates = {}
        for iv in INTERVENTIONS:
            vals = []
            for s in range(N_SEEDS):
                pr = Params(T=T, pi=pi,
                            beta_alpha=a, beta_beta=b, c_R_min=cR, u_R=uR,
                            intervention=iv,
                            intervention_dose=DOSE if iv != "none" else 0.0,
                            N_A=N_SIDE, N_B=N_SIDE, mean_degree=MEANDEG,
                            seed=s * 17 + 3)
                vals.append(long_run_rate(run(pr)))
            rates[iv] = float(np.mean(vals))
        ranking = sorted(rates, key=rates.get, reverse=True)
        # check expected order among the active interventions + baseline
        ok = (rates["bridge_seeding"] >= rates["cost_reduction"] - 0.005 and
              rates["cost_reduction"] >= rates["channel_hardening"] - 0.005 and
              rates["bridge_seeding"] >= rates["none"])
        all_ok = all_ok and ok
        line = "  pi={:.2f}  ".format(pi) + "  ".join(
            f"{SHORT[iv]}={rates[iv]:.3f}" for iv in INTERVENTIONS)
        order = " > ".join(SHORT[iv] for iv in ranking)
        print(line)
        print(f"          order: {order}   {'[OK]' if ok else '[!! ORDER BROKEN]'}")

print("\n" + "=" * 78)
print("RESULT:", "ranking preserved across ALL combinations" if all_ok
      else "ranking NOT uniformly preserved (see flags)")
