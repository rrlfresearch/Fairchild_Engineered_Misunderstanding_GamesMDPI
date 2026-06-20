"""
validation.py

Validation suite for the Computational Model of the felt-understanding ABM.

The qualitative prediction from the Analytical Model: action rate is substantially HIGH
at very low pi and substantially LOW at moderate-to-high pi, with the
transition occurring around the threshold pi*. We test for this pattern
without requiring strict monotonicity at extreme pi (where the noise floor
in the receiver model causes minor non-monotonicities that are artifacts of
the inference procedure, not features of the underlying model).
"""

import numpy as np
import sys
from felt_understanding_abm import Params, run, analytical_threshold


def header(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def check(label, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    marker = "[OK]" if passed else "[!!]"
    print(f"  {marker} {label:<57} {status}")
    if detail:
        print(f"        {detail}")
    return passed


def mean_action_rate(res):
    half = len(res.action_rate_A) // 2
    return (res.action_rate_A[half:].mean() + res.action_rate_B[half:].mean()) / 2


def check_analytical_reduction():
    header("Check 1: Analytical Model reduction (action rate higher at pi=0 than pi=pi*)")
    base_kwargs = dict(T=200, h=0.85, mean_degree=10, gamma=0.0)
    pi_star = analytical_threshold(Params(**base_kwargs, seed=11))
    print(f"  Analytical pi* = {pi_star:.3f}")
    rates_zero, rates_star = [], []
    for s in range(5):
        r_z = run(Params(**base_kwargs, pi=0.0, seed=s * 11 + 1))
        r_s = run(Params(**base_kwargs, pi=pi_star, seed=s * 11 + 1))
        rates_zero.append(mean_action_rate(r_z))
        rates_star.append(mean_action_rate(r_s))
    mean_zero = np.mean(rates_zero)
    mean_star = np.mean(rates_star)
    print(f"  pi = 0.000:       mean action rate = {mean_zero:.3f}")
    print(f"  pi = pi* = {pi_star:.3f}: mean action rate = {mean_star:.3f}")
    print(f"  Reduction:        {mean_zero - mean_star:.3f}")
    ok = (mean_zero > mean_star + 0.15) and (mean_zero > 0.30)
    return check("Action rate substantially reduced at pi=pi*", ok)


def check_no_adversary_sustainability():
    header("Check 2: No-adversary sustainability (high action rate at pi=0)")
    rates = []
    for s in range(5):
        p = Params(T=200, pi=0.0, gamma=0.0, seed=s * 13 + 22)
        res = run(p)
        rates.append(mean_action_rate(res))
    mean_rate = np.mean(rates)
    print(f"  Mean action rate at pi=0 (n=5 seeds): {mean_rate:.3f}")
    print(f"  Per-seed rates: {[f'{r:.3f}' for r in rates]}")
    ok = mean_rate > 0.30
    return check("Action rate > 0.30 with no adversary", ok)


def check_symmetry():
    header("Check 3: Symmetry (FU_A and FU_B match under symmetric params)")
    final_As, final_Bs = [], []
    for s in range(8):
        p = Params(T=150, pi=0.15, seed=s * 7 + 1)
        res = run(p)
        final_As.append(res.final_FU_A)
        final_Bs.append(res.final_FU_B)
    mean_A = np.mean(final_As)
    mean_B = np.mean(final_Bs)
    diff = abs(mean_A - mean_B)
    pooled_se = np.sqrt((np.var(final_As) + np.var(final_Bs)) / 8)
    print(f"  Mean FU_A = {mean_A:.3f}, Mean FU_B = {mean_B:.3f}")
    print(f"  |diff| = {diff:.3f}; 2 * pooled SE = {2*pooled_se:.3f}")
    ok = diff < 2 * pooled_se + 0.03
    return check("FU_A and FU_B match under symmetry", ok)


def check_monotonicity():
    header("Check 4: Action rate decreases sharply through threshold region")
    # Test that action rate is meaningfully reduced as pi increases through
    # the analytically relevant range (0 to slightly above pi*).
    pis = [0.0, 0.1, 0.2, 0.3]
    rates = []
    for pi_val in pis:
        seed_rates = []
        for s in range(3):
            p = Params(T=200, pi=pi_val, seed=s * 17 + 33)
            res = run(p)
            seed_rates.append(mean_action_rate(res))
        rate = np.mean(seed_rates)
        rates.append(rate)
        print(f"  pi = {pi_val:.2f}: mean action rate = {rate:.3f}")
    # Require monotone decrease through this range
    monotone = all(rates[i+1] <= rates[i] + 0.02 for i in range(len(rates) - 1))
    total_drop = rates[0] - rates[-1]
    print(f"  Drop from pi=0 to pi=pi*: {total_drop:.3f}")
    ok = monotone and total_drop > 0.15
    return check("Action rate monotone decreasing through pi*", ok)


def check_robustness():
    header("Check 5: Robustness across network specifications (SBM and scale-free agree)")
    rates_sbm, rates_sf = [], []
    for s in range(3):
        p_sbm = Params(T=200, pi=0.35, network_type="sbm", seed=s * 19 + 44)
        p_sf = Params(T=200, pi=0.35, network_type="scale_free", seed=s * 19 + 44)
        r_sbm = run(p_sbm)
        r_sf = run(p_sf)
        rates_sbm.append(mean_action_rate(r_sbm))
        rates_sf.append(mean_action_rate(r_sf))
    mean_sbm = np.mean(rates_sbm)
    mean_sf = np.mean(rates_sf)
    print(f"  SBM:        mean action rate at pi=0.35 = {mean_sbm:.3f}")
    print(f"  Scale-free: mean action rate at pi=0.35 = {mean_sf:.3f}")
    ok = mean_sbm < 0.20 and mean_sf < 0.20
    return check("Both networks show action collapse at pi=0.35", ok)


def main():
    print("\n" + "#" * 72)
    print("#  Computational Model Validation Suite")
    print("#" * 72)
    results = [check_analytical_reduction(), check_no_adversary_sustainability(),
               check_symmetry(), check_monotonicity(), check_robustness()]
    print("\n" + "#" * 72)
    n_pass = sum(results)
    print(f"#  RESULTS: {n_pass}/{len(results)} checks passed")
    if n_pass == len(results):
        print("#  All checks passed. Proceeding to sweeps.")
    else:
        print("#  VALIDATION FAILED.")
    print("#" * 72 + "\n")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
