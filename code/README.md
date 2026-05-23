# Engineered Misunderstanding under Psychological Warfare

Reproducibility code for Fairchild, R. *Engineered Misunderstanding under Psychological Warfare: A Bayesian Signaling Game of Felt-Understanding Collapse in the German Atomausstieg.* Submitted to *Games* (MDPI), Special Issue on Games with Incomplete Information.

This repository contains the agent-based simulation (the manuscript's Computational Model), the validation suite, and the scripts that generate Figures 2 through 6 plus the scale-free robustness check (Figure 5R).

## Quick start

```bash
pip install -r requirements.txt
python validation.py          # runs the 5-check validation suite (should print 5/5 PASS)
python rerun_fig2.py          # regenerates Figure 2 (trajectories)
python rerun_fig3.py          # regenerates Figure 3 (phase diagram)
python rerun_fig4.py          # regenerates Figure 4 (interventions)
python rerun_fig5.py          # regenerates Figure 5 (targeting, SBM)
python robustness_targeting_scalefree.py  # regenerates Figure 5R (targeting, scale-free)
python fig6_dose_response.py  # regenerates Figure 6 (bridge seeding dose-response)
```

All scripts write their output figures and any associated JSON data to the `figures/` subdirectory (which will be created if it does not exist).

Each script uses fixed random seeds so the figures should reproduce numerically across runs, modulo platform-specific floating-point differences.

## Files

| File | Purpose |
|---|---|
| `felt_understanding_abm.py` | Core simulation module: agent state, network construction, belief update, action selection, adversary, interventions, main `run()` entry point |
| `validation.py` | Five-check validation suite that verifies the Computational Model reduces to the Analytical Model in the appropriate limits |
| `experiments.py` | Master script that runs all experiments in sequence |
| `rerun_fig2.py` | Figure 2: time-course of action rate at three perturbation regimes |
| `rerun_fig3.py` | Figure 3: phase diagram across (π, h) parameter space |
| `rerun_fig4.py` | Figure 4: intervention effectiveness comparison |
| `rerun_fig5.py` | Figure 5: adversary targeting policies on SBM network |
| `robustness_targeting_scalefree.py` | Figure 5R: adversary targeting on scale-free N=5000 (robustness check) |
| `fig6_dose_response.py` | Figure 6: bridge-seeding dose-response across perturbation regimes |

## Model summary

The simulation implements an agent-based extension of the Bayesian signaling game in Section 3 of the paper:

- **Agents:** Two domestic identity groups A and B with continuous recognition capacity θ ∈ [0, 1] drawn from Beta(2, 2)
- **Networks:** Stochastic block model with homophily h, or Barabási–Albert scale-free with randomly permuted group labels
- **Channel:** Perturbation π plus optional adversarial injection γ under three targeting policies (uniform, central, bridge)
- **Update:** Bayesian log-odds posterior under a separating-profile receiver model, with noise floor ε = 0.05 and smoothing η = 0.1
- **Actions:** Quantal response with precision λ = 10
- **Interventions:** Bridge seeding, cost reduction, or channel hardening at adjustable dose

The full specification, including the analytical threshold result that this code is built to validate, is in Section 3 and 4 of the manuscript.

## Note on the volume-vs-precision result

An earlier version of this code contained three defects identified:

1. **Scale-free type assignment** correlated group labels with node IDs and therefore with degree (because Barabási–Albert preferential attachment makes early nodes hubs). Fixed: type labels are now randomly permuted relative to node identifiers.
2. **Adversary injection budget** used per-edge Bernoulli sampling with `gamma * weight`, which saturated at high-weight edges and made central/bridge targeting effectively lose budget relative to uniform. Fixed: injection now samples exactly `round(γ · M)` directed cross-edge transmissions without replacement, with sampling probabilities proportional to targeting weights.
3. **Bridge seeding at dose=0** previously seeded one edge by default (a `max(..., 1)` clamp). Fixed: dose=0 seeds exactly zero edges.

Under the corrected code, the strong "uniform volume beats targeted precision" claim does not survive. Uniform targeting is marginally more damaging than central or bridge targeting (gap typically 0.001 to 0.015 in action rate), but the gap is small and central and bridge are statistically indistinguishable. The manuscript reports this honestly as a qualification of the firehose-of-falsehood literature rather than a confirmation of it. Figures 1, 2, 3, 4, and 6 are unaffected by these fixes.

## Citation

If you use this code, please cite:

Fairchild, R. (2026). *Engineered Misunderstanding under Psychological Warfare: A Bayesian Signaling Game of Felt-Understanding Collapse in the German Atomausstieg.* Manuscript submitted to *MDPI Games*.

## License

MIT (please credit).
