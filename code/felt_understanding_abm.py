"""
felt_understanding_abm.py

Agent-based simulation of felt-understanding dynamics under adversarial
signal perturbation. The Computational Model: An Agent-Based Extension

This module implements:
  - Stochastic block model network construction with homophily parameter h
  - Continuous recognition capacities theta_i ~ Beta(alpha, beta)
  - Bayesian belief update over neighbor signals with channel perturbation pi
  - Three adversary targeting policies (uniform, central, bridge)
  - Three intervention types (bridge seeding, cost reduction, channel hardening)
  - Validation suite reproducing Analytical Model predictions in the appropriate limit

Author: Ryanne Fairchild
For: Games (MDPI) Special Issue on Games with Incomplete Information
"""

from dataclasses import dataclass, field, replace
from typing import Optional, Literal, Callable
import numpy as np
import networkx as nx


# ===========================================================================
# SECTION 1: PARAMETERS
# ===========================================================================

@dataclass
class Params:
    """All tunable parameters for a single simulation run.

    Defaults match the stylized Atomausstieg parameterization.
    """
    # Population
    N_A: int = 500
    N_B: int = 500

    # Prior recognition capacity ~ Beta(alpha, beta)
    # Beta(2,2) has mean 0.5 and is unimodal at 0.5 (the baseline)
    beta_alpha: float = 2.0
    beta_beta: float = 2.0

    # Network: stochastic block model parameters
    # h is homophily: fraction of edges that are within-group
    # mean_degree controls overall connectivity
    h: float = 0.85
    mean_degree: float = 10.0
    network_type: Literal["sbm", "scale_free"] = "sbm"

    # Payoffs and costs
    u_R: float = 1.0       # recognition payoff
    u_I: float = 2.0       # identity-protection penalty
    c_R_min: float = 0.35  # cost of authentic signaling for theta=1
    c_N_max: float = 1.75  # cost of authentic signaling for theta=0

    # Channel perturbation (the adversary's primary lever)
    pi: float = 0.0

    # Adversary injection: fraction of cross-group edges with fabricated s_D
    # added each step beyond the channel perturbation
    gamma: float = 0.0
    targeting: Literal["uniform", "central", "bridge"] = "uniform"

    # Time horizon
    T: int = 200

    # Interventions (off by default)
    intervention: Literal["none", "bridge_seeding", "cost_reduction",
                          "channel_hardening"] = "none"
    intervention_dose: float = 0.0  # fraction of agents / edges affected

    # Reproducibility
    seed: int = 42

    # Initial belief mu(0) for all agents (default 0.9 = established cooperative
    # basin; exposed so robustness to the starting condition can be tested)
    initial_mu: float = 0.9

    @property
    def N(self) -> int:
        return self.N_A + self.N_B


# ===========================================================================
# SECTION 2: NETWORK CONSTRUCTION
# ===========================================================================

def build_network(p: Params) -> nx.Graph:
    """Build the agent network. Returns a NetworkX graph with 'group'
    attribute on each node ('A' or 'B').
    """
    rng = np.random.default_rng(p.seed)
    N = p.N

    if p.network_type == "sbm":
        # Stochastic block model
        # Target mean degree d. Of d edges per node, (h*d) are intra-group,
        # ((1-h)*d) are inter-group.
        # For an SBM with block sizes n_A, n_B:
        #   intra-A: each node has (h*d) intra-A neighbors out of (n_A-1) possible
        #     => p_in = (h*d) / (n_A-1)
        #   inter:   each node has ((1-h)*d) inter neighbors out of n_B (or n_A) possible
        #     => p_out = ((1-h)*d) / n_other
        d = p.mean_degree
        p_in_A = (p.h * d) / max(p.N_A - 1, 1)
        p_in_B = (p.h * d) / max(p.N_B - 1, 1)
        p_out = ((1.0 - p.h) * d) / max(min(p.N_A, p.N_B), 1)
        # Clamp to [0, 1]
        p_in_A = min(max(p_in_A, 0.0), 1.0)
        p_in_B = min(max(p_in_B, 0.0), 1.0)
        p_out = min(max(p_out, 0.0), 1.0)

        sizes = [p.N_A, p.N_B]
        probs = [[p_in_A, p_out], [p_out, p_in_B]]
        G = nx.stochastic_block_model(sizes, probs, seed=p.seed)

    elif p.network_type == "scale_free":
        # Barabasi-Albert preferential attachment. Because early-added nodes
        # accumulate higher degrees in BA, we MUST randomly permute group
        # assignments after graph construction; otherwise Group A would have
        # systematically higher degree than Group B (artifact of node ordering).
        m = max(int(round(p.mean_degree / 2)), 1)
        G = nx.barabasi_albert_graph(N, m, seed=p.seed)
    else:
        raise ValueError(f"unknown network_type {p.network_type}")

    # Assign groups. For SBM the first N_A nodes are by construction group A
    # (the SBM generator partitions by node index). For scale-free, we randomly
    # permute the node-to-group assignment using the seed so that group label
    # is uncorrelated with the structural position of the node in the graph.
    if p.network_type == "scale_free":
        rng = np.random.default_rng(p.seed + 1000)
        node_ids = np.arange(N)
        rng.shuffle(node_ids)
        group_A_nodes = set(node_ids[:p.N_A].tolist())
        for i in range(N):
            G.nodes[i]['group'] = 'A' if i in group_A_nodes else 'B'
    else:
        for i in range(N):
            G.nodes[i]['group'] = 'A' if i < p.N_A else 'B'

    return G


# ===========================================================================
# SECTION 3: AGENT INITIALIZATION
# ===========================================================================

@dataclass
class Population:
    """Vectorized state for all agents."""
    group: np.ndarray       # shape (N,), 0 for A, 1 for B
    theta: np.ndarray       # shape (N,), recognition capacity in [0,1]
    mu: np.ndarray          # shape (N,), belief P(other group is recog-capable)
    c_R: np.ndarray         # shape (N,), per-agent cost of authentic signal
    G: nx.Graph             # the network
    neighbors_A: list       # neighbors_A[i] = list of i's neighbors in group A
    neighbors_B: list       # neighbors_B[i] = list of i's neighbors in group B
    cross_edges: list       # list of (u, v) tuples spanning A-B


def init_population(p: Params, G: nx.Graph,
                    initial_mu: float = 0.9) -> Population:
    """Initialize the population.

    initial_mu: starting belief of each agent about P(other group is
    recognition-capable). Default 0.9 represents an established cooperative
    state from which we test sustainability under perturbation. This matches
    the spec's analytical question: given that a separating equilibrium with
    calibrated felt understanding is achievable, does it survive when an
    adversary perturbs the channel? Starting at the prior would conflate
    convergence and sustainability questions.
    """
    rng = np.random.default_rng(p.seed + 1)
    N = p.N

    group = np.array([0 if G.nodes[i]['group'] == 'A' else 1 for i in range(N)],
                     dtype=np.int8)
    theta = rng.beta(p.beta_alpha, p.beta_beta, size=N)

    mu = np.full(N, initial_mu, dtype=np.float64)

    c_R = p.c_R_min + (p.c_N_max - p.c_R_min) * (1.0 - theta) ** 2

    neighbors_A = [[] for _ in range(N)]
    neighbors_B = [[] for _ in range(N)]
    cross_edges = []
    for u, v in G.edges():
        gu, gv = group[u], group[v]
        if gv == 0:
            neighbors_A[u].append(v)
        else:
            neighbors_B[u].append(v)
        if gu == 0:
            neighbors_A[v].append(u)
        else:
            neighbors_B[v].append(u)
        if gu != gv:
            cross_edges.append((u, v))

    return Population(
        group=group, theta=theta, mu=mu, c_R=c_R, G=G,
        neighbors_A=neighbors_A, neighbors_B=neighbors_B,
        cross_edges=cross_edges
    )


# ===========================================================================
# SECTION 4: THE UPDATE STEP
# ===========================================================================

def choose_actions(pop: Population, p: Params,
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Each agent chooses action a_R (1) or a_D (0) via quantal response.

    Spec Sec 4.3 prescribes the deterministic best response:
      play a_R iff (1 - pi_eff) * u_R * mu_i > c(theta_i)
    For numerical stability and behavioral realism (standard in ABMs) we use a
    logit / quantal-response specification, parameterized by precision lambda.
    As lambda -> infinity this recovers the deterministic best response.

      P(a_R) = sigmoid(lambda * [(1 - pi_eff) * u_R * mu_i - c_i])

    We use lambda = 10, which is sharp enough to recover the deterministic
    best response in expectation while letting agents make occasional
    "mistakes" that prevent absorbing-state lock-in.
    """
    if rng is None:
        rng = np.random.default_rng(p.seed + 13)
    pi_eff = effective_pi(pop, p)
    expected_payoff_aR = (1.0 - pi_eff) * p.u_R * pop.mu
    c = pop.c_R.copy()
    if p.intervention == "cost_reduction":
        rng2 = np.random.default_rng(p.seed + 99)
        n_treated = int(p.intervention_dose * pop.theta.size)
        treated_idx = rng2.choice(pop.theta.size, n_treated, replace=False)
        c[treated_idx] *= 0.5
    # Quantal response
    lam = 10.0
    margin = expected_payoff_aR - c
    prob_aR = 1.0 / (1.0 + np.exp(-lam * margin))
    plays_aR = (rng.random(pop.mu.size) < prob_aR).astype(np.int8)
    return plays_aR


def effective_pi(pop: Population, p: Params) -> float:
    """Channel perturbation, possibly reduced by channel_hardening intervention."""
    if p.intervention == "channel_hardening":
        # Reduce pi by a factor proportional to intervention_dose
        return p.pi * (1.0 - p.intervention_dose)
    return p.pi


def transmit_signals(actions: np.ndarray, pop: Population, p: Params,
                     rng: np.random.Generator) -> dict:
    """For each cross-group edge, produce the signal observed by the receiver.

    Returns: dict mapping (sender, receiver) -> observed signal (0 = s_D, 1 = s_A)
    where the observed signal reflects sender's action passed through the
    perturbed channel and possibly adversarial injection.

    The injection budget gamma is held EXACTLY (not in expectation) equal
    across targeting policies: we sample exactly round(gamma * M) directed
    cross-edge transmissions without replacement, with sampling probabilities
    proportional to the targeting weights. This prevents the probability-
    saturation artifact that would arise if we used per-edge Bernoulli
    sampling with weight-amplified probabilities (which would cause
    central/bridge policies to clip at high-weight edges, effectively
    losing budget compared to uniform).
    """
    pi_eff = effective_pi(pop, p)
    observed = {}

    # ----- Channel transmission first (no adversarial injection yet) -----
    # Enumerate all directed cross-edge transmissions
    directed_transmissions = []
    for u, v in pop.cross_edges:
        directed_transmissions.append((u, v))
        directed_transmissions.append((v, u))

    # Apply channel perturbation to each transmission
    for sender, receiver in directed_transmissions:
        sent = actions[sender]
        flip = rng.random() < pi_eff
        observed[(sender, receiver)] = sent if not flip else 1 - sent

    # ----- Adversarial injection with exact-count budget -----
    if p.gamma > 0:
        M = len(directed_transmissions)
        n_inject = int(round(p.gamma * M))
        n_inject = min(n_inject, M)  # cap at total transmissions
        if n_inject > 0:
            # Compute selection probabilities according to targeting policy
            if p.targeting == "uniform":
                probs = np.ones(M) / M
            else:
                edge_weights = adversary_edge_weights(pop, p)
                raw = np.array([edge_weights.get(dt, 1.0)
                                for dt in directed_transmissions])
                # Normalize so probabilities sum to 1
                raw = np.maximum(raw, 0.0)
                if raw.sum() > 0:
                    probs = raw / raw.sum()
                else:
                    probs = np.ones(M) / M
            # Sample WITHOUT replacement to enforce exact count
            chosen_idx = rng.choice(M, size=n_inject, replace=False, p=probs)
            for idx in chosen_idx:
                sender, receiver = directed_transmissions[idx]
                observed[(sender, receiver)] = 0  # forced s_D

    return observed


_betweenness_cache = {}

def adversary_edge_weights(pop: Population, p: Params) -> dict:
    """Compute targeting weights on directed cross-edges per targeting policy.
    Weights are normalized so the mean across cross-edges is 1.0, so the
    overall injection budget gamma is preserved across policies.

    Bridge-targeting betweenness is cached per (graph_id, seed) since the
    network is static within a run.
    """
    weights = {}
    if p.targeting == "central":
        # Weight by receiver's degree
        deg = dict(pop.G.degree())
        raw = {}
        for u, v in pop.cross_edges:
            raw[(u, v)] = deg[v]
            raw[(v, u)] = deg[u]
    elif p.targeting == "bridge":
        cache_key = (id(pop.G), p.seed)
        if cache_key in _betweenness_cache:
            bc = _betweenness_cache[cache_key]
        else:
            try:
                k_sample = min(50, max(20, int(pop.G.number_of_nodes() ** 0.5)))
                bc = nx.edge_betweenness_centrality(pop.G, k=k_sample,
                                                      seed=p.seed + 7)
            except Exception:
                bc = nx.edge_betweenness_centrality(pop.G)
            _betweenness_cache[cache_key] = bc
        raw = {}
        for u, v in pop.cross_edges:
            key = (u, v) if (u, v) in bc else (v, u)
            w = bc.get(key, 0.0)
            raw[(u, v)] = w
            raw[(v, u)] = w
    else:
        for u, v in pop.cross_edges:
            raw[(u, v)] = 1.0
            raw[(v, u)] = 1.0
    mean_w = np.mean(list(raw.values())) if raw else 1.0
    if mean_w == 0:
        mean_w = 1.0
    for k in raw:
        weights[k] = raw[k] / mean_w
    return weights


def update_beliefs(pop: Population, actions: np.ndarray, observed: dict,
                   p: Params) -> np.ndarray:
    """Bayesian update of each agent's belief mu_i given observed cross-group
    signals.

    Receiver's mental model: assume the opposing
    population is playing the separating strategy in which type R sends s_A
    and type N sends s_D. Under this model and the perturbed channel:
      P(s_A observed | theta_other = R) = 1 - pi_eff
      P(s_A observed | theta_other = N) = pi_eff
    so as pi -> 0 the signal becomes perfectly informative, and as pi -> 1/2
    it becomes uninformative. A noise floor (eps) prevents degenerate
    log-likelihood ratios when pi_eff = 0; this represents irreducible
    observational ambiguity even on a clean channel.

    Aggregation across multiple cross-group neighbor observations uses the
    log-odds form with a learning-rate smoothing.
    """
    pi_eff = effective_pi(pop, p)
    eps = 0.05  # noise floor on receiver inference
    pi_used = max(pi_eff, eps)
    pi_used = min(pi_used, 0.5 - eps)

    # Likelihood ratios for one observation
    # log P(s_A | R) / P(s_A | N) = log[(1-pi)/pi]
    # log P(s_D | R) / P(s_D | N) = log[pi/(1-pi)]
    log_lr_sA = np.log((1.0 - pi_used) / pi_used)
    log_lr_sD = -log_lr_sA

    N = pop.mu.size
    new_mu = pop.mu.copy()
    eta = 0.1  # learning rate

    n_sA = np.zeros(N, dtype=np.int32)
    n_sD = np.zeros(N, dtype=np.int32)
    for (sender, receiver), sig in observed.items():
        if sig == 1:
            n_sA[receiver] += 1
        else:
            n_sD[receiver] += 1

    for i in range(N):
        if n_sA[i] + n_sD[i] == 0:
            continue
        mu_prev = pop.mu[i]
        log_prior_ratio = np.log(mu_prev / (1.0 - mu_prev))
        log_post_ratio = (log_prior_ratio
                          + n_sA[i] * log_lr_sA
                          + n_sD[i] * log_lr_sD)
        log_post_ratio = np.clip(log_post_ratio, -10, 10)
        post = 1.0 / (1.0 + np.exp(-log_post_ratio))
        new_mu[i] = (1 - eta) * mu_prev + eta * post
        new_mu[i] = np.clip(new_mu[i], 0.05, 0.95)
    return new_mu


def step(pop: Population, p: Params, rng: np.random.Generator
         ) -> tuple[Population, np.ndarray]:
    """Single time step. Returns (updated population, actions taken)."""
    actions = choose_actions(pop, p, rng)
    observed = transmit_signals(actions, pop, p, rng)
    new_mu = update_beliefs(pop, actions, observed, p)
    pop = replace_pop_mu(pop, new_mu)
    return pop, actions


def replace_pop_mu(pop: Population, new_mu: np.ndarray) -> Population:
    return Population(
        group=pop.group, theta=pop.theta, mu=new_mu, c_R=pop.c_R, G=pop.G,
        neighbors_A=pop.neighbors_A, neighbors_B=pop.neighbors_B,
        cross_edges=pop.cross_edges
    )


# ===========================================================================
# SECTION 5: INTERVENTIONS
# ===========================================================================

def apply_intervention(pop: Population, p: Params) -> Population:
    """Apply pre-run interventions that modify the population state.
    Note: cost_reduction and channel_hardening are applied inside the step
    functions; only bridge_seeding modifies the population directly.
    """
    if p.intervention == "bridge_seeding":
        rng = np.random.default_rng(p.seed + 200)
        # Identify high-betweenness cross-edges and seed both endpoints
        # with elevated theta
        try:
            bc = nx.edge_betweenness_centrality(pop.G,
                                                k=min(100, pop.G.number_of_nodes()),
                                                seed=p.seed + 200)
        except Exception:
            bc = nx.edge_betweenness_centrality(pop.G)
        # Filter to cross-edges
        cross_set = set()
        for u, v in pop.cross_edges:
            cross_set.add((u, v))
            cross_set.add((v, u))
        cross_bc = []
        for (u, v), b in bc.items():
            if (u, v) in cross_set or (v, u) in cross_set:
                cross_bc.append(((u, v), b))
        cross_bc.sort(key=lambda x: -x[1])

        n_seed_edges = int(round(p.intervention_dose * len(cross_bc)))
        # Note: at dose=0, n_seed_edges=0 (no seeding at all); at the smallest
        # positive dose that rounds to 0, we still seed zero. The previous
        # version forced a floor of 1 edge even at dose=0, contaminating the
        # zero-dose baseline of the dose-response experiment (Figure 6).
        new_theta = pop.theta.copy()
        new_c_R = pop.c_R.copy()
        seeded = set()
        for (u, v), _ in cross_bc[:n_seed_edges]:
            for node in (u, v):
                if node not in seeded:
                    new_theta[node] = max(new_theta[node], 0.9)
                    seeded.add(node)
        # Recompute cost for seeded agents
        new_c_R = p.c_R_min + (p.c_N_max - p.c_R_min) * (1.0 - new_theta) ** 2
        return Population(
            group=pop.group, theta=new_theta, mu=pop.mu, c_R=new_c_R, G=pop.G,
            neighbors_A=pop.neighbors_A, neighbors_B=pop.neighbors_B,
            cross_edges=pop.cross_edges
        )
    return pop


# ===========================================================================
# SECTION 6: OUTCOME MEASURES AND RUN
# ===========================================================================

@dataclass
class RunResult:
    params: Params
    FU_A_trajectory: np.ndarray   # shape (T+1,)
    FU_B_trajectory: np.ndarray   # shape (T+1,)
    action_rate_A: np.ndarray     # shape (T,), fraction of group A playing a_R
    action_rate_B: np.ndarray
    final_FU_A: float
    final_FU_B: float
    time_to_collapse: Optional[int]   # first t where FU < 0.3 for both groups
    basin: Literal["cooperative", "collapsed", "asymmetric", "intermediate"]


def run(p: Params) -> RunResult:
    """Run a single simulation."""
    G = build_network(p)
    pop = init_population(p, G, initial_mu=p.initial_mu)
    pop = apply_intervention(pop, p)
    rng = np.random.default_rng(p.seed + 5)

    FU_A = np.zeros(p.T + 1)
    FU_B = np.zeros(p.T + 1)
    act_A = np.zeros(p.T)
    act_B = np.zeros(p.T)

    FU_A[0] = pop.mu[pop.group == 0].mean()
    FU_B[0] = pop.mu[pop.group == 1].mean()

    for t in range(p.T):
        pop, actions = step(pop, p, rng)
        FU_A[t + 1] = pop.mu[pop.group == 0].mean()
        FU_B[t + 1] = pop.mu[pop.group == 1].mean()
        act_A[t] = actions[pop.group == 0].mean()
        act_B[t] = actions[pop.group == 1].mean()

    # Outcome classification
    final_A = FU_A[-1]
    final_B = FU_B[-1]
    collapsed_threshold = 0.3
    cooperative_threshold = 0.6

    if final_A < collapsed_threshold and final_B < collapsed_threshold:
        basin = "collapsed"
    elif final_A > cooperative_threshold and final_B > cooperative_threshold:
        basin = "cooperative"
    elif ((final_A < collapsed_threshold) != (final_B < collapsed_threshold)):
        basin = "asymmetric"
    else:
        basin = "intermediate"

    # Time to collapse
    ttc = None
    for t in range(p.T + 1):
        if FU_A[t] < collapsed_threshold and FU_B[t] < collapsed_threshold:
            ttc = t
            break

    return RunResult(
        params=p,
        FU_A_trajectory=FU_A,
        FU_B_trajectory=FU_B,
        action_rate_A=act_A,
        action_rate_B=act_B,
        final_FU_A=final_A,
        final_FU_B=final_B,
        time_to_collapse=ttc,
        basin=basin,
    )


# ===========================================================================
# SECTION 7: LAYER A THRESHOLD (analytical reference)
# ===========================================================================

def analytical_threshold(p: Params, prior_p: Optional[float] = None) -> float:
    """The analytical threshold pi* from Proposition 1, with c_R taken at the
    representative agent (theta = 1, the recognition-capable type).
    """
    if prior_p is None:
        prior_p = p.beta_alpha / (p.beta_alpha + p.beta_beta)
    return max(0.0, 1.0 - p.c_R_min / (p.u_R * prior_p))
