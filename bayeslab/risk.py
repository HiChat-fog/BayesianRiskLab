"""
Decision-Theoretic Risk Analysis on Bayesian Posterior Surfaces.

Given a fitted Bayesian model, computes:

1. P(outcome < threshold | conditions) → probability map over parameter grid
2. Risk decision matrix → expected gain/loss of choosing each option,
   enabling optimal decision-making under uncertainty.
"""

import numpy as np
import pandas as pd


def probability_grid(bayes_model, grid_spec, fixed_values=None, threshold=0.0):
    """Evaluate P(outcome < threshold) on a 2D parameter grid.

    bayes_model: fitted BayesModel instance
    grid_spec:   dict like {'x1': np.linspace(1, 100, 20), 'x2': np.linspace(0, 2, 20)}
    fixed_values: dict of fixed values for variables not in grid
    threshold:   float, decision boundary

    Returns: (X_mesh, Y_mesh, prob_grid)
        prob_grid[i, j] = P(outcome < threshold | x1_j, x2_i)
    """
    keys = list(grid_spec.keys())
    if len(keys) != 2:
        raise ValueError("grid_spec must have exactly 2 keys for a 2D grid.")

    g1, g2 = keys
    v1 = np.asarray(grid_spec[g1])
    v2 = np.asarray(grid_spec[g2])
    M1, M2 = np.meshgrid(v1, v2)

    n = M1.size
    df = pd.DataFrame({g1: M1.ravel(), g2: M2.ravel()})
    if fixed_values:
        for k, v in fixed_values.items():
            df[k] = v

    prob = bayes_model.probability_less_than(df, threshold=threshold)
    return M1, M2, prob.reshape(M1.shape)


def risk_decision_matrix(bayes_model, grid_spec, fixed_values=None):
    """Full decision-theoretic risk analysis on a 2D parameter grid.

    For each grid point, computes:
    - prob_safe:  P(outcome < 0 | conditions) — probability that option A is better
    - risk_fp:    E[loss | chose A, but B was better] (false positive cost)
    - risk_fn:    E[loss | chose B, but A was better] (false negative cost)

    Returns: (prob_safe, risk_fp, risk_fn) each as 2D arrays
    """
    keys = list(grid_spec.keys())
    g1, g2 = keys[0], keys[1]
    v1, v2 = np.asarray(grid_spec[g1]), np.asarray(grid_spec[g2])
    M1, M2 = np.meshgrid(v1, v2)

    df = pd.DataFrame({g1: M1.ravel(), g2: M2.ravel()})
    if fixed_values:
        for k, v in fixed_values.items():
            df[k] = v

    prob = bayes_model.probability_less_than(df, threshold=0.0)
    mean, _, _ = bayes_model.predict(df)

    delta_mean = np.asarray(mean).ravel()
    pos_mask = (delta_mean > 0)
    neg_mask = (delta_mean < 0)

    risk_fp = np.zeros_like(delta_mean)
    risk_fn = np.zeros_like(delta_mean)
    risk_fp[pos_mask] = delta_mean[pos_mask]
    risk_fn[neg_mask] = -delta_mean[neg_mask]

    return (
        prob.reshape(M1.shape),
        risk_fp.reshape(M1.shape),
        risk_fn.reshape(M1.shape),
    )
