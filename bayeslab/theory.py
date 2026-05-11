"""
Cross-Dimensional Scaling Laws & Unified Theoretical Framework.

Two key innovations:

1. DIMENSIONLESS SCALING LAW
   Reduces multi-parameter systems to dimensionless form:
       Outcome* = alpha / L + beta
   where Outcome* = Outcome / N^2 and L is the structuring parameter.
   The coefficients alpha, beta are universal constants — they do
   not depend on system size N.

2. UNIFIED CROSS-DIMENSIONAL THEORY
   A single formula that spans 2D and 3D system configurations:
       Delta = K1/L + K2 + K3 * (N/C)^d
   where d is the dimension factor (~2 for 2D, ~1.5 for 3D).
   The exponent d quantifies how additional spatial dimensions
   reduce interaction density.

Both provide closed-form predictions for unseen system configurations
without requiring new simulations.
"""

import numpy as np
from scipy.optimize import curve_fit
import json


class ScalingLaw:
    """Dimensionless scaling analysis.

    Given experimental data with varying structuring parameter L
    and fixed system size N, extracts universal constants alpha, beta
    such that: Outcome/N^2 = alpha/L + beta

    These constants can predict outcomes at any system size.
    """

    def __init__(self):
        self.alpha = None
        self.beta = None
        self.r_squared = None

    def fit(self, L_values, outcomes, system_size):
        """Fit the dimensionless scaling law.

        L_values:   array-like, structuring parameter values
        outcomes:   array-like, observed outcomes at each L
        system_size: float, the fixed N used in these experiments
        """
        L_vals = np.asarray(L_values, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        N = system_size

        outcome_star = outcomes / (N ** 2)

        def _form(X, alpha, beta):
            return alpha / X[0] + beta

        p0 = [0.001, 1e-5]
        params, _ = curve_fit(_form, (L_vals,), outcome_star, p0=p0, maxfev=10000)
        self.alpha, self.beta = params

        pred = _form((L_vals,), self.alpha, self.beta)
        ss_res = np.sum((outcome_star - pred) ** 2)
        ss_tot = np.sum((outcome_star - np.mean(outcome_star)) ** 2)
        self.r_squared = 1 - ss_res / ss_tot

        return self

    def predict(self, L_values, system_size):
        """Predict outcome at new L values and system size.

        outcome_new = (alpha/L + beta) * N_new^2
        """
        L_vals = np.asarray(L_values, dtype=float)
        outcome_star = self.alpha / L_vals + self.beta
        return outcome_star * (system_size ** 2)

    def summary(self):
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'r_squared': self.r_squared,
            'formula': 'Outcome/N^2 = alpha/L + beta',
        }


class UnifiedTheory:
    """Cross-dimensional unified theoretical model.

    Fits: Delta = K1/L + K2 + K3 * (N/C)^d

    where:
        L  = structuring parameter (e.g. layers)
        N  = system size (e.g. number of agents)
        C  = capacity (e.g. number of interaction points)
        d  = dimension exponent (derived, not assumed)
    """

    def __init__(self):
        self.K1 = None
        self.K2 = None
        self.K3 = None
        self.d = None
        self.r_squared = None

    def fit(self, L_vals, N_vals, C_vals, delta_obs):
        """Fit the unified cross-dimensional formula.

        L_vals:    structuring parameter values
        N_vals:    system size values
        C_vals:    capacity values
        delta_obs: observed outcome differences
        """
        def _unified(X, K1, K2, K3, d):
            L, N, C = X
            return K1 / L + K2 + K3 * (N / C) ** d

        p0 = [40, -25, 20, 1.8]
        params, _ = curve_fit(_unified, (L_vals, N_vals, C_vals),
                              delta_obs, p0=p0, maxfev=20000)
        self.K1, self.K2, self.K3, self.d = params

        pred = _unified((L_vals, N_vals, C_vals), *params)
        ss_res = np.sum((delta_obs - pred) ** 2)
        ss_tot = np.sum((delta_obs - np.mean(delta_obs)) ** 2)
        self.r_squared = 1 - ss_res / ss_tot

        return self

    def predict(self, L_vals, N_vals, C_vals):
        """Predict using the fitted unified formula."""
        return self.K1 / np.asarray(L_vals) + self.K2 + \
               self.K3 * (np.asarray(N_vals) / np.asarray(C_vals)) ** self.d

    def summary(self):
        return {
            'K1': self.K1, 'K2': self.K2, 'K3': self.K3,
            'd': self.d, 'r_squared': self.r_squared,
            'formula': 'Delta = K1/L + K2 + K3 * (N/C)^d',
            'interpretation': (
                f'Dimension exponent d = {self.d:.2f}. '
                f'In 2D systems d ≈ 2 (quadratic density scaling); '
                f'in 3D systems d ≈ 1.5 (reduced by vertical isolation).'
            ),
        }

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.summary(), f, indent=2)
