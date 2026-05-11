"""
Example 3: Cross-Dimensional Scaling Law & Unified Theory
==========================================================
Demonstrates the most mathematically innovative component:

1. DIMENSIONLESS SCALING: Outcome/N^2 = alpha/L + beta
   Extracts universal constants from one experiment and
   predicts outcomes for ANY system size.

2. UNIFIED CROSS-DIMENSIONAL THEORY:
   Delta = K1/L + K2 + K3 * (N/C)^d
   A single formula spanning 2D and 3D systems, with
   dimension exponent d derived from data.
"""

import numpy as np
from bayeslab.theory import ScalingLaw, UnifiedTheory


# ------------------------------------------------------------
# Part 1: Dimensionless Scaling Law
# ------------------------------------------------------------
print("=" * 60)
print("PART 1: DIMENSIONLESS SCALING LAW")
print("=" * 60)

# Simulated experiment: varying L (structuring parameter)
# with fixed system size N=100
np.random.seed(42)
L_vals = np.array([1, 3, 5, 10, 20, 30, 50, 100])
N_100 = 100

# Ground truth: alpha=0.0042, beta=0.0018
true_alpha, true_beta = 0.0042, 0.0018
outcome_100 = (true_alpha / L_vals + true_beta) * (N_100 ** 2)
outcome_100 += np.random.normal(0, 2.0, len(L_vals))  # add noise

print(f"\nTraining data (N={N_100}):")
for L, delta in zip(L_vals, outcome_100):
    print(f"  L={L:3d}  Delta={delta:7.2f}")

# Fit scaling law
scaling = ScalingLaw()
scaling.fit(L_vals, outcome_100, system_size=N_100)
print(f"\nFitted: alpha={scaling.alpha:.6f}, beta={scaling.beta:.6f}")
print(f"True:   alpha={true_alpha:.6f}, beta={true_beta:.6f}")
print(f"R-squared: {scaling.r_squared:.4f}")

# Predict at a DIFFERENT system size (N=200) — without any N=200 data
print(f"\nPredictions for N=200 (no N=200 data used in fitting):")
for L in [3, 10, 50]:
    pred = scaling.predict([L], system_size=200)[0]
    print(f"  L={L:3d}, N=200: predicted Delta = {pred:.2f}")

print("\nKey insight: alpha and beta are UNIVERSAL constants.")
print("Once calibrated at one N, they predict outcomes at any N.")


# ------------------------------------------------------------
# Part 2: Unified Cross-Dimensional Theory
# ------------------------------------------------------------
print(f"\n{'=' * 60}")
print("PART 2: UNIFIED CROSS-DIMENSIONAL THEORY")
print("=" * 60)

# 2D data (N=100, C=81 fixed grid points)
L_2d = np.array([1, 3, 5, 10, 20, 30, 50, 100])
N_2d = np.full_like(L_2d, 100, dtype=float)
C_2d = np.full_like(L_2d, 81, dtype=float)
delta_2d = np.array([42, 14, 9, 5, 3.5, 3, 2.5, 2]) + np.random.normal(0, 1.0, 8)

# 3D data (N=100, C=81*v_layers)
np.random.seed(1)
L_3d = np.array([1, 3, 3, 3, 3, 3, 5, 10, 50])
N_3d = np.array([100, 100, 100, 100, 100, 200, 200, 100, 100], dtype=float)
v_layers_3d = np.array([4, 2, 4, 5, 6, 5, 5, 3, 3])
C_3d = 81 * v_layers_3d
delta_3d = np.array([-44, -8, -13, -7, -18, -47, -29, -3, 1]) + np.random.normal(0, 2.0, 9)

# Merge 2D + 3D data
L_all = np.concatenate([L_2d, L_3d])
N_all = np.concatenate([N_2d, N_3d])
C_all = np.concatenate([C_2d, C_3d])
delta_all = np.concatenate([delta_2d, delta_3d])

# Fit unified theory
theory = UnifiedTheory()
theory.fit(L_all, N_all, C_all, delta_all)

summary = theory.summary()
print(f"\nFitted formula: {summary['formula']}")
print(f"  K1 = {summary['K1']:.2f}")
print(f"  K2 = {summary['K2']:.2f}")
print(f"  K3 = {summary['K3']:.2f}")
print(f"  d  = {summary['d']:.4f}")
print(f"  R-squared = {summary['r_squared']:.4f}")
print(f"\nInterpretation: {summary['interpretation']}")

print(f"\nPrediction examples:")
examples = [
    (10, 100, 81),       # 2D, same size
    (10, 200, 81),       # 2D, larger system
    (5, 100, 81*5),      # 3D, 5 vertical layers
    (3, 200, 81*5),      # 3D, larger system
]

print(f"{'L':>4} {'N':>5} {'C':>5} {'Delta_pred':>10}")
print("-" * 28)
for L, N, C in examples:
    pred = theory.predict([L], [N], [C])[0]
    print(f"{L:4d} {N:5d} {C:5d} {pred:10.2f}")

print("\nKey insight: The dimension exponent d emerges from data.")
print("d ≈ 2 means 2D-like scaling (quadratic density growth).")
print("d ≈ 1.5 means 3D with partial vertical isolation.")
print("The theory UNIFIES 2D and 3D in a single formula.")
