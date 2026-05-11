"""
Example 1: Declarative Bayesian Hierarchical Modeling
=====================================================
Demonstrates the BayesSpec → BayesModel pipeline:
1. Declare model structure with BayesSpec
2. Automatic MCMC fitting
3. Posterior summary
4. Predict on new data
5. Probability queries for decision-making

This is the flagship feature of BayesianRiskLab.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bayeslab import BayesSpec, BayesModel


# ------------------------------------------------------------
# 1. Generate synthetic data with known ground truth
# ------------------------------------------------------------
np.random.seed(42)
n = 200

df = pd.DataFrame({
    'X1': np.random.uniform(1, 100, n),       # continuous factor
    'X2': np.random.uniform(0, 2, n),          # noise factor
    'X3': np.random.uniform(0, 5, n),          # environmental factor
    'group': np.random.choice(['A', 'B', 'C'], n),
})

# Ground truth: Y = 2 - 0.5*X1 + 0.3*X2 + 0.1*X1*X2 + group_effect + noise
group_effects = {'A': -1.0, 'B': 0.0, 'C': 1.5}
df['Y'] = (
    2.0
    - 0.5 * (df['X1'] - 50) / 30
    + 0.3 * (df['X2'] - 1.0) / 0.6
    + 0.1 * (df['X1'] - 50) / 30 * (df['X2'] - 1.0) / 0.6
    + df['group'].map(group_effects)
    + np.random.normal(0, 0.3, n)
)

print("Data shape:", df.shape)
print(df.head())
print()


# ------------------------------------------------------------
# 2. Declare model structure
# ------------------------------------------------------------
spec = BayesSpec(
    observed='Y',
    fixed_effects={'X1': 'linear', 'X2': 'linear', 'X3': 'linear'},
    interactions=[('X1', 'X2')],
    group_var='group',
)
print("Model specification:")
print(f"  Observed: {spec.observed}")
print(f"  Fixed effects: {spec.fixed_effects}")
print(f"  Interactions: {spec.interactions}")
print(f"  Groups: {spec.group_var}")
print()


# ------------------------------------------------------------
# 3. Fit with MCMC
# ------------------------------------------------------------
model = BayesModel(spec)
print("Fitting model (MCMC with NUTS)...")
trace = model.fit(df, draws=2000, tune=1000)
print("Done.\n")


# ------------------------------------------------------------
# 4. Posterior summary
# ------------------------------------------------------------
print("=== Posterior Summary (95% HDI) ===")
print(model.summary())
print()


# ------------------------------------------------------------
# 5. Predict on new data
# ------------------------------------------------------------
X_new = pd.DataFrame({
    'X1': [10, 50, 90],
    'X2': [0.5, 1.0, 1.8],
    'X3': [2.5, 2.5, 2.5],
})
mean_pred, lo, hi = model.predict(X_new)
print("=== Predictions (mean, 95% CI) ===")
for i, row in X_new.iterrows():
    print(f"  X1={row['X1']:.0f}, X2={row['X2']:.1f}: "
          f"Y = {mean_pred[i] if hasattr(mean_pred, '__iter__') else mean_pred:.3f} "
          f"[{lo[i] if hasattr(lo, '__iter__') else lo:.3f}, "
          f"{hi[i] if hasattr(hi, '__iter__') else hi:.3f}]")


# ------------------------------------------------------------
# 6. Decision-critical probability query
# ------------------------------------------------------------
prob = model.probability_less_than(X_new, threshold=0.0)
print("\n=== P(Y < 0 | conditions) ===")
if hasattr(prob, '__iter__'):
    for i, row in X_new.iterrows():
        print(f"  X1={row['X1']:.0f}, X2={row['X2']:.1f}: P(Y<0) = {prob[i]:.4f}")
else:
    print(f"  P(Y<0) = {prob:.4f}")

print("\n✓ Example complete. The BayesSpec → BayesModel pipeline")
print("  eliminates hundreds of lines of boilerplate PyMC code.")
