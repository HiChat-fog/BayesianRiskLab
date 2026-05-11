"""
Latin Hypercube Sampling (LHS) for Efficient Experimental Design.

LHS with maximin criterion ensures space-filling coverage of the
parameter space with minimal samples — critical when each sample
requires an expensive Monte Carlo simulation.
"""

import numpy as np
from pyDOE import lhs


def lhs_design(factor_ranges, n_samples, criterion='maximin', seed=42):
    """Generate a Latin Hypercube Sample design.

    factor_ranges: dict {name: (low, high, type)}
        type can be 'float', 'int', or 'log'
        Example: {'layers': (1, 100, 'int'), 'noise': (0, 2.0, 'float')}

    n_samples: int, number of design points
    criterion: str, 'maximin' | 'centermaximin' | 'correlation'
    seed: int, for reproducibility

    Returns: dict {name: np.array of shape (n_samples,)}
    """
    np.random.seed(seed)
    n_factors = len(factor_ranges)
    names = list(factor_ranges.keys())

    design = lhs(n_factors, samples=n_samples, criterion=criterion)

    result = {}
    for i, name in enumerate(names):
        low, high, dtype = factor_ranges[name]
        col = design[:, i] * (high - low) + low
        if dtype == 'int':
            col = np.floor(col).astype(int)
        elif dtype == 'log':
            col = np.exp(col)  # design in log-space
        result[name] = np.round(col, 6) if dtype == 'float' else col

    return result
