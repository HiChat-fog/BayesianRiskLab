"""
BayesianRiskLab — Declarative Bayesian Hierarchical Modeling & Risk Analysis Toolkit.

A domain-neutral framework for building, comparing, and applying
Bayesian hierarchical models with automated MCMC inference,
cross-dimensional scaling laws, and decision-theoretic risk analysis.
"""

from bayeslab.bayes import BayesSpec, BayesModel
from bayeslab.comparison import ModelComparator
from bayeslab.theory import ScalingLaw, UnifiedTheory
from bayeslab.collision import CollisionDetector, orca_velocity
from bayeslab.risk import probability_grid, risk_decision_matrix
try:
    from bayeslab.sampling import lhs_design
except ImportError:
    lhs_design = None  # pyDOE2 not available

__version__ = "1.0.0"
