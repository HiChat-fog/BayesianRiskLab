"""Tests for BayesSpec → BayesModel pipeline."""

import numpy as np
import pandas as pd
import pytest
from bayeslab import BayesSpec, BayesModel


@pytest.fixture
def simple_data():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'X1': np.random.uniform(0, 10, n),
        'X2': np.random.uniform(0, 5, n),
        'group': np.random.choice(['A', 'B'], n),
    })
    df['Y'] = 3.0 + 0.5 * df['X1'] - 0.3 * df['X2'] + np.random.normal(0, 0.5, n)
    return df


def test_bayes_spec_creation():
    spec = BayesSpec(observed='y', fixed_effects={'x': 'linear'})
    assert spec.observed == 'y'
    assert spec.fixed_effects == {'x': 'linear'}


def test_bayes_model_fit(simple_data):
    spec = BayesSpec(observed='Y', fixed_effects={'X1': 'linear', 'X2': 'linear'})
    model = BayesModel(spec)
    trace = model.fit(simple_data, draws=500, tune=200)
    assert trace is not None
    summary = model.summary()
    assert 'beta_X1' in summary.index


def test_bayes_model_predict(simple_data):
    spec = BayesSpec(observed='Y', fixed_effects={'X1': 'linear', 'X2': 'linear'})
    model = BayesModel(spec)
    model.fit(simple_data, draws=500, tune=200)

    new = pd.DataFrame({'X1': [5.0, 8.0], 'X2': [2.5, 3.0]})
    mean, lo, hi = model.predict(new)

    assert mean is not None
    assert np.ndim(mean) <= 1


def test_bayes_model_with_group(simple_data):
    spec = BayesSpec(
        observed='Y',
        fixed_effects={'X1': 'linear'},
        group_var='group',
    )
    model = BayesModel(spec)
    trace = model.fit(simple_data, draws=500, tune=200)
    assert 'group_int' in trace.posterior.data_vars


def test_bayes_model_with_interaction(simple_data):
    spec = BayesSpec(
        observed='Y',
        fixed_effects={'X1': 'linear', 'X2': 'linear'},
        interactions=[('X1', 'X2')],
    )
    model = BayesModel(spec)
    trace = model.fit(simple_data, draws=500, tune=200)
    assert 'beta_X1_X2' in trace.posterior.data_vars


def test_probability_query(simple_data):
    spec = BayesSpec(observed='Y', fixed_effects={'X1': 'linear', 'X2': 'linear'})
    model = BayesModel(spec)
    model.fit(simple_data, draws=500, tune=200)

    new = pd.DataFrame({'X1': [5.0], 'X2': [2.5]})
    prob = model.probability_less_than(new, threshold=3.0)
    assert 0 <= prob <= 1
