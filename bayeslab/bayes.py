"""
Declarative Bayesian Hierarchical Modeling Framework.

Core innovation: a specification-driven approach that automatically
constructs PyMC models from a declarative spec, eliminating boilerplate
while preserving full Bayesian rigor.

Supports:
- Fixed effects (linear, log-transformed)
- Interaction terms
- Quadratic effects
- Group/hierarchical intercepts
- Automatic scaling & posterior predictive inference
- Probability queries: P(target < threshold | conditions)
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


class BayesSpec:
    """Declarative specification of a Bayesian hierarchical model.

    Example:
        spec = BayesSpec(
            observed='outcome',
            fixed_effects={'X1': 'linear', 'X2': 'log'},
            interactions=[('X1', 'X2')],
            group_var='category',
            quadratic=['X1'],
        )
        model = BayesModel(spec)
        trace = model.fit(dataframe)
    """

    def __init__(self, observed, fixed_effects=None, interactions=None,
                 group_var=None, quadratic=None):
        self.observed = observed
        self.fixed_effects = fixed_effects or {}
        self.interactions = interactions or []
        self.group_var = group_var
        self.quadratic = quadratic or []

    def scale_col(self, df, col):
        x = df[col].values.astype(float)
        mu, sigma = x.mean(), x.std()
        if sigma < 1e-8:
            return x, mu, 1.0
        return (x - mu) / sigma, mu, sigma


class BayesModel:
    """Fits a Bayesian hierarchical model from a BayesSpec.

    Uses PyMC with the nutpie NUTS sampler for efficient MCMC.
    Provides posterior prediction, probability queries, and diagnostics.
    """

    def __init__(self, spec):
        self.spec = spec
        self.trace = None
        self.model = None
        self.scalers = {}
        self._group_map = {}

    def fit(self, df, draws=2000, tune=1000, target_accept=0.9, seed=42):
        """Fit the model via MCMC. Returns ArviZ InferenceData trace."""
        spec = self.spec

        # --- Scale fixed effects ---
        scaled = {}
        for col, form in spec.fixed_effects.items():
            if form == 'log':
                raw = np.log(df[col].values.astype(float))
                mu, sigma = raw.mean(), raw.std()
                scaled[col] = (raw - mu) / sigma if sigma > 1e-8 else raw
                self.scalers[col] = ('log', mu, sigma)
            else:
                vals, mu, sigma = spec.scale_col(df, col)
                scaled[col] = vals
                self.scalers[col] = ('linear', mu, sigma)

        # --- Group encoding ---
        if spec.group_var and spec.group_var in df.columns:
            groups = df[spec.group_var].astype('category')
            group_codes = groups.cat.codes.values
            n_groups = groups.cat.categories.size
            self._group_map = dict(enumerate(groups.cat.categories))
        else:
            group_codes = np.zeros(len(df), dtype=int)
            n_groups = 1

        y = df[spec.observed].values.astype(float)

        with pm.Model() as self.model:
            # Hierarchical intercepts
            group_int = pm.Normal('group_int', mu=0, sigma=5, shape=n_groups)
            global_int = pm.Normal('global_int', 0, 10)

            # Fixed effect coefficients
            betas = {}
            for col in spec.fixed_effects:
                betas[col] = pm.Normal(f'beta_{col}', 0, 1)

            # Interaction coefficients
            betas_int = {}
            for c1, c2 in spec.interactions:
                betas_int[(c1, c2)] = pm.Normal(f'beta_{c1}_{c2}', 0, 0.5)

            # Quadratic coefficients
            betas_quad = {}
            for col in spec.quadratic:
                betas_quad[col] = pm.Normal(f'beta_{col}_sq', 0, 0.5)

            # Linear predictor
            mu = global_int + group_int[group_codes]
            for col, beta in betas.items():
                mu = mu + beta * scaled[col]
            for (c1, c2), beta in betas_int.items():
                mu = mu + beta * scaled[c1] * scaled[c2]
            for col, beta in betas_quad.items():
                mu = mu + beta * scaled[col] ** 2

            sigma_obs = pm.HalfNormal('sigma_obs', 5)
            pm.Normal('obs', mu=mu, sigma=sigma_obs, observed=y)

            try:
                import nutpie  # noqa: F401
                sampler = 'nutpie'
            except ImportError:
                sampler = None  # PyMC default NUTS

            sample_kwargs = dict(
                draws=draws, tune=tune, target_accept=target_accept,
                random_seed=seed,
            )
            if sampler is not None:
                sample_kwargs['nuts_sampler'] = sampler

            self.trace = pm.sample(**sample_kwargs)

        return self.trace

    def summary(self, hdi_prob=0.95):
        """Posterior summary with HDI intervals."""
        if self.trace is None:
            raise RuntimeError("Call .fit() first.")
        return az.summary(self.trace, hdi_prob=hdi_prob)

    def predict(self, df_new, lower=2.5, upper=97.5):
        """Posterior predictive mean and credible interval for new data."""
        if self.trace is None:
            raise RuntimeError("Call .fit() first.")
        scaled_new = self._scale_new_data(df_new)
        post = self.trace.posterior.stack(sample=('chain', 'draw'))
        mu = self._build_mu_post(post, scaled_new)

        sigma_s = post['sigma_obs'].values[:, None]
        pred = np.random.normal(mu, sigma_s)

        result_mean = np.mean(pred, axis=0)
        result_lo = np.percentile(pred, lower, axis=0)
        result_hi = np.percentile(pred, upper, axis=0)
        return (
            result_mean.item() if result_mean.size == 1 else result_mean,
            result_lo.item() if result_lo.size == 1 else result_lo,
            result_hi.item() if result_hi.size == 1 else result_hi,
        )

    def probability_less_than(self, df_new, threshold=0.0):
        """P(outcome < threshold | data) — decision-critical probability."""
        if self.trace is None:
            raise RuntimeError("Call .fit() first.")
        scaled_new = self._scale_new_data(df_new)
        post = self.trace.posterior.stack(sample=('chain', 'draw'))
        mu = self._build_mu_post(post, scaled_new)

        sigma_s = post['sigma_obs'].values[:, None]
        pred = np.random.normal(mu, sigma_s)
        prob = np.mean(pred < threshold, axis=0)
        return prob.item() if prob.size == 1 else prob

    # --- Internal helpers ---

    def _scale_new_data(self, df_new):
        scaled_new = {}
        for col in self.spec.fixed_effects:
            form, mu, sigma = self.scalers[col]
            raw = np.log(df_new[col].values.astype(float)) if form == 'log' \
                  else df_new[col].values.astype(float)
            scaled_new[col] = (raw - mu) / sigma if sigma > 1e-8 else raw
        return scaled_new

    def _build_mu_post(self, post, scaled_new):
        n_data = len(next(iter(scaled_new.values())))
        n_samples = len(post['global_int'].values)

        has_group = 'group_int' in post
        group_int_avg = post['group_int'].values.mean(axis=-1) if has_group else 0.0
        mu = (post['global_int'].values + group_int_avg)[:, None] * np.ones(n_data)[None, :]

        for col in self.spec.fixed_effects:
            key = f'beta_{col}'
            if key in post:
                mu = mu + post[key].values[:, None] * scaled_new[col][None, :]

        for (c1, c2) in self.spec.interactions:
            key = f'beta_{c1}_{c2}'
            if key in post:
                mu = mu + post[key].values[:, None] * \
                    (scaled_new[c1][None, :] * scaled_new[c2][None, :])

        for col in self.spec.quadratic:
            key = f'beta_{col}_sq'
            if key in post:
                mu = mu + post[key].values[:, None] * (scaled_new[col][None, :] ** 2)

        return mu
