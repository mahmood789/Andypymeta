"""Bayesian meta-analysis methods."""

import numpy as np
import warnings

# Optional dependencies
try:
    import pymc as pm
    import arviz as az
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False


def bayesian_random_effects(effect_sizes, variances, draws=2000, tune=1000):
    """
    Bayesian random effects meta-analysis.
    
    Parameters
    ----------
    effect_sizes : array_like
        Study effect sizes
    variances : array_like
        Within-study variances
    draws : int, default 2000
        Number of MCMC draws
    tune : int, default 1000
        Number of tuning samples
        
    Returns
    -------
    dict
        Dictionary with Bayesian results
    """
    if not HAS_PYMC:
        warnings.warn("PyMC not available, returning mock results")
        effect_sizes = np.asarray(effect_sizes)
        return {
            'pooled_effect_mean': np.mean(effect_sizes),
            'pooled_effect_sd': 0.2,
            'tau_mean': 0.1,
            'tau_sd': 0.05,
            'hdi_2.5': np.mean(effect_sizes) - 0.4,
            'hdi_97.5': np.mean(effect_sizes) + 0.4,
            'method': 'Bayesian RE (mock)',
            'warning': 'PyMC not available'
        }
    
    effect_sizes = np.asarray(effect_sizes)
    variances = np.asarray(variances)
    k = len(effect_sizes)
    
    with pm.Model() as model:
        # Priors
        mu = pm.Normal('mu', mu=0, sigma=1)  # Overall effect
        tau = pm.HalfNormal('tau', sigma=0.5)  # Between-study SD
        
        # Random effects
        theta = pm.Normal('theta', mu=mu, sigma=tau, shape=k)
        
        # Likelihood
        y_obs = pm.Normal('y_obs', mu=theta, sigma=np.sqrt(variances), 
                         observed=effect_sizes)
        
        # Sample
        trace = pm.sample(draws=draws, tune=tune, return_inferencedata=True)
    
    # Extract results
    posterior = trace.posterior
    mu_samples = posterior['mu'].values.flatten()
    tau_samples = posterior['tau'].values.flatten()
    
    # Summary statistics
    mu_mean = np.mean(mu_samples)
    mu_sd = np.std(mu_samples)
    tau_mean = np.mean(tau_samples)
    tau_sd = np.std(tau_samples)
    
    # Credible intervals
    mu_hdi = az.hdi(trace, var_names=['mu'])['mu'].values
    
    return {
        'pooled_effect_mean': mu_mean,
        'pooled_effect_sd': mu_sd,
        'tau_mean': tau_mean,
        'tau_sd': tau_sd,
        'hdi_2.5': mu_hdi[0],
        'hdi_97.5': mu_hdi[1],
        'trace': trace,
        'method': 'Bayesian Random Effects',
        'n_draws': draws,
        'n_tune': tune
    }


def bayesian_rct_rwe_synthesis(rct_effects, rct_variances, rwe_effects, rwe_variances,
                              rwe_bias_prior_mean=0, rwe_bias_prior_sd=0.1):
    """
    Bayesian synthesis of RCT and real-world evidence.
    
    Parameters
    ----------
    rct_effects : array_like
        RCT effect sizes
    rct_variances : array_like
        RCT variances
    rwe_effects : array_like
        RWE effect sizes
    rwe_variances : array_like
        RWE variances
    rwe_bias_prior_mean : float, default 0
        Prior mean for RWE bias
    rwe_bias_prior_sd : float, default 0.1
        Prior SD for RWE bias
        
    Returns
    -------
    dict
        Dictionary with synthesis results
    """
    if not HAS_PYMC:
        warnings.warn("PyMC not available, returning mock results")
        all_effects = np.concatenate([rct_effects, rwe_effects])
        return {
            'pooled_effect_mean': np.mean(all_effects),
            'pooled_effect_sd': 0.15,
            'rwe_bias_mean': 0.05,
            'rwe_bias_sd': 0.08,
            'method': 'Bayesian RCT/RWE (mock)',
            'warning': 'PyMC not available'
        }
    
    rct_effects = np.asarray(rct_effects)
    rct_variances = np.asarray(rct_variances) 
    rwe_effects = np.asarray(rwe_effects)
    rwe_variances = np.asarray(rwe_variances)
    
    k_rct = len(rct_effects)
    k_rwe = len(rwe_effects)
    
    with pm.Model() as model:
        # Overall treatment effect
        mu = pm.Normal('mu', mu=0, sigma=1)
        
        # Between-study heterogeneity
        tau_rct = pm.HalfNormal('tau_rct', sigma=0.5)
        tau_rwe = pm.HalfNormal('tau_rwe', sigma=0.5)
        
        # RWE bias parameter
        bias_rwe = pm.Normal('bias_rwe', mu=rwe_bias_prior_mean, 
                           sigma=rwe_bias_prior_sd)
        
        # Study-specific effects
        theta_rct = pm.Normal('theta_rct', mu=mu, sigma=tau_rct, shape=k_rct)
        theta_rwe = pm.Normal('theta_rwe', mu=mu + bias_rwe, 
                            sigma=tau_rwe, shape=k_rwe)
        
        # Observations
        y_rct = pm.Normal('y_rct', mu=theta_rct, sigma=np.sqrt(rct_variances),
                         observed=rct_effects)
        y_rwe = pm.Normal('y_rwe', mu=theta_rwe, sigma=np.sqrt(rwe_variances),
                         observed=rwe_effects)
        
        # Sample
        trace = pm.sample(draws=1000, tune=500, return_inferencedata=True)
    
    # Extract results
    posterior = trace.posterior
    mu_samples = posterior['mu'].values.flatten()
    bias_samples = posterior['bias_rwe'].values.flatten()
    
    return {
        'pooled_effect_mean': np.mean(mu_samples),
        'pooled_effect_sd': np.std(mu_samples),
        'rwe_bias_mean': np.mean(bias_samples),
        'rwe_bias_sd': np.std(bias_samples),
        'trace': trace,
        'method': 'Bayesian RCT/RWE Synthesis',
        'n_rct': k_rct,
        'n_rwe': k_rwe
    }