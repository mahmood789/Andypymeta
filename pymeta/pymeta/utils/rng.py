"""
Random number generation utilities for PyMeta.

This module provides consistent random number generation for reproducible results.
"""

import numpy as np
from typing import Optional, Union, Tuple
from ..types import Array


class PyMetaRandomState:
    """
    Random state manager for PyMeta.
    
    This class provides a consistent interface for random number generation
    across the package, ensuring reproducible results when a seed is set.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random state.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        """
        self._rng = np.random.RandomState(seed)
        self._seed = seed
    
    def seed(self, seed: int) -> None:
        """Set random seed."""
        self._rng = np.random.RandomState(seed)
        self._seed = seed
    
    def get_seed(self) -> Optional[int]:
        """Get current seed."""
        return self._seed
    
    def random(self, size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, Array]:
        """Generate random numbers from uniform [0, 1) distribution."""
        return self._rng.random(size)
    
    def normal(self, loc: float = 0.0, scale: float = 1.0, 
              size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, Array]:
        """Generate random numbers from normal distribution."""
        return self._rng.normal(loc, scale, size)
    
    def binomial(self, n: int, p: float, 
                size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[int, Array]:
        """Generate random numbers from binomial distribution."""
        return self._rng.binomial(n, p, size)
    
    def poisson(self, lam: float, 
               size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[int, Array]:
        """Generate random numbers from Poisson distribution."""
        return self._rng.poisson(lam, size)
    
    def beta(self, a: float, b: float, 
            size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, Array]:
        """Generate random numbers from beta distribution."""
        return self._rng.beta(a, b, size)
    
    def gamma(self, shape: float, scale: float = 1.0,
             size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, Array]:
        """Generate random numbers from gamma distribution."""
        return self._rng.gamma(shape, scale, size)
    
    def exponential(self, scale: float = 1.0,
                   size: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[float, Array]:
        """Generate random numbers from exponential distribution."""
        return self._rng.exponential(scale, size)
    
    def choice(self, a: Array, size: Optional[int] = None, 
              replace: bool = True, p: Optional[Array] = None) -> Union[float, Array]:
        """Generate random samples from array."""
        return self._rng.choice(a, size, replace, p)
    
    def permutation(self, x: Union[int, Array]) -> Array:
        """Randomly permute array or generate permutation of integers."""
        return self._rng.permutation(x)
    
    def shuffle(self, x: Array) -> None:
        """Shuffle array in-place."""
        self._rng.shuffle(x)
    
    def multivariate_normal(self, mean: Array, cov: Array, 
                           size: Optional[Union[int, Tuple[int, ...]]] = None) -> Array:
        """Generate random numbers from multivariate normal distribution."""
        return self._rng.multivariate_normal(mean, cov, size)


# Global random state instance
_global_rng = PyMetaRandomState()


def set_random_seed(seed: int) -> None:
    """
    Set global random seed for PyMeta.
    
    Parameters
    ----------
    seed : int
        Random seed for reproducibility
    """
    global _global_rng
    _global_rng.seed(seed)
    np.random.seed(seed)  # Also set numpy's global seed


def get_random_seed() -> Optional[int]:
    """
    Get current global random seed.
    
    Returns
    -------
    int or None
        Current random seed
    """
    return _global_rng.get_seed()


def get_rng() -> PyMetaRandomState:
    """
    Get global random state instance.
    
    Returns
    -------
    PyMetaRandomState
        Global random state
    """
    return _global_rng


def bootstrap_sample(data: Array, n_samples: Optional[int] = None, 
                    rng: Optional[PyMetaRandomState] = None) -> Array:
    """
    Generate bootstrap sample from data.
    
    Parameters
    ----------
    data : array-like
        Original data
    n_samples : int, optional
        Number of samples (default: len(data))
    rng : PyMetaRandomState, optional
        Random state to use
    
    Returns
    -------
    numpy.ndarray
        Bootstrap sample
    """
    if rng is None:
        rng = _global_rng
    
    data = np.asarray(data)
    if n_samples is None:
        n_samples = len(data)
    
    return rng.choice(data, size=n_samples, replace=True)


def generate_effect_sizes(true_effect: float, tau_squared: float, variances: Array,
                         rng: Optional[PyMetaRandomState] = None) -> Array:
    """
    Generate synthetic effect sizes for simulation studies.
    
    Parameters
    ----------
    true_effect : float
        True overall effect size
    tau_squared : float
        Between-study variance
    variances : array-like
        Within-study variances
    rng : PyMetaRandomState, optional
        Random state to use
    
    Returns
    -------
    numpy.ndarray
        Generated effect sizes
    """
    if rng is None:
        rng = _global_rng
    
    variances = np.asarray(variances)
    k = len(variances)
    
    # Generate study-specific true effects
    if tau_squared > 0:
        study_effects = rng.normal(true_effect, np.sqrt(tau_squared), k)
    else:
        study_effects = np.full(k, true_effect)
    
    # Add within-study error
    observed_effects = study_effects + rng.normal(0, np.sqrt(variances))
    
    return observed_effects


def generate_binary_data(true_or: float, baseline_risk: float, sample_sizes: Array,
                        rng: Optional[PyMetaRandomState] = None) -> Tuple[Array, Array, Array, Array]:
    """
    Generate synthetic binary outcome data.
    
    Parameters
    ----------
    true_or : float
        True odds ratio
    baseline_risk : float
        Baseline risk in control group
    sample_sizes : array-like
        Sample sizes for each study (will be split between groups)
    rng : PyMetaRandomState, optional
        Random state to use
    
    Returns
    -------
    tuple of arrays
        Treatment events, treatment totals, control events, control totals
    """
    if rng is None:
        rng = _global_rng
    
    sample_sizes = np.asarray(sample_sizes)
    k = len(sample_sizes)
    
    # Split sample sizes between groups (roughly equal)
    control_n = sample_sizes // 2
    treatment_n = sample_sizes - control_n
    
    # Generate control events
    control_events = rng.binomial(control_n, baseline_risk)
    
    # Calculate treatment probability from odds ratio
    baseline_odds = baseline_risk / (1 - baseline_risk)
    treatment_odds = baseline_odds * true_or
    treatment_prob = treatment_odds / (1 + treatment_odds)
    
    # Generate treatment events
    treatment_events = rng.binomial(treatment_n, treatment_prob)
    
    return treatment_events, treatment_n, control_events, control_n


def generate_continuous_data(true_smd: float, pooled_sd: float, sample_sizes: Array,
                           rng: Optional[PyMetaRandomState] = None) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """
    Generate synthetic continuous outcome data.
    
    Parameters
    ----------
    true_smd : float
        True standardized mean difference
    pooled_sd : float
        Pooled standard deviation
    sample_sizes : array-like
        Sample sizes for each study (will be split between groups)
    rng : PyMetaRandomState, optional
        Random state to use
    
    Returns
    -------
    tuple of arrays
        Treatment means, treatment SDs, treatment Ns, control means, control SDs, control Ns
    """
    if rng is None:
        rng = _global_rng
    
    sample_sizes = np.asarray(sample_sizes)
    k = len(sample_sizes)
    
    # Split sample sizes between groups
    control_n = sample_sizes // 2
    treatment_n = sample_sizes - control_n
    
    # Generate control group data
    control_means = rng.normal(0, pooled_sd / np.sqrt(control_n))
    control_sds = rng.gamma(pooled_sd, 1, k)  # Approximate SD variation
    
    # Generate treatment group data
    treatment_means = control_means + true_smd * pooled_sd + rng.normal(0, pooled_sd / np.sqrt(treatment_n))
    treatment_sds = rng.gamma(pooled_sd, 1, k)
    
    return treatment_means, treatment_sds, treatment_n, control_means, control_sds, control_n


def simulate_publication_bias(effect_sizes: Array, variances: Array, 
                             bias_strength: float = 1.0,
                             rng: Optional[PyMetaRandomState] = None) -> Array:
    """
    Simulate publication bias by selectively removing studies.
    
    Parameters
    ----------
    effect_sizes : array-like
        Original effect sizes
    variances : array-like
        Variances
    bias_strength : float, default 1.0
        Strength of publication bias
    rng : PyMetaRandomState, optional
        Random state to use
    
    Returns
    -------
    numpy.ndarray
        Boolean mask of published studies
    """
    if rng is None:
        rng = _global_rng
    
    effect_sizes = np.asarray(effect_sizes)
    variances = np.asarray(variances)
    
    # Calculate z-scores
    z_scores = effect_sizes / np.sqrt(variances)
    
    # Publication probability based on significance and study size
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    publication_prob = 1 / (1 + np.exp(bias_strength * (p_values - 0.05)))
    
    # Determine which studies are published
    published = rng.random(len(effect_sizes)) < publication_prob
    
    return published