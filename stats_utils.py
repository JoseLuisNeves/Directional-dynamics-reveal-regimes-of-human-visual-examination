from __future__ import annotations
from sklearn.mixture import GaussianMixture
from scipy.stats import mannwhitneyu
from typing import Optional, Tuple, Dict, Any
import numpy as np
eps = 1e-12
PARAMS = {"GMM_SUBSAMPLE_N": 500_000}

def finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]

def subsample(x: np.ndarray, n: int, seed: int) -> np.ndarray:
    x = finite(x)
    if x.size <= n:
        return x
    rng = np.random.default_rng(seed)
    return x[rng.choice(x.size, size=n, replace=False)]

def extract_sorted_params_2comp(gmm: GaussianMixture) -> Tuple[float, float, float, float, float, float, int, int]:
    means = gmm.means_.reshape(-1)
    cov = gmm.covariances_.reshape(-1) # covariances_ shape depends on covariance_type; in our pipeline it's 'full' with 1D features
    stds = np.sqrt(np.maximum(cov, 0.0))
    weights = gmm.weights_.reshape(-1)
    o = np.argsort(means)
    k1, k2 = int(o[0]), int(o[1])
    return (float(means[k1]), float(stds[k1]), float(means[k2]), float(stds[k2]), float(weights[k1]), float(weights[k2]), k1, k2)

def fit_gmm_2comp(x: np.ndarray, valley: Optional[float], seed: int) -> Tuple[GaussianMixture, Dict[str, float]]:
    xs = subsample(x, PARAMS["GMM_SUBSAMPLE_N"], seed)
    X = xs[:, None]
    means_init = None
    weights_init = None
    precisions_init = None
    n_init = 10
    if valley is not None and np.isfinite(valley):
        left = xs[xs <= valley]
        right = xs[xs > valley]
        if left.size >= 10 and right.size >= 10:
            m1, m2 = float(np.median(left)), float(np.median(right))
            means_init = np.array([[min(m1, m2)], [max(m1, m2)]])
            w1 = float(left.size / xs.size)
            weights_init = np.array([w1, 1.0 - w1])
            v1 = max(float(np.var(left)) if left.size > 1 else 1.0, 1e-6)
            v2 = max(float(np.var(right)) if right.size > 1 else 1.0, 1e-6)
            precisions_init = np.array([[[1.0 / v1]], [[1.0 / v2]]])
            n_init = 1
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=seed, n_init=n_init, max_iter=500, reg_covar=1e-6, means_init=means_init, weights_init=weights_init, precisions_init=precisions_init).fit(X)
    mu1, sd1, mu2, sd2, pi1, pi2, _, _ = extract_sorted_params_2comp(gmm)
    pooled_sd = float(np.sqrt(0.5 * (sd1**2 + sd2**2)))
    cohen_d = float((mu2 - mu1) / pooled_sd) if pooled_sd > 0 else np.nan
    return gmm, {"pi_min": float(min(pi1, pi2)), "cohen_d": cohen_d}

# Alias for compatibility with bayesian_focal_model.py
fit_gmm_2comp_with_valley_init = fit_gmm_2comp

def compute_delta_bic(x: np.ndarray, valley: Optional[float], seed: int) -> float:
    """BIC(1-comp) - BIC(2-comp) on subsample. Positive favors 2-comp. Uses valley initialization for turn_std to match reported model"""
    xs = subsample(x, PARAMS["GMM_SUBSAMPLE_N"], seed)
    X = xs[:, None]
    gmm1 = GaussianMixture(n_components=1, covariance_type="full", random_state=seed).fit(X)
    means_init = None
    weights_init = None
    precisions_init = None
    n_init = 10
    if valley is not None and np.isfinite(valley):
        left = xs[xs <= valley]
        right = xs[xs > valley]
        if left.size >= 10 and right.size >= 10:
            m1, m2 = float(np.median(left)), float(np.median(right))
            means_init = np.array([[min(m1, m2)], [max(m1, m2)]])
            w1 = float(left.size / xs.size)
            weights_init = np.array([w1, 1.0 - w1])
            v1 = max(float(np.var(left)) if left.size > 1 else 1.0, 1e-6)
            v2 = max(float(np.var(right)) if right.size > 1 else 1.0, 1e-6)
            precisions_init = np.array([[[1.0 / v1]], [[1.0 / v2]]])
            n_init = 1
    gmm2 = GaussianMixture(n_components=2, covariance_type="full", random_state=seed, n_init=n_init, reg_covar=1e-6, means_init=means_init, weights_init=weights_init, precisions_init=precisions_init).fit(X)
    return float(gmm1.bic(X) - gmm2.bic(X))

def mw_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 10 or b.size < 10: 
        return {"mean_low": np.nan, "mean_high": np.nan, "p": np.nan, "rbc": np.nan}
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    rbc = 1.0 - (2.0 * float(U)) / (float(a.size) * float(b.size))
    return {"mean_low": float(np.mean(a)), "mean_high": float(np.mean(b)), "p": float(p), "rbc": float(rbc)}

def logpdf(x:np.ndarray, mu: float, sd: float) -> np.ndarray:
    s = float(max(sd, eps))
    z = (x - float(mu)) / s
    return -0.5 * (z * z) - np.log(s) - 0.5 * np.log(2.0 * np.pi)

def rank01(x: np.ndarray) -> np.ndarray:
    x = finite(x)
    if x.size == 0:
        return x
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    return ranks / (x.size + 1.0)

# Alias for compatibility
mann_whitney_with_rbc = mw_stats
