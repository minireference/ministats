
import numpy as np

from .estimators import mean
from .estimators import var


# STANDARDIZED EFFECT SIZE MEASURES
################################################################################

def cohend(sample, mu0):
    """
    Compute Cohen's d for one group compared to the theoretical mean `mu0`.
    """
    mean = np.mean(sample)
    std = np.std(sample, ddof=1)
    d = (mean - mu0) / std
    return d


def cohend2(xsample, ysample):
    """
    Compute Cohen's d measure of effect size for two independent samples.
    """
    obsdhat = np.mean(xsample) - np.mean(ysample)
    n, m = len(xsample), len(ysample)
    varX = np.var(xsample, ddof=1)
    varY = np.var(ysample, ddof=1)
    # Calculate the pooled variance and standard deviation
    varp = ((n-1)*varX + (m-1)*varY) / (n + m - 2)
    stdp = np.sqrt(varp)
    d = obsdhat / stdp
    return d



# UTILS
################################################################################

def calcdf(stdX, n, stdY, m):
    """
    The Welch-Satterthwaite formula for the degrees of freedom parameter
    used for confidence intervals and Welch's t-test.
    """
    vX = stdX**2 / n
    vY = stdY**2 / m
    df = (vX + vY)**2 / (vX**2/(n-1) + vY**2/(m-1))
    return df



# KOLMOGOROV-SMIRNOV DISTANCE
################################################################################

def ks_distance(sample, rv):
    """
    Compute the Kolmogorov-Smirnov distance between an data in `sample` and 
    and the CDF function of the random variable `rv`.
    """
    n = len(sample)
    sample = np.sort(sample)

    # Empirical CDF values: i/n for the right-limit, (i-1)/n for left-limit
    ecdf_vals = np.arange(1, n+1) / n
    cdf_vals = rv.cdf(sample)

    # Compute KS distances
    ks_dist_plus = np.max(ecdf_vals - cdf_vals)
    ks_dist_minus = np.max(cdf_vals - (np.arange(0, n) / n))
    max_ks_dist = max(ks_dist_plus, ks_dist_minus)

    return max_ks_dist
