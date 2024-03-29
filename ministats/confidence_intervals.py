import numpy as np
from scipy.stats import chi2
from scipy.stats import t as tdist

from .estimators import mean
from .estimators import var
from .formulas import calcdf
from .sampling import gen_boot_dist



# CONFIDENCE INTERVALS
################################################################################

def ci_mean(sample, alpha=0.1, method="a"):
    """
    Compute confidence interval for the population mean.
    - method="a" analytical approx. based on Student's t-dist
    - method="b" approx. based on bootstrap estimation
    """
    assert method in ["a", "b"]
    if method == "a":        # analytical approximation
        from scipy.stats import t as tdist
        n = len(sample)
        xbar = np.mean(sample)
        sehat = np.std(sample, ddof=1) / np.sqrt(n)
        t_l = tdist(df=n-1).ppf(alpha/2)
        t_u = tdist(df=n-1).ppf(1-alpha/2)
        return [xbar + t_l*sehat, xbar + t_u*sehat]
    elif method == "b":          # bootstrap estimation
        xbars_boot = gen_boot_dist(sample, estfunc=mean)
        return [np.quantile(xbars_boot, alpha/2),
                np.quantile(xbars_boot, 1-alpha/2)]


def ci_var(sample, alpha=0.1, method="a"):
    """
    Compute confidence interval for the population variance.
    - method="a" analytical approx. based on chi-square dist
    - method="b" approx. based on bootstrap estimation
    """
    assert method in ["a", "b"]
    if method == "a":        # analytical approximation
        n = len(sample)
        s2 = np.var(sample, ddof=1)
        q_l = chi2(df=n-1).ppf(alpha/2)
        q_u = chi2(df=n-1).ppf(1-alpha/2)
        return [(n-1)*s2/q_u, (n-1)*s2/q_l]
    elif method == "b":          # bootstrap estimation
        vars_boot = gen_boot_dist(sample, estfunc=var)
        return [np.quantile(vars_boot, alpha/2),
                np.quantile(vars_boot, 1-alpha/2)]


def ci_dmeans(xsample, ysample, alpha=0.1, method="a"):
    """
    Compute confidence interval for the difference between population means.
    - method="a" analytical approx. based on Student's t-dist
    - method="b" approx. based on bootstrap estimation
    """
    assert method in ["a", "b"]
    if method == "a":        # analytical approximation
        stdX, n = np.std(xsample, ddof=1), len(xsample)
        stdY, m = np.std(ysample, ddof=1), len(ysample)
        dhat = np.mean(xsample) - np.mean(ysample)
        seD = np.sqrt(stdX**2/n + stdY**2/m)
        dfD = calcdf(stdX, n, stdY, m)
        t_l = tdist(df=dfD).ppf(alpha/2)
        t_u = tdist(df=dfD).ppf(1-alpha/2)
        return [dhat + t_l*seD, dhat + t_u*seD]
    elif method == "b":          # bootstrap estimation
        xbars_boot = gen_boot_dist(xsample, np.mean)
        ybars_boot = gen_boot_dist(ysample, np.mean)
        dmeans_boot = np.subtract(xbars_boot,ybars_boot)
        return [np.quantile(dmeans_boot, alpha/2),
                np.quantile(dmeans_boot, 1-alpha/2)]

