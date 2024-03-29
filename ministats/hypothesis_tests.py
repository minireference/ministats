import numpy as np
from scipy.stats import bootstrap
from scipy.stats import chi2
from scipy.stats import f_oneway
from scipy.stats import norm
from scipy.stats import rv_continuous
from scipy.stats import t as tdist

from .estimators import mean
from .estimators import std
from .estimators import var
from .estimators import dmeans
from .formulas import calcdf

from .sampling import gen_sampling_dist
from .sampling import gen_boot_dist


# TAIL CALCULATION UTILS
################################################################################

def tailvalues(valuesH0, obs, alt="two-sided"):
    """
    Select the subset of the elements in list `valuesH0` that
    are equal or more extreme than the observed value `obs`.
    """
    assert alt in ["greater", "less", "two-sided"]
    valuesH0 = np.array(valuesH0)
    if alt == "greater":
        tails = valuesH0[valuesH0 >= obs]
    elif alt == "less":
        tails = valuesH0[valuesH0 <= obs]
    elif alt == "two-sided":
        meanH0 = np.mean(valuesH0)
        obsdev = abs(obs - meanH0)
        tails = valuesH0[abs(valuesH0-meanH0) >= obsdev]
    return tails


def tailprobs(rvH0, obs, alt="two-sided"):
    """
    Calculate the probability of all outcomes of the random variable `rvH0`
    that are equal or more extreme than the observed value `obs`.
    """
    assert alt in ["greater", "less", "two-sided"]
    if alt == "greater":
        pvalue = 1 - rvH0.cdf(obs)
    elif alt == "less":
        pvalue = rvH0.cdf(obs)
    elif alt == "two-sided":  # assumes distribution is symmetric
        meanH0 = rvH0.mean()
        obsdev = abs(obs - meanH0)
        pleft = rvH0.cdf(meanH0 - obsdev)
        pright = 1 - rvH0.cdf(meanH0 + obsdev)
        pvalue = pleft + pright
    return pvalue




# BASIC TESTS (used for kombucha data generation)
################################################################################

def ztest(sample, mu0, sigma0, alt="two-sided"):
    """
    Z-test to detect mean deviation from known normal population.
    """
    mean = np.mean(sample)
    n = len(sample)
    se = sigma0 / np.sqrt(n)
    obsz = (mean - mu0) / se
    rvZ = norm(0,1)
    pval = tailprobs(rvZ, obsz, alt=alt)
    return pval


def chi2test_var(sample, sigma0, alt="greater"):
    """
    Run chi2 test to detect if a sample variance deviation
    from the known population variance `sigma0` exists.
    """
    n = len(sample)
    s2 = np.var(sample, ddof=1)
    obschi2 = (n - 1) * s2 / sigma0**2
    rvX2 = chi2(df=n-1)
    pvalue = tailprobs(rvX2, obschi2, alt=alt)
    return pvalue




# SIMULATION TESTS (Section 3.3)
################################################################################

def simulation_test_mean(sample, mu0, sigma0, alt="two-sided"):
    """
    Compute the p-value of the observed mean of `sample`
    under H0 of a normal distribution `norm(mu0,sigma0)`.
    """
    # 1. Compute the sample mean
    obsmean = mean(sample)
    n = len(sample)

    # 2. Get sampling distribution of the mean under H0
    rvXH0 = norm(mu0, sigma0)
    xbars = gen_sampling_dist(rvXH0, estfunc=mean, n=n)

    # 3. Compute the p-value
    tails = tailvalues(xbars, obsmean, alt=alt)
    pvalue = len(tails) / len(xbars)
    return pvalue


def simulation_test_var(sample, mu0, sigma0, alt="greater"):
    """
    Compute the p-value of the observed variance of `sample`
    under H0 of a normal distribution `norm(mu0,sigma0)`.
    """
    # 1. Compute the sample variance
    obsvar = var(sample)
    n = len(sample)

    # 2. Get sampling distribution of variance under H0
    rvXH0 = norm(mu0, sigma0)
    xvars = gen_sampling_dist(rvXH0, estfunc=var, n=n)

    # 3. Compute the p-value
    tails = tailvalues(xvars, obsvar, alt=alt)
    pvalue = len(tails) / len(xvars)
    return pvalue


def simulation_test(sample, rvH0, estfunc, alt="two-sided"):
    """
    Compute the p-value of the observed estimate `estfunc(sample)` under H0
    described by the random variable `rvH0`.
    """
    # 1. Compute the observed value of `estfunc`
    obsest = estfunc(sample)
    n = len(sample)

    # 2. Get sampling distribution of `estfunc` under H0
    sampl_dist_H0 = gen_sampling_dist(rvH0, estfunc, n)

    # 3. Compute the p-value
    tails = tailvalues(sampl_dist_H0, obsest, alt=alt)
    pvalue = len(tails) / len(sampl_dist_H0)
    return pvalue




# BOOTSTRAP TEST FOR THE MEAN (cut material)
################################################################################

def bootstrap_test_mean(sample, mu0, B=10000):
    """
    Compute the p-value of the observed `mean(sample)`
    under H0 with mean `mu0`. Model the variability of
    the distribution using bootstrap estimation.
    """
    # 1. Compute the observed value of the mean
    obsmean = mean(sample)

    # 2. Get sampling distribution of the mean under H0
    #    by "shifting" the sample so its mean is `mu0`
    sample_H0 = np.array(sample) - obsmean + mu0
    bmeans = gen_boot_dist(sample_H0, np.mean, B=B)
    
    # 3. Compute the p-value
    tails = tailvalues(bmeans, obsmean)
    pvalue = len(tails) / len(bmeans)
    return pvalue




# PERMUTATION TEST DMEANS
################################################################################

def resample_under_H0(xsample, ysample):
    """
    Generate new samples from a random permutation of
    the values in the samples `xsample` and `ysample`.
    """
    values = np.concatenate((xsample, ysample))
    shuffled_values = np.random.permutation(values)
    xresample = shuffled_values[0:len(xsample)]
    yresample = shuffled_values[len(xsample):]
    return xresample, yresample


def permutation_test_dmeans(xsample, ysample, P=10000):
    """
    Compute the p-value of the observed difference between means
    `dmeans(xsample,ysample)` under the null hypothesis where
    the group membership is randomized.
    """
    # 1. Compute the observed difference between means
    obsdhat = dmeans(xsample, ysample)

    # 2. Get sampling dist. of `dmeans` under H0
    pdhats = []
    for i in range(0, P):
        rsx, rsy = resample_under_H0(xsample, ysample)
        pdhat = dmeans(rsx, rsy)
        pdhats.append(pdhat)

    # 3. Compute the p-value
    tails = tailvalues(pdhats, obsdhat)
    pvalue = len(tails) / len(pdhats)
    return pvalue


def permutation_test(xsample, ysample, estfunc, P=10000):
    """
    Compute the p-value of the observed estimate `estfunc(xsample,ysample)`
    under the null hypothesis where the group membership is randomized.
    """
    # 1. Compute the observed value of `estfunc`
    obsest = estfunc(xsample, ysample)

    # 2. Get sampling dist. of `estfunc` under H0
    pestimates = []
    for i in range(0, P):
        rsx, rsy = resample_under_H0(xsample, ysample)
        pestimate = estfunc(rsx, rsy)
        pestimates.append(pestimate)

    # 3. Compute the p-value
    tails = tailvalues(pestimates, obsest)
    pvalue = len(tails) / len(pestimates)
    return pvalue




# PERMUTATION ANOVA
################################################################################

def permutation_anova(samples, P=10000, alt="greater"):
    """
    Compute the p-value of the observed F-statistic for `samples` list
    under the null hypothesis where the group membership is randomized.
    """
    ns = [len(sample) for sample in samples]

    # 1. Compute the observed F-statistic
    obsfstat, _ = f_oneway(*samples)

    # 2. Get sampling dist. of F-statistic under H0
    pfstats = []
    for i in range(0, P):
        values = np.concatenate(samples)
        pvalues = np.random.permutation(values)
        psamples = []
        nstart = 0
        for nstep in ns:
            psample = pvalues[nstart:nstart+nstep]
            psamples.append(psample)
            nstart = nstart + nstep
        pfstat, _ = f_oneway(*psamples)
        pfstats.append(pfstat)

    # 3. Compute the p-value
    tails = tailvalues(pfstats, obsfstat, alt=alt)
    pvalue = len(tails) / len(pfstats)
    return pvalue



# T-TESTS
################################################################################

def ttest_mean(sample, mu0, alt="two-sided"):
    """
    T-test to detect mean deviation from a population with known mean `mu0`.
    """
    assert alt in ["greater", "less", "two-sided"]
    obsmean = np.mean(sample)
    n = len(sample)
    std = np.std(sample, ddof=1)
    sehat = std / np.sqrt(n)
    obst = (obsmean - mu0) / sehat
    rvT = tdist(df=n-1)
    pvalue = tailprobs(rvT, obst, alt=alt)
    return pvalue


def ttest_dmeans(xsample, ysample, equal_var=False, alt="two-sided"):
    """
    T-test to detect difference between two populations means
    based on the difference between sample means.
    """
    # Calculate the observed difference between means
    obsdhat = mean(xsample) - mean(ysample)

    # Calculate the sample sizes and the stds
    n, m = len(xsample), len(ysample)
    sx, sy = std(xsample), std(ysample)

    # Calculate the standard error, the degrees of
    # freedom, the null model, and the t-statistic
    if not equal_var:  # Welch's t-test (default)
        seD = np.sqrt(sx**2/n + sy**2/m)
        obst = (obsdhat - 0) / seD
        dfD = calcdf(sx, n, sy, m)
        rvT0 = tdist(df=dfD)
    else:              # Use pooled variance
        varp = ((n-1)*sx**2 + (m-1)*sy**2) / (n+m-2)
        stdp = np.sqrt(varp)
        seDp = stdp * np.sqrt(1/n + 1/m)
        obst = (obsdhat - 0) / seDp
        dfp = n + m - 2
        rvT0 = tdist(df=dfp)

    # Calculate the p-value from the t-distribution
    pvalue = tailprobs(rvT0, obst, alt=alt)
    return pvalue



def ttest_paired(sample1, sample2, alt="two-sided"):
    """
    T-test for comparing relative change in a set of paired measurements.
    """
    n = len(sample1)
    n2 = len(sample2)
    assert n == n2, "Paired t-test assumes both samples are of the same size."
    ds = np.array(sample1) - np.array(sample2)
    std = np.std(ds, ddof=1)
    meand  = np.mean(ds)
    se = std / np.sqrt(n)
    obst = (meand - 0) / se
    rvT = tdist(df=n-1)
    pvalue = tailprobs(rvT, obst, alt=alt)
    return pvalue
