import numpy as np

# SAMPLING DISTRIBUTIONS
################################################################################

def gen_sampling_dist_of_mean(rvX, n, N=1000):
    """
    Generate the sampling distribution of the mean for samples of size `n`
    from the random variable `rvX` based on `N` simulated random samples.
    """
    xbars = []
    for j in range(N):
        xsample = rvX.rvs(n)
        xbar = np.mean(xsample)
        xbars.append(xbar)
    return xbars


# This function is shown in Section 2.8
# The re-definition is identical, but uses `estfunc` instead of `statfunc`
def gen_sampling_dist(rvX, statfunc, n, N=1000):
    """
    Simulate `N` samples of size `n` from the random variable `rvX`
    to generate the sampling distribution of the statistic `statfunc`.
    """
    stats = []
    for j in range(N):
        xsample = rvX.rvs(n)
        stat = statfunc(xsample)
        stats.append(stat)
    return stats



def gen_sampling_dist(rv, estfunc, n, N=10000):
    """
    Simulate `N` samples of size `n` from the random variable `rv`
    to generate the sampling distribution of the estimator `estfunc`.
    """
    estimates = []
    for j in range(N):
        sample = rv.rvs(n)
        estimate = estfunc(sample)
        estimates.append(estimate)
    return estimates




# BOOTSTRAP
################################################################################

def gen_boot_dist(sample, estfunc, B=5000):
    """
    Generate estimates from the sampling distribution of the estimator `estfunc`
    based on `B` bootstrap samples (sampling with replacement) from `sample`.
    """
    n = len(sample)
    bestimates = []
    for j in range(B):
        bsample = np.random.choice(sample, n, replace=True)
        bestimate = estfunc(bsample)
        bestimates.append(bestimate)
    return bestimates

