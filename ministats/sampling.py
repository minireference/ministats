import numpy as np

# SAMPLING DISTRIBUTIONS
################################################################################

# This function is used in Chapter 2 with argument `statfunc` because the concept
# of estimator has not been introduced yet, but it is identical to the function below.

def gen_sampling_dist(rv, statfunc, n, N=1000):
    """
    Simulate `N` samples of size `n` from the random variable `rv` to
    generate the sampling distribution of the statistic `statfunc`.
    """
    stats = []
    for i in range(0, N):
        sample = rv.rvs(n)
        stat = statfunc(sample)
        stats.append(stat)
    return stats



# This function is identical to the one above, but uses the argument `estfunc`.
# It is the main function used in Chapter 3 and later.

def gen_sampling_dist(rv, estfunc, n, N=10000):
    """
    Simulate `N` samples of size `n` from the random variable `rv` to
    generate the sampling distribution of the estimator `estfunc`.
    """
    estimates = []
    for i in range(0, N):
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
    for i in range(0, B):
        bsample = np.random.choice(sample, n, replace=True)
        bestimate = estfunc(bsample)
        bestimates.append(bestimate)
    return bestimates

