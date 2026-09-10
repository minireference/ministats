from collections import defaultdict
import os

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

from .confidence_intervals import ci_mean
from .confidence_intervals import ci_var
from .estimators import var


def boot_ci(sample, estfunc, alpha=0.1, method=None, B=5000):
    """
    An adaptor for calling the function `scipy.stats.bootstrap` without the need
    to specify all all the optional keyword arguments.
    """
    res = bootstrap([sample],
                    statistic=estfunc,
                    confidence_level=1-alpha,
                    n_resamples=B,
                    vectorized=False,
                    method=method)
    return [res.confidence_interval.low,
            res.confidence_interval.high]



def simulate_ci_props(pops, methods=["a", "percentile", "bca"], ns=[20,40],
                      param="mean", alpha=0.1, N=1000, B=10000, seed=42):
    """
    Runs a simulation of confidence intervals for `param` using `methods`
    for sample sizes `ns` from populations `pops` (dict label: model).
    Simulation parameters:
        - pops          # populations
        - methods       = ["a", "percentile", "bca"]
        - ns = [20,40]  # sample sizes
        - param         # population parameter
        - alpha = 0.1   # target error level
        - N = 1000      # number of simulations
        - B = 5000      # number of bootstrap samples
    """
    assert param in ["mean", "var"]

    # check if cached simulation data exists
    filename = "simulate_ci_props_" + param + "__ns_" + "_".join(map(str,ns)) \
                + "__alpha_" + str(alpha) + "__seed_" + str(seed) + ".csv"
    filepath = os.path.join("simdata", filename)
    if os.path.exists(filepath):  # load cached results
        print("loaded cached results from ", filepath)
        results = pd.read_csv(filepath, header=[0,1], index_col=[0,1])
        return results

    # simulation data structures
    rowindextuples = []
    for pop in pops.keys():
        if pop == "E":
            ns_to_simulate = ns + [100]
        else:
            ns_to_simulate = ns
        for n in ns_to_simulate:
            rowindextuples.append( (pop, n))
    rowsindex = pd.MultiIndex.from_tuples(rowindextuples, names=["population", "n"])
    colindex = pd.MultiIndex.from_product((["cov", "wbar"], methods), names=["property", "method"])
    results = pd.DataFrame(index=rowsindex, columns=colindex)

    # Count and interval width data will be stored here
    counts = defaultdict(int)  # keys are tuples (pop,method,n)
    widthscolindex = pd.MultiIndex.from_product((pops.keys(), methods, ns+[100]), names=["pop", "method", "n"])
    widths = pd.DataFrame(index=range(0,N), columns=widthscolindex)

    # run simulation
    np.random.seed(seed+1)
    print(f"Starting simulation for confidence intervals of population {param} :::::::::::::")
    for pop, n in rowsindex:
        print(f"  - running simulation for rv{pop} with {n=} ...")
        rv = pops[pop]
        if param == "mean":
            pop_param = rv.mean()
        else:
            pop_param = rv.var()
        for j in range(0, N):
            sample = rv.rvs(n)
            for method in methods:
                if method == "a":
                    if param == "mean":
                        ci = ci_mean(sample, alpha=alpha, method="a")
                    elif param == "var":
                        ci = ci_var(sample, alpha=alpha, method="a")
                else:
                    if param == "mean":
                        ci = boot_ci(sample, estfunc=np.mean, alpha=alpha, method=method, B=B)
                    elif param == "var":
                        ci = boot_ci(sample, estfunc=var, alpha=alpha, method=method, B=B)
                # Check if `ci`` captures populaiton parameter
                if ci[0] <= pop_param <= ci[1]:
                    counts[(pop,method,n)] += 1  # success
                # Calculate `ci` width
                widths.loc[j,(pop,method,n)] = ci[1] - ci[0]
    for method in methods:
        for pop, n in rowsindex:
            results.loc[(pop, n), ("cov", method)] = counts[(pop, method, n)] / N
            results.loc[(pop, n), ("wbar", method)] = widths.loc[:,pd.IndexSlice[pop,method,:]].mean()[pop,method, n]

    results.to_csv(filepath)
    print("Saved file to " + filepath)
    return results
