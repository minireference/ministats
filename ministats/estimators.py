import numpy as np


# ESTIMATORS
################################################################################

def mean(sample):
    return sum(sample) / len(sample)


def var(sample):
    xbar = mean(sample)
    sumsqdevs = sum([(xi-xbar)**2 for xi in sample])
    return sumsqdevs / (len(sample)-1)


def std(sample):
    s2 = var(sample)
    return np.sqrt(s2)


def dmeans(xsample, ysample):
    dhat = mean(xsample) - mean(ysample)
    return dhat




# DESCRIPTIVE STATISTICS
################################################################################

def median(values):
    n = len(values)
    svalues = sorted(values)
    if n % 2 == 1:            # Case A: n is odd
        mid = n // 2
        return svalues[mid]
    else:                     # Case B: n is even
        j = n // 2
        return 0.5*svalues[j-1] + 0.5*svalues[j]


def quantile(values, q):
    svalues = sorted(values)
    p = q * (len(values)-1)
    i = int(p)
    g = p - int(p)
    return (1-g)*svalues[i] + g*svalues[i+1]
