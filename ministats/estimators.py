import numpy as np


# ESTIMATORS
################################################################################

def mean(sample):
    return np.sum(sample) / len(sample)


def var(sample):
    xbar = mean(sample)
    sumsqdevs = np.sum([(xi-xbar)**2 for xi in sample])
    return sumsqdevs / (len(sample)-1)


def std(sample):
    s2 = var(sample)
    return np.sqrt(s2)


def dmeans(xsample, ysample):
    dhat = mean(xsample) - mean(ysample)
    return dhat


