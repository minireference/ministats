import numpy as np
from scipy.optimize import fmin

def hpdi_from_grid(params, probs, hdi_prob=0.9):
    """
    Compute the highest probability density interval from
    the grid approximation probabilities `probs` evaluated at `params`.
    """
    assert len(params) == len(probs)
    cdfprobs = np.cumsum(probs)

    def ppf(prob):
        "Inverse CDF function calculated from `params` grid and `prob`."
        idx = cdfprobs.searchsorted(prob)
        return params[idx]

    def width(startprob):
        "Calculate the width of the `hdi_prob` interval starting at `startprob`."
        return ppf(startprob + hdi_prob) - ppf(startprob)
    
    idx_left_stop = cdfprobs.searchsorted(1 - hdi_prob)
    widths = []
    for idx_probstart in range(0, idx_left_stop):
        probstart = cdfprobs[idx_probstart]
        widths.append(width(probstart))
    
    idx_left = np.argmin(widths)
    hdi_left = params[idx_left]
    probstart = cdfprobs[idx_left]    
    idx_right = cdfprobs.searchsorted(probstart + hdi_prob)
    hdi_right = params[idx_right]
    return [hdi_left, hdi_right]



def hpdi_from_rv(rv, hdi_prob=0.9):
    probstart0 = (1 - hdi_prob) / 2
    def width(probstart):
        return rv.ppf(probstart + hdi_prob) - rv.ppf(probstart)
    min_probstart = fmin(width, x0=probstart0, xtol=1e-9, disp=False)[0]
    hdi_left = rv.ppf(min_probstart)
    hdi_right = rv.ppf(min_probstart + hdi_prob)
    return [hdi_left, hdi_right]



def hpdi_from_samples(samples, hdi_prob=0.9):
    samples = np.sort(samples)
    n = len(samples)
    idx_width = int(np.floor(hdi_prob*n))
    n_intervals = n - idx_width
    left_endpoints = samples[0:n_intervals]
    right_endpoints = samples[idx_width:n]
    widths = right_endpoints - left_endpoints
    idx_left = np.argmin(widths)
    hdi_left = samples[idx_left]
    hdi_right = samples[idx_left + idx_width]
    return [hdi_left, hdi_right]

