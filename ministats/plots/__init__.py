import numpy as np



# UTILS
################################################################################

def nicebins(stats, obs, nbins=60):
    """
    Choose bins that are aligned with observation `obs` so that
    `tailvalues(stats,obs)` hist. will cover `stats` hist. cleanly.
    """
    stats = np.array(stats)
    xmin, xbar, xmax = stats.min(), stats.mean(), stats.max()
    if not xmin <= obs <= xmax:
        return np.linspace(xmin, xmax, nbins)
    # Find values we want the bins to be aligned to
    dev = abs(xbar - obs)
    x1, x2 = xbar-dev, xbar+dev
    # Calculate prop. of bins to allocate to middle...
    propmid = (x2-x1) / (xmax-xmin)
    # ... and generate the bins for the mid-section
    nmid = int(nbins * propmid)
    binsmid = np.linspace(x1, x2, nmid+1)
    # Generate left and right bins with the same step size as `binsmid`
    step = (x2-x1) / nmid
    binsleft = np.sort(np.arange(x1, xmin, -step)[1:])
    binsright = np.arange(x2, xmax, step)[1:]
    return np.concatenate([binsleft, binsmid, binsright])

