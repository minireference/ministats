import math
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_counties(radon, idata_cp=None, idata_np=None, idata_pp=None, idata_pp2=None,
                  figsize=None, counties=None):
    """
    Generate a 2x4 panel of scatter plots for the `selected_counties`
    and optional line plots models:
    - `idata_cp`: complete pooling model
    - `idata_np`: no pooling model
    - `idata_pp`: partial pooling model (variying intercepts)
    - `idata_pp2`: partial pooling model (variying slopes and intercepts)
    """
    if counties == None:
        counties = [
            "LAC QUI PARLE",
            "AITKIN",
            "KOOCHICHING",
            "DOUGLAS",
            "HENNEPIN",
            "STEARNS",
            "RAMSEY",
            "ST LOUIS",
        ]

    if idata_cp:
        # completely pooled means
        post1_means = idata_cp["posterior"].mean(dim=("chain", "draw"))

    if idata_np:
        # no pooling means
        post2_means = idata_np["posterior"].mean(dim=("chain", "draw"))

    if idata_pp:
        # partial pooling model (variying intercepts)
        post3_means = idata_pp["posterior"].mean(dim=("chain", "draw"))
    
    if idata_pp2:
        # partial pooling model (variying slopes and intercepts)
        post4_means = idata_pp2["posterior"].mean(dim=("chain", "draw"))

    n_rows = math.ceil(len(counties) / 4)
    if figsize is None:
        if n_rows == 1:
            figsize = (10,2)
        elif n_rows == 2:
            figsize = (10,4)
        if n_rows > 2:
            figsize = (10, 2*n_rows)
    fig, axes = plt.subplots(n_rows, 4, figsize=figsize, sharey=True, sharex=True)
    axes = axes.flatten()
    
    for i, c in enumerate(counties):
        y = radon.log_radon[radon.county == c]
        x = radon.floor[radon.county == c]
        x = x.map({"basement":0, "ground":1})
        axes[i].scatter(x + np.random.randn(len(x)) * 0.01, y, alpha=0.4)

        # linspace of x-values 
        xvals = xr.DataArray(np.linspace(0, 1))

        if idata_cp:
            # Plot complete pooling model
            model1_vals = post1_means["Intercept"] + post1_means["floor"].values*xvals
            axes[i].plot(xvals, model1_vals, "C0-")

        if idata_np: 
            # Plot no pooling model
            b = post2_means["county"].sel(county_dim=c)
            m = post2_means["floor"]
            axes[i].plot(xvals, b.values + m.values*xvals, "C1--")

        if idata_pp:
            # Plot variying intercepts model
            post3c = post3_means.sel(county__factor_dim=c)
            # When using 0 + floor model
            # slope = post.floor.values[1] - post.floor.values[0]
            # theta = post["1|county"].values + post.floor.values[0] + slope * xvals
            # When using 1 + floor model
            slope = post3c["floor"].values[0]
            theta = post3c["Intercept"] + post3c["1|county"].values + slope*xvals
            axes[i].plot(xvals, theta, "k:")

        if idata_pp2:
            # Plot variying slopes and intercepts model
            post4c = post4_means.sel(county__factor_dim=c)
            intercept = post4c["Intercept"] + post4c["1|county"].values
            slope = post4c["floor"].values[0] + post4c["floor|county"].values[0]
            theta = intercept + slope*xvals
            axes[i].plot(xvals, theta, "C3-.")

        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(["basement", "ground"])
        axes[i].set_ylim(-1, 3)
        axes[i].set_title(c)
        if i % 4 == 0:
            axes[i].set_ylabel("log radon level")

    return fig