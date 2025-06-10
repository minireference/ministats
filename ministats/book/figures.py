import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr


# Probability theory
################################################################################

def plot_ks_dist_with_inset(sample, rv, label_sample="eCDF(sample)", label_rv="CDF $F_X$"):
    """
    Usage example:
    ```
    def gen_e(lam):
        u = np.random.rand()
        e = -1 * np.log(1-u) / lam
        return e
    np.random.seed(26)
    N = 200  # number of observations to generate
    es2 = [gen_e(lam=0.2) for i in range(0,N)]
    plot_ks_dist_with_inset(es2, rvE, label_sample="eCDF(es2)", label_rv="CDF $F_E$")
    ```
    """
    from matplotlib.patches import Rectangle
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    # KS distance function
    def ks_distance_and_location(sample, dist_cdf):
        sample_sorted = np.sort(sample)
        n = len(sample_sorted)
        ecdf_vals = np.arange(1, n+1) / n
        cdf_vals = dist_cdf(sample_sorted)
        diffs = np.abs(ecdf_vals - cdf_vals)
        D = np.max(diffs)
        idx = np.argmax(diffs)
        return D, sample_sorted[idx], ecdf_vals[idx], cdf_vals[idx]

    # Compute KS distance
    D, x_star, F_emp, F_th = ks_distance_and_location(sample, rv.cdf)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.ecdfplot(sample, label=label_sample, ax=ax)

    xrange = np.linspace(0, 30, 1000)
    sns.lineplot(x=xrange, y=rv.cdf(xrange), ax=ax, label=label_rv, color="C1")
    ax.legend()

    # Zoom range
    zoom_radius = 0.75
    x_min, x_max = x_star - zoom_radius, x_star + zoom_radius
    y_min, y_max = F_th - zoom_radius * 0.1, F_th + zoom_radius * 0.1

    # Add rectangle to show zoom region in main plot
    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                     linewidth=1, edgecolor='black', facecolor='none', linestyle='--')
    ax.add_patch(rect)

    # Inset with 60% size
    axins = inset_axes(ax, width="40%", height="75%", loc="lower right")
    sns.ecdfplot(sample, ax=axins)
    sns.lineplot(x=xrange, y=rv.cdf(xrange), ax=axins, color="C1")

    # Inset limits
    axins.set_xlim(x_min, x_max)
    axins.set_ylim(y_min, y_max)
    axins.set_title("Zoom in near max $D_{KS}$", fontsize=11)

    # Draw short red line between empirical and theoretical CDF at x_star
    axins.plot([x_star, x_star], [F_emp, F_th], color='red', linestyle='--', linewidth=1.5)
    axins.annotate(
        f"$D_{{KS}} = {D:.4f}$",
        xy=(x_star, (F_emp + F_th)/2),             # Point to the middle of red line
        xytext=(x_star + 0.08, (F_emp + F_th)/2-0.05),   # Text placed to the right
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color='red'),
        ha='left', va='center', fontsize=12, color='red'
    )

    # Draw connectors between inset and main plot
    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="0.5")

    # Return figure
    return fig



# Hierarchical models
################################################################################

def plot_counties(radon, idata_cp=None, idata_np=None, idata_pp=None, idata_pp2=None,
                  figsize=None, counties=None):
    """
    Generate a 2x4 panel of scatter plots for the `selected_counties`
    and optional line plots models:
    - `idata_cp`: complete pooling model
    - `idata_np`: no pooling model
    - `idata_pp`: partial pooling model (varying intercepts)
    - `idata_pp2`: partial pooling model (varying slopes and intercepts)
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
        # partial pooling model (varying intercepts)
        post3_means = idata_pp["posterior"].mean(dim=("chain", "draw"))
    
    if idata_pp2:
        # partial pooling model (varying slopes and intercepts)
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
            # Plot varying intercepts model
            post3c = post3_means.sel(county__factor_dim=c)
            # When using 0 + floor model
            # slope = post.floor.values[1] - post.floor.values[0]
            # theta = post["1|county"].values + post.floor.values[0] + slope * xvals
            # When using 1 + floor model
            slope = post3c["floor"].values[0]
            theta = post3c["Intercept"] + post3c["1|county"].values + slope*xvals
            axes[i].plot(xvals, theta, "k:")

        if idata_pp2:
            # Plot varying slopes and intercepts model
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