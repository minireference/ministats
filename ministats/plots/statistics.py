import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns




# Section 3.2: Confidence intervals
################################################################################

def plot_conf_int(rv, alpha=0.1, rv_name="t", dist_name=None, xlims=None, ax=None):
    """
    Plot the PDF of `rv` and highlight the central (1-alpha) probability
    interval [F^{-1}(alpha/2), F^{-1}(1-alpha/2)].
    """
    # Setup axes
    if ax is None:
        fig, ax = plt.subplots()

    # Central interval endpoints
    lower_bound = rv.ppf(alpha/2)
    upper_bound = rv.ppf(1 - alpha/2)

    # Plot limits
    if xlims is not None:
        xmin, xmax = xlims
    elif rv.support()[0] == 0:
        xmin, xmax = 0, rv.ppf(0.99)
    else:
        xmin, xmax = rv.ppf(0.005), rv.ppf(0.995)

    # Dense grid for plotting
    xs = np.linspace(xmin, xmax, 1000)
    fXs = rv.pdf(xs)
    ymax = np.max(fXs)

    # Axis labels
    rv_name = rv_name.lower()
    if dist_name is None:
        dist_name = rv_name.upper()

    # PDF curve
    sns.lineplot(x=xs, y=fXs, ax=ax, color="C0")

    # Shade the central interval
    mask_mid = (xs >= lower_bound) & (xs <= upper_bound)
    ax.fill_between(xs[mask_mid], 0, fXs[mask_mid], color="C0", alpha=0.25)

    # Cutoff lines
    ax.vlines([lower_bound, upper_bound], ymin=0, ymax=[rv.pdf(lower_bound), rv.pdf(upper_bound)], color="C0")

    # Top interval arrow
    y_arrow = 1.08 * ymax
    ax.annotate("", xy=(upper_bound, y_arrow), xytext=(lower_bound, y_arrow),
                arrowprops=dict(arrowstyle="<->", color="C0", lw=1.8, shrinkA=0, shrinkB=0))
    ax.text((lower_bound + upper_bound) / 2, y_arrow + 0.03 * ymax,
            rf"$[{rv_name}_\ell,\ {rv_name}_u]$",
            ha="center", va="bottom")

    # Central probability label
    ax.text((lower_bound + upper_bound) / 2, 0.22 * ymax,
            r"$1-\alpha$", fontsize="large", ha="center", va="center")

    # Axis labels
    ax.set_ylabel(rf"$f_{{{dist_name}}}$")
    ax.set_xlabel("")

    # Only show the two cutoff ticks
    ax.set_xticks([lower_bound, upper_bound])
    ax.set_xticklabels([rf"${rv_name}_\ell$", rf"${rv_name}_u$"])
    ax.set_yticks([])

    # Limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 1.2 * ymax)

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Put rv_name at the right end of the x-axis
    ax.text(1.035, 0.055, rf"${rv_name}$", transform=ax.transAxes, ha="left", va="top")

    return ax
