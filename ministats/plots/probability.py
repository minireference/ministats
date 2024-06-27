import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



# Discrete random variables
################################################################################

def plot_pmf(rv, xlims=None, ylims=None, rv_name="X", ax=None, title=None, label=None):
    """
    Plot the pmf of the discrete random variable `rv` over the `xlims`.
    """
    # Setup figure and axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Compute limits of plot
    if xlims:
        xmin, xmax = xlims
    else:
        xmin, xmax = rv.ppf(0.000000001), rv.ppf(0.99999)
    xs = np.arange(xmin, xmax)

    # Compute the probability mass function and plot it
    fXs = rv.pmf(xs)
    fXs = np.where(fXs == 0, np.nan, fXs)  # set zero fXs to np.nan
    ax.stem(fXs, basefmt=" ", label=label)
    ax.set_xticks(xs)
    ax.set_xlabel(rv_name.lower())
    ax.set_ylabel(f"$f_{{{rv_name}}}$")
    if ylims:
        ax.set_ylim(*ylims)
    if label:
        ax.legend()

    if title and title.lower() == "auto":
        title = "Probability mass function of the random variable " + rv.dist.name + str(rv.args)
    if title:
        ax.set_title(title, y=0, pad=-30)

    # return the axes
    return ax


def plot_cdf(rv, xlims=None, ylims=None, rv_name="X", ax=None, title=None, label=None):
    """
    Plot the CDF of the random variable `rv` (discrete or continuous) over the `xlims`.
    """
    # Setup figure and axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Compute limits of plot
    if xlims:
        xmin, xmax = xlims
    else:
        xmin, xmax = rv.ppf(0.000000001), rv.ppf(0.99999)
    xs = np.linspace(xmin, xmax, 1000)

    # Compute the CDF and plot it
    FXs = rv.cdf(xs)
    sns.lineplot(x=xs, y=FXs, ax=ax)

    # Set plot attributes
    ax.set_xlabel(rv_name.lower())
    ax.set_ylabel(f"$F_{{{rv_name}}}$")
    if ylims:
        ax.set_ylim(*ylims)
    if label:
        ax.legend()
    if title and title.lower() == "auto":
        title = "Cumulative distribution function of the random variable " + rv.dist.name + str(rv.args)
    if title:
        ax.set_title(title, y=0, pad=-30)

    # return the axes
    return ax



# Continuous random variables
################################################################################

def plot_pdf(rv, xlims=None, ylims=None, rv_name="X", ax=None, title=None, **kwargs):
    """
    Plot the pdf of the continuous random variable `rv` over the `xlims`.
    """
    # Setup figure and axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Compute limits of plot
    if xlims:
        xmin, xmax = xlims
    else:
        xmin, xmax = rv.ppf(0.000000001), rv.ppf(0.99999)
    xs = np.linspace(xmin, xmax, 1000)

    # Compute the probability density function and plot it
    fXs = rv.pdf(xs)
    sns.lineplot(x=xs, y=fXs, ax=ax, **kwargs)
    ax.set_xlabel(rv_name.lower())
    ax.set_ylabel(f"$f_{{{rv_name}}}$")
    if ylims:
        ax.set_ylim(*ylims)

    if title and title.lower() == "auto":
        title = "Probability density function of the random variable " + rv.dist.name + str(rv.args)
    if title:
        ax.set_title(title, y=0, pad=-30)

    # return the axes
    return ax





# Diagnostic plots (used in Section 2.7 Random variable generation)
################################################################################
# The function qq_plot tries to imitate the behaviour of the function `qqplot`
# defined in `statsmodels.graphics.api`. Usage: `qqplot(data, dist=norm(0,1), line='q')`. See:
# https://github.com/statsmodels/statsmodels/blob/main/statsmodels/graphics/gofplots.py#L912-L919
#
# TODO: figure out how to plot all of data correctly: currently missing first and last data point

def qq_plot(data, dist, ax=None, xlims=None, filename=None):
    # Setup figure and axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Add the Q-Q scatter plot
    qs = np.linspace(0, 1, len(data)+1)
    xs = dist.ppf(qs)
    ys = np.quantile(data, qs)
    sns.scatterplot(x=xs, y=ys, ax=ax, alpha=0.2)

    # Compute the parameters m and b for the diagonal
    xq25, xq75 = dist.ppf([0.25, 0.75])
    yq25, yq75 = np.quantile(data, [0.25,0.75])
    m = (yq75-yq25)/(xq75-xq25)
    b = yq25 - m * xq25
    # add the line  y = m*x+b  to the plot
    linexs = np.linspace(min(xs[1:]),max(xs[:-1]))
    lineys = m*linexs + b
    sns.lineplot(x=linexs, y=lineys, ax=ax, color="r")

    # Handle keyword arguments
    if xlims:
        ax.set_xlim(xlims)
    if filename:
        basename = filename.replace('.pdf','').replace('.png','')
        fig.tight_layout()
        fig.savefig(basename + '.pdf', dpi=300, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(basename + '.png', dpi=300, bbox_inches="tight", pad_inches=0.02)

    return ax
