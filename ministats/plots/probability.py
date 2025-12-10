import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..utils import savefigure


# Discrete random variables
################################################################################

def plot_pmf(rv, xlims=None, ylims=None, rv_name="X", ax=None, title=None, label=None):
    """
    Plot the PMF of the discrete random variable `rv` over the `xlims`.
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
        xmin = 0
        smax = rv.support()[1]
        xmax = rv.ppf(0.99999) if smax == np.inf else smax + 1
    xs = np.arange(xmin, xmax)

    # Compute the probability mass function and plot it
    fXs = rv.pmf(xs)
    fXs = np.where(fXs == 0, np.nan, fXs)  # set zero fXs to np.nan
    ax.stem(xs, fXs, basefmt=" ", label=label)
    ax.set_xticks(xs)
    ax.set_xlabel("$" + rv_name.lower() + "$")
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


def plot_cdf(rv, xlims=None, ylims=None, rv_name="X", ax=None, title=None, **kwargs):
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
    sns.lineplot(x=xs, y=FXs, ax=ax, **kwargs)

    # Set plot attributes
    ax.set_xlabel("$b$")
    ax.set_ylabel(f"$F_{{{rv_name}}}$")
    if ylims:
        ax.set_ylim(*ylims)
    if title and title.lower() == "auto":
        title = "Cumulative distribution function of the random variable " + rv.dist.name + str(rv.args)
    if title:
        ax.set_title(title, y=0, pad=-30)

    # return the axes
    return ax



# Continuous random variables
################################################################################

def plot_pdf(rv, xlims=None, ylims=None, rv_name="X", a=None, b=None, ax=None, title=None, **kwargs):
    """
    Plot the PDF of the continuous random variable `rv` over the `xlims`.
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
    ax.set_xlabel("$" + rv_name.lower() + "$")
    ax.set_ylabel(f"$f_{{{rv_name}}}$")
    if ylims:
        ax.set_ylim(*ylims)
    
    if a or b:
        # Highlight the area under fX between x=a and x=b
        if a is None:
            a = rv.support()[0]
        if b is None:
            b = rv.support()[1]

        mask = (xs > a) & (xs < b)
        ax.fill_between(xs[mask], y1=fXs[mask], alpha=0.2)
        ax.vlines([a], ymin=0, ymax=rv.pdf(a), linestyle="-", alpha=0.5)
        ax.vlines([b], ymin=0, ymax=rv.pdf(b), linestyle="-", alpha=0.5)

    if title and title.lower() == "auto":
        title = "Probability density function of the random variable " + rv.dist.name + str(rv.args)
    if title:
        ax.set_title(title, y=0, pad=-30)

    # return the axes
    return ax



# Joint distribution plots
################################################################################

def get_meshgrid_and_pos(xlims, ylims, ngrid):
    """
    Create two 1D grids with `ngrid` points in each dimension,
    then combine them using meshgrid and stack the results along a third dimension
    as required to evaluate multivariate probability density function.
    """
    xmin, xmax = xlims
    ymin, ymax = ylims
    xs = np.linspace(xmin, xmax, ngrid)
    ys = np.linspace(ymin, ymax, ngrid)
    X, Y = np.meshgrid(xs, ys)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    # ALT.
    # pos = np.dstack( (X, Y) )
    return X, Y, pos


def plot_joint_pdf_contourf(rvXY, xlims, ylims, ngrid=200, ax=None):
    """
    Filled contour plot of the bivariate joint distribution `rvXY`.
    """
    # Setup figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(7,4))
    else:
        fig = ax.figure
    # Compute the joint-probability density function values
    X, Y, pos = get_meshgrid_and_pos(xlims, ylims, ngrid)
    fXY = rvXY.pdf(pos)
    # Contour plot
    cax = ax.contourf(fXY,
                      origin='lower',
                      extent=(*xlims, *ylims),
                      levels=10,
                      cmap="Greys")
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    return ax


def plot_joint_pdf_contour(rvXY, xlims, ylims, ngrid=200, ax=None, levels=None):
    """
    Contour lines plot of the bivariate joint distribution `rvXY`.
    """
    # Setup figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(7,4))
    else:
        fig = ax.figure
    # Compute the joint-probability density function values
    X, Y, pos = get_meshgrid_and_pos(xlims, ylims, ngrid)
    fXY = rvXY.pdf(pos)
    # Contour plot
    cplot = ax.contour(X, Y, fXY,
                       origin='lower',
                       extent=(*xlims, *ylims),
                       levels=levels,
                       cmap="Greys")
    plt.clabel(cplot, inline=1, fontsize=9) #  fmt='%1.1f')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    return ax


def plot_joint_pdf_surface(rvXY, xlims, ylims, ngrid=200, fig=None, viewdict=None):
    """
    Surface plot of a bivariate joint distribution `rvXY`.
    https://stackoverflow.com/questions/38698277/plot-normal-distribution-in-3d
    """
    # Setup figure and axes
    if fig is None:
        fig = plt.figure(figsize=(7,7))
    ax = plt.axes(projection='3d')

    # Compute the joint-probability density function values
    X, Y, pos = get_meshgrid_and_pos(xlims, ylims, ngrid)
    fXY = rvXY.pdf(pos)

    # Generate the 3D surface plot
    ax.plot_surface(X, Y, fXY, cmap='Greys', linewidth=0)
    ax.set_box_aspect((xlims[1]-xlims[0], ylims[1]-ylims[0], 3))
    if viewdict is not None:
        ax.view_init(**viewdict)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$f_{XY}$')
    ax.set_zticks([])

    return ax




# Diagnostic plots (used in Section 2.7 Random variable generation)
################################################################################


def plot_epmf(data, xlims=None, ylims=None, name="xs", ax=None, title=None, label=None):
    """
    Plot the empirical pmf of the observations in  `data`.
    """
    # Setup figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,1.2))
    else:
        fig = ax.figure

    # Compute limits of plot
    if xlims:
        xmin, xmax = xlims
    else:
        extrax = 0.3 * max(data)  # extend further to the right
        xmin = int(min(data))
        xmax = int(max(data) + extrax)

    # Compute the probability mass function and plot it
    data = np.array(data)
    n = len(data)
    xs, counts = np.unique(data, return_counts=True)
    fxs = counts / n

    label = f"epmf({name})"
    ax.stem(xs, fxs, basefmt=" ", label=label)
    ax.set_xticks(range(xmin,xmax))
    # ax.set_ylim([-0.01,0.22])
    # ax.set_yticks([0, 0.1, 0.2])
    ax.set_xlabel("$b$")
    ax.set_ylabel(f"$f_{{\\text{{{name}}}}}$")
    # ax.set_xticks(xs)
    # ax.set_xlabel(rv_name.lower())
    # ax.set_ylabel(f"$f_{{{rv_name}}}$")
    if ylims:
        ax.set_ylim(*ylims)
    if label:
        ax.legend()

    if title and title.lower() == "auto":
        title = "Empirical probability mass function of the data " + name
    if title:
        ax.set_title(title, y=0, pad=-30)

    # return the axes
    return ax


def plot_ecdf(data, xlims=None, ylims=None, name="xs", ax=None, title=None, label=None):
    """
    Plot the empirical CDF of the observations in  `data`.
    """
    # Setup figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,2))
    else:
        fig = ax.figure

    # Compute limits of plot
    if xlims:
        xmin, xmax = xlims
    else:
        extrax = 0.3 * max(data)  # extend further to the right
        xmin = int(min(data))
        xmax = int(max(data) + extrax)
    

    def ecdf(data, b):
        sorted_data = np.sort(data)
        count = sum(sorted_data <= b)  # num. of obs. <= b
        return count / len(data)       # proportion of total

    # Compute the probability mass function and plot it
    data = np.array(data)
    bs = np.linspace(0, xmax, 1000)
    Fxs = [ecdf(data,b) for b in bs]
    # label = f"eCDF({name})"
    ax = sns.lineplot(x=bs, y=Fxs, drawstyle='steps-post', label=label)
    ax.set_xlabel("$b$")
    ax.set_ylabel(f"$F_{{\\text{{{name}}}}}$")
    ax.set_xlim([0, xmax])
    ax.set_xticks(range(0,xmax))
    if ylims:
        ax.set_ylim(*ylims)
    if label:
        ax.legend()

    if title and title.lower() == "auto":
        title = "Empirical cumulative distribution function of the data " + name
    if title:
        ax.set_title(title, y=0, pad=-30)

    # return the axes
    return ax





def qq_plot(data, dist, ax=None, xlims=None, filename=None, **kwargs):
    """
    This function qq_plot tries to imitate the behaviour of the function `qqplot`
    defined in `statsmodels.graphics.api`. Usage: `qqplot(data, dist=norm(0,1), line='q')`. See:
    https://github.com/statsmodels/statsmodels/blob/main/statsmodels/graphics/gofplots.py#L912-L919
    """
    # Setup figure and axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Add the Q-Q scatter plot
    n = len(data)
    qs = np.linspace(1/(n+1), n/(n+1), n)
    xs = dist.ppf(qs)
    sorted_data = np.sort(data)
    ys = sorted_data
    # ALT. ys = np.quantile(data, qs, method="inverted_cdf")
    sns.scatterplot(x=xs, y=ys, ax=ax, alpha=0.7, **kwargs)

    # Compute the parameters m and b for the diagonal line
    xq25, xq75 = dist.ppf([0.25, 0.75])
    yq25, yq75 = np.quantile(data, [0.25, 0.75])
    m = (yq75 - yq25) / (xq75 - xq25)
    b = yq25 - m * xq25
    # add the line  y = m*x+b  to the plot
    linexs = np.linspace(min(xs), max(xs))
    lineys = m * linexs + b
    sns.lineplot(x=linexs, y=lineys, ax=ax, color="r")

    # Handle keyword arguments
    if xlims:
        ax.set_xlim(xlims)
    if filename:
        savefigure(ax, filename)

    return ax
