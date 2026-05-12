from matplotlib.patches import Rectangle
from matplotlib import gridspec
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


def plot_pmf_series(fX, rv_name="X", ax=None, orientation="vertical"):
    """
    Plot the PMF of the discrete RV stored in the pandas series `fX`.
    """
    # Setup figure and axes
    if ax is None:
        _, ax = plt.subplots()    
    x_labels = list(fX.index)
    xs = range(len(x_labels))
    fXs = fX.values
    ax.stem(xs, fXs, basefmt=" ", orientation=orientation)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)    
    ax.set_xlabel("$" + rv_name.lower() + "$")
    ax.set_ylabel(f"$f_{{{rv_name}}}$")
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




# Discrete joint distribution plots
################################################################################

def plot_joint_pmf_stems(jpmf, flabel=None, ax=None, zmax=None):
    """
    Plot a joint PMF stored in the DataFrame `jpmf` as a 3D stem plot.
    The random variable names can be specified as the `name` attribute
    on the `jpmf.index` and `jpmf.columns` indices.
    """
    # Setup figure and axes
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    else:
        fig = ax.figure

    x_labels = list(jpmf.columns)
    y_labels = list(jpmf.index)

    x_positions = {label: i for i, label in enumerate(x_labels)}
    y_positions = {label: i for i, label in enumerate(y_labels)}


    xs, ys, fXYs = [], [], []
    for row_label in y_labels:
        for col_label in x_labels:
            xs.append(x_positions[col_label])
            ys.append(y_positions[row_label])
            fXYs.append(float(jpmf.loc[row_label, col_label]))

    ax.stem(xs, ys, fXYs, basefmt=" ")

    # X-axis = columns
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    x_rv_name = jpmf.columns.name
    ax.set_xlabel(f"${x_rv_name.lower()}$" if x_rv_name else None)

    # Y-axis = index
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    y_rv_name = jpmf.index.name
    ax.set_ylabel(f"${y_rv_name.lower()}$" if y_rv_name else None)

    # Z-axis 
    if flabel is None:
        row_name = jpmf.index.name or "Y"
        col_name = jpmf.columns.name or "X"
        joint_subscript = col_name + row_name
        flabel = rf"$f_{{{joint_subscript}}}$"
    # OFF due to https://github.com/matplotlib/matplotlib/issues/28117
    # ax.set_zlabel(flabel)
    # /OFF.  See workaround in this comment:
    # github.com/matplotlib/matplotlib/issues/28117#issuecomment-4137120000
    if zmax is None:
        zmax = 1.2 * np.max(fXYs)
    ax.set_zlim(0, zmax)

    return ax




def plot_joint_pmf_balloons(jpmf, flabel="$f_XY$", ax=None,
                            size_exponent=1.5, cell_fraction=0.7,
                            highlight=None):
    """
    Plot the joint PMF in the DataFrame `jpmf` as circles of different sizes.
    We'll plot the rows along the y-axis, and columns on the x-axis.
    This is called a "balloon plot" or "bubble plot".

    The size of the circles are determined by:
    - `size_exponent`: contrast in circle sizes.
    - `cell_fraction`: max size as a fraction of the grid-cell size.
    A subset of the sample space can be highlighted by passing in
    a list of values to the `highlight` option:
    - `highlight=[(col,row), ...]`: highlights individual cells
    - `highlight=[[(col1,row1), (col2,row2)], ...]`: highlights all
      cells in a box with coreners (col1,row1) and (col2,row2).
    """
    # STYLE CONSTANTS
    HIGHLIGHT_FACECOLOR = "C0"
    HIGHLIGHT_EDGECOLOR = "black"
    HIGHLIGHT_ALPHA = 0.25
    NON_HIGHLIGHT_ALPHA = 0.1
    HIGHLIGHT_DOT_ALPHA = 1.0

    # Setup figure and axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    x_labels = list(jpmf.columns)
    y_labels = list(jpmf.index)

    x_positions = {label: i for i, label in enumerate(x_labels)}
    y_positions = {label: i for i, label in enumerate(y_labels)}

    def _axis_pos(key, labels, positions, axis_name):
        """
        Resolve key as label first, then as zero-based integer position.
        """
        if key in positions:
            return positions[key]
        if isinstance(key, (int, np.integer)) and 0 <= key < len(labels):
            return int(key)
        raise KeyError(f"{axis_name} key {key!r} is not a label or valid position.")

    def _cell_pos(cell):
        """
        Resolve a (col, row) cell to numeric (x, y) plot positions.
        """
        col_key, row_key = cell
        x = _axis_pos(col_key, x_labels, x_positions, "column")
        y = _axis_pos(row_key, y_labels, y_positions, "row")
        return x, y

    xs, ys, ps = [], [], []
    for row_label in y_labels:
        for col_label in x_labels:
            xs.append(x_positions[col_label])
            ys.append(y_positions[row_label])
            ps.append(float(jpmf.loc[row_label, col_label]))

    # X-axis = columns
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_xlim(-0.5, len(x_labels) - 0.5)
    x_rv_name = jpmf.columns.name
    ax.set_xlabel(f"${x_rv_name.lower()}$" if x_rv_name else None)

    # Y-axis = index
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_ylim(-0.5, len(y_labels) - 0.5)
    ax.invert_yaxis()
    y_rv_name = jpmf.index.name
    ax.set_ylabel(f"${y_rv_name.lower()}$" if y_rv_name else None)

    # Track which cells are highlighted
    highlighted_cells = set()

    if highlight is not None:
        for item in highlight:

            # Box syntax: [(col1, row1), (col2, row2)]
            is_box = (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and all(isinstance(corner, (list, tuple)) and len(corner) == 2
                        for corner in item)
            )

            if is_box:
                (col1, row1), (col2, row2) = item

                x1, y1 = _cell_pos((col1, row1))
                x2, y2 = _cell_pos((col2, row2))

                xmin_i, xmax_i = sorted([x1, x2])
                ymin_i, ymax_i = sorted([y1, y2])

                for x in range(xmin_i, xmax_i + 1):
                    for y in range(ymin_i, ymax_i + 1):
                        highlighted_cells.add((x, y))

                rect = Rectangle(
                    (xmin_i - 0.5, ymin_i - 0.5),
                    xmax_i - xmin_i + 1,
                    ymax_i - ymin_i + 1,
                    facecolor=HIGHLIGHT_FACECOLOR,
                    edgecolor=HIGHLIGHT_EDGECOLOR,
                    alpha=HIGHLIGHT_ALPHA,
                    linewidth=1.5,
                    zorder=0,
                )
                ax.add_patch(rect)

            # Cell syntax: (col, row)
            else:
                x, y = _cell_pos(item)
                highlighted_cells.add((x, y))

                rect = Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    facecolor=HIGHLIGHT_FACECOLOR,
                    edgecolor=HIGHLIGHT_EDGECOLOR,
                    alpha=HIGHLIGHT_ALPHA,
                    linewidth=1.5,
                    zorder=0,
                )
                ax.add_patch(rect)

    ax.grid(True, alpha=0.3)

    # Need the renderer to know the axes size on screen
    fig.canvas.draw()

    # Axes size in points
    bbox = ax.get_window_extent()
    ax_width_pts = bbox.width * 72 / fig.dpi
    ax_height_pts = bbox.height * 72 / fig.dpi

    # Approximate grid-cell size in points
    cell_width_pts = ax_width_pts / max(len(x_labels), 1)
    cell_height_pts = ax_height_pts / max(len(y_labels), 1)

    # Safe maximum marker diameter
    max_diameter_pts = cell_fraction * min(cell_width_pts, cell_height_pts)

    # For circular markers, s is area in points^2
    max_area = np.pi * (max_diameter_pts / 2) ** 2
    pmax = max(ps)

    if pmax > 0:
        sizes = [max_area * (p / pmax) ** size_exponent for p in ps]
    else:
        sizes = [0 for _ in ps]

    if highlighted_cells:
        xs_hi, ys_hi, s_hi = [], [], []
        xs_lo, ys_lo, s_lo = [], [], []

        for x, y, s in zip(xs, ys, sizes):
            if (x, y) in highlighted_cells:
                xs_hi.append(x)
                ys_hi.append(y)
                s_hi.append(s)
            else:
                xs_lo.append(x)
                ys_lo.append(y)
                s_lo.append(s)

        # Non-highlighted circles: pale
        ax.scatter(xs_lo, ys_lo, s=s_lo, linewidths=1, color="C0", alpha=NON_HIGHLIGHT_ALPHA, zorder=2)

        # Highlighted circles: full opacity
        ax.scatter(xs_hi, ys_hi, s=s_hi, linewidths=1, color="C0", alpha=HIGHLIGHT_DOT_ALPHA, zorder=3)

    else:
        ax.scatter(xs, ys, s=sizes, linewidths=1, zorder=2)

    return ax



def plot_joint_pmf_and_marginals(jpmfXY, fig=None):
    """
    Plot the joint PMF `f_XY` and it marginals `f_X` and `f_Y`.
    """
    # Setup figure and axes
    if fig is None:
        fig = plt.figure(figsize=(7,4))

    # Figure grid
    gs = gridspec.GridSpec(2, 2, width_ratios=[6,1], height_ratios=[4,1], hspace=0.1, wspace=0.2)

    # Dot plot of f_XY
    ax = plt.subplot(gs[0,0])
    ax = plot_joint_pmf_balloons(jpmfXY, ax=ax)
    ax.tick_params(labelbottom=False)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.text(-0.48, -0.45, "$f_{XY}$", va="top", fontsize="x-large")

    # The marginal f_X (bottom)
    fX = jpmfXY.sum(axis=0)
    axb = plt.subplot(gs[1,0], sharex=ax, frameon=False)
    plot_pmf_series(fX, rv_name="X", ax=axb)
    axb.tick_params(labelleft=False)
    axb.set_yticks([0,0.1,0.2,0.3])
    axb.set_yticklabels([0,0.1,0.2,0.3])
    axb.set_xlabel(None)
    axb.set_ylabel(None)
    axb.text(-0.48, 0.05, "$f_{X}$", fontsize="x-large")

    # The marginal f_Y (right)
    fY = jpmfXY.sum(axis=1)
    axr = plt.subplot(gs[0,1], sharey=ax, frameon=False)
    plot_pmf_series(fY, rv_name="Y", ax=axr, orientation="horizontal")
    axr.set_xlim(0,0.4)
    axr.set_xlabel(None)
    axr.set_ylabel(None)
    axr.set_xticks([0,0.1,0.2,0.3,0.4])
    axr.tick_params(labelbottom=False)
    axr.text(0.01, -0.4, "$f_{Y}$", rotation=270, fontsize="x-large")

    return fig



def plot_joint_pmf_and_conditional(jpmfXY, given="y", fig=None):
    """
    Plot the joint PMF `f_XY` and the conditional `f_X|Y`.
    """
    given = given.lower()
    assert given in ["x", "y"], "must specify given='x' or given='y'."
    # Setup figure and axes
    if fig is None:
        fig = plt.figure(figsize=(7.2,3))

    # Figure grid
    axs = fig.subplots(ncols=2, nrows=2, width_ratios=[4,3], height_ratios=[1,1],
                       gridspec_kw=dict(hspace=0.7, wspace=0.4))
    gs = axs[0][0].get_gridspec()
    # Remove the axes (we'll recreate manually below)
    axs[0][0].remove()
    axs[1][0].remove()
    axs[0][1].remove()
    axs[1][1].remove()

    # (a) Dot plot of f_XY
    ax = fig.add_subplot(gs[0:, 0])
    if given == "y":
        plot_joint_pmf_balloons(jpmfXY, ax=ax, highlight=[[(1,"b"),(5,"b")]])
    else:
        plot_joint_pmf_balloons(jpmfXY, ax=ax, highlight=[[(5,"a"),(5,"d")]])
    ax.set_xlabel("$x$", fontsize=7)
    ax.set_ylabel("$y$", fontsize=7)
    ax.set_title("(a) Joint probability mass function $f_{XY}$", fontsize=12)

    # (b) Slice through f_XY at y=b  (top right)
    ax1 = plt.subplot(gs[0,1], frameon=False)
    if given == "y":
        fXY_at_y = jpmfXY.loc["b",:]
        plot_pmf_series(fXY_at_y, rv_name="XY", ax=ax1)
        ax1.set_title("(b) Horizontal slice though $f_{XY}$ at $y=b$           ", fontsize=11)
        ax1.set_yticks([0,0.02,0.04,0.06,0.08,0.1,0.12])
    else:
        fXY_at_x = jpmfXY.loc[:,5]
        plot_pmf_series(fXY_at_x, rv_name="XY", ax=ax1)
        ax1.set_title("(b) Vertical slice though $f_{XY}$ at $x=5$           ", fontsize=11)
        ax1.set_yticks([0,0.02,0.04,0.06])
    ax1.set_xlabel(None)

    # Arrow between (a) and (b)
    arrowprops2 = dict(facecolor='C0', shrink=0.005, width=4, headwidth=10, headlength=12,
                       connectionstyle="arc3,rad=0.1")
    ax.annotate("", xytext=(4.5, 1), xy=(5.5, 0.6),  arrowprops=arrowprops2, annotation_clip=False)

    # Conditional f_X|Y at y=b (bottom right)
    ax2 = plt.subplot(gs[1,1], frameon=False, sharex=ax1)
    if given == "y":
        fYgivenX = jpmfXY.loc["b",:] / jpmfXY.loc["b",:].sum()
        plot_pmf_series(fYgivenX, rv_name="X|Y", ax=ax2)
        ax2.set_title("(c) Conditional distribution $f_{X|Y}(x|b)$", fontsize=11, loc="right")
        ax2.set_xlabel("$x$", fontsize=7)
        ax2.set_yticks([0,0.1,0.2,0.3,0.4])
    else:
        fXgivenY = jpmfXY.loc[:,5] / jpmfXY.loc[:,5].sum()
        plot_pmf_series(fXgivenY, rv_name="Y|X", ax=ax2)
        ax2.set_title("(c) Conditional distribution $f_{Y|X}(y|5)$", fontsize=11, loc="right")
        ax2.set_xlabel("$y$", fontsize=7)
        ax2.set_yticks([0,0.1,0.2,0.3,0.4,0.5])

    return fig




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




# Continuous joint distribution plots
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


def plot_joint_pdf_contourf(rvXY, xlims, ylims, ngrid=200, ax=None, highlight=None):
    """
    Filled contour plot of the bivariate joint distribution `rvXY`.
    Use the option `highlight=[[(x1,y1),(x2,y2)], ... ]` to highlight
    rectangular events with corners (x1,y1) and (x2,y2).
    """
    # Setup figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(7,4))
    else:
        fig = ax.figure

    # Compute the joint-probability density function values
    X, Y, pos = get_meshgrid_and_pos(xlims, ylims, ngrid)
    fXY = rvXY.pdf(pos)

    LEVELS = 10
    CMAP = "Greys"

    # Base contour plot: muted if highlights are requested
    base_alpha = 0.3 if highlight is not None else 1.0
    ax.contourf(X, Y, fXY, levels=LEVELS, cmap=CMAP, alpha=base_alpha)

    # Overlay highlighted regions in full colour
    if highlight is not None:
        for (x1, y1), (x2, y2) in highlight:
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])
            mask = (xmin <= X) & (X <= xmax) & (ymin <= Y) & (Y <= ymax)
            fXY_highlight = np.ma.masked_where(~mask, fXY)
            ax.contourf(X, Y, fXY_highlight, levels=LEVELS, cmap=CMAP, alpha=1.0)
            rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             facecolor="C0", edgecolor="k", alpha=0.25,
                             linewidth=1.5, zorder=10)
            ax.add_patch(rect)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
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
    ax.plot_surface(X, Y, fXY,
                    color="white", edgecolor="black", shade=False,
                    linewidth=0.2, rcount=40, ccount=40)
    ax.set_box_aspect((xlims[1]-xlims[0], ylims[1]-ylims[0], 3))
    if viewdict is not None:
        ax.view_init(**viewdict)
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    # ax.set_zlabel('$f_{XY}$')
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
