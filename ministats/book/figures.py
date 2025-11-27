import math

from matplotlib import gridspec
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.contingency import margins
from scipy.stats import t as tdist
import seaborn as sns
import xarray as xr

from ..plots import nicebins
from ..plots.figures import calc_prob_and_plot
from ..plots.figures import calc_prob_and_plot_tails
from ..plots.probability import get_meshgrid_and_pos


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



# Section 2.4: Multivariate distributions
################################################################################

def plot_joint_pdf_and_marginals(rvXY, xlims, ylims, ngrid=200, fig=None):
    """
    Contour plot of a bivariate joint distribution $f_XY$ (`rvXY.pdf`) while
    also showing the marginals $f_X$ and $f_Y$ on the sides.
    """
    # Setup figure and axes
    if fig is None:
        fig = plt.figure(figsize=(7,4))

    # Compute the joint-probability density function values
    X, Y, pos = get_meshgrid_and_pos(xlims, ylims, ngrid)
    fXY = rvXY.pdf(pos)

    # Figure grid
    gs = gridspec.GridSpec(2, 2, width_ratios=[6,1], height_ratios=[1,4])

    # Contour plot of f_XY
    ax = plt.subplot(gs[1,0])    
    cax = ax.contourf(fXY, origin = 'lower',
                      extent=(*xlims, *ylims),
                      levels=12,
                      cmap="Greys")
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.text(5, 6, "$f_{XY}$", fontsize="x-large")

    # Compute marginal distributions
    fYm, fXm = margins(fXY)
    dx = (xlims[1] - xlims[0]) / (ngrid - 1)
    dy = (ylims[1] - ylims[0]) / (ngrid - 1)
    fX = fXm.flatten() * dy
    fY = fYm.flatten() * dx

    # The marginal f_X (top)
    xs = X[0]
    axt = plt.subplot(gs[0,0], sharex=ax, frameon=False, xlim=xlims, ylim=(0, 1.1*fX.max()))
    axt.plot(xs, fX, color = 'black')
    axt.fill_between(xs, 0, fX, alpha=.5, color = 'gray')
    axt.tick_params(labelbottom=False)
    axt.tick_params(labelleft=False)
    axt.text(5, 0.08, "$f_{X}$", fontsize="x-large")

    # The marginal f_Y (right)
    ys = Y[:,0]
    axr = plt.subplot(gs[1,1], sharey=ax, frameon=False, xlim=(0, 1.05*fY.max()), ylim=ylims)
    axr.plot(0*np.ones_like(ys), ys)
    axr.plot(fY, ys, color = 'black')
    axr.fill_betweenx(ys, 0, fY, alpha=0.5, color="gray")
    axr.tick_params(labelbottom=False)
    axr.tick_params(labelleft=False)
    axr.text(0.3,3.2, "$f_{Y}$", fontsize="x-large")

    return fig


def find_nearest1(array, value):
    """
    Find the index of the `array` entry that is closest to `value`.
    Helper function used by `plot_slices_through_joint_pdf` et al.
    """
    idx, _ = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx


def polygon_under_graph(xs, ys):
    """
    Construct the vertex list that defines the polygon filling the space
    under the curve that passes through the points `[(x,y) in zip(xs,ys)]`.
    Helper function used by `plot_slices_through_joint_pdf` et al.
    """
    return [(xs[0], 0.), *zip(xs, ys), (xs[-1], 0.)]


def plot_slices_through_joint_pdf(rvXY, xlims, ylims, xcuts, ngrid=500, fig=None):
    """
    Plot slices through the joint distribution $f_XY$ at the x-values in `xcuts`.
    """
    # Setup figure and axes
    if fig is None:
        fig = plt.figure(figsize=(7,4))
    ax = fig.add_subplot(projection='3d')

    # Compute the joint-probability density function values
    X, Y, pos = get_meshgrid_and_pos(xlims, ylims, ngrid)
    xs, ys = X[0], Y[:,0]
    fXY = rvXY.pdf(pos)

    # The entry `verts[i]` is a list of (x,y) pairs defining polygon `i`
    verts = []
    for xcut in xcuts:
        xidx = find_nearest1(xs, xcut)
        fXY_at_xcut = fXY[xidx,:]
        vert = polygon_under_graph(ys, fXY_at_xcut)
        verts.append(vert)
    
    # Plot polygons
    facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(verts)))
    poly = PolyCollection(verts, facecolors=facecolors, alpha=.6)
    ax.add_collection3d(poly, zs=xcuts, zdir='x')
    ax.set_box_aspect((9, 5, 4))
    zmax = 0.06
    ax.set(xlim=xlims, ylim=ylims, zlim=(0, zmax), xlabel='$x$', ylabel='$y$', zlabel='probability')
    ax.set_xticks(range(4,17,1))

    return fig


def plot_conditional_fYgivenX(rvXY, xlims, ylims, xcuts, ngrid=500, fig=None):
    """
    Plot the conditional distribution $f_Y|X$ at the x-values in `xcuts`.
    """
    # Setup figure and axes
    if fig is None:
        fig = plt.figure(figsize=(7,4))
    ax = fig.add_subplot(projection='3d')

    # Compute the joint-probability density function values
    X, Y, pos = get_meshgrid_and_pos(xlims, ylims, ngrid)
    xs, ys = X[0], Y[:,0]
    fXY = rvXY.pdf(pos)

    # hack to find normalizing height
    xmid = xcuts[len(xcuts)//2]
    xmididx = find_nearest1(xs, xmid)
    fXY_at_xmid = fXY[xmididx,:]
    fYgiven_xmid = fXY_at_xmid / np.sum(fXY_at_xmid)
    maxfYgiven_xmid = max(fYgiven_xmid)

    # The entry `verts[i]` is a list of (x,y) pairs defining polygon `i`
    verts = []
    for xcut in xcuts:
        xidx = find_nearest1(xs, xcut)
        fXY_at_xcut = fXY[xidx,:]
        fYgiven_xcut = fXY_at_xcut / np.sum(fXY_at_xcut)
        # hack to normalize height
        zscale = max(fYgiven_xcut) / maxfYgiven_xmid
        fYgiven_xcut = fYgiven_xcut / zscale
        # /hack to normalize height
        vert = polygon_under_graph(ys, fYgiven_xcut)
        verts.append(vert)

    # Plot polygons
    facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(verts)))
    poly = PolyCollection(verts, facecolors=facecolors, alpha=.7)
    ax.add_collection3d(poly, zs=xcuts, zdir='x')
    ax.set_box_aspect((9, 5, 4))
    zmax = 0.006
    ax.set(xlim=xlims, ylim=ylims, zlim=(0, zmax), xlabel='$x$', ylabel='$y$', zlabel='probability');
    ax.set_xticks(range(4, 17, 1))

    return fig




# Section 2.4: Bulk and tails of a continuous distribution
################################################################################


def bulk_of_pdf_panel(rvX, rv_name, xlims, xticks=None, ns=[1,2,3], fig=None):
    """
    Print a 1x3 panel figure highlighting the probability mass that lies within
    `ns` standard deviations from the mean of the random variable `rvX`.
    """
    if fig is None:
        fig, axs = plt.subplots(1, 3, figsize=(9.2,2), sharey=True)
    else:
        axs = fig.subplots(1, 3, sharey=True)

    muX = rvX.mean()    # mean of the random variable rvX
    sigmaX = rvX.std()  # standard deviation of rvX

    for i, n in enumerate(ns):
        ax = axs[i]
        bulk_interval = [muX - n*sigmaX, muX + n*sigmaX]
        letter = ["a", "b", "c"][i]
        mu = "\\mu_" + rv_name
        sigma = "\\sigma_" + rv_name
        if n == 1:
            title = f"({letter}) Pr($\\{{{mu}-{sigma} \\leq {rv_name} \\leq {mu}+{sigma}\\}}$)"
        else:
            title = f"({letter}) Pr($\\{{{mu}-{n}{sigma} \\leq {rv_name} \\leq {mu}+{n}{sigma}\\}}$)"
        calc_prob_and_plot(rvX, *bulk_interval, xlims=xlims, ax=ax, title=title)
        if xticks:
            ax.set_xticks(xticks)

    return fig



def tails_of_pdf_panel(rvX, rv_name, xlims, xticks=None, ns=[1,2,3], fig=None):
    """
    Print a 1x3 panel figure highlighting the probability mass that lies within
    `ns` standard deviations from the mean of the random variable `rvX`.
    """
    if fig is None:
        fig, axs = plt.subplots(1, 3, figsize=(9.2,2), sharey=True)
    else:
        axs = fig.subplots(1, 3, sharey=True)

    muX = rvX.mean()    # mean of the random variable rvX
    sigmaX = rvX.std()  # standard deviation of rvX

    for i, n in enumerate(ns):
        ax = axs[i]
        x_l = muX - n * sigmaX
        x_r = muX + n * sigmaX
        letter = ["a", "b", "c"][i]
        mu = "\\mu_" + rv_name
        sigma = "\\sigma_" + rv_name
        if n == 1:
            title = f"({letter}) Pr($\\{{{rv_name} \\leq {mu}-{sigma}\\}} \\cup \\{{{rv_name} \\geq {mu}+{sigma}\\}}$)"
        else:
            title = f"({letter}) Pr($\\{{{rv_name} \\leq {mu}-{n}{sigma}\\}} \\cup \\{{{rv_name} \\geq {mu}+{n}{sigma}\\}}$)"
        calc_prob_and_plot_tails(rvX, x_l, x_r, xlims=xlims, ax=ax, title=title)
        if xticks:
            ax.set_xticks(xticks)

    return fig














# Section 3.4: Tails of a sampling distributions
################################################################################

def plot_panel_pvalue_theta0_tails(figsize=(9,2.3)):
    rvT = tdist(df=9)
    xs = np.linspace(-4, 4, 1000)
    ys = rvT.pdf(xs)

    with plt.rc_context({"figure.figsize":figsize}), sns.axes_style("ticks"):
        fig, (ax3, ax1, ax2) = plt.subplots(1,3)
        ax3.set_ylabel(r"$f_{\widehat{\Theta}_0}$")

        # RIGHT
        title = '(a) right-tailed test'
        ax3.set_title(title, fontsize=13)#, y=-0.26)
        sns.lineplot(x=xs, y=ys, ax=ax3)
        ax3.set_xlim(-4, 4)
        ax3.set_ylim(0, 0.42)
        ax3.set_xticks([0,2])
        ax3.set_xticklabels([])
        ax3.set_yticks([])
        ax3.spines[['right', 'top']].set_visible(False)

        # highlight the right tail
        mask = (xs > 2)
        ax3.fill_between(xs[mask], y1=ys[mask], alpha=0.6, facecolor="red")
        ax3.vlines([2], ymin=0, ymax=rvT.pdf(2), linestyle="-", alpha=0.5, color="red")
        ax3.text(2, -0.03, r"$\hat{\theta}_{\mathbf{x}}$", va="top", ha="center")
        ax3.text(0, -0.04, r"$\theta_0$", va="top", ha="center")


        # LEFT
        title = '(b) left-tailed test'
        ax1.set_title(title, fontsize=13) #, y=-0.26)
        sns.lineplot(x=xs, y=ys, ax=ax1)
        ax1.set_xlim(-4, 4)
        ax1.set_ylim(0, 0.42)
        ax1.set_xticks([-2,0])
        ax1.set_xticklabels([])
        ax1.set_yticks([])
        ax1.spines[['left', 'right', 'top']].set_visible(False)

        # highlight the left tail
        mask = (xs < -2)
        ax1.fill_between(xs[mask], y1=ys[mask], alpha=0.6, facecolor="red")
        ax1.vlines([-2], ymin=0, ymax=rvT.pdf(-2), linestyle="-", alpha=0.5, color="red")
        ax1.text(-2, -0.03, r"$\hat{\theta}_{\mathbf{x}}$", va="top", ha="center")
        ax1.text(0, -0.04, r"$\theta_0$", va="top", ha="center")

        # TWO-TAILED
        title = '(c) two-tailed test'
        ax2.set_title(title, fontsize=13)#, y=-0.26)
        sns.lineplot(x=xs, y=ys, ax=ax2)
        ax2.set_xlim(-4, 4)
        ax2.set_ylim(0, 0.42)
        ax2.set_xticks([-2,0,2])
        ax2.set_xticklabels([])
        ax2.set_yticks([])
        ax2.spines[['left', 'right', 'top']].set_visible(False)

        # highlight the left and right tails
        mask = (xs < -2)
        ax2.fill_between(xs[mask], y1=ys[mask], alpha=0.6, facecolor="red")
        ax2.vlines([-2], ymin=0, ymax=rvT.pdf(-2), linestyle="-", alpha=0.5, color="red")
        # ax2.text(-2, -0.03, r"$-|\hat{\theta}_{\mathbf{x}}|$", va="top", ha="center")
        ax2.text(-2, -0.03, r"$\theta_0$-dev", va="top", ha="center")
        mask = (xs > 2)
        ax2.fill_between(xs[mask], y1=ys[mask], alpha=0.6, facecolor="red")
        ax2.vlines([2], ymin=0, ymax=rvT.pdf(2), linestyle="-", alpha=0.5, color="red")
        ax2.text(2, -0.03, r"$\theta_0$+dev", va="top", ha="center")
        ax2.text(0, -0.04, r"$\theta_0$", va="top", ha="center")

    return fig



def plot_panel_pvalue_t_tails(figsize=(9,2.3)):    
    rvT = tdist(df=9)
    xs = np.linspace(-4, 4, 1000)
    ys = rvT.pdf(xs)

    with plt.rc_context({"figure.figsize":figsize}), sns.axes_style("ticks"):
        fig, (ax3, ax1, ax2) = plt.subplots(1,3)
        ax3.set_ylabel("$f_{T_0}$")

        # RIGHT
        title = '(a) right-tailed test'
        ax3.set_title(title, fontsize=13)#, y=-0.26)
        sns.lineplot(x=xs, y=ys, ax=ax3)
        ax3.set_xlim(-4, 4)
        ax3.set_ylim(0, 0.42)
        ax3.set_xticks([0,2])
        ax3.set_xticklabels([])
        ax3.set_yticks([])
        ax3.spines[['right', 'top']].set_visible(False)

        # highlight the right tail
        mask = (xs > 2)
        ax3.fill_between(xs[mask], y1=ys[mask], alpha=0.6, facecolor="red")
        ax3.vlines([2], ymin=0, ymax=rvT.pdf(2), linestyle="-", alpha=0.5, color="red")
        ax3.text(2, -0.03, r"$t_{\mathbf{x}}$", va="top", ha="center")
        ax3.text(0, -0.04, r"$0$", va="top", ha="center")

        # LEFT
        title = '(b) left-tailed test'
        ax1.set_title(title, fontsize=13) #, y=-0.26)
        sns.lineplot(x=xs, y=ys, ax=ax1)
        ax1.set_xlim(-4, 4)
        ax1.set_ylim(0, 0.42)
        ax1.set_xticks([-2,0])
        ax1.set_xticklabels([])
        ax1.set_yticks([])
        ax1.spines[['left', 'right', 'top']].set_visible(False)

        # highlight the left tail
        mask = (xs < -2)
        ax1.fill_between(xs[mask], y1=ys[mask], alpha=0.6, facecolor="red")
        ax1.vlines([-2], ymin=0, ymax=rvT.pdf(-2), linestyle="-", alpha=0.5, color="red")
        ax1.text(-2, -0.03, r"$t_{\mathbf{x}}$", va="top", ha="center")
        ax1.text(0, -0.04, r"$0$", va="top", ha="center")

        # TWO-TAILED
        title = '(c) two-tailed test'
        ax2.set_title(title, fontsize=13)#, y=-0.26)
        sns.lineplot(x=xs, y=ys, ax=ax2)
        ax2.set_xlim(-4, 4)
        ax2.set_ylim(0, 0.42)
        ax2.set_xticks([-2,0,2])
        ax2.set_xticklabels([])
        ax2.set_yticks([])
        ax2.spines[['left', 'right', 'top']].set_visible(False)

        # highlight the left and right tails
        mask = (xs < -2)
        ax2.fill_between(xs[mask], y1=ys[mask], alpha=0.6, facecolor="red")
        ax2.vlines([-2], ymin=0, ymax=rvT.pdf(-2), linestyle="-", alpha=0.5, color="red")
        ax2.text(-2, -0.03, r"$-|t_{\mathbf{x}}|$", va="top", ha="center")
        mask = (xs > 2)
        ax2.fill_between(xs[mask], y1=ys[mask], alpha=0.6, facecolor="red")
        ax2.vlines([2], ymin=0, ymax=rvT.pdf(2), linestyle="-", alpha=0.5, color="red")
        ax2.text(2, -0.03, r"$|t_{\mathbf{x}}|$", va="top", ha="center")
        ax2.text(0, -0.04, r"$0$", va="top", ha="center")

    return fig





# Section 3.5: Tails of a sampling distributions
################################################################################


def plot_panel_pvalue_D0_hist_and_dist(figsize=(7,2)):
    rvT = tdist(df=9)
    xs = np.linspace(-4, 4, 1000)
    ys = rvT.pdf(xs)
    N = 100000
    np.random.seed(42)
    ts = rvT.rvs(N)
    bins = nicebins(xs, 2, nbins=50)
    with plt.rc_context({"figure.figsize":figsize}), sns.axes_style("ticks"):
        fig, (ax1, ax2) = plt.subplots(1,2)

        # D0 hist
        sns.histplot(ts, ax=ax1, bins=bins, alpha=0.3)
        ax1.set_title("(a) permutation test")
        ax1.set_xlim(-4, 4)
        ax1.set_xticks([2])
        ax1.set_xticklabels([])
        ax1.set_yticks([])
        ax1.set_ylabel(r"$f_{\widehat{D}_0}$")
        # highlight the left and right tails
        tailvaluesl = [t for t in ts if t <= -2]
        sns.histplot(tailvaluesl, bins=bins, ax=ax1, color="red")
        ax1.text(-2.3, -630, r"$-|\hat{d}|$", verticalalignment="top", horizontalalignment="center")
        tailvaluesr = [t for t in ts if t >= 2]
        sns.histplot(tailvaluesr, bins=bins, ax=ax1, color="red")
        ax1.text(2, -630, r"$|\hat{d}|$", verticalalignment="top", horizontalalignment="center")
        # ax1.axvline(2, color="red")
        
        # D0 dist
        sns.lineplot(x=xs, y=ys, ax=ax2)
        ax2.set_title("(b) analytical apprixmation")
        ax2.set_xlim(-4, 4)
        ax2.set_ylim(0, 0.42)
        ax2.set_xticks([-2,2])
        ax2.set_xticklabels([])
        ax2.set_yticks([])
        ax2.set_ylabel(r"$f_{\widehat{D}_0}$")
        # highlight the left and right tails
        mask = (xs < -2)
        ax2.fill_between(xs[mask], y1=ys[mask], alpha=0.6, facecolor="red")
        ax2.text(-2, -0.03, r"$-|\hat{d}|$", verticalalignment="top", horizontalalignment="center")
        mask = (xs > 2)
        ax2.fill_between(xs[mask], y1=ys[mask], alpha=0.6, facecolor="red")
        ax2.text(2, -0.03, r"$|\hat{d}|$", verticalalignment="top", horizontalalignment="center")

    return fig



def plot_panel_pvalue_D0_and_T0(figsize=(7,2)):
    from scipy.stats import t as tdist
    rvT = tdist(df=9)
    xs = np.linspace(-4, 4, 1000)
    ys = rvT.pdf(xs)
    with plt.rc_context({"figure.figsize":figsize}), sns.axes_style("ticks"):
        fig, (ax1, ax2) = plt.subplots(1,2)

        # D0
        sns.lineplot(x=xs, y=ys, ax=ax1)
        ax1.set_xlim(-4, 4)
        ax1.set_ylim(0, 0.42)
        ax1.set_xticks([2])
        ax1.set_xticklabels([])
        ax1.set_yticks([])
        ax1.set_ylabel(r"$f_{\widehat{D}_0}$")
        # highlight the left and right tails
        mask = (xs < -2)
        ax1.fill_between(xs[mask], y1=ys[mask], alpha=0.6, facecolor="red")
        ax1.text(-2, -0.03, r"$-|\hat{d}|$", verticalalignment="top", horizontalalignment="center")
        mask = (xs > 2)
        ax1.fill_between(xs[mask], y1=ys[mask], alpha=0.6, facecolor="red")
        ax1.text(2, -0.03, r"$|\hat{d}|$", verticalalignment="top", horizontalalignment="center")

        # T0
        sns.lineplot(x=xs, y=ys, ax=ax2)
        ax2.set_xlim(-4, 4)
        ax2.set_ylim(0, 0.42)
        ax2.set_xticks([-2,2])
        ax2.set_xticklabels([])
        ax2.set_yticks([])
        ax2.set_ylabel("$f_{T_0}$")
        # highlight the left and right tails
        mask = (xs < -2)
        ax2.fill_between(xs[mask], y1=ys[mask], alpha=0.6, facecolor="red")
        ax2.text(-2, -0.03, r"$-|t|$", verticalalignment="top", horizontalalignment="center")
        mask = (xs > 2)
        ax2.fill_between(xs[mask], y1=ys[mask], alpha=0.6, facecolor="red")
        ax2.text(2, -0.03, r"$|t|$", verticalalignment="top", horizontalalignment="center")

    return fig


# Section 3.6 — Statistical design and error analysis
################################################################################



def plot_rejection_region(ax, xs, ys, rvT, alt, title):
    # design choices
    transp = 0.3
    alpha_color = "#4A25FF"
    beta_color = "#0CB0D6"
    axis_color = "#808080"

    ax.set_title(title, fontsize=11)

    # manually add arrowhead to x-axis + label t at the end
    ax.plot(1, 0, ">", color=axis_color, transform=ax.get_yaxis_transform(), clip_on=False)
    ax.set_xlabel("t")
    ax.xaxis.set_label_coords(1, 0.2)

    sns.lineplot(x=xs, y=ys, ax=ax, color="k")
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 0.42)
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.spines[['left', 'right', 'top']].set_visible(False)
    ax.spines['bottom'].set_color(axis_color)
    ax.tick_params(axis='x', colors=axis_color)
    ax.xaxis.label.set_color(axis_color)
    
    if alt == "greater":
        ax.set_xticks([2])
        # highlight the right tail
        mask = (xs > 2)
        ax.fill_between(xs[mask], y1=ys[mask], alpha=transp, facecolor=alpha_color)
        ax.vlines([2], ymin=0, ymax=rvT.pdf(2), linestyle="-", alpha=transp+0.2, color=alpha_color)
        ax.text(2, -0.03, r"$\mathrm{CV}_{\alpha}^+$", verticalalignment="top", horizontalalignment="center")

    elif alt == "less":
        ax.set_xticks([-2])
        # highlight the left tail
        mask = (xs < -2)
        ax.fill_between(xs[mask], y1=ys[mask], alpha=transp, facecolor=alpha_color)
        ax.vlines([-2], ymin=0, ymax=rvT.pdf(-2), linestyle="-", alpha=transp+0.2, color=alpha_color)
        ax.text(-2, -0.03, r"$\mathrm{CV}_{\alpha}^-$", verticalalignment="top", horizontalalignment="center")

    elif alt == "two-sided":
        ax.set_xticks([-2,2])
        # highlight the left and right tails
        mask = (xs < -2)
        ax.fill_between(xs[mask], y1=ys[mask], alpha=transp, facecolor=alpha_color)
        ax.vlines([-2], ymin=0, ymax=rvT.pdf(-2), linestyle="-", alpha=transp+0.2, color=alpha_color)
        ax.text(-2, -0.03, r"$\mathrm{CV}_{\alpha/2}^-$", verticalalignment="top", horizontalalignment="center")
        mask = (xs > 2)
        ax.fill_between(xs[mask], y1=ys[mask], alpha=transp, facecolor=alpha_color)
        ax.vlines([2], ymin=0, ymax=rvT.pdf(2), linestyle="-", alpha=transp+0.2, color=alpha_color)
        ax.text(2, -0.03, r"$\mathrm{CV}_{\alpha/2}^+$", verticalalignment="top", horizontalalignment="center")



def plot_panel_rejection_regions(figsize=(7,1.6)):
    rvT = tdist(df=9)
    xs = np.linspace(-4, 4, 1000)
    ys = rvT.pdf(xs)
    with sns.axes_style("ticks"), plt.rc_context({"figure.figsize":figsize}):
        fig, (ax3, ax1, ax2) = plt.subplots(1,3)
        # RIGHT
        title = '(a) right-tailed rejection region'
        plot_rejection_region(ax3, xs, ys, rvT, "greater", title)
        # LEFT
        title = '(b) left-tailed rejection region'
        plot_rejection_region(ax1, xs, ys, rvT, "less", title)
        # TWO-TAILED
        title = '(c) two-tailed rejection region'
        plot_rejection_region(ax2, xs, ys, rvT, "two-sided", title)
        fig.tight_layout()
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