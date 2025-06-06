import os

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import beta
from scipy.stats import binom      # used in Bayesian plots
from scipy.stats import randint    # special handling beta+1=beta
from scipy.stats import nbinom     # display parameter n as r
from scipy.stats import hypergeom  # special handling M=a+b, n=a, N=n
from scipy.stats import expon      # hide loc=0 parameter
from scipy.stats import gamma      # hide loc=0 parameter
from scipy.stats import norm
from scipy.stats import uniform    # show alpha and beta; not loc and scale
import seaborn as sns
import statsmodels.formula.api as smf


# Useful colors
snspal = sns.color_palette()
blue, orange, red, purple = snspal[0], snspal[1], snspal[3], snspal[4]

from ..bayes import hdi_from_grid
from .probability import plot_pdf
from ..sampling import gen_sampling_dist
from ..utils import default_labeler
from ..utils import savefigure


# Discrete random variables
################################################################################

def generate_pmf_panel(fname, xs, model, params_matrix,
                       params_to_latex={},
                       xticks=None,
                       fontsize=10,
                       labeler=default_labeler):
    """
    Generate PDF and PNG figures with panel of probability mass function of
    `model` over the sample space `xs` for all RV parameters specified in the
    list-of-lists `params_matrix`.
    """
    # We're drawing a figure with MxN subplots
    M = len(params_matrix)
    N = max( [len(row) for row in params_matrix] )

    # RV generation
    xmax = np.max(xs) + 1
    fX_matrix = np.zeros((M,N,xmax))
    for i in range(0,M):
        for j in range(0,N):
            params = params_matrix[i][j]
            rv = model(**params)
            low, high = rv.support()
            if high == np.inf:
                high = 1000
            calX = range(low, high+1)
            fXs = []
            for x in xs:
                if x in calX:
                    fXs.append(rv.pmf(x))
                else:
                    fXs.append(np.nan)
            fX_matrix[i][j] = fXs

    # Generate the MxN panel of subplots
    fig, axarr = plt.subplots(M, N, sharex=True, sharey=True)
    # We need to ensure `axarr` is an MxN matrix even if M or N are 1
    if M == 1 and N == 1:
        ax = axarr
        axarr = np.ndarray((1,1), object)
        axarr[0,0] = ax
    elif M == 1:
        axarr = axarr[np.newaxis,:]
    elif N == 1:
        axarr = axarr[:, np.newaxis]

    # Construct the panel of plots
    for i in range(0,M):
        for j in range(0,N):
            ax = axarr[i][j]
            fX = fX_matrix[i][j]
            params = params_matrix[i][j]
            if model == randint:
                display_params = {"low":params["low"], "high":params["high"]-1}
            elif model == hypergeom:
                display_params = {"a":params["n"], "b":params["M"]-params["n"], "n":params["N"]}
            elif model == nbinom:
                display_params = {"r":params["n"], "p":params["p"]}
            else:
                display_params = params
            label = labeler(display_params, params_to_latex)
            markerline, _stemlines, _baseline = ax.stem(fX, basefmt=" ")
            plt.setp(markerline, markersize=2)
            if xticks is not None:
                ax.xaxis.set_ticks(xticks)
            ax.text(0.95, 0.86, label,
                    horizontalalignment='right',
                    transform=ax.transAxes,
                    size=fontsize)

    # Save as PDF and PNG
    savefigure(fig, fname)
    return fig





# Continuous random variables
################################################################################

def calc_prob_and_plot(rv, a, b, xlims=None, ax=None, title=None):
    """
    Calculate the probability random variable `rv` falls between a and b,
    and plot the area-under-the-curve visualization of this calculation.
    """

    # 1. calculate Pr(a<X<b) == integral of rv.pdf between x=a and x=b
    p = quad(rv.pdf, a, b)[0]

    # 2. plot the probability density function (pdf)
    if xlims:
        xmin, xmax = xlims
    else:
        xmin, xmax = rv.ppf(0.001), rv.ppf(0.999)
    x = np.linspace(xmin, xmax, 10000)
    pX = rv.pdf(x)
    ax = sns.lineplot(x=x, y=pX, ax=ax)
    if title is None:
        title = "Probability density for the random variable " + rv.dist.name + str(rv.args) \
                 + " between " + str(a) + " and " + str(b)
    ax.set_title(title, y=0, pad=-30)

    # 3. highlight the area under pX between x=a and x=b
    mask = (x > a) & (x < b)
    ax.fill_between(x[mask], y1=pX[mask], alpha=0.2, facecolor=blue)
    ax.vlines([a], ymin=0, ymax=rv.pdf(a), linestyle="-", alpha=0.5, color=blue)
    ax.vlines([b], ymin=0, ymax=rv.pdf(b), linestyle="-", alpha=0.5, color=blue)
    
    # return prob and figure axes
    return p, ax



def calc_prob_and_plot_tails(rv, x_l, x_r, xlims=None, ax=None, title=None,
                             color=blue, facecolor="red", alpha=0.3):
    """
    Plot the area-under-the-curve visualization for the distribution's tails and
    calculate their combined probability mass: Pr({X < x_l}) + Pr({X > x_r}).
    """
    # 1. compute the probability in the left (-∞,x_l] and right [x_r,∞) tails
    p_l = quad(rv.pdf, rv.ppf(0.0000000000001), x_l)[0]
    p_r = quad(rv.pdf, x_r, rv.ppf(0.9999999999999))[0]
    p_tails = p_l + p_r

    # 2. plot the probability density function (pdf)
    if xlims:
        xmin, xmax = xlims
    else:
        xmin, xmax = rv.ppf(0.001), rv.ppf(0.999)
    x = np.linspace(xmin, xmax, 10000)
    pX = rv.pdf(x)
    ax = sns.lineplot(x=x, y=pX, ax=ax, color=color)
    if title is None:
        title = "Tails of the random variable " + rv.dist.name + str(rv.args)
    ax.set_title(title, y=0, pad=-30)

    # 3. highlight the area under pX for the tails
    mask_l = x < x_l   # left tail
    mask_u = x > x_r   # right tail
    ax.fill_between(x[mask_l], y1=pX[mask_l], alpha=alpha, facecolor=facecolor)
    ax.fill_between(x[mask_u], y1=pX[mask_u], alpha=alpha, facecolor=facecolor)
    ax.vlines([x_l], ymin=0, ymax=rv.pdf(x_l), linestyle="-", alpha=alpha+0.2, color=facecolor)
    ax.vlines([x_r], ymin=0, ymax=rv.pdf(x_r), linestyle="-", alpha=alpha+0.2, color=facecolor)

    # return prob and figure axes
    return p_tails, ax



def plot_pdf_and_cdf(rv, b=None, a=-np.inf, xlims=None, rv_name="X", title=None):
    """
    Plot side-by-side figure that shows pdf and CDF of random variable `rv`.
    If `b` is specified, the left plot will shows the area-under-the-curve
    visualization until x=b and tight plot highlights point at (b, F_X(b)).
    """
    fig, axs = plt.subplots(1, 2)
    ax0, ax1 = axs

    # figure title
    if title and title.lower() == "auto":
        title = "Probability distributions of the random variable " \
            + "$" + rv_name + "$" + " ~ " \
            + rv.dist.name + str(rv.args).replace(" ", "")
        fig.suptitle(title, y=1.01)
    if title:
        fig.suptitle(title, y=1.01)

    # 1. plot the probability density function (pdf)
    if xlims:
        xmin, xmax = xlims
    else:
        xmin, xmax = rv.ppf(0.001), rv.ppf(0.999)
    x = np.linspace(xmin, xmax, 1000)
    pX = rv.pdf(x)
    sns.lineplot(x=x, y=pX, ax=ax0)
    ax0.set_title("Probability density function")

    if b:
        # highlight the area under pX between x=a and x=b
        mask = (x > a) & (x < b)
        ax0.fill_between(x[mask], y1=pX[mask], alpha=0.2, facecolor=blue)
        ax0.vlines([b], ymin=0, ymax=rv.pdf(b), linestyle="-", alpha=0.5, color=blue)
        ax0.text(b, 0, "$b$", horizontalalignment="center", verticalalignment="top")
        ax0.text(b, rv.pdf(b)/2.5, r"Pr$(\{" + rv_name + r" \leq b \})$    ",
                 horizontalalignment="right", verticalalignment="center")

    # 2. plot the CDF
    FX = rv.cdf(x)
    sns.lineplot(x=x, y=FX, ax=ax1)
    ax1.set_title("Cumulative distribution function")

    if b:
        # highlight the point x=b
        ax1.vlines([b], ymin=0, ymax=rv.cdf(b), linestyle="-", color=blue)
        ax1.text(b, 0, "$b$", horizontalalignment="center", verticalalignment="top")
        ax1.text(b, rv.cdf(b), "$(b, F_{" + rv_name + "}(b))$",
                 horizontalalignment="right", verticalalignment="bottom")
        ax1.plot([b], [rv.cdf(b)], ".C0", markersize=7)

    # return figure and axes
    return fig, axs


def generate_pdf_panel(fname, xs, model, params_matrix,
                       params_to_latex={},
                       xticks=None, ylims=None,
                       fontsize=10,
                       labeler=default_labeler):
    """
    Generate PDF and PNG figures with panel of probability density function of
    `model` over the sample space `xs` for all RV parameters specified in the
    list-of-lists `params_matrix`.
    """
    # We're drawing a figure with MxN subplots
    M = len(params_matrix)
    N = max([len(row) for row in params_matrix])

    # RV generation
    fXs_matrix = np.zeros( (M,N,len(xs)) )
    for i in range(0,M):
        for j in range(0,N):
            params = params_matrix[i][j]
            rv = model(**params)
            fXs_matrix[i][j] = rv.pdf(xs)

    # Generate the MxN panel of subplots
    fig, axarr = plt.subplots(M, N, sharey=True)
    # We need to ensure `axarr` is an MxN matrix even if M or N are 1
    if M == 1 and N == 1:
        ax = axarr
        axarr = np.ndarray((1,1), object)
        axarr[0,0] = ax
    elif M == 1:
        axarr = axarr[np.newaxis,:]
    elif N == 1:
        axarr = axarr[:, np.newaxis]

    # Construct the panel of plots
    for i in range(0,M):
        for j in range(0,N):
            ax = axarr[i][j]
            fXs = fXs_matrix[i][j]
            params = params_matrix[i][j]
            if model == expon:
                display_params = {"scale": params["scale"]}
            elif model == gamma:
                lam = 1 / params["scale"]
                if lam >= 1:
                    lam = int(lam)
                display_params = {"a": params["a"], "lam":lam}
            elif model == uniform:
                beta = params["loc"] + params["scale"]
                display_params = {"alpha": params["loc"], "beta": beta}
            else:
                display_params = params
            label = labeler(display_params, params_to_latex)
            sns.lineplot(x=xs, y=fXs, ax=ax)
            if ylims:
                ax.set_ylim(*ylims)
            if xticks is not None:
                ax.xaxis.set_ticks(xticks)
            ax.text(0.93, 0.86, label,
                    horizontalalignment='right',
                    transform=ax.transAxes,
                    size=fontsize)

    # Save as PDF and PNG
    savefigure(fig, fname)

    return fig





# Random samples
################################################################################

def gen_samples(rv, n=30, N=10):
    """
    Generate `N` samples of size `n` from the random variable `rv`.
    Returns a pd.DataFrame with `N` columns containing the samples.
    """
    samples = {}
    for i in range(1, N+1):
        column_name = "sample " + str(i)
        samples[column_name] = rv.rvs(n)
    samples_df = pd.DataFrame(samples)
    return samples_df


def plot_samples(samples_df, ax=None, xlims=None, filename=None,
                 showmean=True, showstd=False):
    """
    Draw a strip plots for each of the columns in `samples_df`.
    Annotate each strip plot with the mean for each sample.
    """
    n, N = samples_df.shape  # sample size, number of samples

    # 1. Setup figure and axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # 2. Plot the samples as strip plot
    sns.stripplot(samples_df, orient="h", s=3, palette=[blue]*N, ax=ax, alpha=0.8, jitter=0)

    # 3. Add annotations 
    for i in range(1, N+1):
        column_name = "sample " + str(i)
        xbar_i = samples_df[column_name].mean()
        if showmean:
            # diamond-shaped marker to indicate mean in each sample
            ax.scatter(xbar_i, i-1, marker="D", s=45, color=orange, zorder=10)
        if showstd:
            # vertical bar to indicate xbar-std and xbar+std in each sample
            xstd_i = samples_df[column_name].std()
            stdbars_i = [xbar_i - xstd_i, xbar_i + xstd_i]
            ax.scatter(stdbars_i, [i-1,i-1], marker="|", s=70, color=orange, zorder=10)

    # 4. Handle keyword arguments
    if xlims:
        ax.set_xlim(xlims)
    if filename:
        savefigure(fig, filename)

    return ax





def plot_sampling_dist(stats, label=None, xlims=None, ax=None,
                       binwidth=None, scatter="mean", filename=None):
    """
    Plot a combined histogram and strip plot of the values in `stats`.
    """
    # 1. Setup figure and axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    if binwidth is None:
        if xlims is None:
            xlims = min(stats), max(stats)
        binwidth = (xlims[1]-xlims[0]) / 30            
    
    # 2. Plot a histogram of the sampling distribution
    sns.histplot(stats, binwidth=binwidth, stat="density", color="C1", ax=ax, label=label)

    # 3. add the scatter plot of `stats` below
    y_offset = 1/(100*binwidth)
    if scatter == "mean":
        sns.scatterplot(x=stats, y=-y_offset, ax=ax, color=orange, marker="D", s=30, alpha=0.1)
    elif scatter == "std":
        # sns.scatterplot(x=[0.0], y=-y_offset, color=orange, marker="D", s=30, alpha=0.9)
        sns.scatterplot(x=stats, y=-y_offset, ax=ax, color=orange, marker="|", s=30, alpha=0.1)

    # 4. Handle keyword arguments
    if xlims:
        ax.set_xlim(xlims)
    if filename:
        savefigure(fig, filename)

    return ax

        


# Panels illustrating CLT
################################################################################
        
def plot_samples_panel(rv, xlims, N=10, ns=[10,30,100], filename=None):
    """
    Draw a panel of strip plots for `N` sample with sizes `ns`.
    Need to pass `xlims` because cannot be determined automatically.
    """
    fig, axs = plt.subplots(1, len(ns), sharey=True, figsize=(10,2.5))

    for n, ax in zip(ns, axs):
        samples_df = gen_samples(rv, n=n, N=N)
        plot_samples(samples_df, xlims=xlims, ax=ax)
        ax.set_title(f"Samples of size $n={n}$")

    if filename:
        savefigure(fig, filename)


def plot_sampling_dists_panel(rv, xlims, N=1000, ns=[10,30,100], binwidth=None, filename=None):
    """
    Draw a panel of combined histogram and strip plot of the sampling distributions
    of random variable `rv` for sample sizes `ns`.
    Need to pass appropriate `xlims` and `binwidth` parameters depending on `rv`.
    """
    fig, axs = plt.subplots(1, len(ns), sharey=True, figsize=(10,2.5))

    # plot parameters
    xs = np.linspace(*xlims, 1000)

    xbarss = []
    for n, ax in zip([10,30,100], axs):
        # A. generate and plot sampling distribution
        xbars = gen_sampling_dist(rv, np.mean, n=n, N=N)
        plot_sampling_dist(xbars, ax=ax, xlims=xlims, binwidth=binwidth, label=f"$n={n}$")
        # B. plot the distribution predicted by the CLT
        rvXbar = norm(rv.mean(), rv.std()/np.sqrt(n))
        sns.lineplot(x=xs, y=rvXbar.pdf(xs), ax=ax, color="m")
        xbarss.append(xbars)

    if filename:
        savefigure(fig, filename)
    
    return xbarss



# Illustrating Type I and Type II error rates
################################################################################


def plot_alpha_beta_errors(cohend, ax=None, xlims=None, n=9, alpha=0.05,
                           show_alt=True, show_concl=False, show_dist_labels=False, show_es=False,
                           fontsize=14, alpha_offset=(0,0), beta_offset=(0,0)):
    """
    Plot sampling distribution under H0 and HA on the same graph,
    with Type I and Type II error probabilities highlighted.
    """

    if ax is None:
        fig, ax = plt.subplots()

    if xlims:
        xmin, xmax = xlims
    else:
        xmin, xmax = -3, 5

    # design choices
    transp = 0.1
    alpha_color = "#4A25FF"
    beta_color = "#0CB0D6"
    axis_color = "#808080"

    # default design parameters
    # n = 9         
    # alpha = 0.05


    # populations
    muH0 = 0
    sigma = 2
    muHA = muH0 + cohend*sigma

    # sampling distributions
    se = np.sqrt(sigma**2/n)
    rvXbarH0 = norm(muH0, se)
    rvXbarHA = norm(muHA, se)

    # cutoff value
    CV = norm.ppf(1-alpha) * se

    # plot sampling distributions
    calc_prob_and_plot_tails(rvXbarH0, x_l=xmin, x_r=CV, xlims=[xmin, xmax],
                                ax=ax, color="black", alpha=transp, facecolor=alpha_color)
    if show_alt:
        calc_prob_and_plot_tails(rvXbarHA, x_l=CV, x_r=xmax, xlims=[xmin, xmax],
                                    ax=ax, color="black", alpha=transp, facecolor=beta_color)
        ax.lines[1].set_linestyle("--")
    ax.set_title(None)
    ax.spines[['left', 'right', 'top']].set_visible(False)

    # manually add arrowhead to x-axis + label t at the end
    ax.plot(1, 0, ">", color=axis_color, transform=ax.get_yaxis_transform(), clip_on=False)
    ax.set_xlabel("t")
    ax.spines['bottom'].set_color(axis_color)
    ax.tick_params(axis='x', colors=axis_color)
    ax.xaxis.label.set_color(axis_color)
    aspect_ratio = ax.get_xlim()[1] / ax.get_ylim()[1] 
    if aspect_ratio > 5:
        ax.xaxis.set_label_coords(1, 0.1)
    else:
        ax.xaxis.set_label_coords(1, 0.2)

    # errors
    alpha_x = (CV + rvXbarH0.ppf(0.94)) / 2 + alpha_offset[0]
    alpha_y = rvXbarH0.pdf(alpha_x)/5 + alpha_offset[1]
    ax.annotate(r' $\alpha$', xy=(alpha_x, alpha_y), fontsize=fontsize, va="center", color=alpha_color)

    beta = rvXbarHA.cdf(CV)
    if show_alt:
        if beta > 0.01:
            beta_x = (CV + rvXbarHA.ppf(0.1)) / 2 + beta_offset[0]
            beta_y = rvXbarH0.pdf(beta_x)/5 + beta_offset[1]
            ax.annotate(r'$\beta$  ', xy=(beta_x, beta_y), fontsize=fontsize, color=beta_color, va="center", ha="right")

    # distribution annotations
    if show_dist_labels:
        arrowprops = dict(facecolor='black', shrink=0.05, width=2, headwidth=6, headlength=8)
        H0_x = rvXbarH0.ppf(0.1)
        H0_y = rvXbarH0.pdf(H0_x)
        ax.annotate('$T_0$', xy=(H0_x, H0_y), xytext=(H0_x-1, H0_y+0.1), ha="right", arrowprops=arrowprops)
        if show_alt:
            HA_x = rvXbarHA.ppf(0.90)
            HA_y = rvXbarHA.pdf(HA_x)
            ax.annotate('$T_A$', xy=(HA_x, HA_y), xytext=(HA_x+1, HA_y+0.1), arrowprops=arrowprops)

    # x-axis ticks and labels
    ax.set_yticks([])
    if show_alt:
        ax.set_xticks([0,CV,muHA])
        ax.set_xticklabels(["0", r"CV$_{\alpha}$", r"$\Delta$"])
    else:
        ax.set_xticks([0,CV])
        ax.set_xticklabels(["0", r"CV$_{\alpha}$"])
    # ax.vlines([0,muHA], ymin=0, ymax=rvXbarH0.pdf(0), linestyle="dotted", color="k", linewidth=1)

    # manually set y-limits of plot to avoid gap
    rvXbarH0MAX = rvXbarH0.pdf(0)
    ymax = rvXbarH0MAX*1.15
    ax.set_ylim([0, ymax])

    # cutoff line
    ax.vlines([CV], ymin=0, ymax=ax.get_ylim()[1], linestyle="-", color="red")

    # effect size (thick line segment above pdf plots)
    if show_es:
        esy = rvXbarH0MAX*1.07
        ax.plot([0,muHA], [esy,esy], linewidth=4, pickradius=1, solid_capstyle="butt")

    # decision annotations
    if show_concl:
        offset2 = 0.15
        offset3 = 0.11
        arrowprops2 = dict(facecolor='black', shrink=0.005, width=4, headwidth=10, headlength=12)
        ax.annotate("", xy=(xmax, -offset2), xytext=(CV, -offset2), arrowprops=arrowprops2, annotation_clip=False)
        ax.annotate('Reject $H_0$', xy=(xmax-0.1, -offset3), ha="right", annotation_clip=False, )
        ax.annotate("", xy=(xmin, -offset2), xytext=(CV, -offset2), arrowprops=arrowprops2, annotation_clip=False)
        ax.annotate('Fail to reject $H_0$', xy=(xmin+0.1, -offset3), ha="left", annotation_clip=False)

    # print design params for other info
    print("Design params: n =", n, ", alpha =", alpha, ", beta =", beta, ", Delta =", muHA, ", d =", cohend, ", CV =", CV)

    return ax




# Linear models
################################################################################

def plot_residuals(xdata, ydata, b0, b1, xlims=None, ax=None):
    """
    Plot residuals between the points (x,y) and the line y = b0 + b1*x.
    """
    if ax is None:
        fig, ax = plt.subplots()
    for x, y in zip(xdata, ydata):
        ax.plot([x, x], [y, b0+b1*x], color=red, zorder=0)
    return ax


def plot_residuals2(xdata, ydata, b0, b1, xlims=None, ax=None):
    """
    Plot residuals between the points (x,y) and the line y = b0 + b1*x
    as a square.
    """
    from matplotlib.patches import Rectangle
    ASPECT_CORRECTION = 0.89850746268

    if ax is None:
        _, ax = plt.subplots()

    def get_aspect(ax):
        fig = ax.figure
        ll, ur = ax.get_position() * fig.get_size_inches()
        width, height = ur - ll
        axes_ratio = height / width
        aspect = axes_ratio / ax.get_data_ratio()
        return aspect

    for x, y in zip(xdata, ydata):
        # plot the residual as a vertical line
        ax.set_axisbelow(True)
        ax.plot([x, x], [y, b0+b1*x], color=red, zorder=0, linewidth=0.5)
        # plot the residual squared
        deltay = y - (b0+b1*x)
        deltax = get_aspect(ax)*deltay*ASPECT_CORRECTION
        rect1 = Rectangle([x, b0+b1*x], width=-deltax, height=deltay,
                          linewidth=0, facecolor=red, zorder=2, alpha=0.3)
        rect2 = Rectangle([x, b0+b1*x], width=-deltax, height=deltay,
                          linewidth=0.5, facecolor="none", edgecolor=red, zorder=2)
        ax.add_patch(rect1)
        ax.add_patch(rect2)

    return ax




# Hypothesis tests as linear models
################################################################################

def plot_lm_ttest(data, x, y, ax=None):
    """
    Plot a combined scatterplot, means, and LM slope line
    to illustrate the equivalence between two-sample t-test
    and a linear model with a single binary predictor `x`.
    """
    # Fit the linear model
    lm = smf.ols(formula=f"{y} ~ 1 + C({x})", data=data).fit()
    beta0, beta1 = lm.params
    interceptlab, slopelab = lm.params.index

    # Plot the data
    with plt.rc_context({"text.usetex":True}):
        ax = plt.gca() if ax is None else ax
        sns.stripplot(data=data, x=x, y=y, hue=x, size=3, jitter=0, alpha=0.2)
        sns.pointplot(data=data, x=x, y=y, hue=x, estimator="mean", errorbar=None, marker="D")

        # Customize plot labels
        xlabel0, xlabel1 = [lab.get_text() for lab in ax.get_xticklabels()]
        newxlabel0 = xlabel0 + "\n0"
        newxlabel1 = xlabel1 + "\n1"
        ax.set_xticks([0,1])
        ax.set_xticklabels([newxlabel0, newxlabel1])
        ax.set_xlim([-0.3, 1.3])
        ax.set_xlabel(f"$\\texttt{{{x}}}_{{\\texttt{{{xlabel1}}}}}$")
        ax.xaxis.set_label_coords(0.5, -0.15)

        # Get seaborn colors
        snspal = sns.color_palette()

        # Add h-lines to represent the two group means
        ax.hlines(beta0, xmin=-0.3, xmax=1.3, color=snspal[0])
        ax.hlines(beta0+beta1, xmin=0.8, xmax=1.2, color=snspal[1])

        # Add diagonal to represent difference between means
        ax.plot([0, 1], [beta0, beta0 + beta1], color="k")

        # Draw custom legend
        blue_diamond = mlines.Line2D([], [], color=snspal[0], marker='D', ls="",
            label=f"$\\widehat{{\\beta}}_0$ = \\texttt{{{interceptlab}}} = {xlabel0} mean")
        yellow_diamond = mlines.Line2D([], [], color=snspal[1], marker='D', ls="",
            label=f"$\\widehat{{\\beta}}_0 + \\widehat{{\\beta}}_{{\\texttt{{{xlabel1}}}}}$ = {xlabel1} mean")
        slope_line = mlines.Line2D([], [], color="k",
            label=f"$\\widehat{{\\beta}}_{{\\texttt{{{xlabel1}}}}}$ = \\texttt{{{slopelab}}} slope")
        ax.legend(handles=[blue_diamond, yellow_diamond, slope_line])
        return ax


def plot_lm_anova(data, x, y, ax=None):
    """
    Plot a combined scatterplot, means, and LM slope lines
    to illustrate the equivalence between ANOVA test and
    a linear model with a single categorical predictor `x`.
    """
    # Fit the linear model
    lm = smf.ols(formula=f"{y} ~ 1 + C({x})", data=data).fit()

    # Labels for the different levels of the categorical variable
    labels = sorted(np.unique(data[x].values))

    # Seaborn color palette, line styles, and aesthetics
    snspal = sns.color_palette()
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot',
                  (0, (3, 5, 1, 5, 1, 5)),  # dashdotdotted
                  (5, (10, 3))]             # long dash with offset

    # Plot the data
    with plt.rc_context({"text.usetex":True}):
        ax = plt.gca() if ax is None else ax
        sns.stripplot(data=data, x=x, y=y, hue=x, size=3, jitter=0, alpha=0.2, order=labels, hue_order=labels)
        sns.pointplot(data=data, x=x, y=y, hue=x, estimator="mean", errorbar=None, marker="D", hue_order=labels)

        # Group 1 (baseline)
        beta0 = lm.params.iloc[0]
        interceptlab = lm.params.index[0]
        ax.axhline(beta0, color=snspal[0], linewidth=1,
                label=f"$\\widehat{{\\beta}}_0$ = \\texttt{{{interceptlab}}} = \\texttt{{{labels[0]}}} mean")

        # Remaining groups
        for i in range(1, len(labels)):
            label = labels[i]
            beta = lm.params.iloc[i]
            slopelab = lm.params.index[i]
            linestyle = linestyles[i%len(linestyles)]
            ax.hlines(beta0+beta, xmin=i-0.2, xmax=i+0.2, color=snspal[i])
            ax.plot([i-0.7, i], [beta0, beta0 + beta], color="k", linestyle=linestyle,
                    label=f"$\\widehat{{\\beta}}_{{\\texttt{{{label}}}}}$ = \\texttt{{{slopelab}}} slope")

        # Return axes
        ax.legend()
        return ax
    


# Bayesian statistics
################################################################################

def prior_times_likelihood_eq_posterior(heads=4, n=5, figsize=(5,5), destdir=None):
    """
    Vertical panel showing the prior, likelihood, and posterior functions.
    """
    with plt.rc_context({"figure.figsize":figsize}):
        fig, axs = plt.subplots(3, 1, sharex=True)
        eps = 0.001
        ps = np.linspace(0-eps, 1.0+eps, 1000)

        # prior (slightly sloped uniform)
        slope = -1
        prior = np.ones_like(ps) - slope/2 + slope*ps
        prior[(ps < 0) | (ps > 1)] = 0
        sns.lineplot(x=ps, y=prior, ax=axs[0], color="C1", ls="--", label="prior")
        axs[0].set_ylabel("$f_{\\Theta}$")
        axs[0].set_ylim([-0.1,2.4])
        axs[0].set_yticks([0.0,0.5,1.0,1.5])
        axs[0].set_yticklabels([0.0,0.5,1.0,1.5])
        axs[0].legend(loc="upper left")
        
        # add 1/C at the top
        axs[0].text(-0.2, 2.1, r"$1/C$", size=14, color="black")
        # add x in between
        axs[0].text(-0.2, -0.4, "$\\times$", size=20, color="black")

        # likelihood
        likelihood = binom(n, p=ps).pmf(heads)
        sns.lineplot(x=ps, y=likelihood, ax=axs[1], lw=2, color="black", label="likelihood")
        axs[1].set_ylabel("$L_{\\mathbf{x}}$")
        axs[1].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])

        # add = in between
        axs[1].text(-0.2, -0.1, "$=$", size=20, color="black")

        # posterior
        numerator = likelihood * prior
        posterior = numerator / np.nansum(numerator)
        posteriord = posterior / (ps[1]-ps[0])
        sns.lineplot(x=ps, y=posteriord, ax=axs[2], color="C0", label="posterior")
        axs[2].set_ylabel("$f_{\\Theta|\\mathbf{x}}$")
        axs[2].set_xlabel("$\\theta$")
        axs[2].set_xticks(np.linspace(0,1,11))
        axs[2].set_ylim([-0.1,2.4])
        axs[2].legend(loc="upper left")

        if destdir:
            filename = os.path.join(destdir, "prior_times_likelihood_eq_posterior.pdf")
            savefigure(fig, filename)


def prior_times_likelihood_eq_posterior_grid(heads=4, n=5, ngrid=26, figsize=(5,5), destdir=None):
    """
    Vertical panel showing the grid approximation to the prior, likelihood, and posterior.
    """
    with plt.rc_context({"figure.figsize":figsize}):
        fig, axs = plt.subplots(3, 1, sharex=True)

        # grid
        ps = np.linspace(0, 1, ngrid)

        # prior values
        slope = -1
        prior = np.ones_like(ps) - slope/2 + slope*ps
        axs[0].stem(ps, prior, basefmt=" ", linefmt="C1-", label="prior")
        axs[0].set_ylabel("$f_{\\Theta}$ values")
        axs[0].set_ylim([-0.1,2.4])
        axs[0].set_yticks([0.0,0.5,1.0,1.5])
        axs[0].set_yticklabels([0.0,0.5,1.0,1.5])
        axs[0].legend(loc="upper left")

        # add 1/C at the top
        axs[0].text(-0.2, 2.2, r"$1/C$", size=14, color="black")
        # add x in between
        axs[0].text(-0.2, -0.4, "$\\times$", size=20, color="black")
        
        # likelihood values
        likelihood = binom(n, p=ps).pmf(heads)
        axs[1].stem(ps, likelihood, basefmt=" ", linefmt="k", label="likelihood")
        axs[1].set_ylabel("$L_{\\mathbf{x}}$ values")
        axs[1].legend(loc="upper left")

        # add = in between
        axs[1].text(-0.2, -0.1, "$=$", size=20, color="black")

        # posterior values
        numerator = likelihood * prior
        posterior = numerator / np.nansum(numerator)
        posteriord = posterior / (ps[1]-ps[0])
        axs[2].stem(ps, posteriord, basefmt=" ", linefmt="C0-", label="posterior")
        axs[2].set_ylabel("$f_{\\Theta|\\mathbf{x}}$ values")
        axs[2].set_xlabel("$\\theta$")
        axs[2].set_ylim([-0.1,2.4])
        axs[2].legend(loc="upper left")

        if destdir:
            filename = os.path.join(destdir, "prior_times_likelihood_eq_posterior_grid.pdf")
            savefigure(fig, filename)


def posterior_visualization(heads=4, n=5, ngrid=1000, figsize=(6,2.5), destdir=None):
    """
    Focus on the posterior with annotations for point and interval estimates.
    """
    eps = 0.001
    ps = np.linspace(0-eps, 1.0+eps, ngrid)

    # prior (slightly sloped uniform)
    slope = -1
    prior = np.ones_like(ps) - slope/2 + slope*ps
    prior[(ps < 0) | (ps > 1)] = 0
    # likelihood
    likelihood = binom(n, p=ps).pmf(heads)
    likelihood[np.isnan(likelihood)] = 0
    # posterior
    numerator = likelihood * prior
    posterior = numerator / np.sum(numerator)

    # calculate mean, median, mode
    pmean = np.sum(ps*posterior)
    pmedian = ps[np.cumsum(posterior).searchsorted(0.5)]
    pmode = ps[np.argmax(posterior)]
    hdi90 = hdi_from_grid(ps, posterior, hdi_prob=0.9)

    with plt.rc_context({"figure.figsize":figsize}):
        # plot the posterior
        porteriord = posterior / (ps[1] - ps[0])
        ax = sns.lineplot(x=ps, y=porteriord, color="C0", label="posterior")
        ax.set_ylabel("$f_{\\Theta|\\mathbf{x}}$")
        ax.set_xlabel("$\\theta$")
        ax.set_xticks(np.linspace(0,1,11))
        ax.set_ylim([-0.1,2.4])

        # add mean marker
        ax.plot(pmean, 0, marker="D", color="C0", ls=" ", label="posterior mean")
        # add median line
        post_at_median = porteriord[np.nancumsum(posterior).searchsorted(0.5)]
        ax.vlines(x=pmedian, ymin=0, ymax=post_at_median, color="C0", lw=0.7)
        ax.plot(pmedian, post_at_median/2, marker="s", color="C0", ls=" ", label="posterior median")
        # add mode marker
        ax.plot(pmode, np.nanmax(porteriord), marker="^", color="C0", ls=" ", label="posterior mode")
        # plot 90% credible interval
        ax.hlines(0.15, hdi90[0], hdi90[1], color="C4", lw=2.2, zorder=0, label="90% highest density interval")
        ax.legend()
        
        if destdir:
            filename = os.path.join(destdir, "posterior_visualization.pdf")
            savefigure(plt.gcf(), filename)


def panel_coin_posteriors(ctosses=None, figsize=(6,6), destdir=None):
    """
    Plot a 5x2 panel of prior+posterior snapshots of the coin tosses analysis.
    """
    ns =       [0, 1, 2, 3, 4, 5, 10, 20, 30,  50]
    outcomes = [0, 1, 2, 2, 2, 3,  7, 13, 22,  34]
    # outcomes were generated from coin with true p = 0.7
    assert len(outcomes) == len(ns)
    n_rows = int(len(outcomes))
    if ctosses:
        # check plots are consistent with the `ctosses` data
        assert outcomes == [sum(ctosses[:n]) for n in ns]

    with plt.rc_context({"figure.figsize":figsize}):
        fig, axs_matrix = plt.subplots(n_rows//2, 2, sharex=True)
        axs = [ax for row in axs_matrix for ax in row]
        priorRV = None
        for i, ax in enumerate(axs):
            heads, n = outcomes[i], ns[i]
            rvPpost = beta(a=1+heads, b=1+n-heads)
            plot_pdf(rvPpost, rv_name="P", xlims=[-0.01,1.01], ax=ax)
            # superimpose a plot of the prior as a dashed line
            if priorRV:
                plot_pdf(priorRV, rv_name="P", color="C1", ax=ax, ls="--", xlims=[-0.01,1.01])
            if i==0:
                ax.set_title("flat prior")
                ax.set_ylabel("$f_P$")
            else:
                ax.set_title(f"{heads} heads in {n} tosses")
                ax.set_ylabel("$f_{P|\\mathbf{c}^{(%i)}}$" % n)
            priorRV = rvPpost
            ax.set_ylim(0,6.3)
            ax.set_yticks([0,1,2,3,4,5,6])
        axs[8].set_xlabel("$p$")
        axs[9].set_xlabel("$p$")

        if destdir:
            filename = os.path.join(destdir, "panel_coin_posteriors.pdf")
            savefigure(fig, filename)

