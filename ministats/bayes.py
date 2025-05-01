import copy
import logging

import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fmin
from scipy.stats import gaussian_kde

# Silence annoying warning about missing BLAS library
# WARNING: Using NumPy C-API based implementation for BLAS functions.
logging.getLogger("pytensor.tensor.blas").setLevel(logging.ERROR)


    
# ESTIMATORS
################################################################################

def mode_from_samples(samples, prec=0.0001, **kwargs):
    """
    Calculate the mode (maximum) of the empirical distribution of `samples` by
    approximating the probability density function using KDE and finding its max.
    See also `az.plots.plot_utils.calculate_point_estimate("mode", samples)`.
    """
    values = np.arange(min(samples), max(samples), prec)
    kde = gaussian_kde(samples, **kwargs)
    return values[np.argmax(kde(values))]


# CREDIBILITY INTERVALS
################################################################################

def hdi_from_grid(params, probs, hdi_prob=0.9):
    """
    Compute the highest density interval from the grid approximation specified
    as probabilities `probs` evaluated at grid values `params`.
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


def hdi_from_rv(rv, hdi_prob=0.9):
    probstart0 = (1 - hdi_prob) / 2
    def width(probstart):
        return rv.ppf(probstart + hdi_prob) - rv.ppf(probstart)
    min_probstart = fmin(width, x0=probstart0, xtol=1e-9, disp=False)[0]
    hdi_left = rv.ppf(min_probstart)
    hdi_right = rv.ppf(min_probstart + hdi_prob)
    return [hdi_left, hdi_right]


def hdi_from_samples(samples, hdi_prob=0.9):
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


def hdi_from_idata(idata, var_name, hdi_prob=0.9):
    stats_df = az.summary(idata, kind="stats", hdi_prob=hdi_prob, var_names=var_name)
    hdi_left = stats_df.iloc[0,2]
    hdi_right = stats_df.iloc[0,3]
    return [hdi_left, hdi_right]




# COMPARING TWO GROUPS
################################################################################

def bayes_dmeans(xsample, ysample, priors=None,
                 var_name="var", group_name="group", groups=["x", "y"]):
    """
    Compare the means of two groups using a Bayesian model.
    Returns a tuple containing the Bambi model and the InferenceData object.

    Usage example:
    >>> treated = iqs2[iqs2["group"]=="treat"]["iq"].values
    >>> controls = iqs2[iqs2["group"]=="ctrl"]["iq"].values
    >>> mod, idata = bayes_dmeans(treated, controls, var_name="iq",
                                  group_name="group", groups=["treat", "ctrl"])
    """
    # Pacakge raw data samples as a DataFrame
    m, n = len(xsample), len(ysample)
    groups_col = [groups[0]]*m + [groups[1]]*n
    var_col = list(xsample) + list(ysample)
    df = pd.DataFrame({group_name:groups_col, var_name:var_col})

    # Build the Bambi model
    formula = bmb.Formula(f"{var_name} ~ 1 + {group_name}",
                          f"sigma ~ 0 + {group_name}")
    model = bmb.Model(formula=formula,
                      family="t",
                      link="identity",
                      priors=priors,
                      data=df)

    # Fit the model
    idata = model.fit(draws=2000)
    return model, idata


def _infer_groups_from_idata(idata, group_name="group", other="other"):
    """
    Helper function used in `calc_dmeans_stats` and `plot_dmeans_stats`.
    """
    post = idata["posterior"]
    sigma_group = "sigma_" + group_name
    group_dim = group_name + "_dim"
    if sigma_group in post.data_vars and "sigma" not in post.data_vars:
        # Infer `groups` from the `sigma_{group_name}_dim` coordinate values
        sigma_group_dim = "sigma_" + group_name + "_dim"
        groups = list(post.coords[sigma_group_dim].values)
    elif group_dim in post.coords and len(post.coords[group_dim]) == 2:
        # Infer `groups` from the `{group_name}_dim` coordinate values
        groups = list(post.coords[group_dim].values)
    else:
        # Fallback: infer one group name from the `{group_name}_dim` dimension,
        # and label the other one as generic name `other`
        known_group = list(post.coords[group_dim].values)[0]
        groups = [other, known_group]
    return groups


def calc_dmeans_stats(idata, group_name="group"):
    """
    Calculate derived quantities used for the analysis plots and summaries.
    Handles both cases where formula `y ~ 1 + group` or `y ~ 0 + group` is used,
    and either group-specific `sigma` or common `sigma`.
    """
    post = idata["posterior"]
    groups = _infer_groups_from_idata(idata, group_name=group_name)

    # Add aliases for individual means and calculate the difference between means
    group_dim = group_name + "_dim"
    if "Intercept" in post.data_vars:
        # CASE A: formula is specified as `y ~ 1 + group`
        post["dmeans"] = post[group_name].loc[{group_dim:groups[1]}]
        post["mu_" + groups[0]] = post["Intercept"]
        post["mu_" + groups[1]] = post["Intercept"] + post["dmeans"]
    else:
        # CASE B: formula is specified as `y ~ 0 + group`
        post["mu_" + groups[0]] = post[group_name].loc[{group_dim:groups[0]}]
        post["mu_" + groups[1]] = post[group_name].loc[{group_dim:groups[1]}]
        post["dmeans"] = post["mu_" + groups[1]] - post["mu_" + groups[0]]

    sigma_group = "sigma_" + group_name
    if sigma_group in post.data_vars and "sigma" not in post.data_vars:
        # Calculate sigmas from log-sigmas
        sigma_group_dim = "sigma_" + group_name + "_dim"
        log_sigma_x = post[sigma_group].loc[{sigma_group_dim:groups[0]}]
        log_sigma_y = post[sigma_group].loc[{sigma_group_dim:groups[1]}]
        sigma_x_name = "sigma_" + groups[0]
        sigma_y_name = "sigma_" + groups[1]
        post[sigma_x_name] = np.exp(log_sigma_x)
        post[sigma_y_name] = np.exp(log_sigma_y)
        # Calculate the difference between standard deviations
        post["dsigmas"] = post[sigma_y_name] - post[sigma_x_name]
        # Effect size
        var_pooled = (post[sigma_x_name]**2 + post[sigma_y_name]**2) / 2
        post["cohend"] = post["dmeans"] / np.sqrt(var_pooled)
    else:
        # post["sigma"] is already on the right scale
        # Effect size
        post["cohend"] = post["dmeans"] / post["sigma"]

    return idata



def plot_dmeans_stats(model, idata, group_name="group",
                      figsize=(8,10), ppc_xlims=None, ppc_ylims=None):
    """
    Generate posterior panel of plots similar to the one in BEST paper.

    When model has group-specific `sigma`, the plot will look like this:
    +---------+--------------+
    | mu1     | post pred 1  |
    | mu2     | post pred 2  |
    | sigma1  | dmeans       |
    | sigma2  | dsigmas      |
    | nu      | cohend       |
    +---------+--------------+

    For analyzes with a common `sigma`, the plot will look like this:
    +---------+--------------+
    | mu1     | post pred 1  |
    | mu2     | post pred 2  |
    | dmeans  | dsigmas      |
    | nu      | cohend       |
    +---------+--------------+
    """
    post = idata["posterior"]
    groups = _infer_groups_from_idata(idata, group_name=group_name)

    # Compute posterior predictive checks 
    N_rep = 20
    draws_subset = np.random.choice(post["draw"].values, N_rep, replace=False)
    idata_rep = idata.sel(draw=draws_subset)
    df = model.data
    # PPC group 1
    data1 = df[df[group_name]==groups[1]]
    idata_rep1 = copy.deepcopy(idata_rep).sel(__obs__=data1.index)
    model.predict(idata_rep1, data=data1, kind="response")
    # PPC group 0
    if groups[0] == "other":
        altgroups = list(model.data[group_name].unique())
        altgroups.remove(groups[1])
        data0 = df[df[group_name]==altgroups[0]]
    else:
        data0 = df[df[group_name]==groups[0]]
    idata_rep0 = copy.deepcopy(idata_rep).sel(__obs__=data0.index)
    model.predict(idata_rep0, data=data0, kind="response")
    # Set x-lims automatically based on range of outcome variable
    if ppc_xlims is None:
        oname = model.response_component.term.name
        obs = df[oname]
        omin, omax, orange = obs.min(), obs.max(), obs.max()-obs.min()
        ppc_xlims = [omin-0.1*orange, omax+0.1*orange]

    with plt.rc_context({"figure.figsize":figsize}):

        sigma_group = "sigma_" + group_name
        if sigma_group in post.data_vars and "sigma" not in post.data_vars:
            fig, axs = plt.subplots(5,2)
            axmu1, axpp1        = axs[0,0], axs[0,1]
            axmu2, axpp2        = axs[1,0], axs[1,1]
            axsigma1, axdmeans  = axs[2,0], axs[2,1]
            axsigma2, axdsigmas = axs[3,0], axs[3,1]
            axnu, axcohend      = axs[4,0], axs[4,1]
        else:
            fig, axs = plt.subplots(4,2)
            axmu1, axpp1       = axs[0,0], axs[0,1]
            axmu2, axpp2       = axs[1,0], axs[1,1]
            axdmeans, axsigma  = axs[2,0], axs[2,1]
            axnu, axcohend     = axs[3,0], axs[3,1]
 
        # Top
        ## Left column
        az.plot_posterior(idata, group="posterior", var_names=["mu_" + groups[0]],
                          round_to=3, ax=axmu1)
        az.plot_posterior(idata, group="posterior", var_names=["mu_" + groups[1]],
                          round_to=3, ax=axmu2)
        ## Right column
        az.plot_ppc(idata_rep0, group="posterior", mean=False, ax=axpp1)
        axpp1.set_xlim(ppc_xlims)
        axpp1.set_xlabel(None)
        axpp1.set_title("Posterior predictive for " + groups[0])
        az.plot_ppc(idata_rep1, group="posterior", mean=False, ax=axpp2)
        axpp2.set_xlim(ppc_xlims)
        axpp2.set_xlabel(None)
        axpp2.set_title("Posterior predictive for " + groups[1])
        # Set same y-lims for the two PPC plots
        if ppc_ylims is None:
            ymin = min(axpp1.get_ylim()[0], axpp2.get_ylim()[0])
            ymax = max(axpp1.get_ylim()[1], axpp2.get_ylim()[1])
            axpp1.set_ylim([ymin,ymax])
            axpp2.set_ylim([ymin,ymax])
        else:
            axpp1.set_ylim(ppc_ylims)
            axpp2.set_ylim(ppc_ylims)

        # Middle
        if sigma_group in post.data_vars and "sigma" not in post.data_vars:
            az.plot_posterior(idata, group="posterior", var_names=["sigma_" + groups[0]],
                              point_estimate="mode", round_to=3, ax=axsigma1)
            az.plot_posterior(idata, group="posterior", var_names=["sigma_" + groups[1]],
                              point_estimate="mode", round_to=3, ax=axsigma2)
            az.plot_posterior(idata, group="posterior", var_names=["dsigmas"], hdi_prob=0.95,
                              point_estimate="mode", round_to=3, ax=axdsigmas)
            axdsigmas.axvline(0, c="C2", lw=2)
        else:
            az.plot_posterior(idata, group="posterior", var_names=["sigma"], hdi_prob=0.95,
                              point_estimate="mode", ax=axsigma)
            axsigma.axvline(0, c="C2", lw=2)
        az.plot_posterior(idata, group="posterior", var_names=["dmeans"], hdi_prob=0.95,
                          round_to=3, ax=axdmeans)
        axdmeans.axvline(0, c="C2", lw=2)

        # Bottom
        az.plot_posterior(idata, group="posterior", var_names=["nu"],
                          point_estimate="mode", ax=axnu)
        az.plot_posterior(idata, group="posterior", var_names=["cohend"],
                          point_estimate="mode", round_to=3, ax=axcohend)

    fig.tight_layout()
    return fig




# Bayesian Estimation Supersedes the t-Test (BEST) priors
################################################################################

def best_dmeans_model(xsample, ysample, nuprior="exp"):
    """
    Fit the model described in the "Bayesian Estimation Supersedes the t-Test"
    paper by John K. Kruschke.
    The function supports three different choices for the priors on `nu`:
      - `shiftedexp` = Expon(lam=1/29) + 1: the prior from the original paper
      - `exp` = Expon(lam=1/30): a simplified version without the +1 shift
      - `gamma` = Gamma(alpha=2.0, beta=0.1): the Bambi default prior for `nu`s
    Returns the Bambi model, which you can then analyze and fit.
    """
    pass


def best_dmeans_calc(idata, var_name="z", group_names=["treatment", "control"]):
    """
    Performs various calculations on the inference data object `idata`:
      - `dmeans`: difference between groups means
      - `dsigmas`: difference between groups standard deviaitons
      - 
      - `log10(nu)`: the normality parameter 

    """
    pass


def best_dmeans_plots():
    """
    Generte the panel of plots similar to the BEST paper.
    """
    pass



# BAYES FACTORS
################################################################################

# MAYBE: import grid approximaiotn methods from 50_extra_bayesian_stuff.ipynb
