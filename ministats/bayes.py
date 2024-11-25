import copy
import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd






# CREDIBILITY INTERVALS
################################################################################

# TODO: improt from hpdi.py
# TODO: rename to hdi



# COMPARING TWO GRUPS
################################################################################

def bayes_dmeans(xsample, ysample, priors=None,
                 var_name="var", group_name="group", groups=["x", "y"]):
    """
    Compare the means of two groups using a Bayesian model.
    Returns a tuple containing the Bambi model and the InferenceData object.
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



def calc_dmeans_stats(idata, group_name="group"):
    """
    Calculate derived quantities used for the analyisis plots and summaries.
    """
    post = idata["posterior"]

    # Infer `groups`` from the `sigma_{group_var}_dim` coordinate values
    sigma_group = "sigma_" + group_name
    sigma_group_dim = "sigma_" + group_name + "_dim"
    groups = list(post[sigma_group].coords[sigma_group_dim].values)

    # Add alias for the difference between means
    group_dim = group_name + "_dim"
    post["dmeans"] = post[group_name].loc[{group_dim:groups[1]}]

    # Calculate the group means
    post["mu_" + groups[0]] = post["Intercept"]
    post["mu_" + groups[1]] = post["Intercept"] + post["dmeans"]

    # Calculate sigmas from log-sigmas
    log_sigma_x = post[sigma_group].loc[{sigma_group_dim:groups[0]}]
    log_sigma_y = post[sigma_group].loc[{sigma_group_dim:groups[1]}]
    sigma_x_name = "sigma_" + groups[0]
    sigma_y_name = "sigma_" + groups[1]
    post[sigma_x_name] = np.exp(log_sigma_x)
    post[sigma_y_name] = np.exp(log_sigma_y)

    # Calculate the difference between standard deviations
    post["dstd"] = post[sigma_y_name] - post[sigma_x_name]

    # Effect size
    var_pooled = (post[sigma_x_name]**2 + post[sigma_y_name]**2) / 2
    post["cohend"] = post["dmeans"] / np.sqrt(var_pooled)

    return idata



def plot_dmeans_stats(model, idata, group_name="group", figsize=(8,10), ppc_xlims=None):
    """
    Generate posterior panel of plots similar to the one in BEST paper:
    +---------+-------------+
    | mu1     | post pred 1 |
    | mu2     | post pred 2 |
    | sigma1  | dmeans      |
    | sigma2  | dstd        |
    | nu      | cohend      |
    +---------+-------------+
    """

    # Infer groups from the `sigma_{group_var}_dim` coordinate values
    sigma_group = "sigma_" + group_name
    sigma_group_dim = "sigma_" + group_name + "_dim"
    groups = list(idata["posterior"][sigma_group].coords[sigma_group_dim].values)

    # Compute posterior predictive checks 
    N_rep = 30
    draws_subset = np.random.choice(idata["posterior"]["draw"].values, N_rep, replace=False)
    idata_rep = idata.sel(draw=draws_subset)
    df = model.data
    idata_rep0 = copy.deepcopy(idata_rep)
    data0 = df[df[group_name]==groups[0]]
    model.predict(idata_rep0, data=data0, kind="response")
    idata_rep1 = copy.deepcopy(idata_rep)
    data1 = df[df[group_name]==groups[1]]    
    model.predict(idata_rep1, data=data1, kind="response")

    with plt.rc_context({"figure.figsize":figsize}):
        fig, axs = plt.subplots(5,2)
        # Left column
        az.plot_posterior(idata, group="posterior", var_names=["mu_" + groups[0]], ax=axs[0,0])
        az.plot_posterior(idata, group="posterior", var_names=["mu_" + groups[1]], ax=axs[1,0])
        az.plot_posterior(idata, group="posterior", var_names=["sigma_" + groups[0]], point_estimate="mode", ax=axs[2,0])
        az.plot_posterior(idata, group="posterior", var_names=["sigma_" + groups[1]], point_estimate="mode", ax=axs[3,0])
        az.plot_posterior(idata, group="posterior", var_names=["nu"], point_estimate="mode", ax=axs[4,0])
        # Right column
        az.plot_ppc(idata_rep0, group="posterior", mean=False, ax=axs[0,1])
        axs[0,1].set_xlim(ppc_xlims)
        axs[0,1].set_xlabel(None)
        axs[0,1].set_title("Posterior predictive for " + groups[0])
        az.plot_ppc(idata_rep1, group="posterior", mean=False, ax=axs[1,1])
        axs[1,1].set_xlim(ppc_xlims)
        axs[1,1].set_xlabel(None)
        axs[1,1].set_title("Posterior predictive for " + groups[1])
        az.plot_posterior(idata, group="posterior", var_names=["dmeans"], ref_val=0, ax=axs[2,1])
        az.plot_posterior(idata, group="posterior", var_names=["dstd"], ref_val=0, point_estimate="mode", ax=axs[3,1])
        az.plot_posterior(idata, group="posterior", var_names=["cohend"], point_estimate="mode", ax=axs[4,1])

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
      - `exp` = Expon(lam=1/29): a simplified version without the +1 shift
      - `gamma` = Gamma(alpha=2.0, beta=0.1): the Bambi default prior for `nu`s
    Returns the Bambi model, which you can then analyze and fit.
    """
    pass


def best_dmeans_calc(idata, var_name="z", group_names=["treatment", "control"]):
    """
    Performs various calculations on the inference data object `idata`:
      - `dmeans`: difference between groups means
      - `dstd`: difference between groups standard deviaitons
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
