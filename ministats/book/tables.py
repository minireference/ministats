import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import pingouin as pg

from arviz.plots.plot_utils import calculate_point_estimate as calc_point_est

from ..bayes import calc_dmeans_stats
from ..bayes import hdi_from_idata
from ..confidence_intervals import ci_dmeans
from ..hypothesis_tests import permutation_test_dmeans
from ..hypothesis_tests import ttest_dmeans


# SENSITIVITY ANALYSIS
################################################################################

def fit_bayesian_model_iqs2(iqs2, new_priors, random_seed=42):
    """
    Fits the model like Example 2 in Section 5.4 with `new_priors`
    overwriting the defaults priors in `priors2`.
    Returns the `idata2` object.
    """
    priors2 = {
        "group": bmb.Prior("Normal", mu=100, sigma=35),
        "sigma": {
            "group": bmb.Prior("Normal", mu=1, sigma=2)
        },
        "nu": bmb.Prior("Gamma", alpha=2, beta=0.1),
    }
    priors2.update(new_priors)
    formula2 = bmb.Formula("iq ~ 0 + group", "sigma ~ 0 + group")
    mod2 = bmb.Model(formula=formula2,
                     family="t",
                     link={"mu": "identity", "sigma": "log"},
                     priors=priors2,
                     data=iqs2)
    idata2 = mod2.fit(draws=2000, random_seed=random_seed)
    return idata2



def sens_analysis_dmeans_iqs2(iqs2):
    """
    Generate the table showing the results of the sensitivity analysis
    for Example 2 in Section 5.4.
    """

    M_priors = {
         "orig": {
              "display": r"$\mathcal{N}(100,35)$",
              "bambi": bmb.Prior("Normal", mu=100, sigma=35),
         },
         "wider": {
              "display": r"$\mathcal{N}(100,50)$",
              "bambi": bmb.Prior("Normal", mu=100, sigma=50),
         },
         "tighter": {
              "display": r"$\mathcal{N}(100,10)$",
              "bambi": bmb.Prior("Normal", mu=100, sigma=10),
         },
    }
    logSigma_priors = {
        "orig": {
            "display": r"$\mathcal{N}(1,2)$",
            "bambi": bmb.Prior("Normal", mu=1, sigma=2),
        },
        "low": {
            "display": r"$\mathcal{N}(0,1)$",
            "bambi": bmb.Prior("Normal", mu=0, sigma=1),
        },
    }
    Nu_priors = {
        "orig": {
            "display": r"$\Gamma(2,0.1)$",
            "bambi": bmb.Prior("Gamma", alpha=2, beta=0.1),
        },
        "expon": {
            "display": r"$\textrm{Expon}(1/30)$",
            "bambi": bmb.Prior("Exponential", lam=1/30),
        },
    }

    experiemnt_columns = [
        "M_prior",
        "logSigma_prior",
        "Nu_prior",
    ]

    result_columns = [
        "dmeans_mean",
        "dmeans_95hdi",
        "dsigmas_mode",
        "dsigmas_95hdi",
        "nu_mode",
        "codhend_mode",
    ]


    experiments = [
        dict(name="orig", mean="orig",    sigma="orig", nu="orig"),
        dict(name="orig", mean="wider",   sigma="orig", nu="orig"),
        dict(name="orig", mean="tighter", sigma="orig", nu="orig"),
        dict(name="orig", mean="orig",    sigma="low",  nu="orig"),
        dict(name="orig", mean="orig",    sigma="orig", nu="expon"),
    ]

    # Prepare results table
    results_columns = experiemnt_columns + result_columns
    results_rows = range(len(experiments))
    results = pd.DataFrame(index=results_rows, columns=results_columns)


    for i, exp in enumerate(experiments):
        priors = {}  # priors to be used for current run

        # Set priors based on specificatino in `exp`
        results.loc[i, "M_prior"] = M_priors[exp["mean"]]["display"]
        priors["group"] = M_priors[exp["mean"]]["bambi"]
        results.loc[i, "logSigma_prior"] = logSigma_priors[exp["sigma"]]["display"]
        priors["sigma"] = {"group": logSigma_priors[exp["sigma"]]["bambi"]}
        results.loc[i, "Nu_prior"] = Nu_priors[exp["nu"]]["display"]
        priors["nu"] = Nu_priors[exp["nu"]]["bambi"]

        # Fit model
        idata2 = fit_bayesian_model_iqs2(iqs2, priors, random_seed=42)
        calc_dmeans_stats(idata2, group_name="group")

        # Calculate results
        post2 = idata2["posterior"]
        summary2 = az.summary(post2, kind="stats", hdi_prob=0.95)
        ### Calculate dmeans_mean
        results.loc[i, "dmeans_mean"] = summary2.loc["dmeans", "mean"]
        ### Calculate dmeans_95hdi
        dmeans_ci_low = summary2.loc["dmeans", "hdi_2.5%"]
        dmeans_ci_high = summary2.loc["dmeans","hdi_97.5%"]
        results.loc[i, "dmeans_95hdi"] = [dmeans_ci_low, dmeans_ci_high]
        ### Calculate dsigmas_mode
        dsigmas = post2["dsigmas"].values.flatten()
        results.loc[i, "dsigmas_mode"] = calc_point_est("mode", dsigmas).round(3)
        ### Calculate dsigmas_95hdi
        dsigmas_ci_low = summary2.loc["dsigmas", "hdi_2.5%"]
        dsigmas_ci_high = summary2.loc["dsigmas","hdi_97.5%"]
        results.loc[i, "dsigmas_95hdi"] = [dsigmas_ci_low, dsigmas_ci_high]
        ### Calculate nu_mode
        nus = post2["nu"].values.flatten()
        results.loc[i, "nu_mode"] = calc_point_est("mode", nus).round(3)
        ### Calculate codhend_mode
        cohends = post2["dsigmas"].values.flatten()
        results.loc[i, "codhend_mode"] = calc_point_est("mode", cohends).round(3)

    return results






# PERFORMANCE ANALYSIS
################################################################################

def gen_dmeans_datasets():
	"""
	This function prepares a dictionary of datasets that exhibit different
	combinations of the characteristics we might encounter in the real world.
	To keep things simple,
	we'll assume both populations have unit standard deviation sigma_A = sigma_B = 1
	and group A has mean mu_A=0.
	- Effect size Delta: 0, 0.2, 0.5, 1.0,
	  which means Group B has mean mu_B= 0, 0.2, 0.5, 1.0.
    - Outliers: no outliers, some outliers, lots of outliers.
    - Sample size: $m=n=20$, $m=n=30$, $m=n=50$, $m=n=100$.
    We'll generate samples from each combination of conditions,
    then use various models to perform the hypothesis tests and count
    the number of times we reach the correct decision.
    """
	pass



def is_inside(val, interval):
    """
    Check if the value `val` is inside the interval `interval`.
    """
    if val >= interval[0] and val <= interval[1]:
        return True
    else:
        return False


def calc_dmeans(idata, group_name="group", groups=["ctrl", "treat"]):
    """
    Simplified version of `ministats.bayes.calc_dmeans_stats` used to
    calculate `dmeans` posterior from difference of group means.
    """
    group_dim = group_name + "_dim"
    post = idata["posterior"]
    post["mu_" + groups[0]] = post[group_name].loc[{group_dim:groups[0]}]
    post["mu_" + groups[1]] = post[group_name].loc[{group_dim:groups[1]}]
    post["dmeans"] = post["mu_" + groups[1]] - post["mu_" + groups[0]]



def fit_dmeans_model(dataset, model, random_seed=42):
    """
    Fit a one of the following models for the difference of two means:
    - Permutation test from Section 3.5
    - Welch's two-sample $t$-test from Section 3.5
    - Normal Bayesian model that uses normal as data model
    - Robust Bayesian model that uses t distribution as data model
    - Bayes factor using JZS prior (using `pingouin` library)

    For each model, we run the hypothesis test to decide if the populations are the same or different,
    based on the conventional cutoff 5% and construct interval a 90% interval estimates
    for the difference between population means: Delta = mu_B - mu_A.
    """
    model = model.lower()
    assert model in ["perm", "welch", "norm_bayes", "robust_bayes", "bf"]

    # Helper functions to get decision in each situation
    ################################################################################
    def get_pval_decision(pval, alpha=0.05):
        if pval <= alpha:
            return "reject H0"
        else:
            return "fail to reject H0"

    def get_ci_decision(val, interval):
        if is_inside(val, interval):
            return "fail to reject H0"
        else:
            return "reject H0"

    def get_bf_decision(bfA0, cutoff_reject_H0=3, cutoff_accept_H0=1/3):
        if bfA0 >= cutoff_reject_H0:
            return "reject H0"
        elif bfA0 <= cutoff_accept_H0:
            return "accept H0"
        else:
            return "no decision"

    treated = dataset[dataset["group"]=="treat"]["value"].values
    controls = dataset[dataset["group"]=="ctrl"]["value"].values

    # DIFFERENT MODELS
    ################################################################################
    if model == "perm":
        np.random.seed(random_seed)
        pval = permutation_test_dmeans(treated, controls)
        decision = get_pval_decision(pval)
        ci90 = ci_dmeans(treated, controls, alpha=0.1, method="b")

    elif model == "welch":
        pval = ttest_dmeans(treated, controls, equal_var=False, alt="two-sided")
        decision = get_pval_decision(pval)
        ci90 = ci_dmeans(treated, controls, alpha=0.1, method="a")

    elif model == "norm_bayes":
        priors = {
            "group": bmb.Prior("Normal", mu=0, sigma=2),
            "sigma": {
                "group": bmb.Prior("LogNormal", mu=0, sigma=1)
                # "group": bmb.Prior("Normal", mu=1, sigma=1)
            }
        }
        formula = bmb.Formula("value ~ 0 + group", "sigma ~ 0 + group")
        norm_mod = bmb.Model(formula=formula, family="gaussian", priors=priors, data=dataset)
        idata = norm_mod.fit(draws=2000, random_seed=random_seed)
        calc_dmeans(idata)
        ci95 = hdi_from_idata(idata, var_name="dmeans", hdi_prob=0.95)
        decision = get_ci_decision(0, ci95)
        ci90 = hdi_from_idata(idata, var_name="dmeans", hdi_prob=0.9)

    elif model == "robust_bayes":
        priors = {
            "group": bmb.Prior("Normal", mu=0, sigma=2),
            "sigma": {
                "group": bmb.Prior("Normal", mu=0, sigma=1)
            },
            "nu": bmb.Prior("Gamma", alpha=2, beta=0.1),
        }
        formula = bmb.Formula("value ~ 0 + group", "sigma ~ 0 + group")
        robust_mod = bmb.Model(formula=formula, family="t", priors=priors, data=dataset)
        idata = robust_mod.fit(draws=2000, random_seed=random_seed)
        calc_dmeans(idata)
        ci95 = hdi_from_idata(idata, var_name="dmeans", hdi_prob=0.95)
        decision = get_ci_decision(0, ci95)
        ci90 = hdi_from_idata(idata, var_name="dmeans", hdi_prob=0.9)

    elif model == "bf":
        ttres = ttest_ind(treated, controls, equal_var=True)
        n, m = len(treated), len(controls)
        bfA0 = pg.bayesfactor_ttest(ttres.statistic, nx=n, ny=m, r=0.707)
        decision = get_bf_decision(bfA0)
        ci90 = None

    return decision, ci90


    




"""
## Success metrics
Since we're analyzing synthetic datasets,
we know what the correct decision is for the result of the hypothesis test.
The correction decision when Delta = 0 is to fail to reject $H_0$,
and reject $H_0$ when Delta neq 0.
We'll count the following errors:

- False positives: a dataset generated from $Delta=0$, for which the analysis results rejects $H_0$.
- False negatives: a dataset with $Delta neq 0$ for which the analysis fails to reject $H_0$.
- Incorrect interval estimates: when the interval estimate (ci or hdi)
  doesn't contain the true value $Delta$ used to generate the synthetic dataset.

"""