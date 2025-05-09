from collections import defaultdict
import os
import time

import arviz as az
from arviz.plots.plot_utils import calculate_point_estimate as calc_point_est
import bambi as bmb
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import norm
from scipy.stats import ttest_ind

from ..bayes import calc_dmeans_stats
from ..bayes import hdi_from_idata
from ..confidence_intervals import ci_dmeans
from ..hypothesis_tests import permutation_test_dmeans
from ..hypothesis_tests import ttest_dmeans
from ..utils import loglevel


# BAYESIAN DMEANS SENSITIVITY ANALYSIS
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
    with loglevel("ERROR", module="pymc"):
        idata2 = mod2.fit(draws=2000, random_seed=random_seed, progressbar=False)
    return idata2



def sens_analysis_dmeans_iqs2(iqs2):
    """
    Generate the table showing the results of the sensitivity analysis
    for Example 2 in Section 5.4.
    """
    # Define the priors
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
    
    experiments = [
        dict(name="orig", mean="orig",    sigma="orig", nu="orig",  seed=42),
        dict(name="orig", mean="wider",   sigma="orig", nu="orig",  seed=43),
        dict(name="orig", mean="tighter", sigma="orig", nu="orig",  seed=44),
        dict(name="orig", mean="orig",    sigma="low",  nu="orig",  seed=45),
        dict(name="orig", mean="orig",    sigma="orig", nu="expon", seed=46),
    ]

    # Prepare results table
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
    results_columns = experiemnt_columns + result_columns
    results_rows = range(len(experiments))
    results = pd.DataFrame(index=results_rows, columns=results_columns)

    for i, exp in enumerate(experiments):
        # print("fitting model", i, "...")
        priors = {}  # priors to be used for current run

        # Set priors based on specification in `exp`
        results.loc[i, "M_prior"] = M_priors[exp["mean"]]["display"]
        priors["group"] = M_priors[exp["mean"]]["bambi"]
        results.loc[i, "logSigma_prior"] = logSigma_priors[exp["sigma"]]["display"]
        priors["sigma"] = {"group": logSigma_priors[exp["sigma"]]["bambi"]}
        results.loc[i, "Nu_prior"] = Nu_priors[exp["nu"]]["display"]
        priors["nu"] = Nu_priors[exp["nu"]]["bambi"]

        # Fit model
        idata2 = fit_bayesian_model_iqs2(iqs2, priors, random_seed=exp["seed"])
        calc_dmeans_stats(idata2, group_name="group")

        # Calculate results
        post2 = idata2["posterior"]
        summary2 = az.summary(post2, kind="stats", hdi_prob=0.95)
        ## Calculate dmeans_mean
        results.loc[i, "dmeans_mean"] = summary2.loc["dmeans", "mean"]
        ## Calculate dmeans_95hdi
        dmeans_ci_low = summary2.loc["dmeans", "hdi_2.5%"]
        dmeans_ci_high = summary2.loc["dmeans","hdi_97.5%"]
        results.loc[i, "dmeans_95hdi"] = [dmeans_ci_low, dmeans_ci_high]
        ## Calculate dsigmas_mode
        dsigmas = post2["dsigmas"].values.flatten()
        results.loc[i, "dsigmas_mode"] = calc_point_est("mode", dsigmas).round(3)
        ## Calculate dsigmas_95hdi
        dsigmas_ci_low = summary2.loc["dsigmas", "hdi_2.5%"]
        dsigmas_ci_high = summary2.loc["dsigmas","hdi_97.5%"]
        results.loc[i, "dsigmas_95hdi"] = [dsigmas_ci_low, dsigmas_ci_high]
        ## Calculate nu_mode
        nus = post2["nu"].values.flatten()
        results.loc[i, "nu_mode"] = calc_point_est("mode", nus).round(3)
        ## Calculate codhend_mode
        cohends = post2["dsigmas"].values.flatten()
        results.loc[i, "codhend_mode"] = calc_point_est("mode", cohends).round(3)

    return results






# DMEANS PERFORMANCE ANALYSIS
################################################################################

def gen_dmeans_dataset(n, Delta, prop_outliers=0, random_seed=42):
    """
    Generate a dataset with two-groups of size `n`.
    The mean of the control group is `0`,
    the mean of the treatment group is `Delta`.
    We'll make a proportion `prop_outliers` of the values
    in each group outliers, coming from a population
    with much wider standard deviation.
    """
    np.random.seed(random_seed)
    
    # Generate control group
    controls = norm(0, 1).rvs(n)

    # Generate treated group
    treated = norm(0 + Delta, 1).rvs(n)

    # Add outliers
    n_outliers = int(prop_outliers * n)
    control_outliers = norm(0, 5).rvs(n_outliers)  
    controls[0:n_outliers] = control_outliers
    treated_outliers = norm(0 + Delta, 5).rvs(n_outliers) 
    treated[0:n_outliers] = treated_outliers

    # Package dataset as a Pandas DataFrame
    groups = ["treat"]*len(treated) + ["ctrl"]*len(controls)
    values = np.concatenate((treated, controls))
    dataset = pd.DataFrame({"group": groups, "value": values})
    return dataset


def gen_dmeans_datasets(ns, Deltas, outliers_options, random_seed_start=45):
    """
	This function prepares a dictionary of datasets that exhibit different
	combinations of the characteristics we might encounter in the real world.
	To keep things simple,
	we'll assume both populations have unit standard deviation sigma_A = sigma_B = 1
	and group A has mean mu_A=0.
    - Sample size: n=20, n=30, n=50, or n=100
    - Effect size Delta: 0, 0.2, 0.5, 1.0,
	  which means Group B has mean mu_B= 0, 0.2, 0.5, 0.8, 1.3.
    - Outliers: no outliers, some outliers, lots of outliers.
    We'll generate samples from each combination of conditions,
    then use various models to perform the hypothesis tests and count
    the number of times we reach the correct decision.
    """
    random_seed = random_seed_start
    dataset_specs = []
    for n in ns:
        for Delta in Deltas:
            for outliers in outliers_options:
                spec = dict(n=n,  Delta=Delta, outliers=outliers, random_seed=random_seed)
                dataset_specs.append(spec)
                random_seed += 1

    return dataset_specs



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



def fit_dmeans_models(dataset, random_seed=42):
    """
    Fit the following models for the difference of two means:
    - Permutation test from Section 3.5
    - Welch's two-sample $t$-test from Section 3.5
    - Bayesian model that uses normal as data model
    - Robust Bayesian model that uses t-distribution as data model
    - Bayes factor using JZS prior (using `pingouin` library)
    For each model, we run the hypothesis test to decide if the populations
    are the same or different based on the conventional cutoff level of 5%.
    We also construct a 90% interval estimates for the unknown `Delta`.
    """
    models = ["perm", "welch", "norm_bayes", "robust_bayes", "bf"]

    # Helper functions to get decision in each situation
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

    # Fit the models
    results = {}
    if "perm" in models:
        np.random.seed(random_seed)
        pval = permutation_test_dmeans(treated, controls)
        decision = get_pval_decision(pval)
        ci90 = ci_dmeans(treated, controls, alpha=0.1, method="b")
        results["perm"] = {"decision": decision, "ci90": ci90}

    if "welch" in models:
        pval = ttest_dmeans(treated, controls, equal_var=False, alt="two-sided")
        decision = get_pval_decision(pval)
        ci90 = ci_dmeans(treated, controls, alpha=0.1, method="a")
        results["welch"] = {"decision": decision, "ci90": ci90}

    if "norm_bayes" in models:
        priors = {
            "group": bmb.Prior("Normal", mu=0, sigma=2),
            "sigma": {
                "group": bmb.Prior("LogNormal", mu=0, sigma=1)
                # "group": bmb.Prior("Normal", mu=1, sigma=1)
            }
        }
        formula = bmb.Formula("value ~ 0 + group", "sigma ~ 0 + group")
        norm_mod = bmb.Model(formula=formula, family="gaussian", priors=priors, data=dataset)
        with loglevel("ERROR", module="pymc"):
            idata = norm_mod.fit(draws=2000, random_seed=random_seed, progressbar=False)
        calc_dmeans(idata)
        ci95 = hdi_from_idata(idata, var_name="dmeans", hdi_prob=0.95)
        decision = get_ci_decision(0, ci95)
        ci90 = hdi_from_idata(idata, var_name="dmeans", hdi_prob=0.9)
        results["norm_bayes"] = {"decision": decision, "ci90": ci90}

    if "robust_bayes" in models:
        priors = {
            "group": bmb.Prior("Normal", mu=0, sigma=2),
            "sigma": {
                "group": bmb.Prior("Normal", mu=0, sigma=1)
            },
            "nu": bmb.Prior("Gamma", alpha=2, beta=0.1),
        }
        formula = bmb.Formula("value ~ 0 + group", "sigma ~ 0 + group")
        robust_mod = bmb.Model(formula=formula, family="t", priors=priors, data=dataset)
        with loglevel("ERROR", module="pymc"):
            idata = robust_mod.fit(draws=2000, random_seed=random_seed, progressbar=False)
        calc_dmeans(idata)
        ci95 = hdi_from_idata(idata, var_name="dmeans", hdi_prob=0.95)
        decision = get_ci_decision(0, ci95)
        ci90 = hdi_from_idata(idata, var_name="dmeans", hdi_prob=0.9)
        results["robust_bayes"] = {"decision": decision, "ci90": ci90}

    if "bf" in models:
        ttres = ttest_ind(treated, controls, equal_var=True)
        n, m = len(treated), len(controls)
        bfA0 = pg.bayesfactor_ttest(ttres.statistic, nx=n, ny=m, r=0.707)
        decision = get_bf_decision(bfA0)
        ci90 = None
        results["bf"] = {"decision": decision, "ci90": ci90}

    return results



def calc_dmeans_perf_metrics(
        ns=[20,30,50,100],
        Deltas=[0, 0.2, 0.5, 0.8, 1.3],
        outliers_options=["no", "few", "lots"],
        reps=100,
        random_seed_start=45):
    """
    Calculate the performance of the different statistical analysis procedures
    for each dataset specification, based on `reps` repeated samples.
    We'll count the following outcomes:    
    - `count_reject`: number of times H_0: Delta=0 is rejected
    - `count_fail_to_reject`: number of times we fail to reject H_0: Delta=0
    - `count_captured`: when the interval estimate (ci or hdi) contain the true `Delta`
    - `avg_width`: average width of the interval estimate (ci or hdi) over all `reps`
    """
    dataset_specs = gen_dmeans_datasets(ns=ns,
                                        Deltas=Deltas,
                                        outliers_options=outliers_options,
                                        random_seed_start=random_seed_start)
    print("Simulating a total of", len(dataset_specs), "dataset specs")

    dataset_columns = [
        "n",
        "Delta",
        "outliers",
        "seed",
    ]
    metrics_columns = [
        "count_reject",
        "count_fail_to_reject",
        "count_captured",
        "avg_width",
    ]

    # Check if cached simulation data exists
    filename = "dmeans_perf_metrics" \
                + "__ns_" + "_".join(map(str,ns)) \
                + "__Deltas_" + "_".join(map(str,Deltas)) \
                + "__outs_" + "_".join(map(str,outliers_options)) \
                + "__reps_" + str(reps) + ".csv"
    filepath = os.path.join("simdata", filename)
    if os.path.exists(filepath):
        print("loaded cached results from ", filepath)
        results = pd.read_csv(filepath, index_col=[0,1])
        return results
    
    # Prepare results table
    results_columns = dataset_columns + metrics_columns
    specs_idx = range(len(dataset_specs))
    models_idx = ["perm", "welch", "norm_bayes", "robust_bayes", "bf"]
    results_index = pd.MultiIndex.from_product((specs_idx, models_idx), names=["spec", "model"])
    results = pd.DataFrame(index=results_index, columns=results_columns)

    
    for row_idx, spec in enumerate(dataset_specs):
        print(spec)

        # Get rows that correspond to this `spec` (for all models)
        spec_results = results.loc[(row_idx,),:]

        # Save dataset spec columns
        spec_results.loc[:,"n"] = spec["n"]
        spec_results.loc[:,"Delta"] = spec["Delta"]
        spec_results.loc[:,"outliers"] = spec["outliers"]
        spec_results.loc[:,"seed"] = spec["random_seed"]
        # Initialize metrics columns
        spec_results.loc[:,"count_reject"] = 0
        spec_results.loc[:,"count_fail_to_reject"] = 0
        spec_results.loc[:,"count_captured"] = 0
        spec_results.loc["bf","count_captured"] = pd.NA
        spec_results.loc["bf","avg_width"] = pd.NA

        # Interpret the sparsity specification
        spec["outliers"] in ["no", "few", "lots"]
        if spec["outliers"] == "no":
            prop_outliers = 0
        elif spec["outliers"] == "few":
            prop_outliers = 0.02
        elif spec["outliers"] == "lots":
            prop_outliers = 0.05


        # Repeat for `rep` datasets
        ci_widths = defaultdict(list)
        t_before_reps = time.time()
        for rep in range(reps):
            if rep % 10 == 0:
                print("Running", spec, "rep", rep) 
            rep_random_seed = spec["random_seed"] * 10000 + rep
            dataset = gen_dmeans_dataset(n=spec["n"],
                                         Delta=spec["Delta"],
                                         prop_outliers=prop_outliers,
                                         random_seed=rep_random_seed)
            # Run all the tests
            rep_results = fit_dmeans_models(dataset, random_seed=rep_random_seed)
            for model, model_results in rep_results.items():
                # Hypothesis test metrics
                decision = model_results["decision"]
                if decision == "reject H0":
                    spec_results.loc[model, "count_reject"] += 1
                else:
                    spec_results.loc[model, "count_fail_to_reject"] += 1
                if model == "bf":
                    continue
                # Confidence interval metrics
                ci90 = model_results["ci90"]
                if is_inside(spec["Delta"], ci90):
                    spec_results.loc[model, "count_captured"] += 1
                ci90_width = ci90[1] - ci90[0]
                ci_widths[model].append(ci90_width)

        t_after_reps = time.time()
        reps_time = t_after_reps - t_before_reps
        print(reps, "reps took", reps_time, "to run...")

        for model, widths in ci_widths.items():
            spec_results.loc[model,"avg_width"] = np.mean(widths)

    # Cache results to avoid need to recompute
    results.to_csv(filepath)
    print("Saved file to " + filepath)
    return results




# Tables with success metrics
################################################################################

def get_perf_table_typeI(results):
    """
    Analysis of false positive results for all models.
    For all the datasets generated with Delta=0 (the null hypothesis is true)
    but it is possible the hypothesis test procedure will reject H0,
    which is a Type I error (false positive).
    """
    results = results.copy()

    # Infer the number of `reps` that were simulated for each spec
    first_result = results.iloc[0,:]
    reps = first_result["count_reject"] + first_result["count_fail_to_reject"]

    # Calculate the false positive rate for each (spec,model) combination
    results.loc[:,"false_positives"] = results.loc[:,"count_reject"] / reps

    # Select only relevant rows and columns
    subset_rows = results["Delta"] == 0  # when H0 is true
    subset_cols = ["n", "outliers", "false_positives"]
    subset = results.loc[subset_rows, subset_cols]
    
    # Reshape the data to prepare the Type I errors table
    tableA = subset.reset_index(level="spec", drop=True) \
                   .assign(outliers=pd.Categorical(subset["outliers"],
                                                   categories=["no", "few", "lots"],
                                                   ordered=True)) \
                   .pivot_table(index=["outliers", "n"],
                                columns="model",
                                values="false_positives",
                                observed=True) \
                   .reindex(columns=["perm", "welch", "norm_bayes", "robust_bayes", "bf"]) \
                   .sort_index()
    return tableA


def get_perf_table_power(results, show_all=True):
    """
    Analysis of power (true positives) for all models.
    For datasets Delta ≠ 0, correction decision is to reject H0.
    The `power` is the ability to the test to make the correct decision.
    """
    results = results.copy()

    # Infer the number of `reps` that were simulated for each spec
    first_result = results.iloc[0,:]
    reps = first_result["count_reject"] + first_result["count_fail_to_reject"]

    # Calculate the power for each (spec,model) combination
    results.loc[:,"power"] = 1 - results.loc[:,"count_fail_to_reject"] / reps

    # Select only relevant rows and columns
    cond1 = results["Delta"] != 0    # not part of the experiment
    cond2 = results["Delta"] != 0.2  # near 0% power
    cond3 = results["Delta"] != 1.3  # near 100% power
    cond4 = ~((results["Delta"] == 0.8) & (results["n"] == 100))  # near 100%
    cond5 = ~((results["Delta"] == 0.5) & (results["n"] == 20))   # near 0%
    if show_all:  # For display in notebooks
        subset_rows = cond1
    else:         # For reduced display in print book
        subset_rows = cond1 & cond2 & cond3 & cond4 & cond5
    subset_cols = ["n", "outliers", "Delta", "power"]
    subset = results.loc[subset_rows, subset_cols]
    
    # Reshape the data to prepare the power table
    tableB = subset.reset_index(level="spec", drop=True) \
                   .assign(outliers=pd.Categorical(subset["outliers"],
                                                   categories=["no", "few", "lots"],
                                                   ordered=True)) \
                   .pivot_table(index=["outliers", "Delta", "n"],
                                columns="model",
                                values="power",
                                observed=True) \
                   .reindex(columns=["perm", "welch", "norm_bayes", "robust_bayes", "bf"]) \
                   .sort_index()
    return tableB


def get_perf_table_coverage(results):
    """
    Calculates the summary of the coverage probability
    and width of the interval estimates from different models.
    """
    results = results.copy()

    # Infer the number of `reps` that were simulated for each spec
    first_result = results.iloc[0,:]
    reps = first_result["count_reject"] + first_result["count_fail_to_reject"]

    # Calculate the coverage proportion for each (spec,model) combination
    results.loc[:,"coverage"] = results.loc[:,"count_captured"] / reps

    # Select only relevant rows and columns
    subset_rows = results["Delta"] != 0
    subset_cols = ["n", "outliers", "Delta", "coverage", "avg_width"]
    subset = results.loc[subset_rows, subset_cols]
    # Drop the `bf` model rows because they don't have interval estimates
    bf_rows_to_drop = subset.loc[pd.IndexSlice[:,"bf"],:].index
    subset = subset.drop(index=bf_rows_to_drop)

    # Reshape the data to prepare the interval estimates table
    tableC = subset.reset_index(level="spec", drop=True) \
                   .assign(outliers=pd.Categorical(subset["outliers"],
                                                   categories=["no", "few", "lots"],
                                                   ordered=True)) \
                   .pivot_table(index=["outliers", "n", "Delta"],
                                columns="model",
                                values=["coverage", "avg_width"],
                                observed=True) \
                   .groupby(level=["outliers", "n"],
                            observed=True).mean() 
    tableC.columns = tableC.columns.swaplevel(0, 1)

    # Set the desired sort order of the columns index
    models = ["perm", "welch", "norm_bayes", "robust_bayes"]
    metrics = ["coverage", "avg_width"]
    new_cols = [(model, metric) for model in models for metric in metrics]
    tableC = tableC.loc[:, new_cols]
    return tableC

