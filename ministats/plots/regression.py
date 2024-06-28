import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from scipy.stats import t as tdist
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.nonparametric import smoothers_lowess
import warnings


# Useful colors
snspal = sns.color_palette()
blue, orange, red, purple = snspal[0], snspal[1], snspal[3], snspal[4]



# Simple linear regression
################################################################################

def plot_reg(lmres, ax=None):
    """
    Plot the best-fitting line of a simple linear regression model
    on top of a scatter plot of the dataset.
    """
    ax = plt.gca() if ax is None else ax
    # 1. Extract data
    xname = lmres.model.exog_names[1]
    xs = lmres.model.exog[:,1]
    yname = lmres.model.endog_names
    ys = lmres.model.endog
    # 2. Get model predicitons
    xgrid = np.linspace(np.min(xs), np.max(xs), 100)
    ypred = lmres.get_prediction({xname: xgrid})
    # 3. Draw the scatterplot and plot the best-fit line
    sns.scatterplot(x=xs, y=ys, ax=ax)
    sns.lineplot(x=xgrid, y=ypred.predicted, ax=ax)
    ax.set_xlabel(f"{xname}")
    ax.set_ylabel(f"{yname}")
    return ax


def plot_resid(lmres, pred=None, lowess=False, ax=None):
    """
    Residuals plot for the model `lmres` against the predictor `pred`.
    If `pred` is None, we plot the residuals versus the outcome fitted values.
    The plot shows a dashed horizontal line at `y=0` and an optional LOWESS curve.
    """
    ax = plt.gca() if ax is None else ax
    if pred is None:
        xs = lmres.fittedvalues
        xname = "fitted values"
    else:
        xs = lmres.model.data.orig_exog[pred]
        xname = pred
    ys = lmres.resid
    sns.scatterplot(x=xs, y=ys, color="red", ax=ax)
    ax.axhline(y=0, color="b", linestyle="dotted")
    # ax.axhline(0, ls=":", c=".2")
    if lowess:
        xgrid, ylowess = smoothers_lowess.lowess(xs, ys).T
        sns.lineplot(x=xgrid, y=ylowess)
    ax.set_xlabel(xname)
    ax.set_ylabel("residuals $r_i$")
    return ax


def plot_pred_bands(lmres, ax=None,
                    ci_mean=False, alpha_mean=0.1, lab_mean=True,
                    ci_obs=False, alpha_obs=0.1, lab_obs=True):
    """
    Plot the confidence intervals for the model predcitions.
    If `ci_mean` is True: draw a (1-alpha_mean)-CI for the mean.
    If `ci_obs` is True: draw a (1-ci_obs)-CI for the predicted values.
    """
    ax = plt.gca() if ax is None else ax
    xs = lmres.model.exog[:,1]
    xname = lmres.model.exog_names[1]
    n = lmres.nobs

    # Get model predicitons
    xgrid = np.linspace(np.min(xs), np.max(xs), 100)
    ypred = lmres.get_prediction({xname: xgrid})

    if ci_mean:
        # Draw the confidence interval for the mean predictions
        t_05, t_95 = tdist(df=n-2).ppf([alpha_mean/2, 1-alpha_mean/2])
        lower_mean = ypred.predicted + t_05*ypred.se_mean
        upper_mean = ypred.predicted + t_95*ypred.se_mean
        if lab_mean:
            if isinstance(lab_mean, str):
                label_mean = lab_mean
            else:
                perc_mean = round(100*(1-alpha_mean))
                label_mean = f"{perc_mean}% confidence interval for the mean"
        else:
            label_mean = None
        ax.fill_between(xgrid, lower_mean, upper_mean, alpha=0.4, color="C0", label=label_mean)

    if ci_obs:
        # Draw the confidence interval for the predicted observations
        t_05, t_95 = tdist(df=n-2).ppf([alpha_obs/2, 1-alpha_obs/2])
        lower_obs = ypred.predicted + t_05*ypred.se_obs
        upper_obs = ypred.predicted + t_95*ypred.se_obs
        if lab_obs:
            if isinstance(lab_obs, str):
                label_obs = lab_obs
            else:
                perc_obs = round(100*(1-alpha_obs))
                label_obs = f"{perc_obs}% confidence interval for observations"
        else:
            label_obs = None
        ax.fill_between(xgrid, lower_obs, upper_obs, alpha=0.1, color="C0", label=label_obs)

    if (ci_mean and lab_mean) or (ci_obs and lab_obs):
        ax.legend()



# Multiple linear regression
################################################################################

def plot_partreg(lmres, pred, ax=None):
    """
    Generate a partial regression plot from the model results `lmres`
    for the predictor `pred` given the other predictors.
    We plot the residuals of `outcome ~ other` along the y-axis,
    and the residuals of the model `pred ~ other` on the x-axis.
    """
    ax = plt.gca() if ax is None else ax
    xdata = lmres.model.data.orig_exog
    ydata = lmres.model.data.orig_endog
    data = pd.concat([xdata, ydata], axis=1)

    # Send to specialized function if the model contains categorical predictors
    if any(["C(" in name for name in xdata.columns]):
        return plot_partreg_cat(lmres, pred, ax=ax)

    # Find others= as list of strings
    allpreds = list(xdata.columns)
    names_to_skip = ["Intercept", pred]
    others = [name for name in allpreds if name not in names_to_skip]
    others_formula = "1"
    if others:
        others_formula += "+" + "+".join(others)

    # Fit model for x-axis = residuals of `pred ~ 1 + others`
    lmpred = smf.ols(f"{pred} ~ {others_formula}", data=data).fit()
    xresids = lmpred.resid

    # Fit model for y-axis = residuals of `outcome ~ 1 + others`
    outname = lmres.model.endog_names
    lmoutcome = smf.ols(f"{outname} ~ {others_formula}", data=data).fit()
    yresids = lmoutcome.resid

    # Draw scatter plot
    sns.scatterplot(x=xresids, y=yresids, ax=ax)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    # Plot best-fitting line
    dfresids = pd.DataFrame({"xresids": xresids, "yresids": yresids})
    lmresids = smf.ols("yresids ~ 0 + xresids", data=dfresids).fit()
    slope = lmresids.params.iloc[0]
    xgrid = np.linspace(*ylims, 100)
    ys = slope*xgrid
    sns.lineplot(x=xgrid, y=ys, ax=ax)

    # Add descriptive labels
    if len(others_formula) > 20:
        others_formula = "other"
    ax.set_xlabel(f"{pred} ~ {others_formula}  residuals")
    ax.set_ylabel(f"{outname} ~ {others_formula}  residuals")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    return ax


def plot_projreg(lmres, pred, others=None, ax=None):
    """
    Generate a partial regression plot from the best-fit line
    of the predictor `pred`, where the intercept is calculated
    from the average of the `other` predictors.
    """
    ax = plt.gca() if ax is None else ax
    xdata = lmres.model.data.orig_exog
    xs = xdata[pred]
    ys = lmres.model.endog
    yname = lmres.model.endog_names
    sns.scatterplot(x=xs, y=ys, ax=ax)
    params = lmres.params
    allpreds = set(xdata.columns) - {"Intercept"}
    assert pred in allpreds 
    others = allpreds - {pred} if others is None else others
    intercept = params["Intercept"]
    for other in others:
        intercept += params[other]*xdata[other].mean() 
    slope = params[pred]
    print(pred, "intercept=", intercept, "slope=", slope)
    xgrid = np.linspace(xs.min(), xs.max())
    ypred = intercept + slope*xgrid
    sns.lineplot(x=xgrid, y=ypred, ax=ax)
    ax.set_xlabel(pred)
    ax.set_ylabel(yname)
    return ax


def plot_partreg_cat(lmres, pred, ax=None):
    """
    A version of `plot_partreg` that can handle categorical predictors.
    """
    ax = plt.gca() if ax is None else ax
    xdata = lmres.model.data.orig_exog
    ydata = lmres.model.data.orig_endog
    data = pd.concat([xdata, ydata], axis=1)

    # Find others= as list of strings
    allpreds = list(xdata.columns)
    names_to_skip = ["Intercept", pred]
    others = [name for name in allpreds if name not in names_to_skip]
    others_formula = "1"
    others_display = "1"
    if others:
        others_quoted = ["Q('"+ other + "')" for other in others]
        others_formula += "+" + "+".join(others_quoted)
        for other in others:
            m = re.match("C\((?P<varname>.*)\).*", other)
            if m:
                varname = m.groupdict()["varname"]
                other_clean = "C(" + varname + ")"
            else:
                other_clean = other
            if other_clean not in others_display:
                others_display += "+" + other_clean

    # x-axis = residuals of `pred ~ 1 + others`
    lmpred = smf.ols(f"{pred} ~ {others_formula}", data=data).fit()
    xresids = lmpred.resid

    # y-axis = residuals of `outcome ~ 1 + others`
    outname = lmres.model.endog_names
    lmoutcome = smf.ols(f"{outname} ~ {others_formula}", data=data).fit()
    yresids = lmoutcome.resid

    # scatter plot
    sns.scatterplot(x=xresids, y=yresids, ax=ax)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    # best-fit line between the residuals
    dfresids = pd.DataFrame({"xresids": xresids, "yresids": yresids})
    lmresids = smf.ols("yresids ~ 0 + xresids", data=dfresids).fit()
    slope = lmresids.params.iloc[0]
    xs = np.linspace(*ylims, 100)
    ys = slope*xs
    sns.lineplot(x=xs, y=ys, ax=ax)

    # ax.set_title('Partial regression plot')
    if len(others_display) > 20:
        others_display = "other"
    ax.set_xlabel(f"{pred} ~ {others_display}  residuals")
    ax.set_ylabel(f"{outname} ~ {others_display}  residuals")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    return ax


# Diagnostic plots
################################################################################

def plot_scaleloc(lmres, lowess=True):
    """
    Plot the scale-location plot for the linear model `lmres`.
    """
    sigmahat = np.sqrt(lmres.scale)
    std_resids = lmres.resid / sigmahat
    sqrt_abs_std_resids = np.sqrt(np.abs(std_resids))
    xs = lmres.fittedvalues
    ax = sns.regplot(x=xs, y=sqrt_abs_std_resids, lowess=True)
    # TODO: repalce with scatterplot + manual lowess lineplot
    ax.set_xlabel("fitted values")
    ax.set_ylabel(r"$\sqrt{|\text{standardized residuals}|}$")
    return ax



# Old linear model plotting functions (to be deprecated in next release)
################################################################################

def plot_lm_simple(xs, ys, ax=None, ci_mean=False, alpha_mean=0.1, lab_mean=True,
                   ci_obs=False, alpha_obs=0.1, lab_obs=True): 
    """
    Draw a scatter plot of the data `[xs,ys]`, a regression line,
    and optionally show confidence intervals for the model predcitions.
    If `ci_mean` is True: draw a (1-alpha_mean)-CI for the mean.
    If `ci_obs` is True: draw a (1-ci_obs)-CI for the predicted values.
    """
    warnings.warn("This function is replaced by plot_reg and plot_pred_bands", DeprecationWarning)
    ax = plt.gca() if ax is None else ax

    # Prepare the data
    xname = xs.name if hasattr(xs, "name") else "x"
    yname = ys.name if hasattr(ys, "name") else "y"
    data = pd.DataFrame({xname: xs, yname: ys})

    # Fit the linear model
    formula = f"{yname} ~ 1 + {xname}"
    lm = smf.ols(formula, data=data).fit()

    plot_reg(lm, ax=ax)
    plot_pred_bands(lm, ax=ax,
                    ci_mean=ci_mean, alpha_mean=alpha_mean, lab_mean=lab_mean,
                    ci_obs=ci_obs, alpha_obs=alpha_obs, lab_obs=lab_obs)
    return ax


def plot_lm_partial_old(lmfit, pred, others=None, ax=None):
    """
    Generate a partial regression plot from the best-fit line
    of the predictor `pred`, where the intercept is calculated
    from the average of the `other` predictors.
    """
    warnings.warn("This function is replaced by plot_projreg", DeprecationWarning)
    return plot_projreg(lmfit, pred, others=others, ax=ax)


def plot_lm_partial(lmfit, pred, others=None, ax=None):
    """
    Generate a partial regression plot from the model `lmfit`
    for the predictor `pred`, given the `other` predictors.
    We plot the residuals of `outcome ~ other` along the y-axis,
    and the residuals of the model `pred ~ other` on the x-axis.
    """
    warnings.warn("This function is replaced by plot_partreg", DeprecationWarning)
    return plot_partreg(lmfit, pred=pred, ax=ax)



# DEPRECATED SINCE `plot_partreg` subtracts all other vars
def plot_projreg_cat(lmfit, pred, others=None, color="C0", linestyle="solid", cats=None, ax=None):
    """
    Generate a partial regression plot from the best-fit line
    of the predictor `pred`, where the intercept is calculated
    from the average of the `other` predictors,
    including the value of categorical predictors `cats` in the slope.
    """
    ax = plt.gca() if ax is None else ax
    data = lmfit.model.data.orig_exog
    params = lmfit.params
    allpreds = set(data.columns) - {"Intercept"}
    allnoncatpreds = set([pred for pred in allpreds if "T." not in pred])
    assert pred in allnoncatpreds
    others = allnoncatpreds - {pred} if others is None else others
    intercept = params["Intercept"]
    for other in others:
        intercept += params[other]*data[other].mean() 
    for cat in cats:
        intercept += params[cat]
    slope = params[pred]
    print(pred, "intercept=", intercept, "slope=", slope)
    xs = np.linspace(data[pred].min(), data[pred].max())
    ys = intercept + slope*xs
    sns.lineplot(x=xs, y=ys, color=color, ax=ax, linestyle=linestyle)
    return ax
