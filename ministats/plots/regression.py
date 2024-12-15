import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from scipy.stats import t as tdist
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.nonparametric import smoothers_lowess



# Simple linear regression
################################################################################

def plot_reg(lmres, ax=None):
    """
    Plot the best-fitting line of the simple linear regression
    model `lmres` on top of a scatter plot of the original data.
    """
    ax = plt.gca() if ax is None else ax
    # Draw scatter plot of the data
    xs = lmres.model.data.orig_exog.iloc[:,1]
    ys = lmres.model.data.orig_endog.iloc[:,0]
    sns.scatterplot(x=xs, y=ys, ax=ax)
    # Plot the model's predictions
    xgrid = np.linspace(np.min(xs), np.max(xs), 100)
    ypred = lmres.get_prediction({xs.name: xgrid})
    sns.lineplot(x=xgrid, y=ypred.predicted, ax=ax)
    return ax


def plot_resid(lmres, pred=None, lowess=False, ax=None):
    """
    Residuals plot for the model `lmres` vs. the predictor `pred`.
    If `pred` is None, plot the residuals vs. the fitted values.
    The function can an optional LOWESS curve for the residuals.
    """
    ax = plt.gca() if ax is None else ax
    if pred is None or pred in ["fittedvalues", "predictions"]:
        xs = lmres.fittedvalues
        xname = "fitted values"
    else:
        xs = lmres.model.data.orig_exog[pred]
        xname = pred
    ys = lmres.resid
    sns.scatterplot(x=xs, y=ys, ax=ax)
    ax.axhline(y=0, color="k", linestyle="dotted", alpha=0.6)
    if lowess:
        xgrid, ylowess = smoothers_lowess.lowess(ys, xs, frac=0.72).T
        sns.lineplot(x=xgrid, y=ylowess, ax=ax)
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
    xs = lmres.model.data.orig_exog.iloc[:,1]
    n = lmres.nobs

    # Get model predicitons
    xgrid = np.linspace(np.min(xs), np.max(xs), 100)
    ypred = lmres.get_prediction({xs.name: xgrid})

    if ci_mean:
        # Draw the confidence interval for the mean predictions
        t_05, t_95 = tdist(df=n-2).ppf([alpha_mean/2, 1-alpha_mean/2])
        lower_mean = ypred.predicted + t_05*ypred.se_mean
        upper_mean = ypred.predicted + t_95*ypred.se_mean
        if lab_mean:
            perc_mean = round(100*(1-alpha_mean))
            label_mean = f"{perc_mean}% confidence interval for the mean"
        else:
            label_mean = None
        ax.fill_between(xgrid, lower_mean, upper_mean, color="C0", alpha=0.4, label=label_mean)

    if ci_obs:
        # Draw the confidence interval for the predicted observations
        t_05, t_95 = tdist(df=n-2).ppf([alpha_obs/2, 1-alpha_obs/2])
        lower_obs = ypred.predicted + t_05*ypred.se_obs
        upper_obs = ypred.predicted + t_95*ypred.se_obs
        if lab_obs:
            perc_obs = round(100*(1-alpha_obs))
            label_obs = f"{perc_obs}% confidence interval for observations"
        else:
            label_obs = None
        ax.fill_between(xgrid, lower_obs, upper_obs, color="C0", alpha=0.1, label=label_obs)

    if (ci_mean and lab_mean) or (ci_obs and lab_obs):
        ax.legend()
    return ax



# Multiple linear regression
################################################################################

def plot_partreg(lmres, pred, ax=None):
    """
    Generate a partial regression plot from the linear model `lmres`
    against the predictor `pred`, given the other predictors.
    We plot the residuals of the model `outcome ~ other` along the y-axis,
    versus the residuals of the model `pred ~ other` along the x-axis.
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

    # x-axis = residuals of the model `pred ~ 1 + others`
    lmpred = smf.ols(f"{pred} ~ {others_formula}", data=data).fit()
    xresids = lmpred.resid

    # y-axis = residuals of the model `outcome ~ 1 + others`
    outname = lmres.model.endog_names
    lmoutcome = smf.ols(f"{outname} ~ {others_formula}", data=data).fit()
    yresids = lmoutcome.resid

    # Draw a scatter plot of the y-residuals vs. the x-residuals
    sns.scatterplot(x=xresids, y=yresids, ax=ax)

    # Plot the best-fitting line
    dfresids = pd.DataFrame({"xresids": xresids, "yresids": yresids})
    lmresids = smf.ols("yresids ~ 0 + xresids", data=dfresids).fit()
    slope = lmresids.params.iloc[0]
    xgrid = np.linspace(xresids.min(), xresids.max(), 100)
    ys = slope * xgrid
    sns.lineplot(x=xgrid, y=ys, ax=ax)

    # Add descriptive labels
    if len(others_formula) > 20:
        others_formula = "other"
    ax.set_xlabel(f"$\\mathtt{{{pred}}}$~$\\mathtt{{{others_formula}}}$ residuals")
    ax.set_ylabel(f"$\\mathtt{{{outname}}}$~$\\mathtt{{{others_formula}}}$ residuals")
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
        others_quoted = ["Q('" + other + "')" for other in others]
        others_formula += "+" + "+".join(others_quoted)
        for other in others:
            m = re.match("C\\((?P<varname>.*)\\).*", other)
            if m:
                varname = m.group("varname")
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
    ax.set_ylabel(r"$\sqrt{|\mathrm{standardized residuals}|}$")
    return ax

