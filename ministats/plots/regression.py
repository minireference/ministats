import copy
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from scipy.stats import t as tdist
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm


# Useful colors
snspal = sns.color_palette()
blue, orange, red, purple = snspal[0], snspal[1], snspal[3], snspal[4]



# Linear model plotting functions
################################################################################



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
    ax = plt.gca() if ax is None else ax

    # Prepare the data
    xname = xs.name if hasattr(xs, "name") else "x"
    yname = ys.name if hasattr(ys, "name") else "y"
    data = pd.DataFrame({xname:xs, yname:ys})
    n = len(xs)

    # Fit the linear model
    formula = f"{yname} ~ 1 + {xname}"
    lm = smf.ols(formula, data=data).fit()

    # Get model predicitons
    x_vals = np.linspace(np.min(xs), np.max(xs), 100)
    x_pred = {xname:x_vals}
    y_pred = lm.get_prediction(x_pred)

    # Draw the scatterplot and plot the best-fit line
    sns.scatterplot(x=xs, y=ys, ax=ax)
    sns.lineplot(x=x_vals, y=y_pred.predicted, ax=ax)

    if ci_mean:
        # Draw the confidence interval for the mean
        t_05, t_95 = tdist(df=n-2).ppf([alpha_mean/2, 1-alpha_mean/2])
        lower_mean = y_pred.predicted + t_05*y_pred.se_mean
        upper_mean = y_pred.predicted + t_95*y_pred.se_mean
        if lab_mean:
            if isinstance(lab_mean, str):
                label_mean = lab_mean
            else:
                perc_mean = round(100*(1-alpha_mean))
                label_mean = f"{perc_mean}% confidence interval for the mean"
        else:
            label_mean = None
        ax.fill_between(x_vals, lower_mean, upper_mean, alpha=0.4, color="C0", label=label_mean)

    if ci_obs:
        # Draw the confidence interval for the outcome observations
        t_05, t_95 = tdist(df=n-2).ppf([alpha_obs/2, 1-alpha_obs/2])
        lower_obs = y_pred.predicted + t_05*y_pred.se_obs
        upper_obs = y_pred.predicted + t_95*y_pred.se_obs
        if lab_obs:
            if isinstance(lab_obs, str):
                label_obs = lab_obs
            else:
                perc_obs = round(100*(1-alpha_obs))
                label_obs = f"{perc_obs}% confidence interval for observations"
        else:
            label_obs = None
        ax.fill_between(x_vals, lower_obs, upper_obs, alpha=0.1, color="C0", label=label_obs)

    if (ci_mean and lab_mean) or (ci_obs and lab_obs):
        ax.legend()


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


def plot_lm_partial_old(lmfit, pred, others=None, ax=None):
    """
    Generate a partial regression plot from the best-fit line
    of the predictor `pred`, where the intercept is calculated
    from the average of the `other` predictors.
    """
    ax = plt.gca() if ax is None else ax
    data = lmfit.model.data.orig_exog
    params = lmfit.params
    allpreds = set(data.columns) - {"Intercept"}
    assert pred in allpreds 
    others = allpreds - {pred} if others is None else others
    intercept = params["Intercept"]
    for other in others:
        intercept += params[other]*data[other].mean() 
    slope = params[pred]
    print(pred, "intercept=", intercept, "slope=", slope)
    xs = np.linspace(data[pred].min(), data[pred].max())
    ys = intercept + slope*xs
    sns.lineplot(x=xs, y=ys, ax=ax)


def plot_lm_partial(lmfit, pred, others=None, ax=None):
    """
    Generate a partial regression plot from the model `lmfit`
    for the predictor `pred`, given the `other` predictors.
    We plot the residuals of `outcome ~ other` along the y-axis,
    and the residuals of the model `pred ~ other` on the x-axis.
    """
    ax = plt.gca() if ax is None else ax
    lmfit = copy.copy(lmfit)  # plot function was breaking after re-excution
    xdata = lmfit.model.data.orig_exog
    ydata = lmfit.model.data.orig_endog
    data = pd.concat([xdata, ydata], axis=1)

    # Find others= as list of strings
    allpreds = set(xdata.columns) - {"Intercept"}
    assert pred in allpreds
    others = allpreds - {pred} if others is None else others
    others_formula = "1"
    if others:
        others_formula += "+" + "+".join(others)

    # x-axis = residuals of `pred ~ 1 + others`
    lmpred = smf.ols(f"{pred} ~ {others_formula}", data=data).fit()
    xresids = lmpred.resid

    # y-axis = residuals of `outcome ~ 1 + others`
    outname = lmfit.model.endog_names
    lmoutcome = smf.ols(f"{outname} ~ {others_formula}", data=data).fit()
    yresids = lmoutcome.resid

    # scatter plot
    sns.scatterplot(x=xresids, y=yresids, ax=ax)
    ylims = ax.get_ylim()

    # best-fit line between the residuals
    dfresids = pd.DataFrame({"xresids": xresids, "yresids": yresids})
    lmresids = smf.ols("yresids ~ 0 + xresids", data=dfresids).fit()
    slope = lmresids.params.iloc[0]
    xs = np.linspace(*ylims, 100)
    ys = slope*xs
    sns.lineplot(x=xs, y=ys, ax=ax)

    ax.set_xlabel(f"{pred} ~ {others_formula}  residuals")
    ax.set_ylabel(f"{outname} ~ {others_formula}  residuals")
    ax.set_title('Partial regression plot')
    ax.set_ylim(ylims)

    return ax


def plot_lm_partial_cat(lmfit, pred, others=None, color="C0", linestyle="solid", cats=None, ax=None):
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
    ax = plt.gca() if ax is None else ax
    sns.stripplot(data=data, x=x, y=y, hue=x, size=3, jitter=0, alpha=0.2)
    sns.pointplot(data=data, x=x, y=y, hue=x, estimator="mean", errorbar=None, marker="D")

    # Customize plot labels
    xlabel0, xlabel1 = [l.get_text() for l in ax.get_xticklabels()]
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

    # Return axes
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


def plot_lm_scale_loc(lmfit, pred=None):
    """
    Plot the scale-location plot for the linear model `lmfit`.
    """
    sigmahat = np.sqrt(lmfit.scale)
    std_resids = lmfit.resid / sigmahat
    sqrt_abs_std_resids = np.sqrt(np.abs(std_resids))
    if pred:
        xs = lmfit.model.data.orig_exog[pred]
        xlabel = pred
    else:
        xs = lmfit.fittedvalues
        xlabel = "fitted values"
    ax = sns.regplot(x=xs, y=sqrt_abs_std_resids, lowess=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\sqrt{|\text{standardized residuals}|}$")
    return ax


def calc_lm_vif(lmfit, pred):
    """
    Calculate the variance inflation factor of the `pred` (str)
    for the linear model fit `lmfit`.
    """
    dmatrix = lmfit.model.exog
    pred_idx = lmfit.model.exog_names.index(pred)
    n_cols = dmatrix.shape[1]
    x_i = dmatrix[:, pred_idx]
    mask = np.arange(n_cols) != pred_idx
    X_noti = dmatrix[:, mask]
    r_squared_i = sm.OLS(x_i, X_noti).fit().rsquared
    vif = 1. / (1. - r_squared_i)
    return vif
