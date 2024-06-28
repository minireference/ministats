import numpy as np
import statsmodels.api as sm


def calc_lm_vif(lmres, pred):
    """
    Calculate the variance inflation factor of the `pred` (str)
    for the linear model fit `lmfit`.
    """
    dmatrix = lmres.model.exog
    pred_idx = lmres.model.exog_names.index(pred)
    n_cols = dmatrix.shape[1]
    x_i = dmatrix[:, pred_idx]
    mask = np.arange(n_cols) != pred_idx
    X_noti = dmatrix[:, mask]
    r_squared_i = sm.OLS(x_i, X_noti).fit().rsquared
    vif = 1. / (1. - r_squared_i)
    return vif


