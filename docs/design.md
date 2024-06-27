## Design

### Linear model plotting functions

We need to define some simple functions for plotting linear models for Chapter 4.
The existing functions are not a good fit:

- The `seaborn` functions `lmplot`, `regplot`, and `residplot` are great
  for simple linear regression, but don't generalize to multiple predictors.
- The `statsmodels` functions `plot_partregress` does the right thing,
  but has a complicated API using argument names like `exog`, `endog`, etc. 
  which will be complicated to explain.

Here are the requirements:

- Section 4.1: Simple linear regression
  - plot simple linear regression model (scatterplot and best-fitting line)
  - plot simple linear regression residuals with LOWESS
  - qqplot of residuals
  - plot simple linear regression model with mean CI 
  - plot simple linear regression model with prediction CI
- Section 4.2: Multiple linear regression
  - partial regression plots panel for the 3x predictors
  - residual plots panel for the 3x predictors (no LOWESS)
- Section 4.3: Interpreting linear models
  - residual plots panel with LOWESS (vs. 3x predictors and vs. fitted values)
- Section 4.4: Regression with categorical predictors
  - partial regression with groupby categorical variable
- Section 4.5: Causal effects and confounders
  - partial 
- Section 4.6: Generalized linear models
  - logistic regression viz
  - Poisson regression viz


To get these, we'll define the following core functions:
- `plot_reg(lmres)`: plot a simple linear regression model
- `plot_resid(lmres, pred=None, lowess=False)`: scatter plot of residuals
  versus the predictor `pred`. If `pred` is None, we plot the residuals
  versus the fitted values of the outcome variable. The plot contains shows
  a dashed horizontal line at `y=0` and an optional LOWERSS curve.
- `plot_partreg(lmres, pred)`: partial regression plot that 
  uses regression to "subtract" all other variables from both
  the outcome variable (plot residuals of Y ~ others on y-axis)
  and the predictor `pred` (plot residuals pred ~ others on the x-axis).

Optional methods (for completeness):
- `plot_partreg_proj(lmres, pred)`: use the replace-predictor-by-their-mean to plot
- `plot_scaleloc(lmres, lowess=True)`: scale-location plot

Convenience methods (for generating panels with one command):
- `plot_partregs(lmres)`: calls `plot_partreg` for each predictor
- `plot_resids(lmres)`: calls `plot_resid` to plot residuals vs. each predictor and vs. fitted values
- 

