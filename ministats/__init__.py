"""Top-level package for Mini Statistics."""

__author__ = """Ivan Savov"""
__email__ = 'ivan@minireference.com'
__version__ = '0.3.5'


from .confidence_intervals import (
    ci_mean,
    ci_var,
    ci_dmeans,
)

from .estimators import (
    mean,
    var,
    std,
    dmeans,
    median,
    quantile,
)

from .formulas import (
    cohend,
    cohend2,
    calcdf,
)

from .hypothesis_tests import (
    tailvalues,
    tailprobs,
    ztest,
    chi2test_var,
    simulation_test_mean,
    simulation_test_var,
    # simulation_test,
    bootstrap_test_mean,
    # resample_under_H0,
    permutation_test_dmeans,
    # permutation_test,
    permutation_anova,
    ttest_mean,
    ttest_dmeans,
    ttest_paired,
)

from .linear_models import (
    calc_lm_vif,
)

from .plots import (
    nicebins
)

# TODO: Remove these from namespace in first major release.
#       Notebooks in `figures_generation/` folder must use full path imports
#       e.g. `from ministats.plots.figures import generate_pmf_panel`
from .plots.figures import (
    generate_pmf_panel,
    calc_prob_and_plot,
    calc_prob_and_plot_tails,
    plot_pdf_and_cdf,
    generate_pdf_panel,
    gen_samples,
    plot_samples,
    plot_sampling_dist,
    plot_samples_panel,
    plot_sampling_dists_panel,
    plot_alpha_beta_errors,
    #
    # Linear models
    # plot_residuals,
    # plot_residuals2,
    # plot_lm_ttest,
    # plot_lm_anova,
)


from .plots.probability import (
    plot_pmf,
    plot_cdf,
    plot_pdf,
    qq_plot,
)

from .plots.regression import (
    # NEW
    plot_reg,
    plot_resid,
    plot_partreg,
    plot_scaleloc,
    plot_pred_bands,
    # OPTIONAL
    plot_projreg,
    # OLD
    # plot_lm_simple,
    # plot_lm_partial,
    # plot_lm_partial_old,
    # plot_projreg_cat,
)

from .sampling import (
    gen_sampling_dist,
    gen_boot_dist
)


# Functions that are intentionally left out of the public interface
#  - from probs import MixtureModel, mixnorms
#  - `simulations.simulate_ci_props`
#  - utils.savefigure doesn't need to be part of the public interface
#  - all plotting functions used to generate specific figures `plots.figures`
