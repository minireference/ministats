"""Top-level package for Mini Statistics."""

__author__ = """Ivan Savov"""
__email__ = 'ivan@minireference.com'
__version__ = '0.3.2'


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

from .plots import (
    plot_pmf,
    plot_cdf,
    generate_pmf_panel,
    plot_pdf,
    calc_prob_and_plot,
    calc_prob_and_plot_tails,
    plot_pdf_and_cdf,
    generate_pdf_panel,
    nicebins,
    qq_plot,
    gen_samples,
    plot_samples,
    plot_sampling_dist,
    plot_samples_panel,
    plot_sampling_dists_panel,
    plot_alpha_beta_errors,
    plot_lm_simple,
    plot_residuals,
    plot_residuals2,
    plot_lm_partial,
    plot_lm_ttest,
    plot_lm_anova,
    plot_lm_scale_loc,
    calc_lm_vif,
)

from .sampling import (
    gen_sampling_dist,
    gen_boot_dist
)


# Functions that are intentionally left out of the public interface
#  - from probs import MixtureModel, mixnorms
#  - simulations.simulate_ci_props
#  - utils.savefigure doesn't need to be part of the public interface