"""Top-level package for Mini Statistics."""

__author__ = """Ivan Savov"""
__email__ = 'ivan@minireference.com'
__version__ = '0.2.0'

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
    gen_sampling_dist,
    plot_sampling_dist,
    plot_samples_panel,
    plot_sampling_dists_panel,
    plot_alpha_beta_errors,
    plot_lm_simple,
    plot_residuals,
    plot_residuals2,
    plot_lm_partial,
    plot_lm_ttest,
    plot_lm_anova
)
