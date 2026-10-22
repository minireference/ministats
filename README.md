# Ministats

[![image](https://img.shields.io/pypi/v/ministats.svg)](https://pypi.python.org/pypi/ministats)
[![Documentation Status](https://readthedocs.org/projects/ministats/badge/?version=latest)](https://ministats.readthedocs.io/en/latest/?version=latest)


`ministats` is lightweight Python library of helper functions for learning statistics.
The library contains functions for data analysis, plotting probability distributions,
and performing various statistical analysis procedures implemented "from scratch."
The goal is to show code implementations of the standard stats procedures used for STATS 101
that are easy to read for beginners.

- Free software: MIT license
- Docs: [ministats.readthedocs.io](https://ministats.readthedocs.io)
- Features:
  - Simple, concise, Pythonic code.
  - Code implementations closely matches the math calculations.
  - Leverages probability distributions defined in `scipy.stats`.
  - Includes methods for generating and visualizing sampling distributions.
  - Covers standard STATS 101 methods: confidence intervals, t-tests, etc.
    that give the same answer as the stats functions from `scipy.stats` and `statsmodels`
    but are much easier to read.
  - Covers the modern statistics curriculum: simulations and resampling methods
    like bootstrap estimation and permutations tests.
- Limitations:
  - Hypothesis testing procedures are slower than stats functions from  `scipy.stats` and `statsmodels`.
  - Limited scope: includes only basics stats methods needed for a introductory statistics course.
- Source:
  - The `ministats` library was developed as support code for the 
    [**No Bullshit Guide to Statistics**](https://noBSstats.com/),
    which is an intro-to-statistics textbook that uses a computational approach to teaching.
    Probability and statistics concepts are explained using words, math formulas,
    visualizations, and code examples.
    THe code examples include simulations,
    resampling methods like bootstrap estimation and permutations tests,
    code implementations of analytical methods.
    The power of the scientific Python computing ecosystem make learning statistics easy.



### Installation

To install the `ministats` library,
run this command in a commande line terminal where your Python project lives.

```console
$ python -m pip install ministats
```

If you need to use `ministats` inside a Jupyter notebook or Colab notebook,
you can run the following command:

```Python
%pip install ministats
```

This is the preferred method to install the `ministats` library,
as it will always install the most recent stable release.


<!-- BEGIN-QUICKSTART -->
## Quickstart
INSERT NOTEBOOK quickstart.ipynb AS MARKDOWN SOURCE


<!-- END-QUICKSTART -->



## Roadmap

Here are some tasks I need to take care of before the v1.0 release,
which will be the version that accompanies the first editions
of the [**No Bullshit Guide to Statistics**](https://noBSstats.com/),

- [ ] finish adding `docs/` 
- [ ] move `plots/figures.py` plotting functions to `plots/book/figures.py`
- [ ] remove `plots/figures` plotting functions from `ministats` namespace
- [ ] add more tests:
  - [ ] chi-square tests for categorical data 
  - [ ] one-sample equivalence test
  - [ ] two-sample equivalence test
