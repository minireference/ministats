# Ministats

[![image](https://img.shields.io/pypi/v/ministats.svg)](https://pypi.python.org/pypi/ministats)
[![Documentation Status](https://readthedocs.org/projects/ministats/badge/?version=latest)](https://ministats.readthedocs.io/en/latest/?version=latest)

`ministats` is lightweight Python for learning statistics
that includes helper functions for data analysis,
probability visualizations,
and statistical analysis procedures.
The goal is to show code implementations of the standard stats procedures used in STATS 101 courses
that are easy to understand for Python beginners.

- Docs: [ministats.readthedocs.io](https://ministats.readthedocs.io)
- Features:
  - Code implementations *from scratch* that closely match the math calculations.
  - Leverages probability distributions defined in `scipy.stats`.
  - Includes both classical analytical approximation methods like t-tests
    and modern methods like simulations, bootstrap estimation, and permutation tests.
- Free software: [MIT license](https://github.com/minireference/ministats/blob/main/LICENSE)




## Installation

To install the `ministats` library,
run the following command in a terminal:

```console
$ python -m pip install ministats
```

To use `ministats` inside a Jupyter notebook or Colab notebook,
run the following `%`-magic command in a code cell:

```Python
%pip install ministats
```




<!-- BEGIN-QUICKSTART -->
## Quickstart
INSERT NOTEBOOK quickstart.ipynb AS MARKDOWN SOURCE


<!-- END-QUICKSTART -->





## Textbook

The `ministats` library was developed as the support code
for the [**No Bullshit Guide to Statistics**](https://noBSstats.com/),
which is an intro-to-stats textbook 
that uses a computational approach to teaching probability and statistics concepts.
The book explains concepts using words, math formulas, visualizations, and code examples.
The `ministats` library contains implementations of all the statistical procedures
discussed in the book
written as concise, easy-to-read Python code.

To learn more about the book, consult the links below:

- Table of contents for [Part 1](https://minireference.com/static/excerpts/noBSstats_part1_preview.pdf#page=2)
  and [Part 2](https://minireference.com/static/excerpts/noBSstats_part2_preview.pdf#page=2)
- Book preview PDFs:
  - [`noBSstats_part1_preview.pdf`](https://minireference.com/static/excerpts/noBSstats_part1_preview.pdf)
  - [`noBSstats_part2_preview.pdf`](https://minireference.com/static/excerpts/noBSstats_part2_preview.pdf)
- Concept maps from the books: [`statistics_concepts.pdf`](https://minireference.com/static/conceptmaps/statistics_concepts.pdf)
- Python tutorial: [bit.ly/pytut3](https://bit.ly/pytut3)
- Book website: [noBSstats.com](https://noBSstats.com/)
- Announcement [blog post](https://minireference.com/blog/noBSstats-prerelease/)
- Buy PDF here: [gum.co/noBSstats](https://gum.co/noBSstats)
- Print version coming soon (January 2027).




## Computational notebooks

To see examples of the `ministats` library in action,
you can check out the computational notebooks that accompany the book.

- View notebooks online: [noBSstats.com/notebooks/](https://noBSstats.com/notebooks/)
- Play with the notebooks in the cloud:
  [https://mybinder.org/v2/gh/minireference/noBSstats/main](https://mybinder.org/v2/gh/minireference/noBSstats/main)
- Run the notebooks locally by following the
  [instructions here](https://nobsstats.com/notebooks/12_data_in_practice.html#getting-started-with-jupyterlab).
- Watch this [playlist](https://www.youtube.com/playlist?list=PLGmu4KtWiH6-WQrTReNIIQhLxU8cirdxA)
  with video tutorials that accompany the book.




## Roadmap

Here are some tasks I need to take care of before the v1.0 release,
which will be the version that accompanies the first edition
of the [**No Bullshit Guide to Statistics**](https://noBSstats.com/):

- [ ] add text 
- [ ] finish adding `docs/`
- [ ] move `plots/figures.py` plotting functions to `plots/book/figures.py`
- [ ] remove `plots/figures` plotting functions from `ministats` namespace
- [ ] add more statistical tests:
  - [ ] chi-square tests for categorical data 
  - [ ] one-sample equivalence test
  - [ ] two-sample equivalence test
- [ ] add more tests




## Testing

The statistical procedures implemented in the `ministats` library
are tested against the functions from `scipy.stats` and `statsmodels`
to make sure they give the same answers.


