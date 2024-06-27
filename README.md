# Ministats

[![image](https://img.shields.io/pypi/v/ministats.svg)](https://pypi.python.org/pypi/ministats)
[![Documentation Status](https://readthedocs.org/projects/ministats/badge/?version=latest)](https://ministats.readthedocs.io/en/latest/?version=latest)

Common statistical testing procedures and plotting functions used for STATS 101.
The code is intentionally simple to make it easy to read for beginners.

-   Free software: MIT license
-   Documentation: https://ministats.readthedocs.io


## About

This library contains helper functions for statistical analysis procedures implemented "from scratch."
Many of these procedures can be performed more quickly by simply calling an appropriate function defined in one of the existing libraries for statistical analysis,
but we deliberately show the step by step procedures,
so you'll know what's going on under the hood.



## Features

- Simple, concise code.
- Uses standard prob. libraries `scipy.stats`.
- Tested against other statistical software.



## Roadmap

- [x] import plot helpers from https://github.com/minireference/noBSstatsnotebooks/ repo 
- [x] import stats helpers from https://github.com/minireference/noBSstatsnotebooks/ repo
- [x] add GitHub actions CI
- [x] add some tests
- [x] Move `plots.py` into `plots/__init__.py`:
- [ ] Distribute functions `plots/__init__.py` into submodules:
   - [x] `plots/probability.py`: functions for visualizing probability distributions
   - [x] `plots/regression.py`: linear model visualization functions
   - [x] `plots/figures.py`: special code used for figures in the book (not included in the main namespace)
   - [ ] remove `plots/figures` plotting functions from `ministats` namespace
- [ ] add more tests