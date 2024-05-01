# Mini Stats Helpers

[![image](https://img.shields.io/pypi/v/ministats.svg)](https://pypi.python.org/pypi/ministats)
[![Documentation Status](https://readthedocs.org/projects/ministats/badge/?version=latest)](https://ministats.readthedocs.io/en/latest/?version=latest)

Common statistical testing procedures used for STATS 101 topics.
The code is intentionally simple, to make it easy to read for beginners.

-   Free software: MIT license
-   Documentation: https://ministats.readthedocs.io


## About

This library contains helper functions for statistical analysis procedures implemented "from scratch."
Many of these procedures can be performed more quickly by simply calling an appropriate function defined in one of the existing libraries for statistical analysis,
but we deliberately show the step by step procedures,
so you'll know what's going on under the hood.



## Features

- Simple, concise code
- Uses standard prob. libraries `scipy.stats`
- Tested against other statistical software



## Roadmap

- [x] import plot helpers from https://github.com/minireference/noBSstatsnotebooks/blob/main/notebooks/plot_helpers.py
- [x] import stats helpers from https://github.com/minireference/noBSstatsnotebooks/blob/main/notebooks/stats_helpers.py
- [x] add GitHub actions CI
- [x] add some tests
- [ ] Split `plots.py` into:
   - [ ] `plots/discrete.py`
   - [ ] `plots/continuous.py`
   - [ ] `plots/linear_models.py`
   - [ ] `plots/figures.py` (not in main namespace)
   - [ ] `plots/sampling_dist` ? (not in main namespace)
- [ ] add more tests