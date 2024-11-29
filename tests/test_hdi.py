import math

import arviz
import numpy as np
from scipy.stats import norm
import pytest

from ministats.probs import MixtureModel

# SUT
from ministats import hdi_from_grid
from ministats import hdi_from_rv
from ministats import hdi_from_samples


@pytest.fixture
def rvX1():
    rvX1 = MixtureModel([norm(0,1), norm(3,2)], weights=[0.4,0.6])
    return rvX1

@pytest.fixture
def samplesX1(rvX1, seed=44):
    nsamples = 100_000
    np.random.seed(seed)
    samples = rvX1.rvs(nsamples)
    return samples

@pytest.fixture
def gridX1(rvX1):
    ngrid = 10_000
    lims = [-5, 15]
    params = np.linspace(*lims, ngrid)
    probs = rvX1.pdf(params)
    probs /= np.sum(probs)
    return params, probs


def test_hpdi_from_grid(gridX1, samplesX1):
    arviz_ci = arviz.hdi(samplesX1, hdi_prob=0.9)
    params, probs = gridX1
    ci = hdi_from_grid(params, probs, hdi_prob=0.9)
    # print(arviz_ci - ci)
    assert math.isclose(arviz_ci[0], ci[0], abs_tol=0.03)
    assert math.isclose(arviz_ci[1], ci[1], abs_tol=0.03)

def test_hpdi_from_samples(samplesX1):
    arviz_ci = arviz.hdi(samplesX1, hdi_prob=0.9)
    ci = hdi_from_samples(samplesX1, hdi_prob=0.9)
    assert math.isclose(arviz_ci[0], ci[0])
    assert math.isclose(arviz_ci[1], ci[1])


def test_hpdi_from_rv(rvX1, samplesX1):
    arviz_ci = arviz.hdi(samplesX1, hdi_prob=0.9)
    ci = hdi_from_rv(rvX1, hdi_prob=0.9)
    # print(arviz_ci - ci)
    assert math.isclose(arviz_ci[0], ci[0], abs_tol=0.01)
    assert math.isclose(arviz_ci[1], ci[1], abs_tol=0.01)
