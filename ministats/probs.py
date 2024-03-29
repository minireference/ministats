import numpy as np
from scipy.stats import norm
from scipy.stats import rv_continuous



class mixnorms(object):
    """
    Custom class to represent mixture of normals.
    """

    def __init__(self, locs, scales, weights):
        assert len(locs) == len(scales)
        assert len(locs) == len(weights)
        self.locs = locs
        self.scales = scales
        self.weights = weights

    def pdf(self, x):
        rvNs = [norm(loc, scale) for loc, scale in zip(self.locs, self.scales)]
        terms = [w*rvN.pdf(x) for w, rvN in zip(self.weights, rvNs)]
        return sum(terms)
    
    def mean(self):
        return sum([w*loc for w, loc in zip(self.weights, self.locs)])

    def var(self):
        # via https://stats.stackexchange.com/a/604872/62481
        assert len(self.weights) == 2
        wA, wB = self.weights
        muA, muB = self.locs
        sigmaA, sigmaB = self.scales
        return wA*sigmaA**2 + wB*sigmaB**2 + wA*wB*(muA-muB)**2

    def rvs(self, n):
        rvNs = [norm(loc, scale) for loc, scale in zip(self.locs, self.scales)]
        ids = range(0,len(self.weights))
        choices = np.random.choice(ids, n, p=self.weights)
        values = np.zeros(n)
        for i, choice in enumerate(choices):
            rvN = rvNs[choice]
            values[i] = rvN.rvs(1)
        return values


class MixtureModel(rv_continuous):
    """
    A class for creating a random variable that is mixture of `scipy.stats`
    random variables. Credit: https://stackoverflow.com/a/72315113/127114
    """
    def __init__(self, submodels, *args, weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        if weights is None:
            weights = [1 for _ in submodels]
        if len(weights) != len(submodels):
            raise(ValueError('The number of submodels and weights must be equal.'))
        self.weights = [w / sum(weights) for w in weights]

    def _pdf(self, x):
        pdf = self.submodels[0].pdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            pdf += submodel.pdf(x) * weight
        return pdf

    def _sf(self, x):
        sf = self.submodels[0].sf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            sf += submodel.sf(x) * weight
        return sf

    def _cdf(self, x):
        cdf = self.submodels[0].cdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            cdf += submodel.cdf(x) * weight
        return cdf

    def rvs(self, size):
        submodel_choices = np.random.choice(len(self.submodels), size=size, p=self.weights)
        submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
        rvs = np.choose(submodel_choices, submodel_samples)
        return rvs

