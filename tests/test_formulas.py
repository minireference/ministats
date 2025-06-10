import math
from scipy.stats import expon
from scipy.stats import ks_1samp

# SUT
from ministats.formulas import ks_distance



def test_ks_distance_against_scipy():
    rvE = expon(loc=0, scale=5)
    exp_data = rvE.rvs(10)
    ks_dist = ks_distance(exp_data, rvE)
    ks_dist_scipy = ks_1samp(exp_data, rvE.cdf).statistic
    assert math.isclose(ks_dist, ks_dist_scipy)
