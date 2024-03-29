import math
import random
import pytest
import statistics

# SUT
from ministats.estimators import mean
from ministats.estimators import std


def random_list(n=10, min=0.0, max=100.0):
    """
    Returns a list of length `n` of random floats between `min` and `max`.
    """
    values = []
    for i in range(n):
        r = random.random()
        value = min + r*(max-min)
        values.append(value)
    return values


def test_mean():
    """
    Run a few lists to check if value returned by `mean` matches expected.
    """
    assert mean([1,1,1]) == 1
    assert mean([61,72,85,92]) == 77.5
    list10 = random_list(n=10)
    assert math.isclose(mean(list10), statistics.mean(list10))
    list100 = random_list(n=100)
    assert math.isclose(mean(list100), statistics.mean(list100))


def test_std():
    """
    Run a few lists to check if value returned by `std` matches expected.
    """
    assert std([1,1,1]) == 0
    assert math.isclose(std([61,72,85,92]), 13.771952173409064)
    list10 = random_list(n=10)
    assert math.isclose(std(list10), statistics.stdev(list10))
    list100 = random_list(n=100)
    assert math.isclose(std(list100), statistics.stdev(list100))
