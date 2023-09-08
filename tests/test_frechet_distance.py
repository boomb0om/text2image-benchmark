import numpy as np
from T2IBenchmark.metrics import frechet_distance
from unittest import TestCase
from numpy.testing import assert_almost_equal
from scipy import linalg


class DummyStats:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma


class TestFrechetDistance(TestCase):

    def test_frechet_distance_with_same_stats(self):
        mu1 = np.array([2, 3])
        sigma1 = np.array([[1, 0], [0, 4]])
        stat1 = DummyStats(mu1, sigma1)

        mu2 = np.array([2, 3])
        sigma2 = np.array([[1, 0], [0, 4]])
        stat2 = DummyStats(mu2, sigma2)

        distance = frechet_distance(stat1, stat2)
        assert_almost_equal(distance, 0.0)

    def test_frechet_distance_with_different_stats(self):
        mu1 = np.array([2, 3])
        sigma1 = np.array([[1, 0], [0, 4]])
        stat1 = DummyStats(mu1, sigma1)

        mu2 = np.array([5, 7])
        sigma2 = np.array([[3, 2], [2, 8]])
        stat2 = DummyStats(mu2, sigma2)

        distance = frechet_distance(stat1, stat2)
        self.assertGreater(distance, 0.0)

    def test_frechet_distance_with_different_dimension_stats(self):
        mu1 = np.array([2, 3])
        sigma1 = np.array([[1, 0], [0, 4]])
        stat1 = DummyStats(mu1, sigma1)

        mu2 = np.array([5, 7, 8])
        sigma2 = np.array([[3, 2, 0], [2, 8, 1], [0, 1, 4]])
        stat2 = DummyStats(mu2, sigma2)

        with self.assertRaises(AssertionError):
            frechet_distance(stat1, stat2)

    def test_frechet_distance_with_almost_singular_product(self):
        mu1 = np.array([2, 3])
        sigma1 = np.array([[1, 0], [0, 4]])
        stat1 = DummyStats(mu1, sigma1)

        mu2 = np.array([2.000001, 3.000001])
        sigma2 = np.array([[1, 0], [0, 4]])
        stat2 = DummyStats(mu2, sigma2)

        distance = frechet_distance(stat1, stat2, eps=1e-10)
        self.assertGreater(distance, 0.0)
