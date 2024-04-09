"""Test eps, max and tiny given by numpy against dpmpar and sdpmpar."""

import numpy as np
from numpy.testing import assert_equal

from cminpack_numba import dpmpar, sdpmpar


def test_machine_precision_double() -> None:
    """Test that the machine precision is equal to the numpy float64 precision."""
    double_eps = dpmpar(1)
    assert_equal(double_eps, np.finfo(np.float64).eps)


def test_machine_precision_single() -> None:
    """Test that the machine precision is equal to the numpy float32 precision."""
    single_eps = sdpmpar(1)
    assert_equal(single_eps, np.finfo(np.float32).eps)


def test_smallest_magnitude_double() -> None:
    """Test that the smallest magnitude is equal to the numpy float64 tiny."""
    double_tiny = dpmpar(2)
    assert_equal(double_tiny, np.finfo(np.float64).tiny)


def test_smallest_magnitude_single() -> None:
    """Test that the smallest magnitude is equal to the numpy float32 tiny."""
    single_tiny = sdpmpar(2)
    assert_equal(single_tiny, np.finfo(np.float32).tiny)


def test_largest_magnitude_double() -> None:
    """Test that the largest magnitude is equal to the numpy float64 max."""
    double_max = dpmpar(3)
    assert_equal(double_max, np.finfo(np.float64).max)


def test_largest_magnitude_single() -> None:
    """Test that the largest magnitude is equal to the numpy float32 max."""
    single_max = sdpmpar(3)
    assert_equal(single_max, np.finfo(np.float32).max)
