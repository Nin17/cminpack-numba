"""Wrappers of the `dpmpar` and `sdpmpar` functions."""

from numba import extending, njit, types
from numpy import float32, float64, int32

from ._cminpack import Cminpack

# -------------------------------------- dpmpar -------------------------------------- #


def _dpmpar(i):
    raise NotImplementedError


def _sdpmpar(i):
    raise NotImplementedError


@extending.overload(_dpmpar)
def _dpmpar_overload(i):
    dpmpar_external = Cminpack.dpmpar(types.float64)

    def impl(i):
        return float64(dpmpar_external(i))

    return impl


@extending.overload(_sdpmpar)
def _sdpmpar_overload(i):
    sdpmpar_external = Cminpack.dpmpar(types.float32)

    def impl(i):
        return float32(sdpmpar_external(i))

    return impl


@njit
def dpmpar(i: int32) -> float64:
    """Double precision machine parameters.

    Parameters
    ----------
    i : int32
        1: the machine precision
        2: the smallest magnitude
        3: the largest magnitude

    Returns
    -------
    float64
        machine parameter

    """
    return _dpmpar(i)


@njit
def sdpmpar(i: int32) -> float32:
    """Single precision machine parameters.

    Parameters
    ----------
    i : int32
        1: the machine precision
        2: the smallest magnitude
        3: the largest magnitude

    Returns
    -------
    float32
        machine parameter

    """
    return _sdpmpar(i)
