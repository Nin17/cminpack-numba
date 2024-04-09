"""Wrapper of the `enorm` function."""

from numba import extending, njit
from numpy import floating, int32
from numpy.typing import NDArray

from ._cminpack import Cminpack

# --------------------------------------- enorm -------------------------------------- #


def _enorm(n, x):
    raise NotImplementedError


@extending.overload(_enorm)
def _enorm_overload(n, x):
    enorm_external = Cminpack.enorm(x.dtype)

    def impl(n, x):
        if x.ndim != 1:
            msg = "x must be a 1D array"
            raise ValueError(msg)
        return enorm_external(n, x.ctypes)

    return impl


@njit
def enorm_(n: int32, x: NDArray[floating]) -> float:
    """Euclidean norm of the n-vector x.

    Parameters
    ----------
    n : int
        the length of x
    x : NDArray[floating]
        the vector

    Returns
    -------
    float
        the Euclidean norm of x

    """
    return _enorm(n, x)


@njit
def enorm(x: NDArray[floating]) -> float:
    """Euclidean norm of the n-vector x.

    Parameters
    ----------
    x : NDArray[floating]
        the vector

    Returns
    -------
    float
        the Euclidean norm of x

    """
    return _enorm(x.size, x)
