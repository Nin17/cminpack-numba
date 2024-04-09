"""_summary_"""

from numba import extending, njit

from ._cminpack import Cminpack
from .utils import _check_dtype


def _chkder(m, n, x, fvec, fjac, ldfjac, xp, fvecp, mode, err):
    raise NotImplementedError


@extending.overload(_chkder)
def _chkder_overload(m, n, x, fvec, fjac, ldfjac, xp, fvecp, mode, err):
    _check_dtype((fvec, fjac, xp, fvecp), x.dtype)
    chkder_external = Cminpack.chkder(x.dtype)

    def impl(m, n, x, fvec, fjac, ldfjac, xp, fvecp, mode, err):
        return chkder_external(
            m,
            n,
            x.ctypes,
            fvec.ctypes,
            fjac.ctypes,
            ldfjac,
            xp.ctypes,
            fvecp.ctypes,
            mode,
            err.ctypes,
        )

    return impl


@njit
def chkder(m, n, x, fvec, fjac, ldfjac, xp, fvecp, mode, err):
    return _chkder(m, n, x, fvec, fjac, ldfjac, xp, fvecp, mode, err)
