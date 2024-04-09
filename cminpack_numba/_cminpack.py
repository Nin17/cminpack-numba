"""Signatures for the cminpack functions."""

import warnings
from ctypes.util import find_library

from llvmlite import binding
from numba import types

from .utils import get_extension_path

__all__ = [
    "Cminpack",
]


def _apply_prefix(func: str, dtype: types.Float) -> str:
    """Get the cminpack function name with the correct prefix for the given dtype.

    Parameters
    ----------
    func : str
        The function name
    dtype : types.Float
        The dtype

    Returns
    -------
    str
        The cminpack function name with the correct prefix

    """
    _cminpack_prefix = {types.float32: "s", types.float64: ""}
    return f"{_cminpack_prefix[dtype]}{func}"


def _ensure_cminpack(dtype: str = "") -> None:
    """Ensure that the cminpack library is available, in either 64 or 32 bit precision.

    Parameters
    ----------
    dtype : str, optional
        "" for double precision, "s" for single precision, by default ""

    Raises
    ------
    ImportError
        If the cminpack/cminpacks library is not found

    """
    if (
        find_library(f"cminpack{dtype}") is None
        and get_extension_path(f"libcminpack{dtype}") is None
    ):
        msg = f"cminpack{dtype} library not found"
        raise ImportError(msg)


# Load double and single precision libraries
try:
    _cminpack_path = get_extension_path("libcminpack") or find_library("cminpack")
    _ensure_cminpack()
    CMINPACK = True
    binding.load_library_permanently(_cminpack_path)
except ImportError:
    warnings.warn(
        "cminpack not found. Double precision functions unavailable.",
        stacklevel=1,
    )
    CMINPACK = False
try:
    _cminpacks_path = get_extension_path("libcminpacks") or find_library("cminpacks")
    _ensure_cminpack("s")
    CMINPACKS = True
    binding.load_library_permanently(_cminpacks_path)
except ImportError:
    warnings.warn(
        "cminpacks not found. Single precision functions unavailable.",
        stacklevel=1,
    )
    CMINPACKS = False

if not (CMINPACK or CMINPACKS):
    _msg = "cminpack & cminpacks not found."
    raise ImportError(_msg)


class Cminpack:
    """External functions from the cminpack(s) library."""

    @staticmethod
    def chkder(
        dtype: types.Float,
    ) -> types.ExternalFunction:
        """Return the external function for chkder.

        Parameters
        ----------
        dtype : types.Float
            The dtype

        Returns
        -------
        types.ExternalFunction
            The external function for chkder

        """
        sig = types.void(
            types.int32,  # m
            types.int32,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            types.CPointer(dtype),  # *fjac
            types.int32,  # ldfjac
            types.CPointer(dtype),  # *xp
            types.CPointer(dtype),  # *fvecp
            types.int32,  # mode
            types.CPointer(dtype),  # *err
        )
        return types.ExternalFunction(_apply_prefix("chkder", dtype), sig)

    @staticmethod
    def dpmpar(dtype: types.Float) -> types.ExternalFunction:
        """Return the external function for dpmpar.

        Parameters
        ----------
        dtype : types.Float
            The dtype

        Returns
        -------
        types.ExternalFunction
            The external function for dpmpar

        """
        sig = dtype(types.int32)
        return types.ExternalFunction(_apply_prefix("dpmpar", dtype), sig)

    @staticmethod
    def enorm(dtype: types.Float) -> types.ExternalFunction:
        """Return the external function for enorm.

        Parameters
        ----------
        dtype : types.Float
            The dtype

        Returns
        -------
        types.ExternalFunction
            The external function for enorm

        """
        sig = dtype(
            types.int32,  # n
            types.CPointer(dtype),  # *x
        )
        return types.ExternalFunction(_apply_prefix("enorm", dtype), sig)

    @staticmethod
    def hybrd(dtype: types.Float) -> types.ExternalFunction:
        """Return the external function for hybrd.

        Parameters
        ----------
        dtype : types.Float
            The dtype

        Returns
        -------
        types.ExternalFunction
            The external function for hybrd

        """
        sig = types.int32(
            types.voidptr,  # fcn
            types.voidptr,  # *p / *udata
            types.int32,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            dtype,  # xtol
            types.int32,  # maxfev
            types.int32,  # ml
            types.int32,  # mu
            dtype,  # epsfcn
            types.CPointer(dtype),  # *diag
            types.int32,  # mode
            dtype,  # factor
            types.int32,  # nprint
            types.CPointer(types.int32),  # *nfev
            types.CPointer(dtype),  # *fjac
            types.int32,  # ldfjac
            types.CPointer(dtype),  # *r
            types.int32,  # lr
            types.CPointer(dtype),  # *qtf
            types.CPointer(dtype),  # *wa1
            types.CPointer(dtype),  # *wa2
            types.CPointer(dtype),  # *wa3
            types.CPointer(dtype),  # *wa4
        )
        return types.ExternalFunction(_apply_prefix("hybrd", dtype), sig)

    @staticmethod
    def hybrd1(dtype: types.Float) -> types.ExternalFunction:
        """Return the external function for hybrd1.

        Parameters
        ----------
        dtype : types.Float
            The dtype

        Returns
        -------
        types.ExternalFunction
            The external function for hybrd1

        """
        sig = types.int32(
            types.voidptr,  # fcn
            types.voidptr,  # *p / *udata
            types.int32,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            dtype,  # tol
            types.CPointer(dtype),  # *wa
            types.int32,  # lwa
        )
        return types.ExternalFunction(_apply_prefix("hybrd1", dtype), sig)

    @staticmethod
    def hybrj(dtype: types.Float) -> types.ExternalFunction:
        """Return the external function for hybrj.

        Parameters
        ----------
        dtype : types.Float
            The dtype

        Returns
        -------
        types.ExternalFunction
            The external function for hybrj

        """
        sig = types.int32(
            types.voidptr,  # fcn
            types.voidptr,  # *p / *udata
            types.int32,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            types.CPointer(dtype),  # *fjac
            types.int32,  # ldfjac
            dtype,  # xtol
            types.int32,  # maxfev
            types.CPointer(dtype),  # *diag
            types.int32,  # mode
            dtype,  # factor
            types.int32,  # nprint
            types.CPointer(types.int32),  # *nfev
            types.CPointer(types.int32),  # *njev
            types.CPointer(dtype),  # *r
            types.int32,  # lr
            types.CPointer(dtype),  # *qtf
            types.CPointer(dtype),  # *wa1
            types.CPointer(dtype),  # *wa2
            types.CPointer(dtype),  # *wa3
            types.CPointer(dtype),  # *wa4
        )
        return types.ExternalFunction(_apply_prefix("hybrj", dtype), sig)

    @staticmethod
    def hybrj1(dtype: types.Float) -> types.ExternalFunction:
        """Return the external function for hybrj1.

        Parameters
        ----------
        dtype : types.Float
            The dtype

        Returns
        -------
        types.ExternalFunction
            The external function for hybrj1

        """
        sig = types.int32(
            types.voidptr,  # fcn
            types.voidptr,  # *p / *udata
            types.int32,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            types.CPointer(dtype),  # *fjac
            types.int32,  # ldfjac
            dtype,  # tol
            types.CPointer(dtype),  # *wa
            types.int32,  # lwa
        )
        return types.ExternalFunction(_apply_prefix("hybrj1", dtype), sig)

    @staticmethod
    def lmdif(dtype: types.Float) -> types.ExternalFunction:
        """Return the external function for lmdif.

        Parameters
        ----------
        dtype : types.Float
            The dtype

        Returns
        -------
        types.ExternalFunction
            The external function for lmdif

        """
        sig = types.int32(
            types.voidptr,  # fcn
            types.voidptr,  # *p / *udata
            types.int32,  # m
            types.int32,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            dtype,  # ftol
            dtype,  # xtol
            dtype,  # gtol
            types.int32,  # maxfev
            dtype,  # epsfcn
            types.CPointer(dtype),  # *diag
            types.int32,  # mode
            dtype,  # factor
            types.int32,  # nprint
            types.CPointer(types.int32),  # *nfev
            types.CPointer(dtype),  # *fjac
            types.int32,  # ldfjac
            types.CPointer(types.int32),  # *ipvt
            types.CPointer(dtype),  # *qtf
            types.CPointer(dtype),  # *wa1
            types.CPointer(dtype),  # *wa2
            types.CPointer(dtype),  # *wa3
            types.CPointer(dtype),  # *wa4
        )
        return types.ExternalFunction(_apply_prefix("lmdif", dtype), sig)

    @staticmethod
    def lmdif1(dtype: types.Float) -> types.ExternalFunction:
        """Return the external function for lmdif1.

        Parameters
        ----------
        dtype : types.Float
            The dtype

        Returns
        -------
        types.ExternalFunction
            The external function for lmdif1

        """
        sig = types.int32(
            types.voidptr,  # fcn
            types.voidptr,  # *p / *udata
            types.int32,  # m
            types.int32,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            dtype,  # tol
            types.CPointer(types.int32),  # *iwa
            types.CPointer(dtype),  # *wa
            types.int32,  # lwa
        )
        return types.ExternalFunction(_apply_prefix("lmdif1", dtype), sig)

    @staticmethod
    def lmder(dtype: types.Float) -> types.ExternalFunction:
        """Return the external function for lmder.

        Parameters
        ----------
        dtype : types.Float
            The dtype

        Returns
        -------
        types.ExternalFunction
            The external function for lmder

        """
        sig = types.int32(
            types.voidptr,  # fcn
            types.voidptr,  # *p / *udata
            types.int32,  # m
            types.int32,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            types.CPointer(dtype),  # *fjac
            types.int32,  # ldfjac
            dtype,  # ftol
            dtype,  # xtol
            dtype,  # gtol
            types.int32,  # maxfev
            types.CPointer(dtype),  # *diag
            types.int32,  # mode
            dtype,  # factor
            types.int32,  # nprint
            types.CPointer(types.int32),  # *nfev
            types.CPointer(types.int32),  # *njev
            types.CPointer(types.int32),  # *ipvt
            types.CPointer(dtype),  # *qtf
            types.CPointer(dtype),  # *wa1
            types.CPointer(dtype),  # *wa2
            types.CPointer(dtype),  # *wa3
            types.CPointer(dtype),  # *wa4
        )
        return types.ExternalFunction(_apply_prefix("lmder", dtype), sig)

    @staticmethod
    def lmder1(dtype: types.Float) -> types.ExternalFunction:
        """Return the external function for lmder1.

        Parameters
        ----------
        dtype : types.Float
            The dtype

        Returns
        -------
        types.ExternalFunction
            The external function for lmder1

        """
        sig = types.int32(
            types.voidptr,  # fcn
            types.voidptr,  # *p / *udata
            types.int32,  # m
            types.int32,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            types.CPointer(dtype),  # *fjac
            types.int32,  # ldfjac
            dtype,  # tol
            types.CPointer(types.int32),  # *ipvt
            types.CPointer(dtype),  # *wa
            types.int32,  # lwa
        )
        return types.ExternalFunction(_apply_prefix("lmder1", dtype), sig)

    @staticmethod
    def lmstr(
        dtype: types.Float,
    ) -> types.ExternalFunction:
        """Return the external function for lmstr.

        Parameters
        ----------
        dtype : types.Float
            The dtype

        Returns
        -------
        types.ExternalFunction
            The external function for lmstr

        """
        sig = types.int32(
            types.voidptr,  # fcn
            types.voidptr,  # *p / *udata
            types.int32,  # m
            types.int32,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            types.CPointer(dtype),  # *fjac
            types.int32,  # ldfjac
            dtype,  # ftol
            dtype,  # xtol
            dtype,  # gtol
            types.int32,  # maxfev
            types.CPointer(dtype),  # *diag
            types.int32,  # mode
            dtype,  # factor
            types.int32,  # nprint
            types.CPointer(types.int32),  # *nfev
            types.CPointer(types.int32),  # *njev
            types.CPointer(types.int32),  # *ipvt
            types.CPointer(dtype),  # *qtf
            types.CPointer(dtype),  # *wa1
            types.CPointer(dtype),  # *wa2
            types.CPointer(dtype),  # *wa3
            types.CPointer(dtype),  # *wa4
        )
        return types.ExternalFunction(_apply_prefix("lmstr", dtype), sig)

    @staticmethod
    def lmstr1(dtype: types.Float) -> types.ExternalFunction:
        """Return the external function for lmstr1.

        Parameters
        ----------
        dtype : types.Float
            The dtype

        Returns
        -------
        types.ExternalFunction
            The external function for lmstr1

        """
        sig = types.int32(
            types.voidptr,  # fcn
            types.voidptr,  # *p / *udata
            types.int32,  # m
            types.int32,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            types.CPointer(dtype),  # *fjac
            types.int32, # ldfjac
            dtype,  # tol
            types.CPointer(types.int32),  # *ipvt
            types.CPointer(dtype),  # *wa
            types.int32,  # lwa
        )
        return types.ExternalFunction(_apply_prefix("lmstr1", dtype), sig)
