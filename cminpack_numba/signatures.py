"""Signatures for the functions passed to the cminpack functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import types

if TYPE_CHECKING:
    from numba.core.typing import Signature

__all__ = [
    "hybrd_sig",
    "shybrd_sig",
    "hybrj_sig",
    "shybrj_sig",
    "lmdif_sig",
    "slmdif_sig",
    "lmder_sig",
    "slmder_sig",
    "lmstr_sig",
    "slmstr_sig",
    "CminpackSignature",
]


class CminpackSignature:
    """Signatures for the functions passed to the cminpack functions."""

    @staticmethod
    def hybrd(
        udata_type: types.Type = types.voidptr,
        dtype: types.Float = types.float64,
    ) -> Signature:
        """Signature for the function passed to the `hybrd` & `shybrd` functions.

        Parameters
        ----------
        udata_type : types.Type, optional
            The type of udata, by default types.voidptr
        dtype : types.Float, optional
            The dtype, by default types.float64

        Returns
        -------
        Signature
            The signature of the function

        """
        return types.intc(
            udata_type,  # *udata / *p
            types.intc,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            types.intc,  # iflag
        )

    @staticmethod
    def hybrj(
        udata_type: types.Type = types.voidptr,
        dtype: types.Float = types.float64,
    ) -> Signature:
        """Signature for the function passed to the `hybrj` & `shybrj` functions.

        Parameters
        ----------
        udata_type : types.Type, optional
            The type of udata, by default types.voidptr
        dtype : types.Float, optional
            The dtype, by default types.float64

        Returns
        -------
        Signature
            The signature of the function

        """
        return types.intc(
            udata_type,  # *udata / *p
            types.int32,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            types.CPointer(dtype),  # *fjac
            types.int32,  # ldfjac
            types.int32,  # iflag
        )

    @staticmethod
    def lmdif(
        udata_type: types.Type = types.voidptr,
        dtype: types.Float = types.float64,
    ) -> Signature:
        """Signature for the function passed to the `lmdif` & `slmdif` functions.

        Parameters
        ----------
        udata_type : types.Type, optional
            The type of udata, by default types.voidptr
        dtype : types.Float, optional
            The dtype, by default types.float64

        Returns
        -------
        Signature
            The signature of the function

        """
        return types.intc(
            udata_type,  # *udata / *p
            types.intc,  # m
            types.intc,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            types.intc,  # iflag
        )

    @staticmethod
    def lmder(
        udata_type: types.Type = types.voidptr,
        dtype: types.Float = types.float64,
    ) -> Signature:
        """Signature for the function passed to the `lmder` & `slmder` functions.

        Parameters
        ----------
        udata_type : types.Type, optional
            The type of udata, by default types.voidptr
        dtype : types.Float, optional
            The dtype, by default types.float64

        Returns
        -------
        Signature
            The signature of the function

        """
        return types.intc(
            udata_type,  # *udata / *p
            types.intc,  # m
            types.intc,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            types.CPointer(dtype),  # *fjac
            types.intc,  # ldfjac
            types.intc,  # iflag
        )

    @staticmethod
    def lmstr(
        udata_type: types.Type = types.voidptr,
        dtype: types.Float = types.float64,
    ) -> Signature:
        """Signature for the function passed to the `lmstr` & `slmstr` functions.

        Parameters
        ----------
        udata_type : types.Type, optional
            The type of udata, by default types.voidptr
        dtype : types.Float, optional
            The dtype, by default types.float64

        Returns
        -------
        Signature
            The signature of the function

        """
        return types.intc(
            udata_type,  # *udata / *p
            types.intc,  # m
            types.intc,  # n
            types.CPointer(dtype),  # *x
            types.CPointer(dtype),  # *fvec
            types.CPointer(dtype),  # *fjrow
            types.intc,  # iflag
        )


# __cminpack_type_fcn_nn__
hybrd_sig = CminpackSignature.hybrd()
"""
(udata: void*, n: int32, x: float64*, fvec: float64*, iflag: int32) -> int32
"""

# __cminpack_type_fcn_nn__
shybrd_sig = CminpackSignature.hybrd(dtype=types.float32)
"""
(udata: void*, n: int32, x: float32*, fvec: float32*, iflag: int32) -> int32
"""

# __cminpack_type_fcnder_nn__
hybrj_sig = CminpackSignature.hybrj()
"""
(udata: void*, n: int32, x: float64*, fvec: float64*, fjac: float64*,
    ldfjac: int32, iflag:int32) -> int32
"""

# __cminpack_type_fcnder_nn__
shybrj_sig = CminpackSignature.hybrj(dtype=types.float32)
"""
(udata: void*, n: int32, x: float32*, fvec: float32*, fjac: float32*,
    ldfjac: int32, iflag:int32) -> int32
"""

# __cminpack_type_fcn_mn__
lmdif_sig = CminpackSignature.lmdif()
"""
(udata: void*, m: int32, n: int32, x: float64*, fvec: float64*, iflag: int32)
    -> int32
"""

# __cminpack_type_fcn_mn_s__
slmdif_sig = CminpackSignature.lmdif(dtype=types.float32)
"""
(udata: void*, m: int32, n: int32, x: float32*, fvec: float32*, iflag: int32)
    -> int32
"""

# __cminpack_type_fcnder_mn__
lmder_sig = CminpackSignature.lmder()
"""
(udata: void*, m: int32, n: int32, x: float64*, fvec: float64*, fjac: float64*,
    ldfjac: int32, iflag: int32) -> int32
"""

# __cminpack_type_fcnder_mn__
slmder_sig = CminpackSignature.lmder(dtype=types.float32)
"""
(udata: void*, m: int32, n: int32, x: float32*, fvec: float32*, fjac: float32*,
    ldfjac: int32, iflag: int32) -> int32
"""

# __cminpack_type_fcnderstr_mn__
lmstr_sig = CminpackSignature.lmstr()
"""
(udata: void*, m: int32, n: int32, x: float64*, fvec: float64*, fjac: float64*,
    iflag: int32) -> int32
"""

# __cminpack_type_fcnderstr_mn__
slmstr_sig = CminpackSignature.lmstr(dtype=types.float32)
"""
(udata: void*, m: int32, n: int32, x: float32*, fvec: float32*, fjac: float32*,
    iflag: int32) -> int32
"""
