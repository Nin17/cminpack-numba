"""Utility functions for cminpack_numba."""

from __future__ import annotations

import ctypes as ct
import typing
import warnings
from pathlib import Path

from numba.core import cgutils
from numba.extending import intrinsic
from numba.types import CPointer, int32, voidptr

if typing.TYPE_CHECKING:
    from numba.core.typing import Signature
    from numba.types import FunctionType, Type
    from numpy.typing import ArrayLike, DTypeLike, NDArray

ptr_int32 = CPointer(int32)


@intrinsic
def address_as_void_pointer(
    typingctx,
    src: Type,
) -> tuple[Signature, callable]:
    """Void pointer from given memory address.

    Copied from: https://stackoverflow.com/a/61550054/15456681

    Parameters
    ----------
    typingctx :
        typing context
    src : Type
        type of memory address

    Returns
    -------
    tuple[Signature, callable]
        type signature and implementation function for the intrinsic

    """
    sig = voidptr(src)

    def codegen(context, builder, signature, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)

    return sig, codegen


@intrinsic
def ptr_from_val(typingctx, src: Type) -> tuple[Signature, callable]:
    """Pointer from given value.

    Copied from: https://stackoverflow.com/a/59538114/15456681

    Parameters
    ----------
    typingctx : _type_
        typing context
    src : Type
        type of value

    Returns
    -------
    tuple[Signature, callable]
        type signature and implementation function for the intrinsic

    """

    def impl(context, builder, signature, args):
        return cgutils.alloca_once_value(builder, args[0])

    sig = CPointer(src)(src)
    return sig, impl


@intrinsic
def val_from_ptr(typingctx, src: Type) -> tuple[Signature, callable]:
    """Value from given pointer.

    Copied from: https://stackoverflow.com/a/59538114/15456681

    Parameters
    ----------
    typingctx : _type_
        typing context
    src : Type
        type of pointer

    Returns
    -------
    tuple[Signature, callable]
        type signature and implementation function for the intrinsic

    """

    def impl(context, builder, signature, args):
        return builder.load(args[0])

    sig = src.dtype(src)
    return sig, impl


def _check_dtype(
    args: tuple[ArrayLike, ...],
    dtype: DTypeLike,
    *,
    error: bool = True,
) -> None:
    """Check that all array arguments are of the same dtype.

    Parameters
    ----------
    args : tuple[ArrayLike, ...]
        The array arguments
    dtype : DTypeLike
        The dtype to check against
    error : bool, optional
        Whether to raise an error (warning raised if False), by default True

    Raises
    ------
    ValueError
        If all array arguments are not of the same dtype and error is True

    """
    if not all(i.dtype is dtype for i in args):
        msg = "All array arguments {} be of the same dtype"
        if error:
            raise ValueError(msg.format("must"))
        warnings.warn(msg.format("should"), stacklevel=1)


def check_cfunc(func: FunctionType, *args: NDArray | int) -> ct.c_int:
    """Check a numba cfunc with ctypes.

    Parameters
    ----------
    func : types.FunctionType
        The function to check.
    *args : NDArray | int
        The arguments to pass to the function.

    Returns
    -------
    ct.c_int
        The return value of the function.

    """
    _converter = {
        ct.c_void_p: lambda x: x.ctypes.data,
        ct.c_int: lambda x: x,
        ct.POINTER(ct.c_int): lambda x: x.ctypes.data_as(ct.POINTER(ct.c_int)),
        ct.POINTER(ct.c_double): lambda x: x.ctypes.data_as(ct.POINTER(ct.c_double)),
        ct.POINTER(ct.c_float): lambda x: x.ctypes.data_as(ct.POINTER(ct.c_float)),
    }

    _func = func.ctypes
    _args = [_converter[j](i) for i, j in zip(args, _func.argtypes)]

    return _func(*_args)


def get_extension_path(lib_name: str) -> str:
    """Get the path to the library with the given name in the parent directory.

    Parameters
    ----------
    lib_name : str
        The name of the library to search for.

    Returns
    -------
    str
        The path to the library.

    """
    search_path = Path(__file__).parent
    ext_path = f"**/{lib_name}.*"
    matches = search_path.glob(ext_path)
    try:
        return str(next(matches))
    except StopIteration:
        return None
