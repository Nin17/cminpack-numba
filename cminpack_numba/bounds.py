"""_summary_"""

from numba import njit
from numba.core.errors import NumbaTypeError
from numba.extending import overload
from numba.types import NoneType, Number, Optional
from numpy import arcsin as _arcsin, cos, empty_like, sin, sqrt as _sqrt, pi

@njit
def sqrt(x):
    # print("x=", x)
    return _sqrt(x) if x > 0.0 else 0.0

@njit
def arcsin(x):
    print("hello")
    if x < -1.0:
        return -pi/2.0
    if x > 1.0:
        return pi/2.0
    return _arcsin(x)


def _ext2in(xi, lower, upper):
    raise NotImplementedError


@overload(_ext2in)
def _ext2in_overload(xi, lower, upper):
    if not isinstance(xi, Number):
        raise NumbaTypeError(f"Unsupported types: {xi}")
    impl = None
    if isinstance(lower, NoneType) and isinstance(upper, NoneType):

        def impl(xi, lower, upper):
            return xi

    if isinstance(lower, Number) and isinstance(upper, NoneType):

        def impl(xi, lower, upper):
            return sqrt((xi - lower + 1.0) ** 2.0 - 1.0)

    if isinstance(lower, NoneType) and isinstance(upper, Number):

        def impl(xi, lower, upper):
            return sqrt((upper - xi + 1.0) ** 2.0 - 1.0)

    if isinstance(lower, Number) and isinstance(upper, Number):

        def impl(xi, lower, upper):
            return arcsin((2.0 * (xi - lower) / (upper - lower)) - 1.0)
        
    if isinstance(lower, Optional) and isinstance(upper, Optional):
        def impl(xi, lower, upper):
            if lower is None and upper is None:
                return xi
            if lower is not None and upper is None:
                return sqrt((xi - lower + 1.0) ** 2.0 - 1.0)
            if lower is None and upper is not None:
                return sqrt((upper - xi + 1.0) ** 2.0 - 1.0)
            if lower is not None and upper is not None:
                return arcsin((2.0 * (xi - lower) / (upper - lower)) - 1.0)

    if impl is not None:
        return impl
    error_msg = f"Unsupported types: {lower}, {upper}"
    raise NumbaTypeError(error_msg)


@njit
def ext2in(x, lower, upper, out=None):
    assert len(lower) == len(upper) == len(x)
    assert x.ndim == 1
    _out = out if out is not None else empty_like(x)
    for i in range(len(x)):
        _out[i] = _ext2in(x[i], lower[i], upper[i])

    return _out


def _in2ext(xi, lower, upper):
    raise NotImplementedError


@overload(_in2ext)
def _in2ext_overload(xi, lower, upper):
    if not isinstance(xi, Number):
        raise NumbaTypeError(f"Unsupported types: {xi}")
    impl = None
    if isinstance(lower, NoneType) and isinstance(upper, NoneType):

        def impl(xi, lower, upper):
            return xi

    if isinstance(lower, Number) and isinstance(upper, NoneType):

        def impl(xi, lower, upper):
            return lower - 1.0 + sqrt(xi * xi + 1.0)

    if isinstance(lower, NoneType) and isinstance(upper, Number):

        def impl(xi, lower, upper):
            return upper + 1.0 - sqrt(xi * xi + 1.0)

    if isinstance(lower, Number) and isinstance(upper, Number):

        def impl(xi, lower, upper):
            return lower + ((upper - lower) / 2.0) * (sin(xi) + 1.0)

    if isinstance(lower, Optional) and isinstance(upper, Optional):

        def impl(xi, lower, upper):
            if lower is None and upper is None:
                return xi
            if lower is not None and upper is None:
                return lower - 1.0 + sqrt(xi * xi + 1.0)
            if lower is None and upper is not None:
                return upper + 1.0 - sqrt(xi * xi + 1.0)
            if lower is not None and upper is not None:
                return lower + ((upper - lower) / 2.0) * (sin(xi) + 1.0)

    if impl is not None:
        return impl
    error_msg = f"Unsupported types: {lower}, {upper}"
    raise NumbaTypeError(error_msg)


@njit
def in2ext(x, lower, upper, out=None):
    assert len(lower) == len(upper) == len(x)
    assert x.ndim == 1
    _out = out if out is not None else empty_like(x)
    for i in range(len(x)):
        _out[i] = _in2ext(x[i], lower[i], upper[i])

    return _out


def _in2ext_grad(xi, lower, upper):
    raise NotImplementedError


@overload(_in2ext_grad)
def _in2ext_grad_overload(xi, lower, upper):
    if not isinstance(xi, Number):
        raise NumbaTypeError(f"Unsupported types: {xi}")
    impl = None
    if isinstance(lower, NoneType) and isinstance(upper, NoneType):

        def impl(xi, lower, upper):
            return 1.0

    if isinstance(lower, Number) and isinstance(upper, NoneType):

        def impl(xi, lower, upper):
            return xi / sqrt(xi * xi + 1.0)

    if isinstance(lower, NoneType) and isinstance(upper, Number):

        def impl(xi, lowe, upper):
            return -xi / sqrt(xi * xi + 1.0)

    if isinstance(lower, Number) and isinstance(upper, Number):

        def impl(xi, lower, upper):
            return (upper - lower) * cos(xi) / 2.0
    
    if isinstance(lower, Optional) and isinstance(upper, Optional):
        def impl(xi, lower, upper):
            if lower is None and upper is None:
                return 1.0
            if lower is not None and upper is None:
                return xi / sqrt(xi * xi + 1.0)
            if lower is None and upper is not None:
                return -xi / sqrt(xi * xi + 1.0)
            if lower is not None and upper is not None:
                return (upper - lower) * cos(xi) / 2.0

    if impl is not None:
        return impl
    error_msg = f"Unsupported types: {lower}, {upper}"
    raise NumbaTypeError(error_msg)


@njit
def in2ext_grad(x, lower, upper, out=None):
    assert len(lower) == len(upper) == len(x)
    assert x.ndim == 1
    _out = out if out is not None else empty_like(x)
    for i in range(len(x)):
        _out[i] = _in2ext_grad(x[i], lower[i], upper[i])

    return _out
