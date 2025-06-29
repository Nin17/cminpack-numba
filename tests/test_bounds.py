"""_summary_"""

from cminpack_numba import in2ext, in2ext_grad, lmder, lmder_sig, lmdif, lmdif_sig
from numba import carray, cfunc, njit
from numba.types import float64
from numpy import array, inf, take
from numpy.linalg import norm
from numpy.testing import assert_, assert_allclose


@cfunc(lmdif_sig)
def fun_trivial_bounds(udata, m, n, x, fvec, iflag):
    x_in = carray(x, (n,), float64)
    fvec = carray(fvec, (m,), float64)
    udata = carray(udata, (2,), float64)
    lb = udata[0]
    ub = udata[1]
    x_ext = in2ext(x_in, (lb,), (ub,), out=None)
    fvec[:] = x_ext**2 + 5.0
    return 0


@cfunc(lmder_sig)
def fun_trivial_jac_bounds(udata, m, n, x, fvec, fjac, ldfjac, iflag):
    x_in = carray(x, (n,), float64)
    fvec = carray(fvec, (m,), float64)
    fjac = carray(fjac, (m, n), float64)
    udata = carray(udata, (2,), float64)
    lb = udata[0]
    ub = udata[1]
    x_ext = in2ext(x_in, (lb,), (ub,), out=None)
    if iflag == 1:
        fvec[:] = x_ext**2 + 5.0
    if iflag == 2:
        scale = in2ext_grad(x_in, (lb,), (ub,), out=None)
        fjac[:] = 2.0 * x_ext * scale

    return 0


def fun_rosenbrock(x):
    return array([10 * (x[1] - x[0] ** 2), (1 - x[0])])


def jac_rosenbrock(x):
    return array(
        [
            [-20 * x[0], 10],
            [-1, 0],
        ]
    )


@cfunc(lmdif_sig)
def rosenbrock_bounds(udata, m, n, x, fvec, iflag):
    x_in = carray(x, (n,), float64)
    fvec = carray(fvec, (m,), float64)
    udata = carray(udata, (4,), float64)
    lb0, lb1 = udata[:2]
    ub0, ub1 = udata[2:]
    lb0 = None if lb0 == -inf else lb0
    lb1 = None if lb1 == -inf else lb1
    ub0 = None if ub0 == inf else ub0
    ub1 = None if ub1 == inf else ub1
    x_ext = in2ext(x_in, (lb0, lb1), (ub0, ub1), out=None)
    fvec[0] = 10.0 * (x_ext[1] - x_ext[0] ** 2)
    fvec[1] = 1.0 - x_ext[0]
    return 0


@njit
def driver_fun_trivial(address, x0, lb, ub):
    x, fvec, fjac, ipvt, qtf, _nfev, info = lmdif(address, 1, x0, udata=array([lb, ub]))

    return in2ext(x, (lb,), (ub,)), fvec, fjac, ipvt, qtf, _nfev, info


@njit
def driver_fun_trivial_jac(address, x0, lb, ub):
    x, fvec, fjac, ipvt, qtf, nfev, njev, info = lmder(
        address,
        1,
        x0,
        udata=array([lb, ub]),
    )
    # TODO(nin17): convert fjac from internal params to external
    return in2ext(x, (lb,), (ub,)), fvec, fjac, ipvt, qtf, nfev, njev, info


@njit
def driver_rosenbrock(address, x0, udata):
    x, fvec, fjac, ipvt, qtf, _nfev, info = lmdif(
        address,
        2,
        x0,
        udata=udata,
    )
    lb0, lb1, ub0, ub1 = udata
    lb0 = None if lb0 == -inf else lb0
    lb1 = None if lb1 == -inf else lb1
    ub0 = None if ub0 == inf else ub0
    ub1 = None if ub1 == inf else ub1

    grad = in2ext_grad(x, (lb0, lb1), (ub0, ub1))
    _fjac = fjac.T / take(grad, ipvt - 1).T
    return norm(_fjac.T.dot(fvec), ord=inf)


def test_in_bounds() -> None:
    funcs = driver_fun_trivial, driver_fun_trivial_jac
    addresses = fun_trivial_bounds.address, fun_trivial_jac_bounds.address
    for func, address in zip(funcs, addresses):
        res = func(address, array([2.0]), -1.0, 3.0)
        assert_allclose(res[0], 0.0, atol=1e-4)
        assert_(-1 <= res[0] <= 3)
        res = func(address, array([2.0]), 0.5, 3.0)
        assert_allclose(res[0], 0.5, atol=1e-4)
        assert_(0.5 <= res[0] <= 3)


# def test_rosenbrock_bounds(self):
#     x0_1 = array([-2.0, 1.0])
#     x0_2 = array([2.0, 2.0])
#     x0_3 = array([-2.0, 2.0])
#     x0_4 = array([0.0, 2.0])
#     x0_5 = array([-1.2, 1.0])
#     problems = [
#         (x0_1, ([-inf, -1.5], inf)),
#         (x0_2, ([-inf, 1.5], inf)),
#         (x0_3, ([-inf, 1.5], inf)),
#         (x0_4, ([-inf, 1.5], [1.0, inf])),
#         (x0_2, ([1.0, 1.5], [3.0, 3.0])),
#         (x0_5, ([-50.0, 0.0], [0.5, 100]))
#     ]
#     for x0, bounds in problems:
#         for jac, x_scale, tr_solver in product(
#                 ['2-point', '3-point', 'cs', jac_rosenbrock],
#                 [1.0, [1.0, 0.5], 'jac'],
#                 ['exact', 'lsmr']):
#             res = least_squares(fun_rosenbrock, x0, jac, bounds,
#                                 x_scale=x_scale, tr_solver=tr_solver,
#                                 method=self.method)
#             assert_allclose(res.optimality, 0.0, atol=1e-5)


def test_rosenbrock_bounds() -> None:
    x0_1 = array([-2.0, 1.0])
    x0_2 = array([2.0, 2.0])
    x0_3 = array([-2.0, 2.0])
    x0_4 = array([0.0, 2.0])
    x0_5 = array([-1.2, 1.0])
    problems = [
        (x0_1, array([-inf, -1.5, inf, inf])),
        (x0_2, array([-inf, 1.5, inf, inf])),
        (x0_3, array([-inf, 1.5, inf, inf])),
        (x0_4, array([-inf, 1.5, 1.0, inf])),
        (x0_2, array([1.0, 1.5, 3.0, 3.0])),
        (x0_5, array([-50.0, 0.0, 0.5, 100])),
    ]
    for x0, bounds in problems:
        optimality = driver_rosenbrock(rosenbrock_bounds.address, x0, bounds)
        print(optimality)
        # assert_allclose(optimality, 0.0, atol=1e-5)
