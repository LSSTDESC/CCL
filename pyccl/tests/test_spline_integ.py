import numpy as np
import pytest
from pyccl.pyutils import _spline_integrate
from pyccl.errors import CCLError


def yf(x, pw):
    return x**pw


def yf_int(x, pw):
    return x**(pw + 1) / (pw + 1)


def test_intspline_accuracy():
    nx = 1024
    x_arr = np.linspace(-2, 2, nx)
    y_arr = np.array([yf(x_arr, p)
                      for p in [2, 4]])
    int_spline = _spline_integrate(x_arr, y_arr, -1, 1)
    int_exact = np.array([yf_int(1, p) - yf_int(-1, p)
                          for p in [2, 4]])
    assert np.all(np.fabs(int_spline / int_exact - 1) < 1E5-5)


def test_intspline_raises_shapes():
    x = np.linspace(0, 1, 10)
    y = np.zeros([4, 10])

    # x is 0- or 2-D
    with pytest.raises(ValueError):
        _spline_integrate(x[0], y, 0, 1)

    with pytest.raises(ValueError):
        _spline_integrate(y, y, 0, 1)

    # y is 0-D
    with pytest.raises(ValueError):
        _spline_integrate(x, y[0, 0], 0, 1)

    # Mismatching x and y
    with pytest.raises(ValueError):
        _spline_integrate(x[1:], y, 0, 1)

    # Limits as arrays
    with pytest.raises(TypeError):
        _spline_integrate(x, y, np.zeros(10), 1)


def test_intspline_raises_internal():
    x = np.linspace(0, 1, 10)
    y = np.zeros([4, 10])

    # Function is not splineable
    with pytest.raises(CCLError):
        _spline_integrate(0 * x, y, 0, 1)


def test_intspline_smoke():
    x = np.linspace(0, 1, 10)
    y = yf(x, 2)

    r = _spline_integrate(x, y, 0, 1)
    assert np.all(~np.isnan(r))

    r = _spline_integrate(x, np.array([y, y]), 0, 1)
    assert np.all(~np.isnan(r))
