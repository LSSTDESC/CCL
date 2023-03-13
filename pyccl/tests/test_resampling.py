import numpy as np
import pytest
import pyccl as ccl

EXTRAP_TYPES = [
    'none',
    'constant',
    'linx_liny',
    'linx_logy',
    'logx_liny',
    'logx_logy'
]
R_INI = 0.1
R_END = 100
R_ARR = np.linspace(R_INI, R_END, 512)


def test_resample_raises():
    with pytest.raises(ccl.CCLError):
        ccl.resample_array(R_ARR, R_ARR[:-1], R_ARR)
    with pytest.raises(ValueError):
        ccl.resample_array(R_ARR, R_ARR, R_ARR,
                           'linx_lagy', 'linx_logy')
    with pytest.raises(ValueError):
        ccl.resample_array(R_ARR, R_ARR, R_ARR,
                           'linx_logy', 'linx_lagy')


@pytest.mark.parametrize('extrap', EXTRAP_TYPES)
def test_resample_extrapolation(extrap):
    tilt = -1.
    offset = 1

    def f(r):
        if extrap == 'linx_liny':
            return r * tilt
        elif extrap == 'linx_logy':
            return np.exp(r * tilt)
        elif extrap == 'logx_liny':
            return tilt * np.log(r)
        elif extrap == 'logx_logy':
            return r**tilt
        else:
            trunc_ini = np.heaviside(r - R_INI, 0)
            trunc_end = 1 - np.heaviside(r - R_END, 0)
            trunc = trunc_ini * trunc_end
            return r * tilt * trunc + offset

    f_arr = f(R_ARR)
    r_ini_x = 0.01
    r_end_x = 200
    r_arr_x = np.geomspace(r_ini_x, r_end_x, 2048)
    if extrap == 'none':
        with pytest.raises(ccl.CCLError):
            ccl.resample_array(R_ARR, f_arr, r_arr_x,
                               extrap, extrap,
                               offset, offset)
    else:
        f_arr_x = ccl.resample_array(R_ARR, f_arr, r_arr_x,
                                     extrap, extrap,
                                     offset, offset)
        id_extrap = (r_arr_x > R_END) | (r_arr_x < R_INI)
        f_arr_x_pred = f(r_arr_x)
        res = np.fabs(f_arr_x / f_arr_x_pred - 1)
        assert np.all(res[id_extrap] < 1E-10)
