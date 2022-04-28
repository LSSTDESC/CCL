import pyccl as ccl
import numpy as np


def test_spline1d():
    cosmo = ccl.CosmologyVanillaLCDM()
    cosmo.compute_distances()

    chi_gsl_spline = cosmo.cosmo.data.chi
    a_arr, chi_arr = ccl.pyutils._get_spline1d_arrays(chi_gsl_spline)
    chi = ccl.comoving_radial_distance(cosmo, a_arr)

    assert np.allclose(chi_arr, chi)


def test_spline2d():
    x = np.linspace(0.1, 1, 10)
    log_y = np.linspace(-3, 1, 20)
    zarr_in = np.outer(x, np.exp(log_y))

    pk2d = ccl.Pk2D(a_arr=x, lk_arr=log_y, pk_arr=zarr_in,
                    is_logp=False)

    pk2d_gsl_spline2d = pk2d.psp.fka
    xarr, yarr, zarr_out_spline = \
        ccl.pyutils._get_spline2d_arrays(pk2d_gsl_spline2d)

    cosmo = ccl.CosmologyVanillaLCDM()
    zarr_out_eval = pk2d.eval(k=np.exp(log_y), a=x[-1], cosmo=cosmo)

    assert np.allclose(x, xarr)
    assert np.allclose(log_y, yarr)
    assert np.allclose(zarr_in, zarr_out_spline)
    assert np.allclose(zarr_in[-1], zarr_out_eval)
