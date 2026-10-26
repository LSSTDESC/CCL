import numpy as np
import pyccl as ccl

# Set tolerances
BOOST_TOLERANCE = 1e-5

# Set up the cosmological parameters to be used
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67,
                      sigma8=0.8, n_s=0.96,
                      transfer_function='eisenstein_hu')

# Input data for the custom boost factor: a smooth suppression
# whose amplitude grows with scale factor
k_data = np.geomspace(1E-3, 50, 32)
a_data = np.linspace(0.5, 1.0, 8)


def boost_analytic(k, a):
    a_arr = np.atleast_1d(a)
    k_arr = np.atleast_1d(k)
    dip = np.exp(-0.5*(np.log(k_arr[None, :]/5.0))**2)
    return np.squeeze(1 - 0.2*a_arr[:, None]*dip)


boost_data = boost_analytic(k_data, a_data)


def test_recover_input_arrays():
    # Querying the boost factor on the input grid should
    # return the input boost data
    bar = ccl.BaryonsCustom(boost_data, k_data, a_data)
    fka = bar.boost_factor(cosmo, k_data, a_data)
    assert np.allclose(fka, boost_data, atol=0, rtol=BOOST_TOLERANCE)


def test_boost_factor_shapes():
    bar = ccl.BaryonsCustom(boost_data, k_data, a_data)
    # scalar k, scalar a
    fka = bar.boost_factor(cosmo, k_data[3], a_data[2])
    assert np.ndim(fka) == 0
    assert np.allclose(fka, boost_data[2, 3],
                       atol=0, rtol=BOOST_TOLERANCE)
    # array k, scalar a
    fka = bar.boost_factor(cosmo, k_data, a_data[2])
    assert fka.shape == k_data.shape
    assert np.allclose(fka, boost_data[2],
                       atol=0, rtol=BOOST_TOLERANCE)
    # scalar k, array a
    fka = bar.boost_factor(cosmo, k_data[3], a_data)
    assert fka.shape == a_data.shape
    assert np.allclose(fka, boost_data[:, 3],
                       atol=0, rtol=BOOST_TOLERANCE)


def test_boost_factor_outside_bounds():
    # Outside the input range the boost factor should default to 1
    bar = ccl.BaryonsCustom(boost_data, k_data, a_data)
    assert np.allclose(bar.boost_factor(cosmo, 1E-5, 1.0), 1.0)
    assert np.allclose(bar.boost_factor(cosmo, 100.0, 1.0), 1.0)
    assert np.allclose(bar.boost_factor(cosmo, 1.0, 0.1), 1.0)


def test_include_baryonic_effects():
    # The ratio of power spectra with and without baryons
    # should equal the boost factor
    bar = ccl.BaryonsCustom(boost_data, k_data, a_data)
    pk2D_no_baryons = cosmo.get_nonlin_power()
    pk2D_with_baryons = bar.include_baryonic_effects(
        cosmo, pk2D_no_baryons)
    a_arr, lk_arr, pk_arr = pk2D_no_baryons.get_spline_arrays()
    _, _, pkb_arr = pk2D_with_baryons.get_spline_arrays()
    fka = bar.boost_factor(cosmo, np.exp(lk_arr), a_arr)
    assert np.allclose(pkb_arr/pk_arr, fka,
                       atol=0, rtol=BOOST_TOLERANCE)


def test_update_params():
    bar = ccl.BaryonsCustom(boost_data, k_data, a_data)
    boost_new = 1 + 0.5*(boost_data - 1)
    bar.update_parameters(boost_data=boost_new)
    assert np.allclose(bar.boost_data, boost_new)
    # The interpolator should be rebuilt with the new data
    fka = bar.boost_factor(cosmo, k_data, a_data)
    assert np.allclose(fka, boost_new, atol=0, rtol=BOOST_TOLERANCE)
    # None arguments leave everything untouched
    bar.update_parameters()
    assert np.allclose(bar.boost_data, boost_new)
    assert np.allclose(bar.k_data, k_data)
    assert np.allclose(bar.a_data, a_data)


def test_baryons_from_name():
    bar = ccl.BaryonsCustom(boost_data, k_data, a_data)
    bar2 = ccl.Baryons.from_name('BaryonsCustom')
    assert bar.name == bar2.name
    assert bar.name == 'BaryonsCustom'
