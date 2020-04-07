import numpy as np
import pytest

import pyccl as ccl

COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')

COSMO_NU = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear', m_nu=0.1,
    Omega_k=0.1)

AVALS = [
    1,
    1.0,
    0.8,
    [0.2, 0.4],
    np.array([0.2, 0.4])]

input_a_array_not1 = np.linspace(0.1, 0.9, 100)
input_a_array_descending = np.linspace(1.0, 0.1, 100)
input_a_array = np.linspace(0.1, 1, 100)
input_chi = ccl.background.comoving_radial_distance(COSMO, input_a_array)
input_hoh0 = ccl.background.h_over_h0(COSMO, input_a_array)
input_growth = ccl.background.growth_factor(COSMO, input_a_array)
input_fgrowth = ccl.background.growth_rate(COSMO, input_a_array)


@pytest.mark.parametrize('a', AVALS)
@pytest.mark.parametrize('func', [
    ccl.growth_factor,
    ccl.growth_rate,
    ccl.growth_factor_unnorm,
    ccl.scale_factor_of_chi,
    ccl.comoving_angular_distance,
    ccl.comoving_radial_distance,
    ccl.angular_diameter_distance,
    ccl.luminosity_distance,
    ccl.h_over_h0,
    ccl.distance_modulus,
    ccl.mu_MG,
    ccl.Sig_MG])
def test_background_a_interface(a, func):
    if func is ccl.distance_modulus and np.any(a == 1):
        with pytest.raises(ccl.CCLError):
            func(COSMO, a)
    else:
        val = func(COSMO, a)
        assert np.all(np.isfinite(val))
        assert np.shape(val) == np.shape(a)
        if(func is ccl.angular_diameter_distance):
            val = func(COSMO, a, a)
            assert np.all(np.isfinite(val))
            assert np.shape(val) == np.shape(a)
            if(isinstance(a, float) or isinstance(a, int)):
                val1 = ccl.angular_diameter_distance(COSMO, 1., a)
                val2 = ccl.comoving_angular_distance(COSMO, a)*a
            else:
                val1 = ccl.angular_diameter_distance(COSMO, np.ones(len(a)), a)
                val2 = ccl.comoving_angular_distance(COSMO, a)*a
            assert np.allclose(val1, val2)


@pytest.mark.parametrize('a', AVALS)
@pytest.mark.parametrize('kind', [
    'matter',
    'dark_energy',
    'radiation',
    'curvature',
    'neutrinos_rel',
    'neutrinos_massive'])
def test_background_omega_x(a, kind):
    val = ccl.omega_x(COSMO_NU, a, kind)
    assert np.all(np.isfinite(val))
    assert np.shape(val) == np.shape(a)

    if np.all(a == 1):
        if kind == 'matter':
            val_z0 = (
                COSMO_NU['Omega_b'] +
                COSMO_NU['Omega_c'] +
                COSMO_NU['Omega_nu_mass'])
        elif kind == 'dark_energy':
            val_z0 = COSMO_NU['Omega_l']
        elif kind == 'radiation':
            val_z0 = COSMO_NU['Omega_g']
        elif kind == 'curvature':
            val_z0 = COSMO_NU['Omega_k']
        elif kind == 'neutrinos_rel':
            val_z0 = COSMO_NU['Omega_nu_rel']
        elif kind == 'neutrinos_massive':
            val_z0 = COSMO_NU['Omega_nu_mass']

        assert np.allclose(val, val_z0)


def test_background_omega_x_raises():
    with pytest.raises(ValueError):
        ccl.omega_x(COSMO, 1, 'blah')


@pytest.mark.parametrize('a', AVALS)
@pytest.mark.parametrize('kind', [
    'matter',
    'dark_energy',
    'radiation',
    'curvature',
    'neutrinos_rel',
    'neutrinos_massive'])
@pytest.mark.parametrize('is_comoving', [True, False])
def test_background_rho_x(a, kind, is_comoving):
    val = ccl.rho_x(COSMO_NU, a, kind, is_comoving)
    assert np.all(np.isfinite(val))
    assert np.shape(val) == np.shape(a)


def test_background_rho_x_raises():
    with pytest.raises(ValueError):
        ccl.rho_x(COSMO, 1, 'blah', False)


def test_input_arrays():
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.7, n_s=0.965,
                          A_s=2e-9)
    # Where the input quantities are calculated.
    a_arr = np.linspace(0.1, 1, 100)
    chi_from_ccl = ccl.background.comoving_radial_distance(cosmo, a_arr)
    hoh0_from_ccl = ccl.background.h_over_h0(cosmo, a_arr)
    growth_from_ccl = ccl.background.growth_factor(cosmo, a_arr)
    fgrowth_from_ccl = ccl.background.growth_rate(cosmo, a_arr)

    cosmo_input = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.7, n_s=0.965,
                                A_s=2e-9)
    cosmo_input._set_background_from_arrays(a_array=a_arr,
                                            chi_array=chi_from_ccl,
                                            hoh0_array=hoh0_from_ccl,
                                            growth_array=growth_from_ccl,
                                            fgrowth_array=fgrowth_from_ccl)

    # Where to compare chi(a) from CCL and from CCL with input quantities.
    a_arr = np.linspace(0.102, 0.987, 158)
    chi_ccl_input = ccl.background.comoving_radial_distance(cosmo_input,
                                                            a_arr)
    chi_from_ccl = ccl.background.comoving_radial_distance(cosmo, a_arr)
    # Relative difference (a-b)/b < 1e-5 = 0.001 %
    assert np.allclose(chi_ccl_input, chi_from_ccl, atol=0., rtol=1e-5)

    growth_ccl_input = ccl.background.growth_factor(cosmo_input, a_arr)
    growth_from_ccl = ccl.background.growth_factor(cosmo, a_arr)
    assert np.allclose(growth_ccl_input, growth_from_ccl, atol=0., rtol=1e-5)

    fgrowth_ccl_input = ccl.background.growth_rate(cosmo_input, a_arr)
    fgrowth_from_ccl = ccl.background.growth_rate(cosmo, a_arr)
    assert np.allclose(fgrowth_ccl_input, fgrowth_from_ccl, atol=0., rtol=1e-5)


def test_input_arrays_raises():
    """
    Test for input scale factor array being descending,
    not ending in 1.0, being different size, as well as
    for no input arrays.
    """
    for input_a in [input_a_array_descending, input_a_array_not1,
                    input_a_array[:-2]]:
        cosmo_input = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.7,
                                    n_s=0.965, A_s=2e-9)
        cosmo_input._set_background_from_arrays(a_array=input_a,
                                                chi_array=input_chi,
                                                hoh0_array=input_hoh0,
                                                growth_array=input_growth,
                                                fgrowth_array=input_fgrowth)
        with pytest.raises(ValueError):
            cosmo_input.compute_distances()
            cosmo_input.compute_growth()
    # Test trying to set input arrays when cosmology has been initialized
    with pytest.raises(ValueError):
        cosmo_input._set_background_from_arrays(a_array=input_a_array,
                                                chi_array=input_chi,
                                                hoh0_array=input_hoh0,
                                                growth_array=input_growth,
                                                fgrowth_array=input_fgrowth)
        cosmo_input.compute_growth()
        cosmo_input._set_background_from_arrays(a_array=input_a,
                                                chi_array=input_chi,
                                                hoh0_array=input_hoh0,
                                                growth_array=input_growth,
                                                fgrowth_array=input_fgrowth)
    # Test trying to set background without input arrays
    with pytest.raises(ValueError):
        cosmo_input = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.7,
                                    n_s=0.965, A_s=2e-9)
        cosmo_input._set_background_from_arrays()


def test_input_lin_power_spectrum():
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.7, n_s=0.965,
                          A_s=2e-9)
    a_arr = np.linspace(0.01, 0.99999, 100)
    k_arr = np.logspace(np.log10(2e-4), np.log10(50), 1000)
    pk_arr = np.empty(shape=(len(a_arr), len(k_arr)))
    for i, a in enumerate(a_arr):
        pk_arr[i] = ccl.power.linear_matter_power(cosmo, k_arr, a)

    chi_from_ccl = ccl.background.comoving_radial_distance(cosmo, a_arr)
    hoh0_from_ccl = ccl.background.h_over_h0(cosmo, a_arr)
    growth_from_ccl = ccl.background.growth_factor(cosmo, a_arr)
    fgrowth_from_ccl = ccl.background.growth_rate(cosmo, a_arr)

    cosmo_input = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.7, n_s=0.965,
                                A_s=2e-9)
    cosmo_input._set_background_from_arrays(a_array=a_arr,
                                            chi_array=chi_from_ccl,
                                            hoh0_array=hoh0_from_ccl,
                                            growth_array=growth_from_ccl,
                                            fgrowth_array=fgrowth_from_ccl)
    cosmo_input._set_linear_power_from_arrays(a_arr, k_arr, pk_arr)
    cosmo_input.compute_linear_power()
    pk_CCL_input = ccl.power.linear_matter_power(cosmo_input, k_arr, 0.5)
    pk_CCL = ccl.power.linear_matter_power(cosmo, k_arr, 0.5)

    # The first k's seem to always be somewhat high (10^-3 relative
    # difference).
    assert np.allclose(pk_CCL_input, pk_CCL, atol=0., rtol=1e-2)
