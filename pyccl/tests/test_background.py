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
input_growth = ccl.background.growth_factor_unnorm(COSMO, input_a_array)
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
    ccl.distance_modulus])
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
    growthu_from_ccl = ccl.background.growth_factor_unnorm(cosmo, a_arr)
    fgrowth_from_ccl = ccl.background.growth_rate(cosmo, a_arr)

    background = {'a': a_arr,
                  'chi': chi_from_ccl,
                  'h_over_h0': hoh0_from_ccl}
    growth = {'a': a_arr,
              'growth_factor': growthu_from_ccl,
              'growth_rate': fgrowth_from_ccl}
    cosmo_input = ccl.CosmologyCalculator(Omega_c=0.27, Omega_b=0.05, h=0.7,
                                          n_s=0.965, A_s=2e-9,
                                          background=background,
                                          growth=growth)

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

    growthu_ccl_input = ccl.background.growth_factor_unnorm(cosmo_input, a_arr)
    growthu_from_ccl = ccl.background.growth_factor_unnorm(cosmo, a_arr)
    assert np.allclose(growthu_ccl_input, growthu_from_ccl, atol=0., rtol=1e-5)

    fgrowth_ccl_input = ccl.background.growth_rate(cosmo_input, a_arr)
    fgrowth_from_ccl = ccl.background.growth_rate(cosmo, a_arr)
    assert np.allclose(fgrowth_ccl_input, fgrowth_from_ccl, atol=0., rtol=1e-5)

    # Test that the distance/growth flags have been set
    assert cosmo_input.has_distances
    assert cosmo_input.has_growth


def test_input_arrays_raises():
    """
    Test for input scale factor array being descending,
    not ending in 1.0, being different size, as well as
    for no input arrays.
    """
    for input_a in [input_a_array_descending,
                    input_a_array_not1,
                    input_a_array[:-2]]:
        with pytest.raises(ValueError):
            ccl.CosmologyCalculator(Omega_c=0.27, Omega_b=0.05, h=0.7,
                                    n_s=0.965, A_s=2e-9,
                                    growth={'a': input_a,
                                            'growth_factor': input_growth,
                                            'growth_rate': input_fgrowth})
        with pytest.raises(ValueError):
            ccl.CosmologyCalculator(Omega_c=0.27, Omega_b=0.05, h=0.7,
                                    n_s=0.965, A_s=2e-9,
                                    background={'a': input_a,
                                                'chi': input_chi,
                                                'h_over_h0': input_hoh0})
    # Not a dictionary
    with pytest.raises(TypeError):
        ccl.CosmologyCalculator(Omega_c=0.27, Omega_b=0.05, h=0.7,
                                n_s=0.965, A_s=2e-9,
                                growth=3)
    with pytest.raises(TypeError):
        ccl.CosmologyCalculator(Omega_c=0.27, Omega_b=0.05, h=0.7,
                                n_s=0.965, A_s=2e-9,
                                background=3)
    # Incomplete dictionary
    with pytest.raises(ValueError):
        ccl.CosmologyCalculator(Omega_c=0.27, Omega_b=0.05, h=0.7,
                                n_s=0.965, A_s=2e-9,
                                background={'a': input_a_array,
                                            'h_over_h0': input_hoh0})
    with pytest.raises(ValueError):
        ccl.CosmologyCalculator(Omega_c=0.27, Omega_b=0.05, h=0.7,
                                n_s=0.965, A_s=2e-9,
                                growth={'a': input_a_array,
                                        'growth_rate': input_fgrowth})
