import numpy as np
import pytest
import pyccl as ccl


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='halofit')


@pytest.mark.parametrize('k', [
    1,
    1.0,
    [0.3, 0.5, 10],
    np.array([0.3, 0.5, 10])])
def test_linear_matter_power_smoke(k):
    a = 0.8
    pk = ccl.linear_matter_power(COSMO, k, a)
    assert np.all(np.isfinite(pk))
    assert np.shape(pk) == np.shape(k)


def test_linear_power_raises():
    cosmo = ccl.CosmologyVanillaLCDM(transfer_function='bbks')
    with pytest.raises(KeyError):
        ccl.linear_power(cosmo, 1., 1., p_of_k_a='a:b')


def test_nonlin_power_raises():
    cosmo = ccl.CosmologyVanillaLCDM(transfer_function='bbks')
    with pytest.raises(KeyError):
        ccl.nonlin_power(cosmo, 1., 1., p_of_k_a='a:b')


@pytest.mark.parametrize('k', [
    1,
    1.0,
    [0.3, 0.5, 10],
    np.array([0.3, 0.5, 10])])
def test_nonlin_matter_power_smoke(k):
    a = 0.8
    pk = ccl.nonlin_matter_power(COSMO, k, a)
    assert np.all(np.isfinite(pk))
    assert np.shape(pk) == np.shape(k)


@pytest.mark.parametrize('r', [
    1,
    1.0,
    [0.3, 0.5, 10],
    np.array([0.3, 0.5, 10])])
def test_sigmaR_smoke(r):
    a = 0.8
    sig = ccl.sigmaR(COSMO, r, a)
    assert np.all(np.isfinite(sig))
    assert np.shape(sig) == np.shape(r)


@pytest.mark.parametrize('r', [
    1,
    1.0,
    [0.3, 0.5, 10],
    np.array([0.3, 0.5, 10])])
def test_sigmaV_smoke(r):
    a = 0.8
    sig = ccl.sigmaV(COSMO, r, a)
    assert np.all(np.isfinite(sig))
    assert np.shape(sig) == np.shape(r)


def test_sigma8_consistent():
    assert np.allclose(ccl.sigma8(COSMO), COSMO['sigma8'])
    assert np.allclose(ccl.sigmaR(COSMO, 8 / COSMO['h'], 1), COSMO['sigma8'])


@pytest.mark.parametrize('A', [
    1,
    1.0,
    [0.3, 0.5, 1],
    np.array([0.3, 0.5, 1])])
def test_kNL(A):
    knl = ccl.kNL(COSMO, A)
    assert np.all(np.isfinite(knl))
    assert np.shape(knl) == np.shape(A)


@pytest.mark.parametrize('tf', [
    'boltzmann_class', 'boltzmann_camb', 'boltzmann_isitgr'])
def test_power_sigma8norm_norms_consistent(tf):
    # make a cosmo with A_s
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2e-9, n_s=0.96,
        transfer_function=tf)
    sigma8 = ccl.sigma8(cosmo)

    # remake same but now give sigma8
    cosmo_s8 = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=sigma8, n_s=0.96,
        transfer_function=tf)

    # make sure they come out the same-ish
    assert np.allclose(sigma8, ccl.sigma8(cosmo_s8))

    # and that the power spectra look right
    a = 0.8
    gfac = (
        ccl.growth_factor(cosmo, a) / ccl.growth_factor(cosmo_s8, a))**2
    pk_rat = (
        ccl.linear_matter_power(cosmo, 1e-4, a) /
        ccl.linear_matter_power(cosmo_s8, 1e-4, a))
    assert np.allclose(pk_rat, gfac)


def test_input_lin_power_spectrum():
    # Setup
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.7, n_s=0.965,
                          A_s=2e-9)
    a_arr = np.linspace(0.1, 1.0, 50)
    chi_from_ccl = ccl.background.comoving_radial_distance(cosmo, a_arr)
    hoh0_from_ccl = ccl.background.h_over_h0(cosmo, a_arr)
    growth_from_ccl = ccl.background.growth_factor_unnorm(cosmo, a_arr)
    fgrowth_from_ccl = ccl.background.growth_rate(cosmo, a_arr)
    k_arr = np.logspace(np.log10(2e-4), np.log10(1), 1000)
    pk_arr = np.empty(shape=(len(a_arr), len(k_arr)))
    for i, a in enumerate(a_arr):
        pk_arr[i] = ccl.power.linear_matter_power(cosmo, k_arr, a)

    cosmo_input = ccl.CosmologyCalculator(
        Omega_c=0.27, Omega_b=0.05, h=0.7,
        n_s=0.965, A_s=2e-9,
        background={'a': a_arr,
                    'chi': chi_from_ccl,
                    'h_over_h0': hoh0_from_ccl},
        growth={'a': a_arr,
                'growth_factor': growth_from_ccl,
                'growth_rate': fgrowth_from_ccl},
        pk_linear={'a': a_arr, 'k': k_arr,
                   'delta_matter:delta_matter': pk_arr})

    pk_CCL_input = ccl.power.linear_matter_power(cosmo_input, k_arr, 0.5)
    pk_CCL = ccl.power.linear_matter_power(cosmo, k_arr, 0.5)

    assert np.allclose(pk_CCL_input, pk_CCL, atol=0., rtol=1e-5)

    # Test again with negative power spectrum (so it's not logscaled)
    cosmo_input = ccl.CosmologyCalculator(
        Omega_c=0.27, Omega_b=0.05, h=0.7,
        n_s=0.965, A_s=2e-9,
        background={'a': a_arr,
                    'chi': chi_from_ccl,
                    'h_over_h0': hoh0_from_ccl},
        growth={'a': a_arr,
                'growth_factor': growth_from_ccl,
                'growth_rate': fgrowth_from_ccl},
        pk_linear={'a': a_arr, 'k': k_arr,
                   'delta_matter:delta_matter': -pk_arr})

    pk_CCL_input = -ccl.power.linear_matter_power(cosmo_input, k_arr, 0.5)

    assert np.allclose(pk_CCL_input, pk_CCL, atol=0., rtol=1e-5)

    # Via `linear_power`
    cosmo_input = ccl.CosmologyCalculator(
        Omega_c=0.27, Omega_b=0.05, h=0.7,
        n_s=0.965, A_s=2e-9,
        background={'a': a_arr,
                    'chi': chi_from_ccl,
                    'h_over_h0': hoh0_from_ccl},
        growth={'a': a_arr,
                'growth_factor': growth_from_ccl,
                'growth_rate': fgrowth_from_ccl},
        pk_linear={'a': a_arr, 'k': k_arr,
                   'delta_matter:delta_matter': pk_arr,
                   'a:b': pk_arr})
    pk_CCL_input = ccl.power.linear_power(cosmo_input, k_arr, 0.5,
                                          p_of_k_a='a:b')
    assert np.allclose(pk_CCL_input, pk_CCL, atol=0., rtol=1e-5)

    # Shape mismatch of input arrays
    with pytest.raises(ValueError):
        ccl.CosmologyCalculator(
            Omega_c=0.27, Omega_b=0.05, h=0.7,
            n_s=0.965, A_s=2e-9,
            background={'a': a_arr,
                        'chi': chi_from_ccl,
                        'h_over_h0': hoh0_from_ccl[:-1]})


def test_input_linpower_raises():
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.7,
                          n_s=0.965, sigma8=0.8,
                          transfer_function='bbks')
    a_arr = np.linspace(0.1, 1.0, 50)
    k_arr = np.logspace(np.log10(2e-4), np.log10(1), 1000)
    pk_arr = np.array([ccl.power.linear_matter_power(cosmo, k_arr, a)
                       for a in a_arr])

    # Not a dictionary
    with pytest.raises(TypeError):
        ccl.CosmologyCalculator(
            Omega_c=0.27, Omega_b=0.05, h=0.7,
            n_s=0.965, sigma8=0.8,
            pk_linear=np.pi)

    # a not increasing
    with pytest.raises(ValueError):
        ccl.CosmologyCalculator(
            Omega_c=0.27, Omega_b=0.05, h=0.7,
            n_s=0.965, sigma8=0.8,
            pk_linear={'a': a_arr[::-1], 'k': k_arr,
                       'delta_matter:delta_matter': pk_arr})

    # Dm x Dm not present
    with pytest.raises(ValueError):
        ccl.CosmologyCalculator(
            Omega_c=0.27, Omega_b=0.05, h=0.7,
            n_s=0.965, sigma8=0.8,
            pk_linear={'a': a_arr, 'k': k_arr,
                       'delta_matter;delta_matter': pk_arr})

    # Wrong shape
    with pytest.raises(ValueError):
        ccl.CosmologyCalculator(
            Omega_c=0.27, Omega_b=0.05, h=0.7,
            n_s=0.965, sigma8=0.8,
            pk_linear={'a': a_arr, 'k': k_arr,
                       'delta_matter:delta_matter': pk_arr,
                       'a:b': pk_arr[0]})

    # Wrong pk label
    with pytest.raises(ValueError):
        ccl.CosmologyCalculator(
            Omega_c=0.27, Omega_b=0.05, h=0.7,
            n_s=0.965, sigma8=0.8,
            pk_linear={'a': a_arr, 'k': k_arr,
                       'hello_there': pk_arr,
                       'a:b': pk_arr[0]})

    # Check new power spectrum is stored
    cosmo_input = ccl.CosmologyCalculator(
        Omega_c=0.27, Omega_b=0.05, h=0.7,
        n_s=0.965, sigma8=0.8,
        pk_linear={'a': a_arr, 'k': k_arr,
                   'delta_matter:delta_matter': pk_arr,
                   'a:b': pk_arr})
    assert 'a:b' in cosmo_input._pk_lin
    assert cosmo_input.has_linear_power


def test_input_nonlinear_model():
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.7, n_s=0.965,
                          A_s=2e-9, transfer_function='boltzmann_class')
    a_arr = np.linspace(0.1, 1.0, 50)
    k_arr = np.logspace(np.log10(2e-4), np.log10(1), 1000)
    pk_arr = np.empty(shape=(len(a_arr), len(k_arr)))
    for i, a in enumerate(a_arr):
        pk_arr[i] = ccl.power.nonlin_matter_power(cosmo, k_arr, a)

    pk_CCL = ccl.power.nonlin_matter_power(cosmo, k_arr, 0.5)

    # Test again passing only linear Pk, but letting HALOFIT do its thing
    kl_arr = np.logspace(-4, 1, 1000)
    pkl_arr = np.array([ccl.power.linear_matter_power(cosmo, kl_arr, a)
                        for a in a_arr])
    cosmo_input = ccl.CosmologyCalculator(
        Omega_c=0.27, Omega_b=0.05, h=0.7,
        n_s=0.965, A_s=2e-9,
        pk_linear={'a': a_arr, 'k': kl_arr,
                   'delta_matter:delta_matter': pkl_arr},
        nonlinear_model='halofit')

    pk_CCL_input = ccl.power.nonlin_matter_power(cosmo_input, k_arr, 0.5)

    assert np.allclose(pk_CCL_input, pk_CCL, atol=0., rtol=1e-4)

    # Test extra power spectrum
    kl_arr = np.logspace(-4, 1, 1000)
    pkl_arr = np.array([ccl.power.linear_matter_power(cosmo, kl_arr, a)
                        for a in a_arr])
    cosmo_input = ccl.CosmologyCalculator(
        Omega_c=0.27, Omega_b=0.05, h=0.7,
        n_s=0.965, A_s=2e-9,
        pk_linear={'a': a_arr, 'k': kl_arr,
                   'delta_matter:delta_matter': pkl_arr,
                   'a:b': pkl_arr},
        pk_nonlin={'a': a_arr, 'k': k_arr,
                   'delta_matter:delta_matter': -pk_arr},
        nonlinear_model='halofit')

    pk_CCL_input = cosmo_input.get_nonlin_power('a:b')(k_arr, 0.5,
                                                       cosmo_input)
    assert np.allclose(pk_CCL_input, pk_CCL, atol=0., rtol=1e-4)

    # Via `nonlin_power`
    pk_CCL_input = ccl.power.nonlin_power(cosmo_input, k_arr, 0.5,
                                          p_of_k_a='a:b')
    assert np.allclose(pk_CCL_input, pk_CCL, atol=0., rtol=1e-4)

    # Use dictionary
    cosmo_input = ccl.CosmologyCalculator(
        Omega_c=0.27, Omega_b=0.05, h=0.7,
        n_s=0.965, A_s=2e-9,
        pk_linear={'a': a_arr, 'k': kl_arr,
                   'delta_matter:delta_matter': pkl_arr,
                   'a:b': pkl_arr, 'c:d': pkl_arr},
        pk_nonlin={'a': a_arr, 'k': k_arr,
                   'delta_matter:delta_matter': -pk_arr},
        nonlinear_model={'a:b': 'halofit',
                         'c:d': None})
    pk_CCL_input = ccl.power.nonlin_power(cosmo_input, k_arr, 0.5,
                                          p_of_k_a='a:b')
    assert np.allclose(pk_CCL_input, pk_CCL, atol=0., rtol=1e-4)
    assert 'c:d' not in cosmo_input._pk_nl


def test_input_nonlin_power_spectrum():
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.7, n_s=0.965,
                          A_s=2e-9, transfer_function='boltzmann_class')
    a_arr = np.linspace(0.1, 1.0, 50)
    chi_from_ccl = ccl.background.comoving_radial_distance(cosmo, a_arr)
    hoh0_from_ccl = ccl.background.h_over_h0(cosmo, a_arr)
    growth_from_ccl = ccl.background.growth_factor_unnorm(cosmo, a_arr)
    fgrowth_from_ccl = ccl.background.growth_rate(cosmo, a_arr)
    k_arr = np.logspace(np.log10(2e-4), np.log10(1), 1000)
    pk_arr = np.empty(shape=(len(a_arr), len(k_arr)))
    for i, a in enumerate(a_arr):
        pk_arr[i] = ccl.power.nonlin_matter_power(cosmo, k_arr, a)

    cosmo_input = ccl.CosmologyCalculator(
        Omega_c=0.27, Omega_b=0.05, h=0.7,
        n_s=0.965, A_s=2e-9,
        background={'a': a_arr,
                    'chi': chi_from_ccl,
                    'h_over_h0': hoh0_from_ccl},
        growth={'a': a_arr,
                'growth_factor': growth_from_ccl,
                'growth_rate': fgrowth_from_ccl},
        pk_nonlin={'a': a_arr, 'k': k_arr,
                   'delta_matter:delta_matter': pk_arr})

    pk_CCL_input = ccl.power.nonlin_matter_power(cosmo_input, k_arr, 0.5)
    pk_CCL = ccl.power.nonlin_matter_power(cosmo, k_arr, 0.5)

    assert np.allclose(pk_CCL_input, pk_CCL, atol=0., rtol=1e-5)

    # Test again with negative power spectrum (so it's not logscaled)
    cosmo_input = ccl.CosmologyCalculator(
        Omega_c=0.27, Omega_b=0.05, h=0.7,
        n_s=0.965, A_s=2e-9,
        background={'a': a_arr,
                    'chi': chi_from_ccl,
                    'h_over_h0': hoh0_from_ccl},
        growth={'a': a_arr,
                'growth_factor': growth_from_ccl,
                'growth_rate': fgrowth_from_ccl},
        pk_nonlin={'a': a_arr, 'k': k_arr,
                   'delta_matter:delta_matter': -pk_arr})

    pk_CCL_input = -ccl.power.nonlin_matter_power(cosmo_input, k_arr, 0.5)

    assert np.allclose(pk_CCL_input, pk_CCL, atol=0., rtol=1e-5)


def test_input_nonlinear_model_raises():
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.7,
                          n_s=0.965, sigma8=0.8,
                          transfer_function='bbks')
    a_arr = np.linspace(0.1, 1.0, 50)
    k_arr = np.logspace(np.log10(2e-4), np.log10(1), 1000)
    pkl_arr = np.array([ccl.power.linear_matter_power(cosmo, k_arr, a)
                        for a in a_arr])

    # If no non-linear model provided, delta_matter:delta_matter
    # should be there.
    with pytest.raises(ValueError):
        ccl.CosmologyCalculator(
            Omega_c=0.27, Omega_b=0.05, h=0.7,
            n_s=0.965, sigma8=0.8,
            pk_nonlin={'a': a_arr, 'k': k_arr,
                       'a:b': pkl_arr})

    with pytest.raises(ValueError):
        ccl.CosmologyCalculator(
            Omega_c=0.27, Omega_b=0.05, h=0.7,
            n_s=0.965, sigma8=0.8,
            pk_linear={'a': a_arr, 'k': k_arr,
                       'delta_matter:delta_matter': pkl_arr},
            nonlinear_model=np.pi)

    with pytest.raises(ValueError):
        ccl.CosmologyCalculator(
            Omega_c=0.27, Omega_b=0.05, h=0.7,
            n_s=0.965, sigma8=0.8,
            nonlinear_model='halofit')

    with pytest.raises(KeyError):
        ccl.CosmologyCalculator(
            Omega_c=0.27, Omega_b=0.05, h=0.7,
            n_s=0.965, sigma8=0.8,
            pk_linear={'a': a_arr, 'k': k_arr,
                       'delta_matter:delta_matter': pkl_arr},
            nonlinear_model={'y:z': 'halofit'})

    with pytest.raises(ValueError):
        ccl.CosmologyCalculator(
            Omega_c=0.27, Omega_b=0.05, h=0.7,
            n_s=0.965, sigma8=0.8,
            pk_linear={'a': a_arr, 'k': k_arr,
                       'delta_matter:delta_matter': pkl_arr},
            nonlinear_model={'delta_matter:delta_matter': None})

    with pytest.raises(KeyError):
        ccl.CosmologyCalculator(
            Omega_c=0.27, Omega_b=0.05, h=0.7,
            n_s=0.965, sigma8=0.8,
            pk_linear={'a': a_arr, 'k': k_arr,
                       'delta_matter:delta_matter': pkl_arr},
            nonlinear_model={'delta_matter:delta_matter': 'unknown_model'})


def test_input_nonlin_raises():
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.7,
                          n_s=0.965, sigma8=0.8,
                          transfer_function='bbks')
    a_arr = np.linspace(0.1, 1.0, 50)
    k_arr = np.logspace(np.log10(2e-4), np.log10(1), 1000)
    pkl_arr = np.array([ccl.power.linear_matter_power(cosmo, k_arr, a)
                        for a in a_arr])
    pk_arr = np.array([ccl.power.nonlin_matter_power(cosmo, k_arr, a)
                       for a in a_arr])

    # Not a dictionary
    with pytest.raises(TypeError):
        ccl.CosmologyCalculator(
            Omega_c=0.27, Omega_b=0.05, h=0.7,
            n_s=0.965, sigma8=0.8,
            pk_nonlin=np.pi)

    # k not present
    with pytest.raises(KeyError):
        ccl.CosmologyCalculator(
            Omega_c=0.27, Omega_b=0.05, h=0.7,
            n_s=0.965, sigma8=0.8,
            pk_nonlin={'a': a_arr, 'kk': k_arr,
                       'delta_matter;delta_matter': pk_arr})

    # a not increasing
    with pytest.raises(ValueError):
        ccl.CosmologyCalculator(
            Omega_c=0.27, Omega_b=0.05, h=0.7,
            n_s=0.965, sigma8=0.8,
            pk_nonlin={'a': a_arr[::-1], 'k': k_arr,
                       'delta_matter:delta_matter': pk_arr})

    # delta_matter:delta_matter not present
    with pytest.raises(ValueError):
        ccl.CosmologyCalculator(
            Omega_c=0.27, Omega_b=0.05, h=0.7,
            n_s=0.965, sigma8=0.8,
            pk_nonlin={'a': a_arr, 'k': k_arr,
                       'delta_matter;delta_matter': pk_arr})

    # Wrong shape
    with pytest.raises(ValueError):
        ccl.CosmologyCalculator(
            Omega_c=0.27, Omega_b=0.05, h=0.7,
            n_s=0.965, sigma8=0.8,
            pk_nonlin={'a': a_arr, 'k': k_arr,
                       'delta_matter:delta_matter': pk_arr,
                       'a:b': pk_arr[0]})

    # Linear Pk not set for halofit
    with pytest.raises(ValueError):
        ccl.CosmologyCalculator(
            Omega_c=0.27, Omega_b=0.05, h=0.7,
            n_s=0.965, sigma8=0.8,
            nonlinear_model='halofit')

    # Check new power spectrum is stored
    cosmo_input = ccl.CosmologyCalculator(
        Omega_c=0.27, Omega_b=0.05, h=0.7,
        n_s=0.965, sigma8=0.8,
        pk_linear={'a': a_arr, 'k': k_arr,
                   'delta_matter:delta_matter': pkl_arr,
                   'a:b': pkl_arr},
        pk_nonlin={'a': a_arr, 'k': k_arr,
                   'delta_matter:delta_matter': pk_arr},
        nonlinear_model='halofit')
    assert 'a:b' in cosmo_input._pk_nl
    assert cosmo_input.has_nonlin_power


def test_camb_de_model():
    """Check that the dark energy model for CAMB has been properly defined."""
    with pytest.raises(ValueError):
        cosmo = ccl.CosmologyVanillaLCDM(
            transfer_function='boltzmann_camb',
            extra_parameters={"camb": {"dark_energy_model": "pf"}})
        ccl.linear_matter_power(cosmo, 1, 1)

    """Check that w is not less than -1, if the chosen dark energy model for
    CAMB is fluid."""
    with pytest.raises(ValueError):
        cosmo = ccl.CosmologyVanillaLCDM(
            transfer_function='boltzmann_camb', w0=-1, wa=-1)
        ccl.linear_matter_power(cosmo, 1, 1)

    """Check that ppf is running smoothly."""
    cosmo = ccl.CosmologyVanillaLCDM(
        transfer_function='boltzmann_camb', w0=-1, wa=-1,
        extra_parameters={"camb": {"dark_energy_model": "ppf"}})
    assert np.isfinite(ccl.linear_matter_power(cosmo, 1, 1))
