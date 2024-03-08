import pickle
import tempfile
import pytest
import numpy as np
import pyccl as ccl
import copy
import warnings
from .test_cclobject import check_eq_repr_hash


def test_Cosmology_eq_repr_hash():
    # Test eq, repr, hash for Cosmology and CosmologyCalculator.
    # 1. Using a complicated Cosmology object.
    extras = {"camb": {"halofit_version": "mead2020", "HMCode_logT_AGN": 7.8}}
    kwargs = {"transfer_function": "bbks",
              "matter_power_spectrum": "linear",
              "extra_parameters": extras}
    COSMO1 = ccl.CosmologyVanillaLCDM(**kwargs)
    COSMO2 = ccl.CosmologyVanillaLCDM(**kwargs)
    assert check_eq_repr_hash(COSMO1, COSMO2)

    # 2. Now make a copy and change it.
    kwargs = copy.deepcopy(kwargs)
    kwargs["extra_parameters"]["camb"]["halofit_version"] = "mead2020_feedback"
    COSMO3 = ccl.CosmologyVanillaLCDM(**kwargs)
    assert check_eq_repr_hash(COSMO1, COSMO3, equal=False)

    # 3. Using a CosmologyCalculator.
    COSMO1.compute_linear_power()
    a_arr, lk_arr, pk_arr = COSMO1.get_linear_power().get_spline_arrays()
    pk_linear = {"a": a_arr,
                 "k": np.exp(lk_arr),
                 "delta_matter:delta_matter": pk_arr}
    COSMO4 = ccl.CosmologyCalculator(
        Omega_c=0.25, Omega_b=0.05, h=0.67, n_s=0.96, sigma8=0.81,
        pk_linear=pk_linear, pk_nonlin=pk_linear)
    COSMO5 = ccl.CosmologyCalculator(
        Omega_c=0.25, Omega_b=0.05, h=0.67, n_s=0.96, sigma8=0.81,
        pk_linear=pk_linear, pk_nonlin=pk_linear)
    assert check_eq_repr_hash(COSMO4, COSMO5)

    pk_linear = {"a": a_arr,
                 "k": np.exp(lk_arr),
                 "delta_matter:delta_matter": 2*pk_arr}
    COSMO6 = ccl.CosmologyCalculator(
        Omega_c=0.25, Omega_b=0.05, h=0.67, n_s=0.96, sigma8=0.81,
        pk_linear=pk_linear, pk_nonlin=pk_linear)
    assert check_eq_repr_hash(COSMO4, COSMO6, equal=False)


def test_cosmo_methods():
    """ Check that all pyccl functions that take cosmo
    as their first argument are methods of the Cosmology object.
    """
    from inspect import getmembers, isfunction, signature
    from pyccl import background, boltzmann, \
        cells, correlations, covariances, neutrinos, \
        pk2d, power, tk3d, tracers, halos, nl_pt
    cosmo = ccl.CosmologyVanillaLCDM()
    subs = [background, boltzmann, cells, correlations, covariances,
            neutrinos, pk2d, power, tk3d, tracers, halos, nl_pt]
    funcs = [getmembers(sub, isfunction) for sub in subs]
    funcs = [func for sub in funcs for func in sub]
    for name, func in funcs:
        if name.startswith("_"):  # no private functions
            continue
        pars = signature(func).parameters
        if pars and list(pars)[0] == "cosmo":
            _ = getattr(cosmo, name)

    # quantitative
    assert ccl.sigma8(cosmo) == cosmo.sigma8()
    assert ccl.rho_x(cosmo, 1., "matter", is_comoving=False) == \
        cosmo.rho_x(1., "matter", is_comoving=False)
    assert ccl.get_camb_pk_lin(cosmo)(1., 1., cosmo) == \
        cosmo.get_camb_pk_lin()(1., 1., cosmo)
    prof = ccl.halos.HaloProfilePressureGNFW(mass_def="200m")
    hmf = ccl.halos.MassFuncTinker08(mass_def="200m")
    hbf = ccl.halos.HaloBiasTinker10(mass_def="200m")
    hmc = ccl.halos.HMCalculator(mass_function=hmf, halo_bias=hbf,
                                 mass_def="200m")
    assert ccl.halos.halomod_power_spectrum(cosmo, hmc, 1., 1., prof) == \
        cosmo.halomod_power_spectrum(hmc, 1., 1., prof)


def test_cosmology_critical_init():
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        Neff=0,
        m_nu=0.0,
        w0=-1.0,
        wa=0.0,
        mass_split='normal',
        Omega_g=0,
        Omega_k=0)
    assert np.allclose(cosmo.cosmo.data.growth0, 1)


def test_cosmology_As_sigma8_populates():
    # Check that cosmo.sigma8() pupulates sigma8 if it is missing.
    cosmo = ccl.Cosmology(Omega_c=0.265, Omega_b=0.045, h=0.675,
                          n_s=0.965, A_s=2e-9)
    assert np.isnan(cosmo["sigma8"])
    cosmo.sigma8()
    assert cosmo["sigma8"] == cosmo.sigma8()


def test_cosmology_init():
    """
    Check that Cosmology objects can only be constructed in a valid way.
    """
    # Make sure error raised if invalid transfer/power spectrum etc. passed
    with pytest.raises(KeyError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
                      matter_power_spectrum='x')
    with pytest.raises(KeyError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
                      transfer_function='x')
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
                      matter_power_spectrum=None)
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
                      transfer_function=None)
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
                      m_nu=np.array([0.1, 0.1, 0.1, 0.1]))
    with pytest.raises(ValueError):
        ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
                      m_nu=ccl)


def test_cosmology_output():
    """
    Check that status messages and other output from Cosmology() object works
    correctly.
    """
    # Create test cosmology object
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9,
                          n_s=0.96)

    # Return and print status messages
    assert cosmo.cosmo.status == 0

    # Test status methods for different precomputable quantities
    assert not cosmo.has_distances
    assert not cosmo.has_growth
    assert not cosmo.has_linear_power
    assert not cosmo.has_nonlin_power
    assert not cosmo.has_sigma

    cosmo.compute_distances()
    cosmo.compute_growth()
    cosmo.compute_linear_power()
    cosmo.compute_nonlin_power()
    cosmo.compute_sigma()

    assert cosmo.has_distances
    assert cosmo.has_growth
    assert cosmo.has_linear_power
    assert cosmo.has_nonlin_power
    assert cosmo.has_sigma


def test_cosmology_pickles():
    """Check that a Cosmology object pickles."""
    cosmo = ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        m_nu=[0.02, 0.1, 0.05], mass_split='list')

    with tempfile.TemporaryFile() as fp:
        pickle.dump(cosmo, fp)

        fp.seek(0)
        cosmo2 = pickle.load(fp)

    assert np.allclose(ccl.comoving_radial_distance(cosmo, 0.5),
                       ccl.comoving_radial_distance(cosmo2, 0.5),
                       atol=0, rtol=0)


def test_cosmology_lcdm():
    """Check that the default vanilla cosmology behaves
    as expected"""
    c1 = ccl.Cosmology(Omega_c=0.25,
                       Omega_b=0.05,
                       h=0.67, n_s=0.96,
                       sigma8=0.81)
    c2 = ccl.CosmologyVanillaLCDM()
    assert np.allclose(ccl.comoving_radial_distance(c1, 0.5),
                       ccl.comoving_radial_distance(c2, 0.5),
                       atol=0, rtol=0)


def test_cosmology_p18lcdm_raises():
    with pytest.raises(ValueError):
        kw = {'Omega_c': 0.1}
        ccl.CosmologyVanillaLCDM(**kw)


def test_cosmology_context():
    """Check that using a Cosmology object in a context manager
    frees C resources properly."""
    with ccl.Cosmology(
            Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
            m_nu=np.array([0.02, 0.1, 0.05]), mass_split='list') as cosmo:
        # make sure it works
        assert not cosmo.has_distances
        ccl.comoving_radial_distance(cosmo, 0.5)
        assert cosmo.has_distances

    # make sure it does not!
    assert not hasattr(cosmo, "cosmo")
    assert not hasattr(cosmo, "_params")

    with pytest.raises(AttributeError):
        cosmo.has_growth


def test_pyccl_default_params():
    """Check that the Python-layer for setting the gsl and spline parameters
    works on par with the C-layer.
    """
    EPS_SCALEFAC_GROWTH = ccl.gsl_params["EPS_SCALEFAC_GROWTH"]

    # we will test with this parameter
    assert EPS_SCALEFAC_GROWTH == 1e-6

    # can be accessed as an attribute and as a dictionary item
    assert ccl.gsl_params.EPS_SCALEFAC_GROWTH == \
        ccl.gsl_params["EPS_SCALEFAC_GROWTH"]

    # can be assigned as an attribute
    ccl.gsl_params.EPS_SCALEFAC_GROWTH = 1e-5
    assert ccl.gsl_params["EPS_SCALEFAC_GROWTH"] == 1e-5  # cross-check

    ccl.gsl_params["EPS_SCALEFAC_GROWTH"] = 2e-6
    assert ccl.gsl_params.EPS_SCALEFAC_GROWTH == 2e-6

    # does not accept extra assignment
    with pytest.raises(KeyError):
        ccl.gsl_params.test = "hello_world"
    with pytest.raises(KeyError):
        ccl.gsl_params["test"] = "hello_world"

    # complains when we try to set A_SPLINE_MAX != 1.0
    ccl.spline_params.A_SPLINE_MAX = 1.0
    with pytest.raises(ValueError):
        ccl.spline_params.A_SPLINE_MAX = 0.9

    # complains when we try to change the spline type
    ccl.spline_params.A_SPLINE_TYPE = None
    with pytest.raises(TypeError):
        ccl.spline_params.A_SPLINE_TYPE = "something_else"

    # complains when we try to change the physical constants
    with pytest.raises(AttributeError):
        ccl.physical_constants.CLIGHT = 1

    # but if we unfreeze them, we can change them
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ccl.physical_constants.unfreeze()
        ccl.physical_constants.CLIGHT = 1
    assert ccl.physical_constants.CLIGHT == 1
    ccl.physical_constants.freeze()
    ccl.physical_constants.reload()

    # verify that this has changed
    assert ccl.gsl_params.EPS_SCALEFAC_GROWTH != EPS_SCALEFAC_GROWTH

    # but now we reload it, so it should be the default again
    ccl.gsl_params.reload()
    assert ccl.gsl_params.EPS_SCALEFAC_GROWTH == EPS_SCALEFAC_GROWTH


def test_cosmology_default_params():
    """Check that the default params within Cosmology work as intended."""
    cosmo1 = ccl.CosmologyVanillaLCDM()
    v1 = cosmo1.cosmo.gsl_params.EPS_SCALEFAC_GROWTH

    ccl.gsl_params.EPS_SCALEFAC_GROWTH = v1*10
    cosmo2 = ccl.CosmologyVanillaLCDM()
    v2 = cosmo2.cosmo.gsl_params.EPS_SCALEFAC_GROWTH
    assert v2 == v1*10
    assert v2 != v1

    ccl.gsl_params.reload()
    cosmo3 = ccl.CosmologyVanillaLCDM()
    v3 = cosmo3.cosmo.gsl_params.EPS_SCALEFAC_GROWTH
    assert v3 == v1


def test_ccl_physical_constants_smoke():
    assert ccl.physical_constants.CLIGHT == ccl.ccllib.cvar.constants.CLIGHT


def test_ccl_global_parameters_repr():
    ccl.spline_params.reload()
    assert eval(repr(ccl.spline_params)) == ccl.spline_params._bak


def test_camb_sigma8_input():
    sigma8 = 0.85
    cosmo = ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.67, n_s=0.96, sigma8=sigma8,
        transfer_function="boltzmann_camb"
    )
    assert np.isclose(cosmo.sigma8(), sigma8)
