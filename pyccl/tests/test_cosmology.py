from __future__ import print_function
import pickle
import tempfile

import pytest

import numpy as np
from numpy.testing import assert_raises, assert_, assert_no_warnings

import pyccl as ccl


def test_cosmo_methods():
    """ Check that all pyccl functions that take cosmo
    as their first argument are methods of the Cosmology object.
    """
    from inspect import getmembers, isfunction, signature
    from pyccl import background, bcm, boltzmann, \
        cls, correlations, covariances, neutrinos, \
        pk2d, power, tk3d, tracers, halos, nl_pt
    from pyccl.core import CosmologyVanillaLCDM
    cosmo = CosmologyVanillaLCDM()
    subs = [background, boltzmann, bcm, cls, correlations, covariances,
            neutrinos, pk2d, power, tk3d, tracers, halos, nl_pt]
    funcs = [getmembers(sub, isfunction) for sub in subs]
    funcs = [func for sub in funcs for func in sub]
    for name, func in funcs:
        pars = signature(func).parameters
        if list(pars)[0] == "cosmo":
            _ = getattr(cosmo, name)

    # quantitative
    assert ccl.sigma8(cosmo) == cosmo.sigma8()
    assert ccl.rho_x(cosmo, 1., "matter", is_comoving=False) == \
        cosmo.rho_x(1., "matter", is_comoving=False)
    assert ccl.get_camb_pk_lin(cosmo).eval(1., 1., cosmo) == \
        cosmo.get_camb_pk_lin().eval(1., 1., cosmo)
    prof = ccl.halos.HaloProfilePressureGNFW()
    hmd = ccl.halos.MassDef200m()
    hmf = ccl.halos.MassFuncTinker08(cosmo)
    hbf = ccl.halos.HaloBiasTinker10(cosmo)
    hmc = ccl.halos.HMCalculator(cosmo, massfunc=hmf, hbias=hbf, mass_def=hmd)
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
        m_nu_type='normal',
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
    assert_raises(
        ValueError, ccl.Cosmology,
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        matter_power_spectrum='x')
    assert_raises(
        ValueError, ccl.Cosmology,
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        transfer_function='x')
    assert_raises(
        ValueError, ccl.Cosmology,
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        baryons_power_spectrum='x')
    assert_raises(
        ValueError, ccl.Cosmology,
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        mass_function='x')
    assert_raises(
        ValueError, ccl.Cosmology,
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        halo_concentration='x')
    assert_raises(
        ValueError, ccl.Cosmology,
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        emulator_neutrinos='x')
    assert_raises(
        ValueError, ccl.Cosmology,
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        m_nu=np.array([0.1, 0.1, 0.1, 0.1]))
    assert_raises(
        ValueError, ccl.Cosmology,
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        m_nu=ccl)
    assert_raises(
        ValueError, ccl.Cosmology,
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        m_nu=np.array([0.1, 0.1, 0.1]),
        m_nu_type='normal')


def test_cosmology_setitem():
    cosmo = ccl.CosmologyVanillaLCDM()
    with pytest.raises(NotImplementedError):
        cosmo['a'] = 3


def test_cosmology_output():
    """
    Check that status messages and other output from Cosmology() object works
    correctly.
    """
    # Create test cosmology object
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9,
                          n_s=0.96)

    # Return and print status messages
    assert_no_warnings(cosmo.status)
    assert_no_warnings(print, cosmo)

    # Test status methods for different precomputable quantities
    assert_(cosmo.has_distances is False)
    assert_(cosmo.has_growth is False)
    assert_(cosmo.has_linear_power is False)
    assert_(cosmo.has_nonlin_power is False)
    assert_(cosmo.has_sigma is False)

    # Check that quantities can be precomputed
    assert_no_warnings(cosmo.compute_distances)
    assert_no_warnings(cosmo.compute_growth)
    assert_no_warnings(cosmo.compute_linear_power)
    assert_no_warnings(cosmo.compute_nonlin_power)
    assert_no_warnings(cosmo.compute_sigma)
    assert_(cosmo.has_distances is True)
    assert_(cosmo.has_growth is True)
    assert_(cosmo.has_linear_power is True)
    assert_(cosmo.has_nonlin_power is True)
    assert_(cosmo.has_sigma is True)


def test_cosmology_pickles():
    """Check that a Cosmology object pickles."""
    cosmo = ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        m_nu=[0.02, 0.1, 0.05], m_nu_type='list',
        z_mg=[0.0, 1.0], df_mg=[0.01, 0.0])

    with tempfile.TemporaryFile() as fp:
        pickle.dump(cosmo, fp)

        fp.seek(0)
        cosmo2 = pickle.load(fp)

    assert_(
        ccl.comoving_radial_distance(cosmo, 0.5) ==
        ccl.comoving_radial_distance(cosmo2, 0.5))


def test_cosmology_lcdm():
    """Check that the default vanilla cosmology behaves
    as expected"""
    c1 = ccl.Cosmology(Omega_c=0.25,
                       Omega_b=0.05,
                       h=0.67, n_s=0.96,
                       sigma8=0.81)
    c2 = ccl.CosmologyVanillaLCDM()
    assert_(ccl.comoving_radial_distance(c1, 0.5) ==
            ccl.comoving_radial_distance(c2, 0.5))


def test_cosmology_p18lcdm_raises():
    with pytest.raises(ValueError):
        kw = {'Omega_c': 0.1}
        ccl.CosmologyVanillaLCDM(**kw)


def test_cosmology_repr():
    """Check that we can make a Cosmology object from its repr."""
    import pyccl  # noqa: F401

    cosmo = ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        m_nu=[0.02, 0.1, 0.05], m_nu_type='list',
        z_mg=[0.0, 1.0], df_mg=[0.01, 0.0])

    cosmo2 = eval(str(cosmo))
    assert_(
        ccl.comoving_radial_distance(cosmo, 0.5) ==
        ccl.comoving_radial_distance(cosmo2, 0.5))

    cosmo3 = eval(repr(cosmo))
    assert_(
        ccl.comoving_radial_distance(cosmo, 0.5) ==
        ccl.comoving_radial_distance(cosmo3, 0.5))

    # same test with arrays to be sure
    cosmo = ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        m_nu=np.array([0.02, 0.1, 0.05]), m_nu_type='list',
        z_mg=np.array([0.0, 1.0]), df_mg=np.array([0.01, 0.0]))

    cosmo2 = eval(str(cosmo))
    assert_(
        ccl.comoving_radial_distance(cosmo, 0.5) ==
        ccl.comoving_radial_distance(cosmo2, 0.5))

    cosmo3 = eval(repr(cosmo))
    assert_(
        ccl.comoving_radial_distance(cosmo, 0.5) ==
        ccl.comoving_radial_distance(cosmo3, 0.5))


def test_cosmology_context():
    """Check that using a Cosmology object in a context manager
    frees C resources properly."""
    with ccl.Cosmology(
            Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
            m_nu=np.array([0.02, 0.1, 0.05]), m_nu_type='list',
            z_mg=np.array([0.0, 1.0]), df_mg=np.array([0.01, 0.0])) as cosmo:
        # make sure it works
        assert not cosmo.has_distances
        ccl.comoving_radial_distance(cosmo, 0.5)
        assert cosmo.has_distances

    # make sure it does not!
    assert_(not hasattr(cosmo, "cosmo"))
    assert_(not hasattr(cosmo, "_params"))

    with pytest.raises(AttributeError):
        cosmo.has_growth


def test_pyccl_default_params():
    """Check that the Python-layer for setting the gsl and spline parameters
    works on par with the C-layer.
    """
    HM_MMIN = ccl.gsl_params["HM_MMIN"]

    # we will test with this parameter
    assert HM_MMIN == 1e7

    # can be accessed as an attribute and as a dictionary item
    assert ccl.gsl_params.HM_MMIN == ccl.gsl_params["HM_MMIN"]

    # can be assigned as an attribute
    ccl.gsl_params.HM_MMIN = 1e5
    assert ccl.gsl_params["HM_MMIN"] == 1e5  # cross-check

    ccl.gsl_params["HM_MMIN"] = 1e6
    assert ccl.gsl_params.HM_MMIN == 1e6

    # does not accept extra assignment
    with pytest.raises(KeyError):
        ccl.gsl_params.test = "hello_world"
    with pytest.raises(KeyError):
        ccl.gsl_params["test"] = "hallo_world"

    # complains when we try to set A_SPLINE_MAX != 1.0
    ccl.spline_params.A_SPLINE_MAX = 1.0
    with pytest.raises(RuntimeError):
        ccl.spline_params.A_SPLINE_MAX = 0.9

    # complains when we try to change the spline type
    ccl.spline_params.A_SPLINE_TYPE = None
    with pytest.raises(TypeError):
        ccl.spline_params.A_SPLINE_TYPE = "something_else"

    # complains when we try to change the physical constants
    with pytest.raises(AttributeError):
        ccl.physical_constants.CLIGHT = 1

    # verify that this has changed
    assert ccl.gsl_params.HM_MMIN != HM_MMIN

    # but now we reload it, so it should be the default again
    ccl.gsl_params.reload()
    assert ccl.gsl_params.HM_MMIN == HM_MMIN


def test_cosmology_default_params():
    """Check that the default params within Cosmology work as intended."""
    cosmo1 = ccl.CosmologyVanillaLCDM()
    v1 = cosmo1.cosmo.gsl_params.HM_MMIN

    ccl.gsl_params.HM_MMIN = v1*10
    cosmo2 = ccl.CosmologyVanillaLCDM()
    v2 = cosmo2.cosmo.gsl_params.HM_MMIN
    assert v2 == v1*10
    assert v2 != v1

    ccl.gsl_params.reload()
    cosmo3 = ccl.CosmologyVanillaLCDM()
    v3 = cosmo3.cosmo.gsl_params.HM_MMIN
    assert v3 == v1

    # warns when we try to mutate instantiated `cvar` objects
    with pytest.warns(ccl.CCLDeprecationWarning):
        cosmo1.cosmo.spline_params.A_SPLINE_MIN = 0.5


def test_ccl_physical_constants_smoke():
    assert ccl.physical_constants.CLIGHT == ccl.ccllib.cvar.constants.CLIGHT


def test_ccl_global_parameters_repr():
    ccl.spline_params.reload()
    assert eval(repr(ccl.spline_params)) == ccl.spline_params._bak
