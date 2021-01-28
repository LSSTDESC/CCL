from __future__ import print_function
import pickle
import tempfile

import pytest

import numpy as np
from numpy.testing import assert_raises, assert_, assert_no_warnings

import pyccl as ccl


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
