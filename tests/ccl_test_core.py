from __future__ import print_function
import pickle
import tempfile
import numpy as np
from numpy.testing import assert_raises, assert_warns, assert_no_warnings, \
                          assert_, run_module_suite, assert_almost_equal
import pyccl as ccl


def test_parameters_valid_input():
    """
    Check that valid parameter arguments are accepted.
    """
    assert_no_warnings(ccl.Cosmology, Omega_c=0.25, Omega_b=0.05, h=0.7,
                                       A_s=2.1e-9, n_s=0.96)
    assert_no_warnings(ccl.Cosmology, Omega_c=0.25, Omega_b=0.05, h=0.7,
                                       A_s=2.1e-9, n_s=0.96, Omega_k=0.05)
    assert_no_warnings(ccl.Cosmology, Omega_c=0.25, Omega_b=0.05, h=0.7,
                                       A_s=2.1e-9, n_s=0.96, Neff=2.046)
    assert_no_warnings(ccl.Cosmology, Omega_c=0.25, Omega_b=0.05, h=0.7,
                                       A_s=2.1e-9, n_s=0.96, Neff=3.046, m_nu=0.06)

    assert_no_warnings(ccl.Cosmology, Omega_c=0.25, Omega_b=0.05, h=0.7,
                                       A_s=2.1e-9, n_s=0.96, w0=-0.9)
    assert_no_warnings(ccl.Cosmology, Omega_c=0.25, Omega_b=0.05, h=0.7,
                                       A_s=2.1e-9, n_s=0.96, w0=-0.9, wa=0.1)

    # Check that kwarg order doesn't matter
    assert_no_warnings(ccl.Cosmology, h=0.7, Omega_c=0.25, Omega_b=0.05,
                                       A_s=2.1e-9, n_s=0.96)

def test_parameters_missing():
    """
    Check that errors are raised when compulsory parameters are missing, but
    not when non-compulsory ones are.
    """
    assert_raises(ValueError, ccl.Cosmology, Omega_c=0.25)

    # Check that a single missing compulsory parameter is noticed
    assert_raises(ValueError, ccl.Cosmology, Omega_c=0.25, Omega_b=0.05,
                                              h=0.7, A_s=2.1e-9)
    assert_raises(ValueError, ccl.Cosmology, Omega_c=0.25, Omega_b=0.05,
                                              h=0.7, n_s=0.96)
    assert_raises(ValueError, ccl.Cosmology, Omega_c=0.25, Omega_b=0.05,
                                              A_s=2.1e-9, n_s=0.96)
    assert_raises(ValueError, ccl.Cosmology, Omega_c=0.25,
                                              h=0.7, A_s=2.1e-9, n_s=0.96)
    assert_raises(ValueError, ccl.Cosmology, Omega_b=0.05,
                                              h=0.7, A_s=2.1e-9, n_s=0.96)

    # Make sure that compulsory parameters are compulsory
    assert_raises(
        ValueError, ccl.Cosmology,
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        Omega_k=None)
    assert_raises(
            ValueError, ccl.Cosmology,
            Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
            w0=None)
    assert_raises(
            ValueError, ccl.Cosmology,
            Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
            wa=None)

    # Check that sigma8 vs A_s is handled ok.
    assert_raises(
        ValueError, ccl.Cosmology,
        Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.8,
        A_s=2.1e-9, sigma8=0.7)
    assert_raises(
        ValueError, ccl.Cosmology,
        Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.8)

    # Make sure that optional parameters are optional
    assert_no_warnings(
        ccl.Cosmology,
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        z_mg=None, df_mg=None)
    assert_no_warnings(
        ccl.Cosmology,
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        z_mg=None)
    assert_no_warnings(
        ccl.Cosmology,
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        df_mg=None)


def test_parameters_set():
    """
    Check that a Cosmology object doesn't let parameters be set.
    """
    params = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9,
                            n_s=0.96)

    # Check that values of sigma8 and A_s won't be misinterpreted by the C code
    assert_raises(ValueError, ccl.Cosmology, Omega_c=0.25, Omega_b=0.05,
                                              h=0.7, A_s=2e-5, n_s=0.96)
    assert_raises(ValueError, ccl.Cosmology, Omega_c=0.25, Omega_b=0.05,
                                              h=0.7, sigma8=9e-6, n_s=0.96)

    # Check that error is raised when unrecognized parameter requested
    assert_raises(KeyError, lambda: params['wibble'])


def test_parameters_mgrowth():
    """
    Check that valid modified growth inputs are allowed, and invalid ones are
    rejected.
    """
    zarr = np.linspace(0., 1., 15)
    dfarr = 0.1 * np.ones(15)
    f_func = lambda z: 0.1 * z

    # Valid constructions
    for omega_g in [None, 0.0, 0.1]:
        assert_no_warnings(
            ccl.Cosmology,
            Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
            z_mg=zarr, df_mg=dfarr, Omega_g=omega_g)
        assert_no_warnings(
            ccl.Cosmology,
            Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
            z_mg=[0., 0.1, 0.2],
            df_mg=[0.1, 0.1, 0.1], Omega_g=omega_g)

        # Invalid constructions
        assert_raises(
            ValueError, ccl.Cosmology,
            Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
            z_mg=zarr, Omega_g=omega_g)
        assert_raises(
            ValueError, ccl.Cosmology,
            Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
            df_mg=dfarr, Omega_g=omega_g)
        assert_raises(
            ValueError, ccl.Cosmology,
            Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
            z_mg=None,
            df_mg=dfarr, Omega_g=omega_g)
        assert_raises(
            ValueError, ccl.Cosmology,
            Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
            z_mg=zarr,
            df_mg=0.1, Omega_g=omega_g)
        assert_raises(
            ValueError, ccl.Cosmology,
            Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
            z_mg=zarr,
            df_mg=f_func, Omega_g=omega_g)

        # Mis-matched array sizes and dimensionality
        assert_raises(
            ValueError, ccl.Cosmology,
            Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
            z_mg=zarr,
            df_mg=dfarr[1:], Omega_g=omega_g)
        assert_raises(
            ValueError, ccl.Cosmology,
            Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
            z_mg=zarr,
            df_mg=np.column_stack((dfarr, dfarr)), Omega_g=omega_g)


def test_parameters_read_write():
    """Check that Cosmology objects can be read and written"""
    params = ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        m_nu=[0.02, 0.1, 0.05], mnu_type='list',
        z_mg=[0.0, 1.0], df_mg=[0.01, 0.0])

    # Make a temporary file name
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        temp_file_name = tmpfile.name

    # Write out and then eead in the parameters from that file
    params.write_yaml(temp_file_name)
    params2 = ccl.Cosmology.read_yaml(temp_file_name)

    # Check the read-in params are equal to the written out ones
    assert_almost_equal(params['Omega_c'], params2['Omega_c'])
    assert_almost_equal(params['Neff'], params2['Neff'])
    assert_almost_equal(params['sum_nu_masses'], params2['sum_nu_masses'])

    # Now make a file that will be deleted so it does not exist
    # and check the right error is raise
    with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
        temp_file_name = tmpfile.name

    assert_raises(IOError, ccl.Cosmology.read_yaml, filename=temp_file_name)
    assert_raises(
        IOError,
        params.read_yaml,
        filename=temp_file_name+"/nonexistent_directory/params.yml")


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
    assert_(cosmo.has_distances() is False)
    assert_(cosmo.has_growth() is False)
    assert_(cosmo.has_power() is False)
    assert_(cosmo.has_sigma() is False)

    # Check that quantities can be precomputed
    assert_no_warnings(cosmo.compute_distances)
    assert_no_warnings(cosmo.compute_growth)
    assert_no_warnings(cosmo.compute_power)
    assert_(cosmo.has_distances() is True)
    assert_(cosmo.has_growth() is True)
    assert_(cosmo.has_power() is True)


def test_cosmology_pickles():
    """Check that a Cosmology object pickles."""
    cosmo = ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        m_nu=[0.02, 0.1, 0.05], mnu_type='list',
        z_mg=[0.0, 1.0], df_mg=[0.01, 0.0])

    with tempfile.TemporaryFile() as fp:
        pickle.dump(cosmo, fp)

        fp.seek(0)
        cosmo2 = pickle.load(fp)

    assert_(
        ccl.comoving_radial_distance(cosmo, 0.5) ==
        ccl.comoving_radial_distance(cosmo2, 0.5))


def test_cosmology_repr():
    """Check that we can make a Cosmology object from its repr."""
    import pyccl  # noqa: F401

    cosmo = ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        m_nu=[0.02, 0.1, 0.05], mnu_type='list',
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
        m_nu=np.array([0.02, 0.1, 0.05]), mnu_type='list',
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
    """Check that using a Cosmology object in a context manager frees C resources properly."""
    with ccl.Cosmology(
            Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
            m_nu=np.array([0.02, 0.1, 0.05]), mnu_type='list',
            z_mg=np.array([0.0, 1.0]), df_mg=np.array([0.01, 0.0])) as cosmo:
        # make sure it works
        assert not cosmo.has_distances()
        ccl.comoving_radial_distance(cosmo, 0.5)
        assert cosmo.has_distances()

    # make sure it does not!
    assert_(not hasattr(cosmo, "cosmo"))
    assert_(not hasattr(cosmo, "_params"))
    assert_raises(AttributeError, cosmo.has_growth)
    
def test_cosmology_neutrinos():
    """ Make sure an error is raised if inconsistent neutrino options /
    parameters are passed."""
	
    assert_raises(
        ValueError, ccl.Cosmology,
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        m_nu=0.05)
        
    assert_raises(
        ValueError, ccl.Cosmology,
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        m_nu=0.05, mnu_type='sum')
        
    assert_raises(
        ValueError, ccl.Cosmology,
        Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, n_s=0.96,
        m_nu=0.08, mnu_type='sum_inverted')     


if __name__ == '__main__':
    run_module_suite()
