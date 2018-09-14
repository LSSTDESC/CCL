from __future__ import print_function
import numpy as np
from numpy.testing import assert_raises, assert_warns, assert_no_warnings, \
                          assert_, run_module_suite, assert_almost_equal
import pyccl as ccl


def test_parameters_valid_input():
    """
    Check that valid parameter arguments are accepted.
    """
    assert_no_warnings(ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96)
    assert_no_warnings(ccl.Parameters, Omega_c=0.25, Omega_b=0.05, h=0.7, 
                                       A_s=2.1e-9, n_s=0.96)
    assert_no_warnings(ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, w0=-0.9)
    assert_no_warnings(ccl.Parameters, Omega_c=0.25, Omega_b=0.05, h=0.7, 
                                       A_s=2.1e-9, n_s=0.96, Omega_k=0.05)
    assert_no_warnings(ccl.Parameters, Omega_c=0.25, Omega_b=0.05, h=0.7, 
                                       A_s=2.1e-9, n_s=0.96, Neff=2.046)
    assert_no_warnings(ccl.Parameters, Omega_c=0.25, Omega_b=0.05, h=0.7, 
                                       A_s=2.1e-9, n_s=0.96, Neff=3.046, m_nu=0.06)                                   

    assert_no_warnings(ccl.Parameters, Omega_c=0.25, Omega_b=0.05, h=0.7, 
                                       A_s=2.1e-9, n_s=0.96, w0=-0.9)
    assert_no_warnings(ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                       w0=-0.9, wa=0.1)
    assert_no_warnings(ccl.Parameters, Omega_c=0.25, Omega_b=0.05, h=0.7, 
                                       A_s=2.1e-9, n_s=0.96, w0=-0.9, wa=0.1)
    
    # Check that kwarg order doesn't matter
    assert_no_warnings(ccl.Parameters, h=0.7, Omega_c=0.25, Omega_b=0.05,
                                       A_s=2.1e-9, n_s=0.96)
    
def test_parameters_missing():
    """
    Check that errors are raised when compulsory parameters are missing, but 
    not when non-compulsory ones are.
    """
    # Make sure that compulsory parameters are compulsory
    assert_raises(ValueError, ccl.Parameters, 0.25)
    assert_raises(ValueError, ccl.Parameters, 0.25, 0.05)
    assert_raises(ValueError, ccl.Parameters, 0.25, 0.05, 0.7)
    assert_raises(ValueError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9)
    assert_raises(ValueError, ccl.Parameters, Omega_c=0.25)
    assert_raises(ValueError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                              Omega_k=None)
    assert_raises(ValueError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                              w0=None)
    assert_raises(ValueError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                              wa=None)                                                                                 

    # Check that a single missing compulsory parameter is noticed
    assert_raises(ValueError, ccl.Parameters, Omega_c=0.25, Omega_b=0.05, 
                                              h=0.7, A_s=2.1e-9)
    assert_raises(ValueError, ccl.Parameters, Omega_c=0.25, Omega_b=0.05, 
                                              h=0.7, n_s=0.96)
    assert_raises(ValueError, ccl.Parameters, Omega_c=0.25, Omega_b=0.05, 
                                              A_s=2.1e-9, n_s=0.96)
    assert_raises(ValueError, ccl.Parameters, Omega_c=0.25, 
                                              h=0.7, A_s=2.1e-9, n_s=0.96)
    assert_raises(ValueError, ccl.Parameters, Omega_b=0.05, 
                                              h=0.7, A_s=2.1e-9, n_s=0.96)
    
    # Make sure that optional parameters are optional
    assert_no_warnings(ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                       z_mg=None, df_mg=None)
    assert_no_warnings(ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                       z_mg=None)
    assert_no_warnings(ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                       df_mg=None)
    assert_no_warnings(ccl.Parameters, Omega_c=0.25, Omega_b=0.05, h=0.7, 
                                       A_s=2.1e-9, n_s=0.96, 
                                       z_mg=None, df_mg=None)
    
    # Check that Cosmology() object can instantiate a Parameters() object itself
    assert_no_warnings(ccl.Cosmology, Omega_c=0.25, Omega_b=0.05, h=0.7, 
                                      A_s=2.1e-9, n_s=0.96)
    
def test_parameters_spelling():
    """
    Check for misspelled parameters arguments.
    """
    # Mis-spelled or incorrectly-named compulsory parameters
    assert_raises(TypeError, ccl.Parameters, Omega_m=0.3, Omega_b=0.05, 
                                             h=0.7, A_s=2.1e-9, n_s=0.96)
    assert_raises(TypeError, ccl.Parameters, Omega_c=0.25, Omega_b=0.05, 
                                             Omega_m=0.3,
                                             h=0.7, A_s=2.1e-9, n_s=0.96)
    assert_raises(TypeError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                             w_0=-0.9)
    assert_raises(TypeError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                             w_a=-0.9)
    assert_raises(TypeError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                             omega_k=0.05)
    assert_raises(TypeError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                             Omega_K=0.05)
    assert_raises(TypeError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                             omega_n=0.05)
    assert_raises(TypeError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                             Omega_N=0.05)
    
    # Mis-spelled optional parameters
    assert_raises(TypeError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                             nonsense=0.)
    assert_raises(TypeError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                             zarrmgrowth=None)
    assert_raises(TypeError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                             dfarrmgrowth=None)

def test_parameters_set():
    """
    Check that Parameters object allows parameters to be set in a sensible way.
    """
    params = ccl.Parameters(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, 
                            n_s=0.96)
    
    # Check that values of sigma8 and A_s won't be misinterpreted by the C code
    assert_raises(ValueError, ccl.Parameters, Omega_c=0.25, Omega_b=0.05, 
                                              h=0.7, A_s=2e-5, n_s=0.96)
    assert_raises(ValueError, ccl.Parameters, Omega_c=0.25, Omega_b=0.05, 
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
    assert_no_warnings(ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                       z_mg=zarr, df_mg=dfarr)
    assert_no_warnings(ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                       df_mg=dfarr, z_mg=zarr)
    assert_no_warnings(ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, w0=-1.,
                                       z_mg=zarr, df_mg=dfarr)
    assert_no_warnings(ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, w0=-1.,
                                       z_mg=[0., 0.1, 0.2], 
                                       df_mg=[0.1, 0.1, 0.1])
    
    # Invalid constructions
    assert_raises(ValueError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                             z_mg=zarr)
    assert_raises(ValueError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                             df_mg=dfarr)
    assert_raises(ValueError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                             z_mg=None,
                                             df_mg=dfarr)
    assert_raises(AssertionError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                                  z_mg=zarr,
                                                  df_mg=0.1)
    assert_raises(AssertionError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                                  z_mg=zarr,
                                                  df_mg=f_func)
    
    # Mis-matched array sizes and dimensionality
    assert_raises(AssertionError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                                  z_mg=zarr,
                                                  df_mg=dfarr[1:])
    assert_raises(AssertionError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, 
                                                  z_mg=zarr,
                                 df_mg=np.column_stack((dfarr, dfarr)) )

def test_parameters_read_write():
    """Check that Parameters objects can be read and written"""
    import tempfile
    params = ccl.Parameters(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, 
                            n_s=0.96)

    # Make a temporary file name
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        temp_file_name = tmpfile.name

    # Write out and then eead in the parameters from that file
    params.write_yaml(temp_file_name)
    params2 = ccl.Parameters.read_yaml(temp_file_name)

    # Check the read-in params are equal to the written out ones
    assert_almost_equal(params['Omega_c'], params2['Omega_c'])
    assert_almost_equal(params['Neff'], params2['Neff'])
    assert_almost_equal(params['sum_nu_masses'], params2['sum_nu_masses'])

    # Now make a file that will be deleted so it does not exist
    # and check the right error is raise
    with tempfile.NamedTemporaryFile(delete=True) as tmpfile:
        temp_file_name = tmpfile.name

    assert_raises(IOError, ccl.Parameters.read_yaml, filename=temp_file_name)




def test_cosmology_init():
    """
    Check that Cosmology objects can only be constructed in a valid way.
    """
    # Create test cosmology object
    params = ccl.Parameters(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9, 
                            n_s=0.96)
    
    # Make sure error raised if incorrect type of Parameters object passed
    assert_raises(TypeError, ccl.Cosmology, params=params.parameters)
    assert_raises(TypeError, ccl.Cosmology, params="x")
    
    # Make sure error raised if wrong config type passed
    assert_raises(TypeError, ccl.Cosmology, params=params, config="string")
    
    # Make sure error raised if invalid transfer/power spectrum etc. type passed
    assert_raises(KeyError, ccl.Cosmology, params=params, 
                  matter_power_spectrum='x')
    assert_raises(KeyError, ccl.Cosmology, params=params, 
                  transfer_function='x')
    assert_raises(KeyError, ccl.Cosmology, params=params, 
                  baryons_power_spectrum='x')
    assert_raises(KeyError, ccl.Cosmology, params=params, 
                  mass_function='x')
    assert_raises(KeyError, ccl.Cosmology, params=params, 
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


if __name__ == '__main__':
    run_module_suite()
