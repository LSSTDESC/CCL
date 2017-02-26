import numpy as np
from numpy.testing import assert_raises, assert_warns, assert_no_warnings, \
                          run_module_suite
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
                                       A_s=2.1e-9, n_s=0.96, Omega_n=0.05)
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
    assert_raises(ValueError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96,
                                              Omega_n=None)

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
                                       zarr_mgrowth=None, dfarr_mgrowth=None)
    assert_no_warnings(ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96,
                                       zarr_mgrowth=None)
    assert_no_warnings(ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96,
                                       dfarr_mgrowth=None)
    assert_no_warnings(ccl.Parameters, Omega_c=0.25, Omega_b=0.05, h=0.7,
                                       A_s=2.1e-9, n_s=0.96,
                                       zarr_mgrowth=None, dfarr_mgrowth=None)

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
                                       zarr_mgrowth=zarr, dfarr_mgrowth=dfarr)
    assert_no_warnings(ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96,
                                       dfarr_mgrowth=dfarr, zarr_mgrowth=zarr)
    assert_no_warnings(ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, w0=-1.,
                                       zarr_mgrowth=zarr, dfarr_mgrowth=dfarr)
    assert_no_warnings(ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96, w0=-1.,
                                       zarr_mgrowth=[0., 0.1, 0.2],
                                       dfarr_mgrowth=[0.1, 0.1, 0.1])

    # Invalid constructions
    assert_warns(UserWarning, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96,
                                              zarr_mgrowth=zarr)
    assert_warns(UserWarning, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96,
                                              dfarr_mgrowth=dfarr)
    assert_warns(UserWarning, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96,
                                              zarr_mgrowth=None,
                                              dfarr_mgrowth=dfarr)
    assert_raises(AssertionError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96,
                                                  zarr_mgrowth=zarr,
                                                  dfarr_mgrowth=0.1)
    assert_raises(AssertionError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96,
                                                  zarr_mgrowth=zarr,
                                                  dfarr_mgrowth=f_func)

    # Mis-matched array sizes and dimensionality
    assert_raises(AssertionError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96,
                                                  zarr_mgrowth=zarr,
                                                  dfarr_mgrowth=dfarr[1:])
    assert_raises(AssertionError, ccl.Parameters, 0.25, 0.05, 0.7, 2.1e-9, 0.96,
                                                  zarr_mgrowth=zarr,
                                 dfarr_mgrowth=np.column_stack((dfarr, dfarr)) )


if __name__ == '__main__':
    run_module_suite()
