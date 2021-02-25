import copy
import numpy as np
from numpy.testing import assert_raises
import pyccl
from pyccl import ccllib
from pyccl import CCLError

PYCOSMO = pyccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67,
                          sigma8=0.8, n_s=0.96,
                          transfer_function='bbks')
PYCOSMO.compute_nonlin_power()
COSMO = PYCOSMO.cosmo


def test_swig_tracer():
    z = np.linspace(0, 1, 10)
    bias_ia = z * 2
    dNdz = z * 8

    assert_raises(
        ValueError,
        pyccl.WeakLensingTracer,
        PYCOSMO,
        dndz=(z, dNdz[0:2]),
        ia_bias=(z, bias_ia))

    assert_raises(
        ValueError,
        pyccl.WeakLensingTracer,
        PYCOSMO,
        dndz=(z, dNdz),
        ia_bias=(z, bias_ia[0:2]))


def test_swig_background():
    status = 0
    for func in [
            ccllib.growth_factor_vec,
            ccllib.growth_factor_unnorm_vec,
            ccllib.growth_rate_vec,
            ccllib.comoving_radial_distance_vec,
            ccllib.comoving_angular_distance_vec,
            ccllib.h_over_h0_vec,
            ccllib.luminosity_distance_vec,
            ccllib.distance_modulus_vec]:
        assert_raises(
            CCLError,
            func,
            COSMO,
            [0.0, 1.0],
            1,
            status)

    assert_raises(
        CCLError,
        ccllib.omega_x_vec,
        COSMO,
        0,
        [0.0, 1.0],
        1,
        status)

    assert_raises(
        CCLError,
        ccllib.rho_x_vec,
        COSMO,
        0,
        0,
        [0.0, 1.0],
        1,
        status)

    assert_raises(
        CCLError,
        ccllib.scale_factor_of_chi_vec,
        COSMO,
        [0.0, 1.0],
        1,
        status)


def test_swig_cls():
    status = 0
    base_args = [
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0],
        [0.0],
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0, 3.0],
        0, 1, 0, 0, 0, 0, 2,
        status]
    args = copy.deepcopy(base_args)
    args[1] = [1.0] * 8
    assert_raises(
        CCLError,
        ccllib.cl_tracer_t_new_wrapper,
        COSMO, 0, 0,
        *args)

    assert_raises(
        CCLError,
        ccllib.angular_cl_vec,
        COSMO,
        None, None, None,
        1, 0, pyccl.pyutils.integ_types['spline'],
        "none",
        status)


def test_swig_core():
    status = 0
    assert_raises(
        CCLError,
        ccllib.parameters_create_nu_vec,
        0.25, 0.05, 0.0, 3.0, -1.0, 0.0, 0.7, 2e-9, 0.95, 1, 0.0, 0.0,
        0.0, 0.0, 1.0, 1.0, 0.0, [1.0, 2.0],
        [0.0, 0.3, 0.5],
        [0.02, 0.01, 0.2],
        status)


def test_swig_correlation():
    status = 0
    assert_raises(
        CCLError,
        ccllib.correlation_vec,
        COSMO,
        [1, 2],
        [1, 2, 3],
        [0, 1, 2, 3],
        0, 0,
        4,
        status)

    assert_raises(
        CCLError,
        ccllib.correlation_vec,
        COSMO,
        [1, 2, 3],
        [1, 2, 3],
        [0, 1, 2, 3],
        0, 0,
        5,
        status)

    assert_raises(
        CCLError,
        ccllib.correlation_3d_vec,
        COSMO,
        PYCOSMO._pk_nl['delta_matter:delta_matter'].psp,
        1.0,
        [1, 2, 3],
        4,
        status)


def test_swig_neurtinos():
    status = 0
    assert_raises(
        CCLError,
        ccllib.Omeganuh2_vec,
        3, 2.7,
        [0.0, 1.0],
        [0.05, 0.1, 0.2],
        4,
        status)

    assert_raises(
        CCLError,
        ccllib.Omeganuh2_vec,
        3, 2.7,
        [0.0, 1.0],
        [0.1, 0.2],
        2,
        status)


def test_swig_power():
    status = 0

    for func in [ccllib.sigmaR_vec,
                 ccllib.sigmaV_vec]:
        assert_raises(
            CCLError,
            func,
            COSMO,
            None,
            1.0,
            [1.0, 2.0],
            3,
            status)

    assert_raises(
        CCLError,
        ccllib.kNL_vec,
        COSMO,
        None,
        [0.5, 1.0],
        3,
        status)


def test_swig_haloprofile():
    status = 0
    assert_raises(CCLError,
                  ccllib.einasto_norm,
                  [0.1, 1.0],
                  [0.1, 1.0],
                  [0.1, 1.0],
                  4, status)
    assert_raises(CCLError,
                  ccllib.hernquist_norm,
                  [0.1, 1.0],
                  [0.1, 1.0],
                  4, status)
