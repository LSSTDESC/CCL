from numpy.testing import assert_raises, run_module_suite
import pyccl
from pyccl import ccllib
from pyccl import CCLError

COSMO = pyccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=1e-10, n_s=0.96).cosmo


def test_background():
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


def test_cls():
    assert False


def test_core():
    status = 0
    assert_raises(
        CCLError,
        ccllib.parameters_create_nu_vec,
        0.25, 0.05, 0.0, 3.0, -1.0, 0.0, 0.7, 2e-9, 0.95, 1, 0.0, 0.0,
        [1.0, 2.0],
        [0.0, 0.3, 0.5],
        0,
        [0.02, 0.01, 0.2],
        status)


def test_correlation():
    assert False


def test_neurtinos():
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


def test_power():
    status = 0
    for func in [ccllib.linear_matter_power_vec,
                 ccllib.nonlin_matter_power_vec]:
        assert_raises(
            CCLError,
            func,
            COSMO,
            1.0,
            [1.0, 2.0],
            3,
            status)

    for func in [ccllib.sigmaR_vec,
                 ccllib.sigmaV_vec]:
        assert_raises(
            CCLError,
            func,
            COSMO,
            1.0,
            [1.0, 2.0],
            3,
            status)


if __name__ == '__main__':
    run_module_suite()
