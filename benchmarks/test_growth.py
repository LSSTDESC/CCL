import numpy as np
import pytest

import pyccl as ccl
from pyccl.modified_gravity import MuSigmaMG


GROWTH_HIZ_TOLERANCE = 6.0e-6
GROWTH_TOLERANCE = 1e-4

# Set up the cosmological parameters to be used in each of the models
# Values that are the same for all 5 models
Omega_c = 0.25
Omega_b = 0.05
Neff = 0.
m_nu = 0.
h = 0.7
A_s = 2.1e-9
n_s = 0.96

# Values that are different for the different models
Omega_v_vals = np.array([0.7, 0.7, 0.7, 0.65, 0.75, 0.7, 0.7, 0.7, 0.7])
w0_vals = np.array([-1.0, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9, -0.9])
wa_vals = np.array([0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0])
mu0_vals = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, -0.1, 0.1, -0.1])
Sig0_vals = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1, -0.1, -0.1, 0.1])


def read_growth_lowz_benchmark_file():
    """
    Read the file containing growth factor benchmarks for the low redshifts.
    """
    # Load data from file
    dat = np.genfromtxt("benchmarks/data/growth_model1-5.txt").T
    assert (dat.shape == (6, 6))

    # Split into redshift column and growth(z) columns
    z = dat[0]
    gfac = dat[1:]
    return z, gfac


def read_growth_highz_benchmark_file():
    """
    Read the file containing growth factor benchmarks for the high redshifts.
    """
    # Load data from file
    dat = np.genfromtxt("benchmarks/data/growth_hiz_model1-3.txt").T
    assert (dat.shape == (4, 7))

    # Split into redshift column and growth(z) columns
    z = dat[0]
    gfac = dat[1:]
    return z, gfac


def read_growth_allz_benchmark_file():
    """
    Read the file containing growth factor benchmarks for
    the whole redshift range.
    """
    # Load data from file
    dat = np.genfromtxt("benchmarks/data/growth_cosmomad_allz.txt").T
    assert (dat.shape == (6, 10))

    # Split into redshift column and growth(z) columns
    z = dat[0]
    gfac = dat[1:]

    return z, gfac


# Set-up test data
z_lowz, gfac_lowz = read_growth_lowz_benchmark_file()
z_highz, gfac_highz = read_growth_highz_benchmark_file()
z_allz, gfac_allz = read_growth_allz_benchmark_file()


def compare_growth(
        z, gfac_bench, Omega_v, w0, wa, mu_0, sigma_0, high_tol=False):
    """
    Compare growth factor calculated by pyccl with the values in the benchmark
    file. This test only works if radiation is explicitly set to 0.
    """

    # Set Omega_K in a consistent way
    Omega_k = 1.0 - Omega_c - Omega_b - Omega_v

    cosmo = ccl.Cosmology(
        Omega_c=Omega_c, Omega_b=Omega_b, Neff=Neff, m_nu=m_nu,
        h=h, A_s=A_s, n_s=n_s, Omega_k=Omega_k, Omega_g=0,
        w0=w0, wa=wa,
        mg_parametrization=MuSigmaMG(mu_0=mu_0, sigma_0=sigma_0))

    # Calculate distance using pyccl
    a = 1. / (1. + z)
    gfac = ccl.growth_factor_unnorm(cosmo, a)

    # Compare to benchmark data
    if high_tol:
        assert np.allclose(gfac, gfac_bench,
                           atol=1e-12, rtol=GROWTH_HIZ_TOLERANCE)
    else:
        assert np.allclose(gfac, gfac_bench,
                           atol=1e-12, rtol=GROWTH_TOLERANCE)


@pytest.mark.parametrize('i', list(range(5)))
def test_growth_lowz_model(i):
    compare_growth(z_lowz, gfac_lowz[i], Omega_v_vals[i], w0_vals[i],
                   wa_vals[i], mu0_vals[i], Sig0_vals[i])


@pytest.mark.parametrize('i', list(range(3)))
def test_growth_highz_model(i):
    compare_growth(z_highz, gfac_highz[i], Omega_v_vals[i], w0_vals[i],
                   wa_vals[i], mu0_vals[i], Sig0_vals[i], high_tol=True)


# 0.01 < z < 1000 tests
@pytest.mark.parametrize('i', list(range(5)))
def test_growth_allz_model(i):
    compare_growth(z_allz, gfac_allz[i], Omega_v_vals[i], w0_vals[i],
                   wa_vals[i], mu0_vals[i], Sig0_vals[i])
