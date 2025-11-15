"""Unit tests for angular power spectra (Cells) calculations."""

import pytest

import numpy as np

import pyccl as ccl
from pyccl.modified_gravity import MuSigmaMG


ccl.gsl_params.LENSING_KERNEL_SPLINE_INTEGRATION = False
COSMO = ccl.Cosmology(
    Omega_c=0.27,
    Omega_b=0.045,
    h=0.67,
    sigma8=0.8,
    n_s=0.96,
    transfer_function="bbks",
    matter_power_spectrum="linear",
)
PKA = ccl.Pk2D.from_function(pkfunc=lambda k, a: np.log(a / k))
ZZ = np.linspace(0.0, 1.0, 200)
NN = np.exp(-(((ZZ - 0.5) / 0.1) ** 2))
LENS = ccl.WeakLensingTracer(COSMO, dndz=(ZZ, NN))
LENS_2 = ccl.WeakLensingTracer(COSMO, dndz=(ZZ, np.zeros(len(ZZ))))


@pytest.mark.parametrize("p_of_k_a", [ccl.DEFAULT_POWER_SPECTRUM, PKA, None])
def test_cells_smoke(p_of_k_a):
    """Tests that angular_cl runs without errors and returns finite values."""
    z = np.linspace(0.0, 1.0, 200)
    n = np.exp(-(((z - 0.5) / 0.1) ** 2))
    b = np.sqrt(1.0 + z)
    lens1 = ccl.WeakLensingTracer(COSMO, dndz=(z, n))
    lens2 = ccl.WeakLensingTracer(COSMO, dndz=(z, n), ia_bias=(z, n))
    nc1 = ccl.NumberCountsTracer(
        COSMO, has_rsd=False, dndz=(z, n), bias=(z, b)
    )
    nc2 = ccl.NumberCountsTracer(COSMO, has_rsd=True, dndz=(z, n), bias=(z, b))
    nc3 = ccl.NumberCountsTracer(
        COSMO, has_rsd=True, dndz=(z, n), bias=(z, b), mag_bias=(z, b)
    )
    cmbl = ccl.CMBLensingTracer(COSMO, z_source=1100.0)
    tracers = [lens1, lens2, nc1, nc2, nc3, cmbl]

    ell_scl = 4.0
    ell_int = 4
    ell_lst = [2, 3, 4, 5]
    ell_arr = np.arange(2, 5)
    ells = [ell_int, ell_scl, ell_lst, ell_arr]

    for i in range(len(tracers)):
        for j in range(i, len(tracers)):
            for ell in ells:
                corr = ccl.angular_cl(
                    COSMO, tracers[i], tracers[j], ell, p_of_k_a=p_of_k_a
                )
                assert np.all(np.isfinite(corr))
                assert np.shape(corr) == np.shape(ell)

                # reversing should be fine
                corr_rev = ccl.angular_cl(
                    COSMO, tracers[j], tracers[i], ell, p_of_k_a=p_of_k_a
                )
                assert np.allclose(corr, corr_rev)

    # Check invalid dndz
    with pytest.raises(ValueError):
        ccl.NumberCountsTracer(COSMO, has_rsd=False, dndz=z, bias=(z, b))
    with pytest.raises(ValueError):
        ccl.NumberCountsTracer(
            COSMO, has_rsd=False, dndz=(z, n, n), bias=(z, b)
        )
    with pytest.raises(ValueError):
        ccl.NumberCountsTracer(COSMO, has_rsd=False, dndz=(z,), bias=(z, b))
    with pytest.raises(ValueError):
        ccl.NumberCountsTracer(COSMO, has_rsd=False, dndz=(1, 2), bias=(z, b))
    with pytest.raises(ValueError):
        ccl.WeakLensingTracer(COSMO, dndz=z)
    with pytest.raises(ValueError):
        ccl.WeakLensingTracer(COSMO, dndz=(z, n, n))
    with pytest.raises(ValueError):
        ccl.WeakLensingTracer(COSMO, dndz=(z,))
    with pytest.raises(ValueError):
        ccl.WeakLensingTracer(COSMO, dndz=(1, 2))


@pytest.mark.parametrize("ells", [[3, 2, 1], [1, 3, 2], [2, 3, 1]])
def test_cells_raise_ell_reversed(ells):
    """Tests that angular_cl raises error for non-increasing ells."""
    with pytest.raises(ValueError):
        ccl.angular_cl(COSMO, LENS, LENS, ells)


def test_cells_raise_integ_method():
    """Tests that angular_cl raises error for invalid integration methods."""
    ells = [10, 11]
    with pytest.raises(ValueError):
        ccl.angular_cl(
            COSMO, LENS, LENS, ells, limber_integration_method="guad"
        )

    with pytest.raises(ValueError):
        ccl.angular_cl(
            COSMO, LENS, LENS_2, ells, limber_integration_method="quad"
        )


def test_cells_raise_nonlimber_methods():
    """Tests that angular_cl raises error for invalid non-Limber methods."""

    cosmo = ccl.CosmologyVanillaLCDM()
    z = np.linspace(0.01, 1.0, 5)
    nz = z**2 * np.exp(-z)
    bz = np.ones_like(z)
    tracer = ccl.NumberCountsTracer(
        cosmo,
        has_rsd=False,
        dndz=(z, nz),
        bias=(z, bz),
    )

    ells = [10, 11]

    # bogus non-Limber method name -> must raise
    with pytest.raises(ValueError, match="Non-Limber integration method"):
        ccl.angular_cl(
            cosmo,
            tracer,
            tracer,
            ells,
            non_limber_integration_method="bogus",
        )


def test_cells_raise_weird_pk():
    """Tests that angular_cl raises error for invalid power spectra."""
    ells = [10, 11]
    with pytest.raises(ValueError):
        ccl.angular_cl(COSMO, LENS, LENS, ells, p_of_k_a=lambda k, a: 10)


def test_fkem_chi_params():
    """Tests that angular_cl FKEM chi params work as intended."""
    # Redshift distribution
    z = np.linspace(0, 4.72, 60)
    nz = z**2*np.exp(-0.5*((z-1.5)/0.7)**2)

    # Galaxy bias
    bz = np.ones_like(z)

    # Power spectra
    ls = np.unique(np.geomspace(2, 2000, 128).astype(int)).astype(float)
    cosmo = ccl.CosmologyVanillaLCDM()
    tracer_gal = ccl.NumberCountsTracer(cosmo, has_rsd=False,
                                        dndz=(z, nz), bias=(z, bz))
    cl_gg = ccl.angular_cl(cosmo, tracer_gal, tracer_gal, ls,
                           ell_limber=-1)
    cl_ggn = ccl.angular_cl(cosmo, tracer_gal, tracer_gal, ls,
                            ell_limber=1000)
    cl_ggn_b = ccl.angular_cl(cosmo, tracer_gal, tracer_gal, ls,
                              ell_limber=1000, fkem_chi_min=1.0,
                              fkem_nchi=100)

    ell_good = ls > 100

    # Check that, above ell, the non-Limber calculation does not
    # agree with Limber when using the default FKEM sampling params
    assert not np.all(np.fabs(cl_ggn/cl_gg-1)[ell_good] < 0.01)
    # Check that it works with custom ones.
    assert np.all(np.fabs(cl_ggn_b/cl_gg-1)[ell_good] < 0.01)


def test_cells_mg():
    """Tests that angular_cl works with MG cosmologies."""
    # Check that if we feed the non-linear matter power spectrum from a MG
    # cosmology into a Calculator and get Cells using MG tracers, we get the
    # same results.

    # set up a MG cosmology
    cosmo_mg = ccl.CosmologyVanillaLCDM(mg_parametrization=MuSigmaMG(
                                        mu_0=0.5, sigma_0=0.5),
                                        transfer_function="bbks",
                                        matter_power_spectrum="linear")
    cosmo_mg.compute_nonlin_power()
    pk2d = cosmo_mg.get_nonlin_power()

    # copy it into a calculator
    a, lk, pk = pk2d.get_spline_arrays()
    pk_nonlin = {"a": a, "k": np.exp(lk), "delta_matter:delta_matter": pk}
    cosmo_calc = ccl.CosmologyCalculator(
        Omega_c=0.25, Omega_b=0.05, h=0.67, n_s=0.96, sigma8=0.81,
        mg_parametrization=MuSigmaMG(mu_0=0.5, sigma_0=0.5),
        pk_nonlin=pk_nonlin)

    # get the Cells
    ell = np.geomspace(2, 2000, 128)
    tr_mg = ccl.CMBLensingTracer(cosmo_mg, z_source=1100.0)
    tr_calc = ccl.CMBLensingTracer(cosmo_calc, z_source=1100.0)

    cl0 = ccl.angular_cl(cosmo_mg, tr_mg, tr_mg, ell)
    cosmo_calc.compute_growth()
    cl1 = ccl.angular_cl(cosmo_calc, tr_calc, tr_calc, ell)
    assert np.all(np.fabs(1 - cl1 / cl0) < 1e-10)


ccl.gsl_params.reload()  # reset to the default parameters
