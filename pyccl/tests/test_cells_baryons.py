"""Tests for propagating baryonic correction models into C_ells."""

from __future__ import annotations

import pytest

import numpy as np

import pyccl as ccl


# Base cosmology for non-HMCode baryon tests
BASE_COSMO = ccl.CosmologyVanillaLCDM()

# Simple redshift distribution
Z = np.linspace(0.0, 2.0, 200)
NZ = Z**2 * np.exp(-0.5 * ((Z - 1.0) / 0.3) ** 2)
BZ = np.ones_like(Z)  # simple constant bias for clustering

# Angular multipoles
ELLS = np.geomspace(10.0, 2000.0, 64)


def _make_tracers(cosmo: ccl.Cosmology):
    """Returns WL and NC tracers for a given cosmology."""
    tracer_wl = ccl.WeakLensingTracer(cosmo, dndz=(Z, NZ))
    tracer_gg = ccl.NumberCountsTracer(
        cosmo, has_rsd=False, dndz=(Z, NZ), bias=(Z, BZ)
    )
    return tracer_wl, tracer_gg


def _build_cosmo_from_pk(pk2d: ccl.Pk2D) -> ccl.CosmologyCalculator:
    """Builds a CosmologyCalculator using pk2d as the nonlinear P(k)."""
    a_arr, lk_arr, pk_arr = pk2d.get_spline_arrays()
    pk_nonlin = {
        "a": a_arr,
        "k": np.exp(lk_arr),
        "delta_matter:delta_matter": pk_arr,
    }

    cosmo_calc = ccl.CosmologyCalculator(
        Omega_c=BASE_COSMO["Omega_c"],
        Omega_b=BASE_COSMO["Omega_b"],
        h=BASE_COSMO["h"],
        n_s=BASE_COSMO["n_s"],
        sigma8=BASE_COSMO["sigma8"],
        pk_nonlin=pk_nonlin,
    )
    cosmo_calc.compute_growth()
    return cosmo_calc


@pytest.mark.parametrize(
    "baryon_model",
    [
        ccl.BaryonsSchneider15(log10Mc=14.0),
        ccl.BaryonsvanDaalen19(fbar=0.7, mass_def="500c"),
    ],
)
def test_cells_baryons_consistent_with_pk(baryon_model):
    """Tests that two routes to Cls with baryons agree for 3x2pt probes.

    Route 1: use pk_bar directly as p_of_k_a.
    Route 2: build a CosmologyCalculator that uses pk_bar as pk_nonlin.
    """
    BASE_COSMO.compute_nonlin_power()
    pk_dmo = BASE_COSMO.get_nonlin_power()

    # Apply baryons to get baryon-modified P(k)
    pk_bar = baryon_model.include_baryonic_effects(BASE_COSMO, pk_dmo)

    # Tracers for BASE_COSMO
    wl_base, gg_base = _make_tracers(BASE_COSMO)

    # Route 1: use pk_bar directly as p_of_k_a
    cl_bar_mm_direct = ccl.angular_cl(
        BASE_COSMO, wl_base, wl_base, ELLS, p_of_k_a=pk_bar
    )
    cl_bar_gg_direct = ccl.angular_cl(
        BASE_COSMO, gg_base, gg_base, ELLS, p_of_k_a=pk_bar
    )
    cl_bar_gm_direct = ccl.angular_cl(
        BASE_COSMO, gg_base, wl_base, ELLS, p_of_k_a=pk_bar
    )

    # Route 2: build a CosmologyCalculator that uses pk_bar as pk_nonlin
    cosmo_calc = _build_cosmo_from_pk(pk_bar)
    wl_calc, gg_calc = _make_tracers(cosmo_calc)

    cl_bar_mm_calc = ccl.angular_cl(cosmo_calc, wl_calc, wl_calc, ELLS)
    cl_bar_gg_calc = ccl.angular_cl(cosmo_calc, gg_calc, gg_calc, ELLS)
    cl_bar_gm_calc = ccl.angular_cl(cosmo_calc, gg_calc, wl_calc, ELLS)

    np.testing.assert_allclose(
        cl_bar_mm_calc, cl_bar_mm_direct, rtol=1e-4, atol=0.0)
    np.testing.assert_allclose(
        cl_bar_gg_calc, cl_bar_gg_direct, rtol=1e-4, atol=0.0)
    np.testing.assert_allclose(
        cl_bar_gm_calc, cl_bar_gm_direct, rtol=1e-4, atol=0.0)


@pytest.mark.parametrize(
    "baryon_model",
    [
        ccl.BaryonsSchneider15(log10Mc=14.0),
        ccl.BaryonsvanDaalen19(fbar=0.7, mass_def="500c"),
    ],
)
def test_cells_baryons_modify_cls(baryon_model):
    """Tests that baryonic effects modify LL, GG, and GL angular Cls."""
    BASE_COSMO.compute_nonlin_power()
    pk_dmo = BASE_COSMO.get_nonlin_power()
    pk_bar = baryon_model.include_baryonic_effects(BASE_COSMO, pk_dmo)

    wl_base, gg_base = _make_tracers(BASE_COSMO)

    cl_dmo_mm = ccl.angular_cl(
        BASE_COSMO, wl_base, wl_base, ELLS, p_of_k_a=pk_dmo)
    cl_bar_mm = ccl.angular_cl(
        BASE_COSMO, wl_base, wl_base, ELLS, p_of_k_a=pk_bar)

    cl_dmo_gg = ccl.angular_cl(
        BASE_COSMO, gg_base, gg_base, ELLS, p_of_k_a=pk_dmo)
    cl_bar_gg = ccl.angular_cl(
        BASE_COSMO, gg_base, gg_base, ELLS, p_of_k_a=pk_bar)

    cl_dmo_gm = ccl.angular_cl(
        BASE_COSMO, gg_base, wl_base, ELLS, p_of_k_a=pk_dmo)
    cl_bar_gm = ccl.angular_cl(
        BASE_COSMO, gg_base, wl_base, ELLS, p_of_k_a=pk_bar)

    for cl_dmo, cl_bar in [(cl_dmo_mm, cl_bar_mm),
                           (cl_dmo_gg, cl_bar_gg),
                           (cl_dmo_gm, cl_bar_gm)]:
        ratio = cl_bar / cl_dmo
        assert np.all(np.isfinite(ratio))
        assert np.any(np.abs(ratio - 1.0) > 1e-3)


def _make_ccl_mead2020_cosmo(logT_AGN: float = 7.93) -> ccl.Cosmology:
    """Creates a CCL Cosmology using CAMB with Mead2020+HMCode baryons."""
    extras = {
        "camb": {
            "halofit_version": "mead2020_feedback",
            "HMCode_logT_AGN": logT_AGN,
        }
    }
    return ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        m_nu=0.0,
        A_s=2.1e-9,
        n_s=0.97,
        transfer_function="boltzmann_camb",
        matter_power_spectrum="camb",
        extra_parameters=extras,
    )


def test_cells_mead20_cls_finite():
    """Tests that angular Cls are finite for 3x2pt probes."""
    cosmo = _make_ccl_mead2020_cosmo()
    z = np.linspace(0.0, 2.0, 100)
    nz = z**2 * np.exp(-0.5 * ((z - 1.0) / 0.3) ** 2)
    bz = np.ones_like(z)

    tracer_wl = ccl.WeakLensingTracer(cosmo, dndz=(z, nz))
    tracer_gg = ccl.NumberCountsTracer(
        cosmo, has_rsd=False, dndz=(z, nz), bias=(z, bz)
    )
    ells = np.geomspace(10.0, 1000.0, 32)

    cl_mm = ccl.angular_cl(cosmo, tracer_wl, tracer_wl, ells)
    cl_gg = ccl.angular_cl(cosmo, tracer_gg, tracer_gg, ells)
    cl_gm = ccl.angular_cl(cosmo, tracer_gg, tracer_wl, ells)

    assert np.all(np.isfinite(cl_mm))
    assert np.all(np.isfinite(cl_gg))
    assert np.all(np.isfinite(cl_gm))


def test_cells_mead20_cls_respond_to_logT_AGN():
    """Tests that changing logT_AGN changes mm, gg, and gm angular Cls."""
    cosmo_lo = _make_ccl_mead2020_cosmo(logT_AGN=7.0)
    cosmo_hi = _make_ccl_mead2020_cosmo(logT_AGN=8.3)

    z = np.linspace(0.0, 2.0, 100)
    nz = z**2 * np.exp(-0.5 * ((z - 1.0) / 0.3) ** 2)
    bz = np.ones_like(z)

    tracer_wl_lo = ccl.WeakLensingTracer(cosmo_lo, dndz=(z, nz))
    tracer_wl_hi = ccl.WeakLensingTracer(cosmo_hi, dndz=(z, nz))

    tracer_gg_lo = ccl.NumberCountsTracer(
        cosmo_lo, has_rsd=False, dndz=(z, nz), bias=(z, bz)
    )
    tracer_gg_hi = ccl.NumberCountsTracer(
        cosmo_hi, has_rsd=False, dndz=(z, nz), bias=(z, bz)
    )

    ells = np.geomspace(10.0, 1000.0, 32)

    cl_mm_lo = ccl.angular_cl(cosmo_lo, tracer_wl_lo, tracer_wl_lo, ells)
    cl_mm_hi = ccl.angular_cl(cosmo_hi, tracer_wl_hi, tracer_wl_hi, ells)

    cl_gg_lo = ccl.angular_cl(cosmo_lo, tracer_gg_lo, tracer_gg_lo, ells)
    cl_gg_hi = ccl.angular_cl(cosmo_hi, tracer_gg_hi, tracer_gg_hi, ells)

    cl_gm_lo = ccl.angular_cl(cosmo_lo, tracer_gg_lo, tracer_wl_lo, ells)
    cl_gm_hi = ccl.angular_cl(cosmo_hi, tracer_gg_hi, tracer_wl_hi, ells)

    for cl_lo, cl_hi in [(cl_mm_lo, cl_mm_hi),
                         (cl_gg_lo, cl_gg_hi),
                         (cl_gm_lo, cl_gm_hi)]:
        ratio = cl_hi / cl_lo
        assert np.all(np.isfinite(ratio))
        assert np.any(np.abs(ratio - 1.0) > 1e-3)
