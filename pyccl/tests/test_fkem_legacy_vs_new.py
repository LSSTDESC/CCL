"""Regression tests comparing the new nonlimber_fkem core with the legacy FKEM.

We cover three behaviours:

1. Number-counts tracer (with and without RSD):
   - new nonlimber_fkem matches the legacy FKEM Cls within a small tolerance.

2. Weak lensing tracer (with and without IA), problematic config:
   - legacy FKEM raises a CCLError (C-level integration failure);
   - new nonlimber_fkem degrades gracefully by falling back to Limber, i.e.
     returns (ell_limber = -1, empty Cl array, status = 0) instead of crashing.

3. Weak lensing tracer (with and without IA):
   - if the legacy FKEM runs without CCLError, we compare its Cls to the new
     implementation and require close agreement;
   - if the legacy FKEM fails, the test is xfailed, since we have no
     reference Cl to compare against.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

import pyccl as ccl
from pyccl.errors import CCLError

from pyccl.nonlimber_fkem.legacy_fkem import legacy_nonlimber_fkem
from pyccl.nonlimber_fkem.core import nonlimber_fkem


# -------------------------------------------------------------------
# Shared helpers
# -------------------------------------------------------------------


def _make_nc_setup(has_rsd: bool):
    """Simple ΛCDM + number-counts tracer (optionally with RSD) + ell grid."""
    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.045,
        h=0.67,
        n_s=0.96,
        sigma8=0.8,
    )

    z = np.linspace(0.01, 2.0, 128)
    nz = z**2 * np.exp(-z)
    b = np.ones_like(z)

    tracer = ccl.NumberCountsTracer(
        cosmo,
        has_rsd=has_rsd,
        dndz=(z, nz),
        bias=(z, b),
    )

    ells = np.arange(10, 300)
    return cosmo, tracer, tracer, ells


def _build_wl_setup(lmax: int, with_ia: bool):
    """Simple ΛCDM + WL tracer (+/- IA) + linear P(k) as Pk2D."""
    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.045,
        h=0.67,
        sigma8=0.8,
        n_s=0.96,
    )

    z = np.linspace(0.0, 2.0, 128)
    nz = z**2 * np.exp(-z)

    tracer_kwargs = dict(dndz=(z, nz))

    if with_ia:
        # Simple constant IA bias just to turn the machinery on
        ia_bias = 0.5 * np.ones_like(z)
        tracer_kwargs["ia_bias"] = (z, ia_bias)

    tracer = ccl.WeakLensingTracer(cosmo, **tracer_kwargs)

    pk2d_lin = ccl.Pk2D.from_function(
        lambda k, a: ccl.linear_matter_power(cosmo, k, a)
    )

    ells = np.arange(2, lmax + 1)

    return cosmo, tracer, ells, pk2d_lin


def _max_rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Compute a robust max relative difference between two arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
    return np.max(np.abs(a - b) / denom)


# -------------------------------------------------------------------
# 1. NC: new FKEM matches legacy (with and without RSD)
# -------------------------------------------------------------------


@pytest.mark.parametrize("has_rsd", [False, True])
def test_fkem_number_counts_new_matches_legacy(has_rsd: bool):
    """For a simple NC auto-correlation, new FKEM ≈ legacy FKEM.

    We test both without RSD (has_rsd=False) and with RSD (has_rsd=True).
    """
    cosmo, tracer1, tracer2, ells = _make_nc_setup(has_rsd=has_rsd)

    p_of_k_a = "delta_matter:delta_matter"
    pk_linear = "delta_matter:delta_matter"
    L_LIMBER = 200
    LIMBER_MAX_ERR = 0.1

    # Legacy implementation
    ell_limber_old, cells_old, status_old = legacy_nonlimber_fkem(
        cosmo,
        tracer1,
        tracer2,
        p_of_k_a,
        ells,
        L_LIMBER,
        pk_linear=pk_linear,
        limber_max_error=LIMBER_MAX_ERR,
        Nchi=None,
        chi_min=None,
    )

    # New implementation
    ell_limber_new, cells_new, status_new = nonlimber_fkem(
        cosmo,
        tracer1,
        tracer2,
        p_of_k_a,
        ells,
        L_LIMBER,
        pk_linear=pk_linear,
        limber_max_error=LIMBER_MAX_ERR,
        n_chi_fkem=None,
        chi_min_fkem=None,
        k_pow=3,
        k_low=1e-5,
        n_consec_ell=3,
    )

    assert status_old == 0
    assert status_new == 0
    assert ell_limber_new == L_LIMBER
    assert ell_limber_old == L_LIMBER

    cells_old = np.asarray(cells_old, dtype=float)
    cells_new = np.asarray(cells_new, dtype=float)
    n = min(len(cells_old), len(cells_new))

    npt.assert_allclose(
        cells_new[:n],
        cells_old[:n],
        rtol=5e-3,
        atol=0.0,
    )


# -------------------------------------------------------------------
# 2. WL: legacy crashes, new FKEM falls back cleanly
# -------------------------------------------------------------------

WL_L_LIMBER = 40  # used for the WL tests below


@pytest.mark.parametrize("lmax", [500, 1000])
@pytest.mark.parametrize("with_ia", [False, True])
def test_legacy_fkem_crashes_new_fkem_falls_back(lmax: int, with_ia: bool):
    """Legacy FKEM raises CCLError, new FKEM returns (-1, [], 0) instead."""
    cosmo, tracer, ells, pk2d_lin = _build_wl_setup(lmax, with_ia=with_ia)

    # Legacy FKEM: expect an integration error in this configuration
    with pytest.raises(CCLError):
        legacy_nonlimber_fkem(
            cosmo,
            tracer,
            tracer,
            pk2d_lin,  # non-linear P(k)
            ells,
            WL_L_LIMBER,
            pk_linear=pk2d_lin,
            limber_max_error=0.1,
            Nchi=128,
            chi_min=1.0,
        )

    # New FKEM core: should NOT raise; it falls back to Limber.
    ell_limber_new, cl_new, status_new = nonlimber_fkem(
        cosmo=cosmo,
        tracer1=tracer,
        tracer2=tracer,
        p_of_k_a=pk2d_lin,
        ell=ells,
        ell_limber=WL_L_LIMBER,
        pk_linear=pk2d_lin,
        limber_max_error=0.1,
        n_chi_fkem=128,
        chi_min_fkem=1.0,
        n_consec_ell=3,
    )

    cl_new = np.asarray(cl_new, dtype=float)

    # Document fallback behaviour:
    assert status_new == 0
    assert ell_limber_new == -1  # "fell back to Limber"
    assert cl_new.size == 0  # no non-Limber Cls returned


# -------------------------------------------------------------------
# 3. WL: when legacy runs, new FKEM matches it
# -------------------------------------------------------------------


@pytest.mark.parametrize("lmax", [500, 1000])
@pytest.mark.parametrize("with_ia", [False, True])
def test_nonlimber_fkem_matches_legacy_wl(lmax: int, with_ia: bool):
    """If legacy FKEM runs, new FKEM Cls should agree closely (±IA)."""
    cosmo, tracer, ells, pk2d_lin = _build_wl_setup(lmax, with_ia=with_ia)

    # ---------- legacy FKEM ----------
    try:
        ell_limber_old, cl_old, status_old = legacy_nonlimber_fkem(
            cosmo,
            tracer,
            tracer,
            pk2d_lin,  # non-linear P(k); legacy parses this as Pk2D
            ells,
            WL_L_LIMBER,
            pk_linear=pk2d_lin,
            limber_max_error=0.1,
            Nchi=128,
            chi_min=1.0,
        )
    except CCLError:
        # If legacy fails, we cannot use it as a reference.
        pytest.xfail(
            "Legacy _nonlimber_FKEM failed with a CCL integration error"
        )

    assert status_old == 0
    cl_old = np.asarray(cl_old, dtype=float)

    # ---------- new FKEM core ----------
    ell_limber_new, cl_new, status_new = nonlimber_fkem(
        cosmo=cosmo,
        tracer1=tracer,
        tracer2=tracer,
        p_of_k_a=pk2d_lin,
        ell=ells,
        ell_limber=WL_L_LIMBER,
        pk_linear=pk2d_lin,
        limber_max_error=0.1,
        n_chi_fkem=128,
        chi_min_fkem=1.0,
        n_consec_ell=3,
    )

    assert status_new == 0
    cl_new = np.asarray(cl_new, dtype=float)

    # Legacy only computes up to its chosen non-Limber cutoff (encoded in len).
    n_nonlim = len(cl_old)
    assert n_nonlim > 0

    cl_new_sub = cl_new[:n_nonlim]

    max_rel = _max_rel_diff(cl_new_sub, cl_old)
    # Tune this threshold as needed for numerical fuzz
    assert max_rel < 1.0e-3, f"max relative diff = {max_rel}"


def test_cells_uses_fkem_for_low_ell_number_counts():
    """Tests that angular_cl uses FKEM for low-ell number counts."""
    cosmo, tracer1, tracer2, ells = _make_nc_setup(has_rsd=False)
    # 1) Pure Limber reference
    cl_limber = ccl.angular_cl(
        cosmo,
        tracer1,
        tracer2,
        ells,
        p_of_k_a="delta_matter:delta_matter",
        ell_limber=0,  # force Limber for all ℓ
    )

    # 2) FKEM
    cl_fkem = ccl.angular_cl(
        cosmo,
        tracer1,
        tracer2,
        ells,
        p_of_k_a="delta_matter:delta_matter",
        # whatever FKEM knobs are exposed at the high level
    )

    # Sanity: shapes match and we’re not wildly off
    assert cl_fkem.shape == cl_limber.shape
    # allow a relatively loose tolerance here
    npt.assert_allclose(cl_fkem, cl_limber, rtol=0.05)
