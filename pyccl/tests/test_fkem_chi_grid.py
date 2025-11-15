"""Unit tests for chi grid construction in FKEM non-Limber module."""

from __future__ import annotations

import pytest

import numpy as np

from pyccl.nonlimber_fkem.chi_grid import build_chi_grid


def simple_chis():
    """Creates simple chi arrays for two tracers for testing."""
    chis_t1 = [np.array([10.0, 20.0, 30.0]), np.array([15.0, 25.0, 35.0])]
    chis_t2 = [np.array([5.0, 12.0, 40.0]), np.array([8.0, 18.0, 50.0])]
    return chis_t1, chis_t2


def test_build_chi_grid_infers_min_max_and_nchi():
    """Test build_chi_grid infers chi_min, chi_max, and n_chi correctly."""
    chis_t1, chis_t2 = simple_chis()

    chi_log, dlnr, chi_min_eff, chi_max_eff, n_chi_eff = build_chi_grid(
        chis_t1,
        chis_t2,
        chi_min=None,
        n_chi=None,
        warn=False,
    )

    # Should span min and max of all chis
    assert chi_min_eff > 0.0
    assert chi_min_eff <= 5.0 + 1e-6
    assert chi_max_eff >= 50.0 - 1e-6

    # n_chi inferred from min of lengths (here we hav 3)
    assert n_chi_eff == 3

    # chi_log is 1D, monotonic, all finite
    assert chi_log.ndim == 1
    assert chi_log.size == n_chi_eff
    assert np.all(np.isfinite(chi_log))
    assert np.all(np.diff(chi_log) > 0)
    assert dlnr > 0.0


def test_build_chi_grid_respects_explicit_nchi():
    """Tests that build_chi_grid respects an explicit n_chi parameter."""
    chis_t1, chis_t2 = simple_chis()

    chi_log, dlnr, chi_min_eff, chi_max_eff, n_chi_eff = build_chi_grid(
        chis_t1,
        chis_t2,
        chi_min=None,
        n_chi=10,
        warn=False,
    )

    assert n_chi_eff == 10
    assert chi_log.size == 10
    assert dlnr == pytest.approx(np.log(chi_max_eff / chi_min_eff) / (10 - 1))


def test_build_chi_grid_rejects_negative_chi():
    """Tests that build_chi_grid rejects chi arrays with negative values."""
    chis_t1 = [np.array([-1.0, 1.0])]
    chis_t2 = [np.array([1.0, 2.0])]

    with pytest.raises(ValueError, match="negative values"):
        build_chi_grid(chis_t1, chis_t2, chi_min=None, n_chi=None, warn=False)


def test_build_chi_grid_rejects_empty_lists():
    """Tests that build_chi_grid rejects empty chi array lists."""
    with pytest.raises(ValueError, match="tracer1"):
        build_chi_grid([], [np.array([1.0, 2.0])], chi_min=None, n_chi=None)

    with pytest.raises(ValueError, match="tracer2"):
        build_chi_grid([np.array([1.0, 2.0])], [], chi_min=None, n_chi=None)


def test_build_chi_grid_rejects_too_small_nchi():
    """Tests that build_chi_grid rejects n_chi < 2."""
    chis_t1, chis_t2 = simple_chis()
    with pytest.raises(ValueError, match="Nchi=1 is too small"):
        build_chi_grid(chis_t1, chis_t2, chi_min=None, n_chi=1, warn=False)


def test_build_chi_grid_rejects_chi_max_leq_min():
    """Tests that build_chi_grid rejects chi_max <= chi_min."""
    chis_t1, chis_t2 = simple_chis()
    with pytest.raises(ValueError, match="chi_max <= chi_min"):
        build_chi_grid(chis_t1, chis_t2, chi_min=1e3, n_chi=None, warn=False)


def test_build_chi_grid_explicit_chimin_clipped():
    """Explicit chi_min smaller than data min should be handled sensibly."""
    chis_t1 = [np.array([10.0, 20.0, 30.0])]
    chis_t2 = [np.array([5.0, 15.0, 25.0])]

    chi_log, dlnr, chi_min_eff, chi_max_eff, n_chi_eff = build_chi_grid(
        chis_t1,
        chis_t2,
        chi_min=1.0,
        n_chi=None,
        warn=False,
    )

    # It should not crash and should give a sensible grid
    assert chi_min_eff > 0.0
    # effective min should not be below the actual finite min
    assert chi_min_eff <= 5.0 + 1e-6
    assert chi_max_eff >= 30.0 - 1e-6
    assert n_chi_eff == 3
    assert chi_log.size == 3
    assert chi_log.ndim == 1
    assert np.all(np.isfinite(chi_log))
    assert np.all(np.diff(chi_log) > 0)
    assert dlnr > 0.0


def test_build_chi_grid_ignores_nonfinite_entries():
    """Non-finite chi values should not crash build_chi_grid."""
    chis_t1 = [np.array([10.0, np.nan, 30.0]), np.array([15.0, 25.0, 35.0])]
    chis_t2 = [np.array([5.0, 12.0, np.inf]), np.array([8.0, 18.0, 50.0])]

    chi_log, dlnr, chi_min_eff, chi_max_eff, n_chi_eff = build_chi_grid(
        chis_t1,
        chis_t2,
        chi_min=None,
        n_chi=None,
        warn=False,
    )

    # Should still return a finite, monotonic grid
    assert n_chi_eff >= 1
    assert chi_log.ndim == 1
    assert chi_log.size == n_chi_eff
    assert np.all(np.isfinite(chi_log))
    assert np.all(np.diff(chi_log) > 0)
    assert dlnr > 0.0
    assert np.isfinite(chi_min_eff)
    assert np.isfinite(chi_max_eff)
