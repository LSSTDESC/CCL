"""Unit tests for `pyccl.baryons.fedeli14`."""

from __future__ import annotations

import numpy as np
import pytest

import pyccl as ccl
from pyccl import Pk2D

from pyccl.baryons.fedeli14 import BaryonsFedeli14


def _cosmo() -> ccl.Cosmology:
    """Return a CCL cosmology."""
    return ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.965, sigma8=0.8)


def _make_pk2d(*, is_logp: bool, absurd_log_flag_data: bool = False) -> Pk2D:
    """Make a small but spline-valid Pk2D with known behavior for testing."""
    # Use >=4 points to avoid spline allocation issues.
    a_arr = np.array([0.05, 0.2, 0.5, 1.0], dtype=float)
    k_arr = np.array([1e-4, 1e-2, 1e-1, 1.0], dtype=float)
    lk_arr = np.log(k_arr)

    pk_lin = np.ones((a_arr.size, k_arr.size), dtype=float) * 2.0

    if is_logp:
        if absurd_log_flag_data:
            # Must be > 200 to trigger fix,
            # but small enough to avoid exp overflow.
            pk_arr = np.ones_like(pk_lin) * 300.0
        else:
            pk_arr = np.log(pk_lin)
    else:
        pk_arr = pk_lin

    return Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr, is_logp=is_logp)


def test_update_parameters_ignores_none_casts_and_rejects_unknown() -> None:
    """update_parameters should ignore None, cast numeric types, and reject
    unknown keys."""
    b = BaryonsFedeli14()

    old = b.Fg
    b.update_parameters(Fg=None)
    assert b.Fg == old

    b.update_parameters(Fg=0.5, bd=1, n_m=16)
    assert isinstance(b.Fg, float) and b.Fg == 0.5
    assert isinstance(b.bd, float) and b.bd == 1.0
    assert isinstance(b.n_m, int) and b.n_m == 16

    with pytest.raises(AttributeError, match=r"Unknown parameter"):
        b.update_parameters(nope=1)


def test_boost_factor_broadcasts_and_calls_bhm_boost(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """boost_factor should broadcast (a,k) and call BaryonHaloModel.boost
    per-a."""
    b = BaryonsFedeli14(renormalize_large_scales=False)

    calls: list[tuple[float, np.ndarray, str]] = []

    class DummyBHM:
        def __init__(self) -> None:
            self.interpolation_grid = {
                "dark_matter": {"k": np.array([1e-6, 1e2])}}

        def boost(self, *, k: np.ndarray, a: float, pk_ref: str) -> np.ndarray:
            calls.append((float(a), np.asarray(k, float).copy(), str(pk_ref)))
            return np.ones_like(np.asarray(k, float)) * (1.0 + float(a))

    monkeypatch.setattr(
        BaryonsFedeli14, "_build_bhm", lambda self, cosmo: DummyBHM())

    cosmo = _cosmo()
    k = np.array([1e-4, 1e-2, 1.0, 10.0], dtype=float)
    a = np.array([0.5, 1.0], dtype=float)

    out = b.boost_factor(cosmo, k, a)
    assert out.shape == (a.size, k.size)
    assert np.allclose(out[0], 1.0 + 0.5)
    assert np.allclose(out[1], 1.0 + 1.0)
    assert len(calls) == a.size
    for (aval, kvals, pkref) in calls:
        assert pkref == b.pk_ref
        assert np.allclose(kvals, k)  # no k->k/h conversion


def test_boost_factor_renormalizes_large_scales_to_unity(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """If renormalize_large_scales=True, mean boost for k<=k_renorm_max is
    unity per-a."""
    b = BaryonsFedeli14(renormalize_large_scales=True, k_renorm_max=1e-2)

    class DummyBHM:
        def __init__(self) -> None:
            self.interpolation_grid = {
                "dark_matter": {
                    "k": np.array([1e-6, 1e2], dtype=float),
                },
            }

        def boost(self, *, k: np.ndarray, a: float, pk_ref: str) -> np.ndarray:
            k = np.asarray(k, float)
            _, _ = a, pk_ref  # unused
            # k<=1e-2 => 2; else => 4
            return np.where(k <= 1e-2, 2.0, 4.0)

    monkeypatch.setattr(
        BaryonsFedeli14, "_build_bhm", lambda self, cosmo: DummyBHM())

    cosmo = _cosmo()
    k = np.array([1e-4, 1e-2, 1e-1, 1.0], dtype=float)
    a = np.array([0.5, 1.0], dtype=float)

    out = b.boost_factor(cosmo, k, a)
    # first two points are <= 1e-2 => mean is 2 => normalized to 1
    assert np.allclose(out[:, :2], 1.0)
    # high-k was 4 => becomes 2 after /2
    assert np.allclose(out[:, 2:], 2.0)


def test_incl_bary_eff_applies_unity_for_early_times_and_outside_k_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply boost only for a>=0.1 and within model k support."""
    b = BaryonsFedeli14(renormalize_large_scales=False)

    # Force boost_factor to return 10 wherever called
    monkeypatch.setattr(
        BaryonsFedeli14,
        "boost_factor",
        lambda self, cosmo, k, a: np.ones((np.size(a), np.size(k))) * 10.0,
    )

    # Dummy BHM: k support [1e-3, 1e-1]
    class DummyBHM1:
        def __init__(self) -> None:
            self.interpolation_grid = {
                "dark_matter": {"k": np.array([1e-3, 1e-1],
                                              dtype=float)}}

    monkeypatch.setattr(
        BaryonsFedeli14, "_build_bhm", lambda self, cosmo: DummyBHM1())

    pk = _make_pk2d(is_logp=False)
    out = b._include_baryonic_effects(_cosmo(), pk)

    a_arr, lk_arr, pk_out = out.get_spline_arrays()
    k_arr = np.exp(lk_arr)

    # baseline linear P=2 everywhere
    assert np.allclose(pk_out[0, :], 2.0)  # a=0.05 < 0.1 => unity

    # k grid is [1e-4, 1e-2, 1e-1, 1]
    # supported is [1e-3, 1e-1] => indices 1 and 2 are boosted for a>=0.1
    for i in range(1, a_arr.size):
        assert np.allclose(pk_out[i, 0], 2.0)  # k=1e-4 outside
        assert np.allclose(pk_out[i, 1], 20.0)  # k=1e-2 inside
        assert np.allclose(pk_out[i, 2], 20.0)  # k=1e-1 inside
        assert np.allclose(pk_out[i, 3], 2.0)  # k=1 outside

    # sanity: grids unchanged
    a0, lk0, _ = pk.get_spline_arrays()
    assert np.allclose(a_arr, a0)
    assert np.allclose(k_arr, np.exp(lk0))

    # Dummy BHM: k support [1e-3, 1e-1]
    class DummyBHM2:
        """Mock BHM class."""

        def __init__(self) -> None:
            self.interpolation_grid = {
                "dark_matter": {
                    "k": np.array([1e-3, 1e-1], dtype=float),
                },
            }

    monkeypatch.setattr(
        BaryonsFedeli14, "_build_bhm", lambda self, cosmo: DummyBHM2())

    pk = _make_pk2d(is_logp=False)
    out = b._include_baryonic_effects(_cosmo(), pk)

    a_arr, lk_arr, pk_arr = out.get_spline_arrays()
    k_arr = np.exp(lk_arr)

    # baseline linear P=2
    assert np.allclose(pk_arr[0, :], 2.0)  # a=0.05 < 0.1 => unity

    # our k grid: [1e-4, 1e-2, 1e-1, 1]
    # support is [1e-3,1e-1] inclusive => boosts apply to 1e-2 and 1e-1
    # for a>=0.1 rows (a=0.2,0.5,1.0): boosted at indices 1,2
    for i in range(1, a_arr.size):
        assert np.allclose(pk_arr[i, 0], 2.0)  # k=1e-4 outside
        assert np.allclose(pk_arr[i, 1], 20.0)  # k=1e-2 inside
        assert np.allclose(pk_arr[i, 2], 20.0)  # k=1e-1 inside
        assert np.allclose(pk_arr[i, 3], 2.0)  # k=1 outside

    assert np.allclose(a_arr, pk.get_spline_arrays()[0])
    assert np.allclose(k_arr, np.exp(pk.get_spline_arrays()[1]))


def test_incl_bary_eff_respects_logp_representation(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """If pk is logp, boost is applied additively in log space
    (but get_spline_arrays returns linear)."""
    b = BaryonsFedeli14(renormalize_large_scales=False)

    monkeypatch.setattr(
        BaryonsFedeli14,
        "boost_factor",
        lambda self, cosmo, k, a: np.ones((np.size(a), np.size(k))) * 10.0,
    )

    class DummyBHM:
        """Mock BHM class."""
        def __init__(self) -> None:
            self.interpolation_grid = {
                "dark_matter": {
                    "k": np.array([1e-6, 1e2], dtype=float),
                },
            }

    monkeypatch.setattr(
        BaryonsFedeli14, "_build_bhm", lambda self, cosmo: DummyBHM())

    pk = _make_pk2d(is_logp=True)
    out = b._include_baryonic_effects(_cosmo(), pk)

    a_arr, lk_arr, pk_out = out.get_spline_arrays()

    # get_spline_arrays() returns linear P(k), not log(P)
    assert np.allclose(pk_out[0, :], 2.0)     # a<0.1 => unity
    assert np.allclose(pk_out[1:, :], 20.0)   # boosted by 10 for a>=0.1


def test_incl_bary_eff_fixes_absurd_log_flag_data(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """If marked log but pk_arr.max>200, treat pk_arr as linear and convert to
     log before boosting."""
    b = BaryonsFedeli14(renormalize_large_scales=False)

    monkeypatch.setattr(
        BaryonsFedeli14,
        "boost_factor",
        lambda self, cosmo, k, a: np.ones((np.size(a), np.size(k))) * 10.0,
    )

    class DummyBHM:
        def __init__(self) -> None:
            self.interpolation_grid = {
                "dark_matter": {"k": np.array([1e-6, 1e2],
                                              dtype=float)}}

    monkeypatch.setattr(
        BaryonsFedeli14, "_build_bhm", lambda self, cosmo: DummyBHM())

    pk = _make_pk2d(is_logp=True, absurd_log_flag_data=True)
    out = b._include_baryonic_effects(_cosmo(), pk)

    a_arr, lk_arr, pk_out = out.get_spline_arrays()

    # After fix: pk_arr stored becomes log(300).
    # get_spline_arrays returns exp => 300 (for a<0.1).
    assert np.allclose(pk_out[0, :], 300.0)
    # For a>=0.1: add log(10) => exp(log(300)+log(10)) = 3000
    assert np.allclose(pk_out[1:, :], 3000.0)


def test_incl_bary_eff_preserves_extrap_orders_and_log_flag(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Output Pk2D preserves extrapolation orders and log flag choice."""
    b = BaryonsFedeli14(renormalize_large_scales=False)

    # Ensure we don't call heavy stuff; keep boost unity everywhere
    monkeypatch.setattr(
        BaryonsFedeli14,
        "boost_factor",
        lambda self, cosmo, k, a: np.ones((np.size(a), np.size(k))) * 1.0,
    )

    class DummyBHM:
        def __init__(self) -> None:
            self.interpolation_grid = {
                "dark_matter": {"k": np.array([1e-6, 1e2],
                                              dtype=float)}}

    monkeypatch.setattr(
        BaryonsFedeli14, "_build_bhm", lambda self, cosmo: DummyBHM())

    pk0 = _make_pk2d(is_logp=False)
    out0 = b._include_baryonic_effects(_cosmo(), pk0)
    assert out0.extrap_order_lok == pk0.extrap_order_lok
    assert out0.extrap_order_hik == pk0.extrap_order_hik
    assert bool(out0.psp.is_log) == bool(pk0.psp.is_log)

    pk1 = _make_pk2d(is_logp=True)
    out1 = b._include_baryonic_effects(_cosmo(), pk1)
    assert out1.extrap_order_lok == pk1.extrap_order_lok
    assert out1.extrap_order_hik == pk1.extrap_order_hik
    assert bool(out1.psp.is_log) == bool(pk1.psp.is_log)
