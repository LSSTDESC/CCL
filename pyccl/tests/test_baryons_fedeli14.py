"""Unit tests for `pyccl.baryons.fedeli14`."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import pyccl as ccl
from pyccl import Pk2D

from pyccl.baryons.fedeli14 import BaryonsFedeli14


class DummyBHM:
    """Mock BaryonHaloModel class."""

    def __init__(
        self,
        *,
        k_support: tuple[float, float] = (1e-6, 1e2),
        boost_value: float | None = 1.0,
        calls: list[tuple[float, np.ndarray, str]] | None = None,
    ) -> None:
        self.interpolation_grid = {
            "dark_matter": {"k": np.array(k_support, dtype=float)}}
        self.boost_value = boost_value
        self.calls = calls

    def boost(self, *, k: np.ndarray, a: float, pk_ref: str) -> np.ndarray:
        """Tests that boost returns a configurable mock boost."""
        k = np.asarray(k, float)

        if self.calls is not None:
            self.calls.append((float(a), k.copy(), str(pk_ref)))

        if self.boost_value is None:
            return np.where(k <= 1e-2, 2.0, 4.0)

        return np.ones_like(k) * self.boost_value


def _cosmo() -> ccl.Cosmology:
    """Return a CCL cosmology."""
    return ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.965, sigma8=0.8)


def _make_pk2d(*, is_logp: bool) -> Pk2D:
    """Make a small but spline-valid Pk2D with known behavior for testing."""
    a_arr = np.array([0.05, 0.2, 0.5, 1.0], dtype=float)
    k_arr = np.array([1e-4, 1e-2, 1e-1, 1.0], dtype=float)
    lk_arr = np.log(k_arr)

    pk_lin = np.ones((a_arr.size, k_arr.size), dtype=float) * 2.0
    pk_arr = np.log(pk_lin) if is_logp else pk_lin

    return Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr, is_logp=is_logp)


@pytest.mark.parametrize(
    "name,value,expected_type,expected_value",
    [
        ("Fg", None, float, None),
        ("bd", None, float, None),
        ("m0_star", None, float, None),
        ("sigma_star", None, float, None),
        ("mmin_star", None, float, None),
        ("mmax_star", None, float, None),
        ("m0_gas", None, float, None),
        ("sigma_gas", None, float, None),
        ("gas_beta", None, float, None),
        ("gas_r_co", None, float, None),
        ("gas_r_ej", None, float, None),
        ("star_x_delta", None, float, None),
        ("star_alpha", None, float, None),
        ("density_mmin", None, float, None),
        ("density_mmax", None, float, None),
        ("a_min", None, float, None),
        ("n_m", None, int, None),
        ("Fg", 0.5, float, 0.5),
        ("bd", 1, float, 1.0),
        ("n_m", 16.0, int, 16),
        ("pk_ref", 123, str, "123"),
        ("renormalize_large_scales", 0, bool, False),
        ("update_fftlog_precision", 1, bool, True),
    ],
)
def test_update_parameters_casts_valid_values(
    name: str,
    value: object,
    expected_type: type,
    expected_value: object,
) -> None:
    """Tests that update_parameters casts valid values and ignores None."""
    b = BaryonsFedeli14()

    old_value = getattr(b, name)
    b.update_parameters(**{name: value})

    if value is None:
        assert getattr(b, name) == old_value
    else:
        assert isinstance(getattr(b, name), expected_type)
        assert getattr(b, name) == expected_value


def test_update_parameters_accepts_parameter_combinations() -> None:
    """Tests that update_parameters accepts valid parameter combinations."""
    values = [
        ("Fg", 0.5),
        ("bd", 1),
        ("n_m", 16),
        ("pk_ref", "pk_lin"),
        ("renormalize_large_scales", False),
        ("a_min", None),
    ]

    for first, second in itertools.product(values, repeat=2):
        b = BaryonsFedeli14()
        updates = dict([first, second])
        b.update_parameters(**updates)


def test_update_parameters_rejects_unknown() -> None:
    """Tests that update_parameters rejects unknown keys."""
    b = BaryonsFedeli14()

    with pytest.raises(AttributeError, match=r"Unknown parameter"):
        b.update_parameters(nope=1)


def test_boost_factor_broadcasts_and_calls_bhm_boost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that boost_factor broadcasts inputs and calls BHM per a."""
    b = BaryonsFedeli14(renormalize_large_scales=False)

    calls: list[tuple[float, np.ndarray, str]] = []
    monkeypatch.setattr(
        BaryonsFedeli14,
        "_build_bhm",
        lambda self, cosmo: DummyBHM(calls=calls),
    )

    k = np.array([1e-4, 1e-2, 1.0, 10.0], dtype=float)
    a = np.array([0.5, 1.0], dtype=float)

    out = b.boost_factor(_cosmo(), k, a)

    assert out.shape == (a.size, k.size)
    assert np.allclose(out, 1.0)
    assert len(calls) == a.size

    for _, kvals, pkref in calls:
        assert pkref == b.pk_ref
        assert np.allclose(kvals, k)


def test_boost_factor_renormalizes_large_scales_to_unity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that boost_factor renormalizes large scales to unity."""
    b = BaryonsFedeli14(renormalize_large_scales=True, k_renorm_max=1e-2)

    monkeypatch.setattr(
        BaryonsFedeli14,
        "_build_bhm",
        lambda self, cosmo: DummyBHM(boost_value=None),
    )

    k = np.array([1e-4, 1e-2, 1e-1, 1.0], dtype=float)
    a = np.array([0.5, 1.0], dtype=float)

    out = b.boost_factor(_cosmo(), k, a)

    assert np.allclose(out[:, :2], 1.0)
    assert np.allclose(out[:, 2:], 2.0)


def test_incl_bary_eff_applies_unity_for_early_times_and_outside_k_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that boost is applied only for a>=0.1 and supported k."""
    b = BaryonsFedeli14(renormalize_large_scales=False)

    monkeypatch.setattr(
        BaryonsFedeli14,
        "boost_factor",
        lambda self, cosmo, k, a: np.ones((np.size(a), np.size(k))) * 10.0,
    )
    monkeypatch.setattr(
        BaryonsFedeli14,
        "_build_bhm",
        lambda self, cosmo: DummyBHM(k_support=(1e-3, 1e-1)),
    )

    pk = _make_pk2d(is_logp=False)
    out = b._include_baryonic_effects(_cosmo(), pk)

    a_arr, lk_arr, pk_out = out.get_spline_arrays()
    k_arr = np.exp(lk_arr)

    assert np.allclose(pk_out[0, :], 2.0)

    for i in range(1, a_arr.size):
        assert np.allclose(pk_out[i, 0], 2.0)
        assert np.allclose(pk_out[i, 1], 20.0)
        assert np.allclose(pk_out[i, 2], 20.0)
        assert np.allclose(pk_out[i, 3], 2.0)

    a0, lk0, _ = pk.get_spline_arrays()
    assert np.allclose(a_arr, a0)
    assert np.allclose(k_arr, np.exp(lk0))


def test_incl_bary_eff_respects_logp_representation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that log-space Pk2D inputs are returned in log-space form."""
    b = BaryonsFedeli14(renormalize_large_scales=False)

    monkeypatch.setattr(
        BaryonsFedeli14,
        "boost_factor",
        lambda self, cosmo, k, a: np.ones((np.size(a), np.size(k))) * 10.0,
    )
    monkeypatch.setattr(
        BaryonsFedeli14,
        "_build_bhm",
        lambda self, cosmo: DummyBHM(),
    )

    pk = _make_pk2d(is_logp=True)
    out = b._include_baryonic_effects(_cosmo(), pk)

    _, _, pk_out = out.get_spline_arrays()

    assert np.allclose(pk_out[0, :], 2.0)
    assert np.allclose(pk_out[1:, :], 20.0)


def test_incl_bary_eff_preserves_extrap_orders_and_log_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that output Pk2D preserves extrap orders and log flag."""
    b = BaryonsFedeli14(renormalize_large_scales=False)

    monkeypatch.setattr(
        BaryonsFedeli14,
        "boost_factor",
        lambda self, cosmo, k, a: np.ones((np.size(a), np.size(k))),
    )
    monkeypatch.setattr(
        BaryonsFedeli14,
        "_build_bhm",
        lambda self, cosmo: DummyBHM(),
    )

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


def test_boost_factor_squeezes_for_scalar_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that boost_factor returns a scalar when k is a scalar."""
    b = BaryonsFedeli14(renormalize_large_scales=False)

    monkeypatch.setattr(
        BaryonsFedeli14,
        "_build_bhm",
        lambda self, cosmo: DummyBHM(boost_value=2.0),
    )

    out = b.boost_factor(_cosmo(), k=1e-2, a=np.array([0.5, 1.0]))

    assert out.shape == (2,)
    assert np.allclose(out, 2.0)


def test_boost_factor_renorm_guard_replaces_bad_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that boost factor renormalization guard replaces bad norms."""
    b = BaryonsFedeli14(renormalize_large_scales=True, k_renorm_max=1e-2)

    class BadNormDummyBHM(DummyBHM):
        """Mock BHM class with bad renormalization values."""

        def boost(self, *, k: np.ndarray, a: float, pk_ref: str) -> np.ndarray:
            """Tests that boost returns zero on renormalization scales."""
            _, _ = a, pk_ref
            k = np.asarray(k, float)
            return np.where(k <= 1e-2, 0.0, 4.0)

    monkeypatch.setattr(
        BaryonsFedeli14,
        "_build_bhm",
        lambda self, cosmo: BadNormDummyBHM(),
    )

    k = np.array([1e-4, 1e-2, 1e-1], dtype=float)
    a = np.array([0.5, 1.0], dtype=float)

    out = b.boost_factor(_cosmo(), k, a)

    assert np.all(np.isfinite(out))
    assert np.all(out > 0.0)
