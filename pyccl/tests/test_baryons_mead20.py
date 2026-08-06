"""Unit tests for ``pyccl.baryons.mead20.BaryonsMead20``."""

from __future__ import annotations

import numpy as np
import pytest

import pyccl.baryons.mead20 as m


class _FakePk2D:
    """Minimal Pk2D-like object for tests (no C splines)."""

    def __init__(
        self,
        *,
        a_arr,
        lk_arr,
        pk_arr,
        is_logp=False,
        extrap_order_lok=1,
        extrap_order_hik=2,
    ):
        self._a = np.asarray(a_arr, float)
        self._lk = np.asarray(lk_arr, float)
        self._pk = np.asarray(pk_arr, float)
        self.is_logp = bool(is_logp)
        self.extrap_order_lok = int(extrap_order_lok)
        self.extrap_order_hik = int(extrap_order_hik)

    def get_spline_arrays(self):
        return self._a, self._lk, self._pk


class _FakeCosmology:
    """Minimal Cosmology-like object for tests."""

    def __init__(self, *, nonlin_fn=None, pk2d=None, d=None, baryons=None):
        self._nonlin_fn = nonlin_fn
        self._pk2d = pk2d
        self._d = dict(d or {})
        self.baryons = baryons

    def to_dict(self):
        return dict(self._d)

    def nonlin_matter_power(self, k, a):
        if self._nonlin_fn is None:
            raise RuntimeError("nonlin_matter_power not configured.")
        return self._nonlin_fn(np.asarray(k), np.asarray(a))

    def get_nonlin_power(self):
        if self._pk2d is None:
            raise RuntimeError("get_nonlin_power not configured.")
        return self._pk2d


def _cosmo_sig_constant(_cosmo):
    """Return a deterministic signature for caching tests."""
    return 12345


def _const_grid(value, na, nk):
    """Return a (na, nk) grid filled with a constant value."""
    return np.full((na, nk), float(value))


def _make_nonlin_fn(value):
    """Return nonlin power fn that matches requested (a, k) grid shape."""

    def fn(k, a):
        a = np.atleast_1d(a)
        k = np.atleast_1d(k)
        return np.full((a.size, k.size), float(value))

    return fn


def test_init_sets_attributes() -> None:
    """Test that constructor stores parameters with correct coercions."""
    b = m.BaryonsMead20(
        HMCode_logT_AGN=7.9,
        HMCode_A_baryon=3.0,
        HMCode_eta_baryon=None,
        kmax=12.0,
        lmax=100,
        dark_energy_model="ppf",
        camb_overrides={"kmax": 999.0},
        cache_cosmologies=False,
    )
    assert b.HMCode_logT_AGN == 7.9
    assert b.HMCode_A_baryon == 3.0
    assert b.HMCode_eta_baryon is None
    assert b.kmax == 12.0
    assert b.lmax == 100
    assert b.dark_energy_model == "ppf"
    assert b.camb_overrides == {"kmax": 999.0}
    assert b.cache_cosmologies is False


def test_get_internal_cosmo_no_cache_rebuilds(monkeypatch) -> None:
    """Test that disabling caching rebuilds internal cosmologies."""
    calls = []

    def fake_make(_cosmo, **kwargs):
        calls.append(dict(kwargs))
        return _FakeCosmology()

    monkeypatch.setattr(m, "_make_internal_camb_cosmology", fake_make)

    b = m.BaryonsMead20(cache_cosmologies=False)
    base = _FakeCosmology()

    c1 = b._get_internal_cosmo(
        base, halofit_version="v", include_feedback=True
    )
    c2 = b._get_internal_cosmo(
        base, halofit_version="v", include_feedback=True
    )

    assert c1 is not c2
    assert len(calls) == 2


def test_get_internal_cosmo_cache_reuses(monkeypatch) -> None:
    """Test that caching reuses the same internal cosmology object."""
    monkeypatch.setattr(m, "_cosmo_signature", _cosmo_sig_constant)

    made = []

    def fake_make(_cosmo, **_kwargs):
        out = _FakeCosmology()
        made.append(out)
        return out

    monkeypatch.setattr(m, "_make_internal_camb_cosmology", fake_make)

    b = m.BaryonsMead20(cache_cosmologies=True)
    base = _FakeCosmology()

    c1 = b._get_internal_cosmo(
        base, halofit_version="v", include_feedback=False
    )
    c2 = b._get_internal_cosmo(
        base, halofit_version="v", include_feedback=False
    )

    assert c1 is c2
    assert len(made) == 1


def test_boost_factor_returns_ratio(monkeypatch) -> None:
    """Test that boost_factor returns P_fb / P_dmo on the (a, k) grid."""

    def fake_make(_cosmo, *, include_feedback, **_kwargs):
        val = 2.0 if include_feedback else 1.0
        return _FakeCosmology(nonlin_fn=_make_nonlin_fn(val))

    monkeypatch.setattr(m, "_make_internal_camb_cosmology", fake_make)

    b = m.BaryonsMead20(cache_cosmologies=False)

    out = b.boost_factor(
        _FakeCosmology(),
        k=np.array([1.0, 2.0, 3.0]),
        a=np.array([0.5, 1.0]),
    )

    assert out.shape == (2, 3)
    assert np.allclose(out, 2.0)


def test_boost_factor_restores_scalar_shapes(monkeypatch) -> None:
    """Test that scalar inputs restore scalar/1D outputs correctly."""

    def fake_make(_cosmo, *, include_feedback, **_kwargs):
        val = 4.0 if include_feedback else 2.0
        return _FakeCosmology(nonlin_fn=_make_nonlin_fn(val))

    monkeypatch.setattr(m, "_make_internal_camb_cosmology", fake_make)

    b = m.BaryonsMead20(cache_cosmologies=False)

    out1 = b.boost_factor(_FakeCosmology(), k=np.array([1.0, 2.0, 3.0]), a=1.0)
    assert out1.shape == (3,)
    assert np.allclose(out1, 2.0)

    out2 = b.boost_factor(_FakeCosmology(), k=1.0, a=np.array([0.5, 1.0]))
    assert out2.shape == (2,)
    assert np.allclose(out2, 2.0)

    out3 = b.boost_factor(_FakeCosmology(), k=1.0, a=1.0)
    assert isinstance(out3, float)
    assert out3 == 2.0


def test__include_baryonic_effects_shape_mismatch_raises(monkeypatch) -> None:
    """Test that boost/Pk2D shape mismatch raises RuntimeError."""
    b = m.BaryonsMead20()

    def fake_boost(_cosmo, _k, _a):
        return np.ones((99, 99))

    b.boost_factor = fake_boost

    a = np.array([0.5, 1.0])
    lk = np.log(np.array([1.0, 2.0, 3.0]))
    pk = _FakePk2D(a_arr=a, lk_arr=lk, pk_arr=np.ones((2, 3)), is_logp=False)

    with pytest.raises(RuntimeError, match=r"Boost shape"):
        b._include_baryonic_effects(_FakeCosmology(), pk)


def test__include_baryonic_effects_log_mode(monkeypatch) -> None:
    """Test that log(P) stays log(P) when boost is applied."""
    monkeypatch.setattr(m, "Pk2D", _FakePk2D)

    b = m.BaryonsMead20()

    def fake_boost(_cosmo, _k, _a):
        return np.full((2, 3), 2.0)

    b.boost_factor = fake_boost

    a = np.array([0.5, 1.0])
    lk = np.log(np.array([1.0, 2.0, 3.0]))
    pk_in = np.log(np.full((2, 3), 5.0))
    pk = _FakePk2D(a_arr=a, lk_arr=lk, pk_arr=pk_in, is_logp=True)

    out = b._include_baryonic_effects(_FakeCosmology(), pk)
    _, _, pk_out = out.get_spline_arrays()

    assert out.is_logp is True
    assert np.allclose(pk_out, pk_in + np.log(2.0))


def test_include_baryonic_effects_uses_internal_dmo_baseline(
    monkeypatch
) -> None:
    """Test that include_baryonic_effects uses internal DMO baseline."""
    # It sohuld not use Pk
    monkeypatch.setattr(m, "Pk2D", _FakePk2D)

    b = m.BaryonsMead20(cache_cosmologies=False)

    a = np.array([0.5, 1.0])
    lk = np.log(np.array([1.0, 2.0, 3.0]))

    pk_dmo = _FakePk2D(a_arr=a, lk_arr=lk, pk_arr=np.full((2, 3), 7.0))
    internal_dmo = _FakeCosmology(pk2d=pk_dmo)

    def fake_get_internal(_cosmo, *, include_feedback, **_kwargs):
        if include_feedback:
            return _FakeCosmology(nonlin_fn=_make_nonlin_fn(2.0))
        return internal_dmo

    monkeypatch.setattr(b, "_get_internal_cosmo", fake_get_internal)

    captured = {"cosmo": None, "pk": None}

    def fake_include(cosmo, pk):
        captured["cosmo"] = cosmo
        captured["pk"] = pk
        return pk

    monkeypatch.setattr(b, "_include_baryonic_effects", fake_include)

    pk_user = _FakePk2D(a_arr=a, lk_arr=lk, pk_arr=np.full((2, 3), 999.0))

    out = b.include_baryonic_effects(_FakeCosmology(), pk_user)

    assert out is pk_dmo
    assert captured["cosmo"] is internal_dmo
    assert captured["pk"] is pk_dmo
