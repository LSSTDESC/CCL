"""Unit tests for `pyccl.baryons.fedeli14_bhm.baryon_halo_model`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import pyccl as ccl

from pyccl.baryons.fedeli14_bhm.baryon_halo_model import BaryonHaloModel
from pyccl.baryons.fedeli14_bhm.numerics import _require_a, _require_k


def _cosmo() -> ccl.Cosmology:
    """Return a simple, valid CCL cosmology."""
    return ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.965, sigma8=0.8)


def _kgrid() -> np.ndarray:
    """Return a small, strictly increasing positive k grid."""
    return np.array([0.1, 1.0, 10.0], dtype=float)


def test_require_a_validates_finite_and_positive() -> None:
    """Tests that _require_a rejects non-finite and non-positive values."""
    assert _require_a(1.0) == 1.0

    with pytest.raises(ValueError, match=r"a must be finite"):
        _ = _require_a(float("nan"))

    with pytest.raises(ValueError, match=r"a must be > 0"):
        _ = _require_a(0.0)

    with pytest.raises(ValueError, match=r"a must be > 0"):
        _ = _require_a(-1.0)


def test_require_k_validates_shape_finiteness_sign_and_monotonicity() -> None:
    """Tests that _require_k enforces 1D, finite, positive, strictly
    increasing grids."""
    k = _kgrid()
    out = _require_k(k)
    assert isinstance(out, np.ndarray) and out.shape == k.shape
    assert np.all(out == k)

    with pytest.raises(ValueError, match=r"non-empty 1D array"):
        _ = _require_k(np.ones((2, 2)))

    with pytest.raises(ValueError, match=r"non-empty 1D array"):
        _ = _require_k(np.array([], dtype=float))

    with pytest.raises(ValueError, match=r"contain only finite values"):
        _ = _require_k(np.array([0.1, np.nan, 1.0], dtype=float))

    with pytest.raises(ValueError, match=r"k must be > 0"):
        _ = _require_k(np.array([0.0, 0.1, 1.0], dtype=float))

    with pytest.raises(ValueError, match=r"strictly increasing"):
        _ = _require_k(np.array([0.1, 0.1, 1.0], dtype=float))


def test_init_validates_Fg_bd_and_density_mass_range() -> None:
    """Tests that BaryonHaloModel validates Fg, bd, and density integral
    mass range."""
    cosmo = _cosmo()

    with pytest.raises(ValueError, match=r"Fg must be in \[0, 1\]"):
        _ = BaryonHaloModel(cosmo=cosmo, Fg=1.5)

    with pytest.raises(ValueError, match=r"bd must be finite"):
        _ = BaryonHaloModel(cosmo=cosmo, bd=float("nan"))

    with pytest.raises(
            ValueError,
            match=r"density_mmin/density_mmax must satisfy 0 < min < max"):
        _ = BaryonHaloModel(cosmo=cosmo, density_mmin=0.0, density_mmax=1.0e6)

    with pytest.raises(
            ValueError,
            match=r"density_mmin/density_mmax must satisfy 0 < min < max"):
        _ = BaryonHaloModel(
            cosmo=cosmo, density_mmin=1.0e6, density_mmax=1.0e6
        )

    with pytest.raises(
            ValueError,
            match=r"density_mmin/density_mmax must satisfy 0 < min < max"):
        _ = BaryonHaloModel(
            cosmo=cosmo, density_mmin=1.0e10, density_mmax=1.0e6
        )


def test_clear_cache_empties_all_internal_caches() -> None:
    """Tests that clear_cache empties all per-a caches."""
    bhm = BaryonHaloModel(cosmo=_cosmo())

    def _y_dm(M, k):
        """Helper function that returns a constant function of mass and k."""
        return M * 0 + k

    def _y_dmo(M, k):
        """Helper function that returns a constant function of mass and k."""
        return M * 0 + k

    # seed caches directly (unit test: OK)
    bhm._frac_cache[1.0] = (object(), object(), object())
    bhm._profile_cache[1.0] = {"dark_matter": object()}
    bhm._interp_cache[1.0] = {"dark_matter": _y_dm}
    bhm._dmo_dm_interp_cache[1.0] = _y_dmo

    bhm.clear_cache()

    assert bhm._frac_cache == {}  # noqa: SLF001
    assert bhm._profile_cache == {}  # noqa: SLF001
    assert bhm._interp_cache == {}  # noqa: SLF001
    assert bhm._dmo_dm_interp_cache == {}  # noqa: SLF001


def test_mass_frac_kwargs_only_returns_non_none_keys() -> None:
    """Tests that _mass_frac_kwargs drops None-valued entries
    (rho_star by default)."""
    bhm = BaryonHaloModel(cosmo=_cosmo(), rho_star=None)
    kw = bhm._mass_frac_kwargs()  # noqa: SLF001
    assert "rho_star" not in kw
    assert "m0_star" in kw
    assert "sigma_star" in kw

    bhm2 = BaryonHaloModel(cosmo=_cosmo(), rho_star=123.0)
    kw2 = bhm2._mass_frac_kwargs()  # noqa: SLF001
    assert "rho_star" in kw2
    assert float(kw2["rho_star"]) == 123.0


def test_get_mass_fractions_is_cached(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests that _get_mass_fractions caches results per scale factor a."""
    calls = {"n": 0}

    def fake_mass_fractions(**kwargs: Any):
        """Mock mass_fractions that returns constant functions."""
        calls["n"] += 1

        def f_g(m):
            """Mock f_g that returns constant functions."""
            return np.zeros_like(np.asarray(m, float)) + 0.1

        def f_s(m):
            """Mock f_s that returns constant functions."""
            return np.zeros_like(np.asarray(m, float)) + 0.01

        def f_d(m):
            """Mock f_d that returns constant functions."""
            return np.zeros_like(np.asarray(m, float)) + 0.89

        return f_g, f_s, f_d

    import pyccl.baryons.fedeli14_bhm.baryon_halo_model as mod

    monkeypatch.setattr(mod, "mass_fractions", fake_mass_fractions)

    bhm = BaryonHaloModel(cosmo=_cosmo())

    fg1, fs1, fd1 = bhm._get_mass_fractions(1.0)  # noqa: SLF001
    fg2, fs2, fd2 = bhm._get_mass_fractions(1.0)  # noqa: SLF001

    assert calls["n"] == 1
    assert fg1 is fg2
    assert fs1 is fs2
    assert fd1 is fd2


def test_get_profiles_uses_override_and_does_not_cache() -> None:
    """Tests that a profiles override is returned directly and bypasses
    profile building."""
    dummy = {"dark_matter": object(), "gas": object(), "stars": object()}
    bhm = BaryonHaloModel(cosmo=_cosmo(), profiles=dummy)

    out1 = bhm._get_profiles(1.0)  # noqa: SLF001
    out2 = bhm._get_profiles(0.5)  # noqa: SLF001

    assert out1 is dummy
    assert out2 is dummy


def test_get_interpolators_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that _get_interpolators caches interpolators per scale factor."""
    calls = {"n": 0}

    def fake_build_profile_interpolators(**kwargs: Any):
        """Mock build_profile_interpolators that returns constant functions."""
        calls["n"] += 1

        def y_dm(M, k):
            shape = np.broadcast(M, k).shape
            return np.zeros(shape, dtype=float) + 1.0

        def y_gas(M, k):
            shape = np.broadcast(M, k).shape
            return np.zeros(shape, dtype=float) + 2.0

        def y_stars(M, k):
            shape = np.broadcast(M, k).shape
            return np.zeros(shape, dtype=float) + 3.0

        return {
            "dark_matter": y_dm,
            "gas": y_gas,
            "stars": y_stars,
        }

    import pyccl.baryons.fedeli14_bhm.baryon_halo_model as mod

    monkeypatch.setattr(
        mod,
        "build_profile_interpolators",
        fake_build_profile_interpolators)

    bhm = BaryonHaloModel(cosmo=_cosmo())

    i1 = bhm._get_interpolators(1.0)  # noqa: SLF001
    i2 = bhm._get_interpolators(1.0)  # noqa: SLF001

    assert calls["n"] == 1
    assert i1 is i2
    assert set(i1.keys()) == {"dark_matter", "gas", "stars"}


def test_pk_components_validates_inputs_before_work() -> None:
    """Tests that pk_components validates (k,a) and fails fast."""
    bhm = BaryonHaloModel(cosmo=_cosmo())

    with pytest.raises(ValueError, match=r"a must be > 0"):
        _ = bhm.pk_components(k=_kgrid(), a=0.0)

    with pytest.raises(ValueError, match=r"k must contain only finite values"):
        _ = bhm.pk_components(k=np.array([0.1, np.nan, 1.0]), a=1.0)

    with pytest.raises(ValueError, match=r"strictly increasing"):
        _ = bhm.pk_components(k=np.array([0.1, 0.1, 1.0]), a=1.0)


def test_pk_components_wires_calculator_and_returns_packet(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests that pk_components wires the calculator, calls ensure_densities,
    and returns a packet."""
    calls = {"ensure": 0}

    def fake_get_interpolators(self: Any, a: float):
        return {"dark_matter": object(), "gas": object(), "stars": object()}

    def fake_get_mass_fractions(self: Any, a: float):
        """Mock mass_fractions that returns constant functions."""
        def f_g(m):
            """Mock f_g that returns constant functions."""
            return np.zeros_like(np.asarray(m, float)) + 0.1

        def f_s(m):
            """Mock f_s that returns constant functions."""
            return np.zeros_like(np.asarray(m, float)) + 0.01

        def f_d(m):
            """Mock f_d that returns constant functions."""
            return np.zeros_like(np.asarray(m, float)) + 0.89

        return f_g, f_s, f_d

    def fake_get_dmo_dm_interpolator(self: Any, a: float):
        return object()

    monkeypatch.setattr(
        BaryonHaloModel,
        "_get_interpolators",
        fake_get_interpolators
    )
    monkeypatch.setattr(
        BaryonHaloModel,
        "_get_mass_fractions",
        fake_get_mass_fractions
    )
    monkeypatch.setattr(
        BaryonHaloModel,
        "_get_dmo_dm_interpolator",
        fake_get_dmo_dm_interpolator
    )

    class FakeCalc:
        def __init__(self, **kwargs: Any) -> None:
            self.k = np.asarray(kwargs["k"], float)

        def ensure_densities(self, **kwargs: Any) -> None:
            calls["ensure"] += 1

        def pk_packet(self, **kwargs: Any) -> dict[str, Any]:
            k = self.k
            return {
                "grid": {"k": k, "a": 1.0},
                "pk_ref": {
                    "pk_lin": np.ones_like(k) * 10.0,
                    "pk_nlin": np.ones_like(k) * 20.0,
                    "pk_dmo": np.ones_like(k) * 30.0,
                },
                "pk": {"total": np.ones_like(k) * 7.0},
                "halo_pairs": {},
                "meta": {},
            }

    import pyccl.baryons.fedeli14_bhm.baryon_halo_model as mod

    monkeypatch.setattr(mod, "FedeliPkCalculator", FakeCalc)

    bhm = BaryonHaloModel(cosmo=_cosmo(), n_m=8)

    out = bhm.pk_components(k=_kgrid(), a=1.0)
    assert isinstance(out, dict)
    assert set(out.keys()) >= {"grid", "pk_ref", "pk"}
    assert np.allclose(out["grid"]["k"], _kgrid())
    assert np.allclose(out["pk"]["total"], 7.0)
    assert calls["ensure"] == 1


def test_pk_total_is_packet_total_view(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests that pk_total returns pk_components()['pk']['total']."""

    def fake_pk_components(
        self,
        *,
        k: np.ndarray,
        a: float
    ) -> dict[str, Any]:
        """Mock pk_components that returns a packet with pk_ref=pref."""
        k = np.asarray(k, float)
        return {"grid": {"k": k, "a": float(a)},
                "pk_ref": {}, "pk": {"total": k * 0.0 + 7.0}}

    monkeypatch.setattr(
        BaryonHaloModel, "pk_components", fake_pk_components)

    bhm = BaryonHaloModel(cosmo=_cosmo())
    out = bhm.pk_total(k=_kgrid(), a=1.0)
    assert isinstance(out, np.ndarray)
    assert out.shape == _kgrid().shape
    assert np.all(out == 7.0)


def test_pk_total_dmo_wires_calculator_and_returns_array(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests that pk_total_dmo wires the calculator and returns its DMO
    halo-model baseline."""

    def fake_get_interpolators(self: Any, a: float):
        return {"dark_matter": object(), "gas": object(), "stars": object()}

    def fake_get_dmo_dm_interpolator(self: Any, a: float):
        return object()

    monkeypatch.setattr(
        BaryonHaloModel,
        "_get_interpolators",
        fake_get_interpolators
    )
    monkeypatch.setattr(
        BaryonHaloModel,
        "_get_dmo_dm_interpolator",
        fake_get_dmo_dm_interpolator
    )

    class FakeCalc:
        def __init__(self, **kwargs: Any) -> None:
            self.k = np.asarray(kwargs["k"], float)

        def pk_total_dmo(self, **kwargs: Any) -> np.ndarray:
            return np.ones_like(self.k) * 2.0

    import pyccl.baryons.fedeli14_bhm.baryon_halo_model as mod

    monkeypatch.setattr(mod, "FedeliPkCalculator", FakeCalc)
    monkeypatch.setattr(ccl, "rho_x", lambda cosmo, a, what: 123.0)

    bhm = BaryonHaloModel(cosmo=_cosmo(), n_m=8)

    out = bhm.pk_total_dmo(k=_kgrid(), a=1.0)
    assert isinstance(out, np.ndarray)
    assert out.shape == _kgrid().shape
    assert np.all(out == 2.0)


def test_boost_validates_pk_ref_and_returns_ratio(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests that boost validates pk_ref and returns pk_bar/pk_ref."""
    k = _kgrid()
    pb = np.array([2.0, 4.0, 6.0], dtype=float)
    pref = np.array([1.0, 2.0, 3.0], dtype=float)

    def fake_pk_components(
        self,
        *,
        k: np.ndarray,
        a: float
    ) -> dict[str, Any]:
        """Fake pk_components that returns a packet with pk_ref=pref."""
        k = np.asarray(k, float)
        return {
            "grid": {"k": k, "a": float(a)},
            "pk": {"total": pb},
            "pk_ref": {"pk_dmo": pref, "pk_nlin": pref, "pk_lin": pref},
        }

    monkeypatch.setattr(
        BaryonHaloModel, "pk_components", fake_pk_components)

    bhm = BaryonHaloModel(cosmo=_cosmo())

    b1 = bhm.boost(k=k, a=1.0, pk_ref="pk_dmo")
    assert np.allclose(b1, pb / pref)

    b2 = bhm.boost(k=k, a=1.0, pk_ref="pk_nlin")
    assert np.allclose(b2, pb / pref)

    b3 = bhm.boost(k=k, a=1.0, pk_ref="pk_lin")
    assert np.allclose(b3, pb / pref)

    with pytest.raises(ValueError, match=r"pk_ref must be one of"):
        _ = bhm.boost(k=k, a=1.0, pk_ref="nope")


def test_validate_ranges_vs_interp_raises_when_ranges_exceed_grid() -> None:
    """Tests that validate_ranges_vs_interp raises when mass_ranges exceed the
    interpolation grid."""
    cosmo = _cosmo()

    # interpolation mass grid only spans [1e10, 1e12]
    mgrid = np.logspace(10, 12, 8)
    kgrid = np.logspace(-3, 0, 4)

    interp = {
        "dark_matter": {"mass": mgrid, "k": kgrid},
        "gas": {"mass": mgrid, "k": kgrid},
        "stars": {"mass": mgrid, "k": kgrid},
    }

    # mass_ranges exceed the grid max -> should raise
    ranges = {
        "dark_matter": {"min": 1e10, "max": 1e13},
        "gas": {"min": 1e10, "max": 1e12},
        "stars": {"min": 1e10, "max": 1e12},
    }

    with pytest.raises(ValueError, match=r"exceed interpolation mass grid"):
        _ = BaryonHaloModel(
            cosmo=cosmo,
            interpolation_grid=interp,
            mass_ranges=ranges)


def test_halo_profiles_space_real_fourier_and_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that halo_profiles validates space and returns real or fourier."""
    bhm = BaryonHaloModel(cosmo=_cosmo())

    monkeypatch.setattr(
        BaryonHaloModel,
        "_get_profiles",
        lambda self, a: {"x": "real"})
    monkeypatch.setattr(
        BaryonHaloModel,
        "_get_interpolators",
        lambda self, a: {"x": "fourier"})

    out_real = bhm.halo_profiles(a=1.0, space="real")
    assert out_real == {"x": "real"}

    out_fourier = bhm.halo_profiles(a=1.0, space="fourier")
    assert out_fourier == {"x": "fourier"}

    with pytest.raises(ValueError, match=r"space must be"):
        _ = bhm.halo_profiles(a=1.0, space="nope")


def test_mass_fractions_wrapper_calls_get_mass_fractions_and_validates_a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that mass_fractions validates a and calls _get_mass_fractions."""
    bhm = BaryonHaloModel(cosmo=_cosmo())

    sentinel = (object(), object(), object())
    monkeypatch.setattr(
        BaryonHaloModel,
        "_get_mass_fractions",
        lambda self, a: sentinel)

    out = bhm.mass_fractions(a=1.0)
    assert out is sentinel

    with pytest.raises(ValueError, match=r"a must be > 0"):
        _ = bhm.mass_fractions(a=0.0)


def test_halo_radius_physical_comoving_and_invalid_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that halo_radius validates frame and returns physical or
    comoving."""
    bhm = BaryonHaloModel(cosmo=_cosmo())

    class _FakeMassDef:
        def get_radius(self, cosmo, M, a):
            return np.asarray(M, float) * 0 + (2.0 * float(a))

    bhm.mass_def = _FakeMassDef()

    r_phys = bhm.halo_radius(M=1e14, a=0.5, frame="physical")
    assert np.allclose(r_phys, 1.0)  # 2*a = 1

    r_com = bhm.halo_radius(M=np.array([1.0, 2.0]), a=0.5, frame="comoving")
    assert np.allclose(r_com, 2.0)  # (2*a)/a = 2

    with pytest.raises(ValueError, match=r"frame must be"):
        _ = bhm.halo_radius(M=1e14, a=1.0, frame="nope")


def test_get_dmo_dm_interpolator_is_cached_and_builds_dm_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that _get_dmo_dm_interpolator caches interpolators per
    scale factor."""
    calls = {"n": 0, "components": None}

    def fake_build_profile_interpolators(**kwargs: Any):
        calls["n"] += 1
        calls["components"] = kwargs.get("components", None)
        # must return mapping containing "dark_matter"
        return {"dark_matter": object()}

    import pyccl.baryons.fedeli14_bhm.baryon_halo_model as mod
    monkeypatch.setattr(mod,
                        "build_profile_interpolators",
                        fake_build_profile_interpolators)

    # nfw_profile isn't important here; but keep it cheap/stable
    monkeypatch.setattr(mod, "nfw_profile", lambda **kwargs: object())

    bhm = BaryonHaloModel(cosmo=_cosmo())

    y1 = bhm._get_dmo_dm_interpolator(1.0)  # noqa: SLF001
    y2 = bhm._get_dmo_dm_interpolator(1.0)  # noqa: SLF001

    assert y1 is y2
    assert calls["n"] == 1
    assert calls["components"] == ("dark_matter",)


def test_get_interpolators_with_concentration_does_not_touch_main_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that _get_interpolators with concentration does not touch the
    main cache and passes the override to nfw_profile."""
    import pyccl.baryons.fedeli14_bhm.baryon_halo_model as mod

    bhm = BaryonHaloModel(cosmo=_cosmo())
    a = 1.0

    bhm._interp_cache[a] = {"sentinel": True}  # noqa: SLF001

    seen = {"conc": None}

    def fake_nfw_profile(*, mass_def, concentration):
        seen["conc"] = concentration
        return object()

    monkeypatch.setattr(mod,
                        "nfw_profile",
                        fake_nfw_profile)
    monkeypatch.setattr(mod,
                        "GasHaloProfile",
                        lambda **kwargs: object())
    monkeypatch.setattr(mod,
                        "StellarHaloProfile",
                        lambda **kwargs: object())
    monkeypatch.setattr(mod,
                        "build_profile_interpolators",
                        lambda **kwargs: {"dark_matter": object(),
                                          "gas": object(),
                                          "stars": object()}
                        )

    conc_override = object()
    interps = bhm._get_interpolators_with_concentration(  # noqa: SLF001
        a, concentration_override=conc_override
    )

    assert set(interps.keys()) == {"dark_matter", "gas", "stars"}
    assert seen["conc"] is conc_override
    assert bhm._interp_cache[a] == {"sentinel": True}  # noqa: SLF001


def test_pk_dmo_terms_wires_calc_pk_halo_pair_and_returns_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that pk_dmo_terms wires the calculator and returns its DMO
    halo-model baseline terms, and that the total is the sum of the 1- and
    2-halo terms."""
    import pyccl.baryons.fedeli14_bhm.baryon_halo_model as mod

    k = _kgrid()
    P1 = np.array([1.0, 2.0, 3.0])
    P2 = np.array([10.0, 20.0, 30.0])

    def fake_get_interps_with_conc(self, a, *, concentration_override):
        return {"dark_matter": object(), "gas": object(), "stars": object()}

    monkeypatch.setattr(
        BaryonHaloModel,
        "_get_interpolators_with_concentration",
        fake_get_interps_with_conc,
    )
    monkeypatch.setattr(ccl, "rho_x", lambda cosmo, a, what: 5.0)

    class FakeCalc:
        def __init__(self, **kwargs: Any) -> None:
            self.k = np.asarray(kwargs["k"], float)

        def pk_halo_pair(self, *, comp1, comp2, rho1, rho2):
            assert comp1 == "dark_matter"
            assert comp2 == "dark_matter"
            return P1, P2

    monkeypatch.setattr(mod, "FedeliPkCalculator", FakeCalc)

    bhm = BaryonHaloModel(cosmo=_cosmo(), n_m=8)

    out1, out2, outt = bhm.pk_dmo_terms(k=k,
                                        a=1.0,
                                        concentration_override=object())
    assert np.allclose(out1, P1)
    assert np.allclose(out2, P2)
    assert np.allclose(outt, P1 + P2)
