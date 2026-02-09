"""Unit tests for `pyccl.baryons.fedeli14_bhm.power_spectra`."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest

import pyccl as ccl

import pyccl.baryons.fedeli14_bhm.power_spectra as ps
from pyccl.baryons.fedeli14_bhm.power_spectra import (
    FedeliPkCalculator,
    _dndm_from_dndlog10m,
    _k_cache_key,
)


def _cosmo() -> ccl.Cosmology:
    """Return a small valid CCL cosmology."""
    return ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.965, sigma8=0.8)


def _kgrid() -> np.ndarray:
    """Return a tiny strictly increasing positive k grid."""
    return np.array([0.1, 1.0, 10.0], dtype=float)


def _mass_ranges(
    mmin: float = 1.0e13, mmax: float = 1.0e15
) -> dict[str, dict[str, float]]:
    """Return valid per-component mass ranges."""
    return {
        "dark_matter": {"min": float(mmin), "max": float(mmax)},
        "gas": {"min": float(mmin), "max": float(mmax)},
        "stars": {"min": float(mmin), "max": float(mmax)},
    }


def _gas_params(Fg: float = 0.7, bd: float = 1.2) -> dict[str, float]:
    """Return valid gas mixing parameters."""
    return {"Fg": float(Fg), "bd": float(bd)}


def _hmf_const(
    val: float = 1.0
) -> Callable[[Any, np.ndarray, float], np.ndarray]:
    """Return dn/dlog10M = constant on the provided mass grid."""
    v = float(val)

    def hmf(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        """Halo mass function, constant on the provided mass grid."""
        _ = a
        M = np.asarray(M, dtype=float)
        return np.full_like(M, v, dtype=float)

    return hmf


def _hb_const(
    val: float = 1.0
) -> Callable[[Any, np.ndarray, float], np.ndarray]:
    """Return b(M) = constant on the provided mass grid."""
    v = float(val)

    def hb(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        _ = a
        M = np.asarray(M, dtype=float)
        return np.full_like(M, v, dtype=float)

    return hb


def _y_const(
    val: float = 1.0
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return y(M,k)=const with the correct broadcasted (nM,nk) shape."""
    v = float(val)

    def y(M: np.ndarray, k: np.ndarray) -> np.ndarray:
        M = np.asarray(M, dtype=float)
        k = np.asarray(k, dtype=float)

        # The production code calls y(M[:,None], k[None,:]) i.e. both 2D.
        # But keep this robust to 1D callers too.
        if M.ndim == 1 and k.ndim == 1:
            return np.full((M.size, k.size), v, dtype=float)

        MM, KK = np.broadcast_arrays(M, k)
        return np.full(MM.shape, v, dtype=float)

    return y


def _profiles() -> dict[str, Any]:
    """Return a valid mapping of u_over_m evaluators."""
    return {"dark_matter": _y_const(1.0),
            "gas": _y_const(1.0),
            "stars": _y_const(1.0)}


def _densities_full(
    *,
    matter: float = 1.0,
    dark_matter: float = 1.0,
    gas: float = 1.0,
    stars: float = 1.0,
) -> dict[str, float]:
    """Return a complete density dict for packet building."""
    return {
        "matter": float(matter),
        "dark_matter": float(dark_matter),
        "gas": float(gas),
        "stars": float(stars),
    }


def _make_calc(
    *,
    cosmo: Any | None = None,
    a: float = 1.0,
    k: np.ndarray | None = None,
    profiles_u_over_m: dict[str, Any] | None = None,
    dmo_dm_u_over_m: Any | None = None,
    mass_function: Any | None = None,
    halo_bias: Any | None = None,
    mass_ranges: dict[str, dict[str, float]] | None = None,
    densities: dict[str, float] | None = None,
    gas_params: dict[str, float] | None = None,
    n_m: int = 16,
) -> FedeliPkCalculator:
    """Build a minimal valid FedeliPkCalculator."""
    use_cosmo = _cosmo() if cosmo is None else cosmo
    use_k = _kgrid() if k is None else k
    use_prof = _profiles() if profiles_u_over_m is None else profiles_u_over_m
    use_mf = _hmf_const(1.0) if mass_function is None else mass_function
    use_hb = _hb_const(1.0) if halo_bias is None else halo_bias
    use_ranges = _mass_ranges() if mass_ranges is None else mass_ranges
    use_rho = _densities_full() if densities is None else densities
    use_gas = _gas_params() if gas_params is None else gas_params

    return FedeliPkCalculator(
        cosmo=use_cosmo,
        a=a,
        k=use_k,
        profiles_u_over_m=use_prof,
        dmo_dm_u_over_m=dmo_dm_u_over_m,
        mass_function=use_mf,
        halo_bias=use_hb,
        mass_ranges=use_ranges,
        densities=use_rho,
        gas_params=use_gas,
        n_m=n_m,
    )


def test_dndm_from_dndlog10m_validates_shapes_and_positive_m() -> None:
    """Tests dn/dM conversion checks shapes and enforces m > 0."""
    m = np.array([1.0, 10.0, 100.0], dtype=float)
    d = np.ones_like(m)

    out = _dndm_from_dndlog10m(d, m)
    assert out.shape == m.shape
    assert np.all(np.isfinite(out))
    assert np.all(out > 0.0)

    with pytest.raises(ValueError):
        _ = _dndm_from_dndlog10m(np.ones(2), m)

    with pytest.raises(ValueError):
        _ = _dndm_from_dndlog10m(d,
                                 np.array([1.0, 0.0, 2.0],
                                          dtype=float))


def test_init_validates_k_grid_shape_monotonicity_and_sign() -> None:
    """Tests that k must be 1D, finite, strictly increasing, and positive."""
    with pytest.raises(Exception):
        _ = _make_calc(k=np.ones((2, 2)))  # type: ignore[arg-type]

    with pytest.raises(Exception):
        _ = _make_calc(k=np.array([], dtype=float))

    with pytest.raises(Exception):
        _ = _make_calc(k=np.array([0.1, 0.1, 1.0], dtype=float))

    with pytest.raises(Exception):
        _ = _make_calc(k=np.array([0.1, np.nan, 1.0], dtype=float))

    with pytest.raises(Exception):
        _ = _make_calc(k=np.array([0.0, 0.1, 1.0], dtype=float))


def test_init_validates_profiles_mapping_and_components() -> None:
    """Tests that profiles_u_over_m must be a mapping with DM/gas/stars
    callables."""
    with pytest.raises(Exception):
        _ = _make_calc(profiles_u_over_m=123)  # type: ignore[arg-type]

    with pytest.raises(Exception):
        _ = _make_calc(
            profiles_u_over_m={"dark_matter": _y_const(1.0),
                               "gas": _y_const(1.0)}
        )

    bad = {"dark_matter": _y_const(1.0), "gas": 123, "stars": _y_const(1.0)}
    with pytest.raises(Exception):
        _ = _make_calc(profiles_u_over_m=bad)


def test_init_validates_mass_ranges_structure_and_values() -> None:
    """Tests that mass_ranges must be present for all components with
    valid min/max."""
    with pytest.raises(Exception):
        _ = _make_calc(mass_ranges=123)  # type: ignore[arg-type]

    bad_missing = {"dark_matter": {"min": 1.0, "max": 2.0},
                   "gas": {"min": 1.0, "max": 2.0}}
    with pytest.raises(Exception):
        _ = _make_calc(mass_ranges=bad_missing)  # type: ignore[arg-type]

    bad_keys = _mass_ranges()
    bad_keys["gas"] = {"min": 1.0}  # missing max
    with pytest.raises(Exception):
        _ = _make_calc(mass_ranges=bad_keys)

    bad_vals = _mass_ranges()
    bad_vals["stars"] = {"min": 1.0e15, "max": 1.0e14}
    with pytest.raises(Exception):
        _ = _make_calc(mass_ranges=bad_vals)


def test_init_validates_gas_params_and_densities_types() -> None:
    """Tests that gas_params has Fg/bd and densities is a mapping of floats."""
    with pytest.raises(Exception):
        _ = _make_calc(gas_params={"Fg": 0.5})  # type: ignore[arg-type]

    with pytest.raises(Exception):
        _ = _make_calc(gas_params=_gas_params(Fg=1.5, bd=1.0))

    with pytest.raises(Exception):
        _ = _make_calc(gas_params=_gas_params(Fg=0.5, bd=float("nan")))

    with pytest.raises(Exception):
        _ = _make_calc(densities=123)  # type: ignore[arg-type]


def test_y_grid_mm_requires_u_over_m_return_shape() -> None:
    """Tests that u_over_m must return (nM,nk) for the internal
    (M[:,None],k[None,:]) call."""
    calc = _make_calc(n_m=16)
    mmin, mmax = 1.0e13, 1.0e14

    y = calc._y_grid_mm("dark_matter", mmin, mmax)  # noqa: SLF001
    assert y.shape == (calc.n_m, calc.k.size)
    assert np.all(np.isfinite(y))

    def y_bad(M: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Returns (nM,) instead of (nM,nk)."""
        _ = k
        M = np.asarray(M, dtype=float)
        return np.ones((M.shape[0], 1), dtype=float)

    with pytest.raises(ValueError, match=r"u_over_m"):
        _ = calc._y_grid_mm(  # noqa: SLF001
            "dark_matter", mmin, mmax, y_fn=y_bad, cache_tag="bad"
        )


def test_I2_and_Ib_validate_components_and_profile_shapes_and_cache() -> None:
    """Tests that _I2_vec/_Ib_vec are finite, have shape (nk,), and cache hits
    work."""
    calc = _make_calc(n_m=16)
    mmin, mmax = 1.0e13, 1.0e14

    Ib1 = calc._Ib_vec("gas", mmin, mmax)  # noqa: SLF001
    Ib2 = calc._Ib_vec("gas", mmin, mmax)  # noqa: SLF001
    assert Ib1 is Ib2
    assert Ib1.shape == (calc.k.size,)
    assert np.all(np.isfinite(Ib1))

    I21 = calc._I2_vec("dark_matter", "stars", mmin, mmax)
    I22 = calc._I2_vec("dark_matter", "stars", mmin, mmax)
    assert I21 is I22
    assert I21.shape == (calc.k.size,)
    assert np.all(np.isfinite(I21))


def test_pk_gas_auto_dm_gas_star_gas_return_finite_arrays() -> None:
    """Tests that the convenience spectra return finite arrays with shape
    (nk,)."""
    rho = _densities_full(matter=1.0, dark_matter=1.0, gas=1.0, stars=1.0)
    calc = _make_calc(densities=rho,
                      gas_params=_gas_params(Fg=0.7, bd=1.2),
                      n_m=32)

    Pg = calc.pk_gas_auto()
    Pdg = calc.pk_dm_gas()
    Psg = calc.pk_star_gas()

    for arr in (Pg, Pdg, Psg):
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (calc.k.size,)
        assert np.all(np.isfinite(arr))


def test_pk_total_dmo_adds_1h_and_2h_and_caches() -> None:
    """Tests that pk_total_dmo returns a finite (nk,) array and caches when
    requested."""
    rho = _densities_full(matter=1.0, dark_matter=1.0, gas=1.0, stars=1.0)
    calc = _make_calc(densities=rho, n_m=16)

    p1 = calc.pk_total_dmo(use_cache=True)
    p2 = calc.pk_total_dmo(use_cache=True)

    assert p1.shape == (calc.k.size,)
    assert np.all(np.isfinite(p1))
    assert np.allclose(p1, p2)
    assert calc._pk_dmo is not None  # noqa: SLF001


def test_boost_hm_over_hm_is_ratio_of_total_to_dmo() -> None:
    """Tests that boost_hm_over_hm equals pk_total / pk_total_dmo."""
    rho = _densities_full(matter=1.0, dark_matter=1.0, gas=1.0, stars=1.0)
    calc = _make_calc(densities=rho,
                      gas_params=_gas_params(Fg=0.7, bd=1.2),
                      n_m=16)

    b = calc.boost_hm_over_hm()
    assert b.shape == (calc.k.size,)
    assert np.all(np.isfinite(b))

    pb = calc.pk_total()
    pd = calc.pk_total_dmo(use_cache=True)
    assert np.allclose(b, pb / pd, rtol=0.0, atol=0.0)


def test_pk_packet_adds_commutative_aliases_for_pairs_and_weighted_pairs(

) -> None:
    """Tests that pk_packet adds dm_gas==gas_dm etc. and weighted aliases."""
    rho = _densities_full(matter=1.0, dark_matter=1.0, gas=1.0, stars=1.0)
    calc = _make_calc(densities=rho,
                      gas_params=_gas_params(Fg=0.7, bd=1.2),
                      n_m=16)

    pkt = calc.pk_packet(use_cache=False)
    pk = pkt["pk"]

    assert np.allclose(pk["dm_gas"], pk["gas_dm"])
    assert np.allclose(pk["dm_stars"], pk["stars_dm"])
    assert np.allclose(pk["stars_gas"], pk["gas_stars"])

    assert np.allclose(pk["w_dm_gas"], pk["w_gas_dm"])
    assert np.allclose(pk["w_dm_stars"], pk["w_stars_dm"])
    assert np.allclose(pk["w_stars_gas"], pk["w_gas_stars"])


def test_y_grid_mm_cache_depends_on_y_fn() -> None:
    """Tests that _y_grid_mm cache keys include y_fn (or otherwise respect it),
    so changing y_fn changes the returned cached result."""
    calc = _make_calc(n_m=8)
    mmin, mmax = 1e13, 1e14

    y0 = calc._y_grid_mm("dark_matter", mmin, mmax)  # noqa: SLF001
    assert y0.shape == (calc.n_m, calc.k.size)

    def y_alt(M: np.ndarray, k: np.ndarray) -> np.ndarray:
        return 2.0 * np.ones((M.shape[0], k.shape[1]), dtype=float)

    y1 = calc._y_grid_mm("dark_matter", mmin, mmax, y_fn=y_alt)
    assert y1.shape == (calc.n_m, calc.k.size)
    assert np.allclose(y1, 2.0)

    # Optional: calling again with the same y_fn should return the same values.
    y1b = calc._y_grid_mm("dark_matter", mmin, mmax, y_fn=y_alt)
    assert np.allclose(y1b, 2.0)


def test_ensure_densities_invalidates_final_caches_when_changed() -> None:
    """Tests that ensure_densities clears cached outputs when it fills missing
    density entries."""
    calc = _make_calc(n_m=16)

    # Pre-fill caches as if we computed something already.
    calc._pk_dmo = np.ones(calc.k.size)  # noqa: SLF001
    calc._pk_packet_cache = {"sentinel": True}  # noqa: SLF001

    # Remove gas/stars so ensure_densities must fill them.
    calc.rho.pop("gas", None)
    calc.rho.pop("stars", None)

    calc.ensure_densities(
        f_gas=lambda m: 0.1 * np.ones_like(m),
        f_star=lambda m: 0.02 * np.ones_like(m),
        mmin=1e10,
        mmax=1e16,
    )

    assert calc._pk_dmo is None  # noqa: SLF001
    assert calc._pk_packet_cache is None  # noqa: SLF001


def test_pk_packet_adds_commutative_aliases() -> None:
    """Tests that pk_packet adds commutative alias keys for cross-terms and
    weighted cross-terms."""
    calc = _make_calc(n_m=32)

    packet = calc.pk_packet(use_cache=False)
    pk = packet["pk"]

    # unweighted aliases
    assert np.allclose(pk["dm_gas"], pk["gas_dm"])
    assert np.allclose(pk["dm_stars"], pk["stars_dm"])
    assert np.allclose(pk["stars_gas"], pk["gas_stars"])

    # weighted aliases
    assert np.allclose(pk["w_dm_gas"], pk["w_gas_dm"])
    assert np.allclose(pk["w_dm_stars"], pk["w_stars_dm"])
    assert np.allclose(pk["w_stars_gas"], pk["w_gas_stars"])


def test_pk_total_matches_packet_total() -> None:
    """Tests that pk_total is the packet total spectrum and uses cached
    packet."""
    rho = _densities_full(matter=1.0, dark_matter=1.0, gas=1.0, stars=1.0)
    calc = _make_calc(
        densities=rho, gas_params=_gas_params(Fg=0.7, bd=1.2), n_m=32)

    p0 = calc.pk_total()
    p1 = calc.pk_packet(use_cache=True)["pk"]["total"]
    assert np.allclose(p0, p1)


def test_pk_halo_pair_raises_for_non_overlapping_mass_ranges() -> None:
    """Tests that pk_halo_pair errors out when the two components have no
    overlap in mass range."""
    calc = _make_calc(n_m=16)

    calc.mass_ranges["dark_matter"] = {"min": 1e10, "max": 1e11}
    calc.mass_ranges["stars"] = {"min": 1e12, "max": 1e13}

    with pytest.raises(ValueError, match=r"No overlap mass range"):
        calc.pk_halo_pair(
            comp1="dark_matter",
            comp2="stars",
            rho1=1.0,
            rho2=1.0,
        )


def test_k_cache_key_rounding():
    """Tests that k_cache_key rounds to the nearest 1e-3."""
    assert _k_cache_key(1.23456789, ndp=3) == 1.235


def test_fedeli_pk_calculator_requires_nm_ge_2() -> None:
    """Tests that FedeliPkCalculator requires n_m >= 2."""
    with pytest.raises(ValueError, match=r"n_m must be >= 2"):
        _ = _make_calc(n_m=1)


def test_fedeli_pk_calculator_requires_nm_ge_2_direct() -> None:
    """Tests that FedeliPkCalculator requires n_m >= 2."""
    with pytest.raises(ValueError, match=r"n_m must be >= 2"):
        _ = FedeliPkCalculator(
            cosmo=_cosmo(),
            a=1.0,
            k=_kgrid(),
            profiles_u_over_m=_profiles(),
            mass_function=_hmf_const(1.0),
            halo_bias=_hb_const(1.0),
            mass_ranges=_mass_ranges(),
            densities=_densities_full(),
            gas_params=_gas_params(),
            n_m=1,
        )


def test_rho_from_fraction_validations_and_cache_hit() -> None:
    """Tests that rho_from_fraction validates inputs and respects the cache."""
    calc = _make_calc(n_m=8)

    with pytest.raises(TypeError, match=r"callable"):
        _ = calc.rho_from_fraction(  # type: ignore[arg-type]
            f_of_m=123, mmin=1e13, mmax=1e14
        )

    with pytest.raises(ValueError, match=r"Invalid mass range"):
        _ = calc.rho_from_fraction(
            f_of_m=lambda m: np.ones_like(m), mmin=1e14, mmax=1e13
        )

    with pytest.raises(ValueError, match=r"same shape"):
        _ = calc.rho_from_fraction(
            f_of_m=lambda m: np.ones(m.size + 1), mmin=1e13, mmax=1e14, n_m=8
        )

    def f_nan(m: np.ndarray) -> np.ndarray:
        out = np.ones_like(m, dtype=float)
        out[0] = np.nan
        return out

    with pytest.raises(ValueError, match=r"must be finite"):
        _ = calc.rho_from_fraction(f_of_m=f_nan, mmin=1e13, mmax=1e14, n_m=8)

    key = ("gas", 1e13, 1e14, 8)
    calc._rho_cache[key] = 123.0  # noqa: SLF001
    out = calc.rho_from_fraction(
        f_of_m=lambda m: np.ones_like(m),
        mmin=1e13,
        mmax=1e14,
        n_m=8,
        cache_key="gas",
    )
    assert out == 123.0


def test_ensure_densities_requires_cosmo_keys_and_fills_matter_and_dm(
    monkeypatch,
) -> None:
    """Tests that ensure_densities requires cosmo keys and fills matter and
    dm."""
    calc = _make_calc(cosmo={"Omega_m": 0.3}, densities={})
    with pytest.raises(KeyError, match=r"cosmo must provide"):
        calc.ensure_densities(
            f_gas=lambda m: 0.1 * np.ones_like(m),
            f_star=lambda m: 0.02 * np.ones_like(m),
            mmin=1e13,
            mmax=1e14,
            n_m=8,
        )

    monkeypatch.setattr(ps.ccl, "rho_x", lambda cosmo, a, what: 10.0)

    calc = _make_calc(cosmo={"Omega_c": 0.25, "Omega_m": 0.30}, densities={})
    calc.ensure_densities(
        f_gas=lambda m: 0.1 * np.ones_like(m),
        f_star=lambda m: 0.02 * np.ones_like(m),
        mmin=1e13,
        mmax=1e14,
        n_m=8,
    )

    assert calc.rho["matter"] == 10.0
    assert np.isclose(calc.rho["dark_matter"], 10.0 * (0.25 / 0.30))
    assert "gas" in calc.rho and "stars" in calc.rho


def test_linpk_caches_single_k(monkeypatch) -> None:
    """Tests that linpk caches the result for a single k."""
    calc = _make_calc()

    calls = {"n": 0}

    def fake_linear(cosmo, k, a):
        _, _, _ = cosmo, k, a
        calls["n"] += 1
        return 3.0

    monkeypatch.setattr(ps.ccl, "linear_matter_power", fake_linear)

    p1 = calc.linpk(0.123456)
    p2 = calc.linpk(0.123456)  # this should hit cache

    assert p1 == 3.0 and p2 == 3.0
    assert calls["n"] == 1


def test_P_lin_falls_back_to_scalar_loop(monkeypatch) -> None:
    """Tests that P_lin falls back to scalar loop when linear_matter_power
    fails on vector k input."""
    calc = _make_calc()

    def fake_linear(cosmo, k, a):
        """Raise an error if k is a vector."""
        _, _ = cosmo, a
        if np.ndim(k) > 0:
            raise RuntimeError("no vector")
        return 5.0

    monkeypatch.setattr(ps.ccl, "linear_matter_power", fake_linear)

    out = calc.P_lin()
    assert out.shape == (calc.k.size,)
    assert np.allclose(out, 5.0)


def test_mass_grid_raises_for_invalid_range() -> None:
    """Tests that mass_grid raises for invalid mass ranges."""
    calc = _make_calc()
    with pytest.raises(ValueError, match=r"Invalid mass range"):
        _ = calc._mass_grid(1e14, 1e13)  # noqa: SLF001


def test_dndm_raises_if_mass_function_wrong_shape() -> None:
    """Tests that _dndm raises if mass_function returns wrong shape."""
    def hmf_bad(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        """Mass function that returns wrong shape."""
        _ = a
        return np.ones(M.size + 1, dtype=float)

    calc = _make_calc(mass_function=hmf_bad)

    with pytest.raises(ValueError, match=r"mass_function must return"):
        _ = calc._dndm(1e13, 1e14)  # noqa: SLF001


def test_bias_raises_if_halo_bias_wrong_shape() -> None:
    """Tests that _bias raises if halo_bias returns wrong shape."""
    def hb_bad(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        """Bias function that returns wrong shape."""
        _ = a
        return np.ones(M.size + 1, dtype=float)

    calc = _make_calc(halo_bias=hb_bad)

    with pytest.raises(ValueError, match=r"halo_bias must return"):
        _ = calc._bias(1e13, 1e14)  # noqa: SLF001


def test_Ib_and_I2_internal_shape_guards(monkeypatch) -> None:
    """Tests that _Ib_vec and _I2_vec raise if their inputs have wrong shape."""
    calc = _make_calc()

    monkeypatch.setattr(ps, "_trapz_compat", lambda y, x, axis=0: np.zeros(2))

    with pytest.raises(ValueError, match=r"Ib has wrong shape"):
        _ = calc._Ib_vec("gas", 1e13, 1e14)  # noqa: SLF001

    with pytest.raises(ValueError, match=r"I2 has wrong shape"):
        _ = calc._I2_vec("dark_matter", "stars", 1e13, 1e14)  # noqa: SLF001


def test_pk_halo_pair_validates_rhos_and_components() -> None:
    """Tests that pk_halo_pair validates rhos and components."""
    calc = _make_calc()

    with pytest.raises(ValueError, match=r"rho1 and rho2 must be > 0"):
        calc.pk_halo_pair(comp1="gas", comp2="gas", rho1=0.0, rho2=1.0)

    with pytest.raises(KeyError, match=r"mass_ranges"):
        calc.pk_halo_pair(comp1="not_a_comp", comp2="gas", rho1=1.0, rho2=1.0)

    calc2 = _make_calc()
    calc2.y.pop("gas")
    with pytest.raises(KeyError, match=r"profiles"):
        calc2.pk_halo_pair(comp1="gas", comp2="stars", rho1=1.0, rho2=1.0)


def test_pair_mixed_cross_packet_validates_comp() -> None:
    """Tests that pair_mixed_cross_packet validates comp."""
    calc = _make_calc()
    with pytest.raises(ValueError, match=r"comp must be"):
        _ = calc.pair_mixed_cross_packet(comp="gas")


def test_pk_packet_requires_all_densities() -> None:
    """Tests that pk_packet requires all densities."""
    calc = _make_calc(densities={"matter": 1.0, "dark_matter": 1.0, "gas": 1.0})
    with pytest.raises(KeyError, match=r"Missing density"):
        _ = calc.pk_packet(use_cache=False)


def test_pk_total_dmo_fills_matter_density_if_missing(monkeypatch) -> None:
    """Tests that pk_total_dmo fills matter density if missing."""
    rho = {"dark_matter": 1.0, "gas": 1.0, "stars": 1.0}
    calc = _make_calc(densities=rho)

    monkeypatch.setattr(ps.ccl, "rho_x", lambda cosmo, a, what: 7.0)

    _ = calc.pk_total_dmo(use_cache=False)
    assert "matter" in calc.rho
    assert calc.rho["matter"] == 7.0


def test_boost_hm_over_hm_raises_if_dmo_nonpositive(monkeypatch) -> None:
    """Tests that boost_hm_over_hm raises if dmo is nonpositive."""
    calc = _make_calc()

    def fake_pk_packet(*, use_cache=True):
        """Fake pk_packet that returns non-positive dmo."""
        _ = use_cache
        return {
            "pk": {"total": np.ones(calc.k.size)},
            "pk_ref": {"pk_dmo": np.zeros(calc.k.size)},  # non-positive
        }

    monkeypatch.setattr(calc, "pk_packet", fake_pk_packet)

    with pytest.raises(ValueError, match=r"non-positive|non-finite"):
        _ = calc.boost_hm_over_hm()
