"""Unit tests for the Fedeli14 P(k) calculator and helpers."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest

import pyccl as ccl

from pyccl.baryons.fedeli14_bhm.power_spectra import (
    FedeliPkCalculator,
    _dndm_from_dndlog10m,
)


def _cosmo() -> ccl.Cosmology:
    """Return a simple, valid CCL cosmology."""
    return ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.965, sigma8=0.8)


def _kgrid() -> np.ndarray:
    """Return a small, strictly increasing positive k grid."""
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


def _densities_empty() -> dict[str, float]:
    """Return an initially empty density dict."""
    return {}


def _densities_full(
    *,
    matter: float = 10.0,
    dark_matter: float = 6.0,
    gas: float = 3.0,
    stars: float = 1.0,
) -> dict[str, float]:
    """Return a minimal complete density dict for running full packets."""
    return {
        "matter": float(matter),
        "dark_matter": float(dark_matter),
        "gas": float(gas),
        "stars": float(stars),
    }


def _gas_params(Fg: float = 0.7, bd: float = 1.0) -> dict[str, float]:
    """Return a valid gas mixing parameter dict."""
    return {"Fg": float(Fg), "bd": float(bd)}


def _hmf_const(
    val: float = 1.0
) -> Callable[[Any, np.ndarray, float], np.ndarray]:
    """Return dn/dlog10M = constant on the input mass grid."""
    v = float(val)

    def hmf(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        M = np.asarray(M, dtype=float)
        return np.full_like(M, v, dtype=float)

    return hmf


def _hb_const(
    val: float = 1.0
) -> Callable[[Any, np.ndarray, float], np.ndarray]:
    """Return b(M) = constant on the input mass grid."""
    v = float(val)

    def hb(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        _ = a
        M = np.asarray(M, dtype=float)
        return np.full_like(M, v, dtype=float)

    return hb


def _y_const(
    val: float = 1.0
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return y(M,k) = constant, broadcasting to (nM, nk)."""
    v = float(val)

    def y(M: np.ndarray, k: np.ndarray) -> np.ndarray:
        M = np.asarray(M, dtype=float)
        k = np.asarray(k, dtype=float)
        M, _ = np.broadcast_arrays(M, k)
        return np.full_like(M, v, dtype=float)

    return y


def _profiles() -> dict[str, Any]:
    """Return a valid profiles_u_over_m mapping with constant evaluators."""
    return {"dark_matter": _y_const(1.0),
            "gas": _y_const(1.0),
            "stars": _y_const(1.0)}


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
    n_m: int = 64,
) -> FedeliPkCalculator:
    """Build a minimal valid FedeliPkCalculator."""
    use_cosmo = _cosmo() if cosmo is None else cosmo
    use_k = _kgrid() if k is None else k
    use_prof = _profiles() if profiles_u_over_m is None else profiles_u_over_m
    use_mf = _hmf_const(1.0) if mass_function is None else mass_function
    use_hb = _hb_const(1.0) if halo_bias is None else halo_bias
    use_ranges = _mass_ranges() if mass_ranges is None else mass_ranges
    use_rho = _densities_empty() if densities is None else densities
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


def test_dndm_from_dndlog10m_validates_shapes_and_positive_masses() -> None:
    """Tests that dn/dM conversion validates shapes and requires m > 0."""
    m = np.array([1.0, 10.0, 100.0], dtype=float)
    d = np.ones_like(m)

    out = _dndm_from_dndlog10m(d, m)
    assert isinstance(out, np.ndarray) and out.shape == m.shape
    assert np.all(np.isfinite(out))
    assert np.all(out > 0.0)

    with pytest.raises(ValueError, match=r"same shape"):
        _ = _dndm_from_dndlog10m(np.ones(2), m)

    with pytest.raises(ValueError, match=r"m must be > 0"):
        _ = _dndm_from_dndlog10m(d, np.array([1.0, 0.0, 2.0],
                                             dtype=float))


def test_init_validates_k_grid_shape_monotonicity_and_sign() -> None:
    """Tests that FedeliPkCalculator validates k as 1D, finite, increasing,
    and > 0."""
    with pytest.raises(ValueError, match=r"k"):
        _ = _make_calc(k=np.ones((2, 2)))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=r"k"):
        _ = _make_calc(k=np.array([], dtype=float))

    with pytest.raises(ValueError, match=r"increasing|strict"):
        _ = _make_calc(k=np.array([0.1, 0.1, 1.0], dtype=float))

    with pytest.raises(ValueError, match=r"finite|nan|increasing|strict"):
        _ = _make_calc(k=np.array([0.1, np.nan, 1.0], dtype=float))

    with pytest.raises(ValueError, match=r"> 0|positive"):
        _ = _make_calc(k=np.array([0.0, 0.1, 1.0], dtype=float))


def test_init_requires_callable_mass_function_and_halo_bias() -> None:
    """Tests that FedeliPkCalculator requires callable mass_function and
    halo_bias."""
    with pytest.raises(TypeError, match=r"mass_function|callable"):
        _ = _make_calc(mass_function=123)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=r"halo_bias|callable"):
        _ = _make_calc(halo_bias=123)  # type: ignore[arg-type]


def test_init_validates_profiles_mapping_and_components() -> None:
    """Tests that FedeliPkCalculator validates profile mapping type, required
     keys, and callability."""
    with pytest.raises(TypeError, match=r"profiles|dict"):
        _ = _make_calc(profiles_u_over_m=123)  # type: ignore[arg-type]

    with pytest.raises(KeyError, match=r"stars|missing"):
        _ = _make_calc(
            profiles_u_over_m={"dark_matter": _y_const(1.0),
                               "gas": _y_const(1.0)}
        )

    bad = {"dark_matter": _y_const(1.0), "gas": 123, "stars": _y_const(1.0)}
    with pytest.raises(TypeError, match=r"callable"):
        _ = _make_calc(profiles_u_over_m=bad)


def test_init_validates_mass_ranges_structure_and_values() -> None:
    """Tests that FedeliPkCalculator validates mass_ranges structure and
    min/max ordering."""
    with pytest.raises(TypeError, match=r"mass_ranges|dict"):
        _ = _make_calc(mass_ranges=123)  # type: ignore[arg-type]

    bad_missing = {"dark_matter": {"min": 1.0, "max": 2.0},
                   "gas": {"min": 1.0, "max": 2.0}}
    with pytest.raises(KeyError, match=r"stars|missing|mass_ranges"):
        _ = _make_calc(mass_ranges=bad_missing)  # type: ignore[arg-type]

    bad_keys = _mass_ranges()
    bad_keys["gas"] = {"min": 1.0}  # missing max
    with pytest.raises(KeyError, match=r"min|max|mass_ranges"):
        _ = _make_calc(mass_ranges=bad_keys)

    bad_vals = _mass_ranges()
    bad_vals["stars"] = {"min": 1.0e15, "max": 1.0e14}
    with pytest.raises(ValueError, match=r"Invalid|range|min|max"):
        _ = _make_calc(mass_ranges=bad_vals)


def test_init_validates_gas_params_and_densities_types() -> None:
    """Tests that FedeliPkCalculator validates gas_params content and
    densities type."""
    with pytest.raises((KeyError, ValueError), match=r"gas_params|Fg|bd"):
        _ = _make_calc(gas_params={"Fg": 0.5})  # type: ignore[arg-type]

    with pytest.raises((KeyError, ValueError), match=r"Fg|gas_params"):
        _ = _make_calc(gas_params=_gas_params(Fg=1.5, bd=1.0))

    with pytest.raises((KeyError, ValueError), match=r"bd|finite|gas_params"):
        _ = _make_calc(gas_params=_gas_params(Fg=0.5, bd=float("nan")))

    with pytest.raises(TypeError, match=r"densit|dict"):
        _ = _make_calc(densities=123)  # type: ignore[arg-type]


def test_init_validates_n_m() -> None:
    """Tests that FedeliPkCalculator validates n_m >= 2."""
    with pytest.raises(ValueError, match=r"n_m"):
        _ = _make_calc(n_m=1)


def test_rho_from_fraction_validates_callable_range_and_shape() -> None:
    """Tests that rho_from_fraction validates f_of_m, mass range, and output
     shape/values."""
    calc = _make_calc()

    with pytest.raises(TypeError, match=r"callable"):
        _ = calc.rho_from_fraction(f_of_m=123, mmin=1.0, mmax=10.0)

    with pytest.raises(ValueError, match=r"Invalid mass range"):
        def f_of_m(m: np.ndarray) -> np.ndarray:
            return np.ones_like(m, dtype=float)

        _ = calc.rho_from_fraction(f_of_m=f_of_m, mmin=0.0, mmax=10.0)

    def bad_shape(m: np.ndarray) -> np.ndarray:
        return np.ones(m.size - 1)

    with pytest.raises(ValueError, match=r"same shape"):
        _ = calc.rho_from_fraction(f_of_m=bad_shape,
                                   mmin=1.0e13, mmax=1.0e14, n_m=16)

    def bad_nan(m: np.ndarray) -> np.ndarray:
        out = np.ones_like(m)
        out[0] = np.nan
        return out

    with pytest.raises(ValueError, match=r"finite"):
        _ = calc.rho_from_fraction(f_of_m=bad_nan,
                                   mmin=1.0e13, mmax=1.0e14, n_m=16)


def test_rho_from_fraction_caches_when_cache_key_given() -> None:
    """Tests that rho_from_fraction caches results when cache_key is
    provided."""
    calls = {"n": 0}

    def hmf_count(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        _ = a
        calls["n"] += 1
        return np.ones_like(np.asarray(M, float))

    calc = _make_calc(mass_function=hmf_count)

    def f(m: np.ndarray) -> np.ndarray:
        return np.ones_like(m)

    r1 = calc.rho_from_fraction(f_of_m=f, mmin=1.0e13, mmax=1.0e14,
                                n_m=32, cache_key="gas")
    r2 = calc.rho_from_fraction(f_of_m=f, mmin=1.0e13, mmax=1.0e14,
                                n_m=32, cache_key="gas")
    assert np.isfinite(r1) and r1 >= 0.0
    assert r2 == r1
    assert calls["n"] == 1


def test_ensure_densities_requires_cosmo_keys_and_fills_missing_entries(

) -> None:
    """Tests that ensure_densities requires Omega_c/Omega_m and fills
    matter/dark_matter/gas/stars if missing."""
    cosmo = _cosmo()
    calc = _make_calc(cosmo=cosmo, densities={})

    def f_g(m: np.ndarray) -> np.ndarray:
        return np.full_like(m, 0.1, dtype=float)

    def f_s(m: np.ndarray) -> np.ndarray:
        return np.full_like(m, 0.01, dtype=float)

    calc.ensure_densities(
        f_gas=f_g, f_star=f_s, mmin=1.0e13, mmax=1.0e14, n_m=64)

    for key in ("matter", "dark_matter", "gas", "stars"):
        assert key in calc.rho
        assert np.isfinite(float(calc.rho[key]))
        assert float(calc.rho[key]) > 0.0

    cosmo_bad = {"Omega_c": 0.25}  # missing Omega_m
    calc2 = _make_calc(cosmo=cosmo_bad, densities={})
    with pytest.raises(KeyError, match=r"Omega_m"):
        calc2.ensure_densities(
            f_gas=f_g, f_star=f_s, mmin=1.0e13, mmax=1.0e14, n_m=16)


def test_ensure_densities_invalidates_final_caches() -> None:
    """Tests that ensure_densities clears pk caches when it changes rho
    entries."""
    cosmo = _cosmo()
    calc = _make_calc(cosmo=cosmo, densities={})
    # seed caches to non-None
    calc._pk_dmo = np.ones_like(calc.k)  # noqa: SLF001
    calc._pk_packet_cache = {"hello": "world"}  # noqa: SLF001

    def f_g(m: np.ndarray) -> np.ndarray:
        return np.full_like(m, 0.1, dtype=float)

    def f_s(m: np.ndarray) -> np.ndarray:
        return np.full_like(m, 0.01, dtype=float)

    calc.ensure_densities(
        f_gas=f_g, f_star=f_s, mmin=1.0e13, mmax=1.0e14, n_m=16)

    assert calc._pk_dmo is None  # noqa: SLF001
    assert calc._pk_packet_cache is None  # noqa: SLF001


def test_linpk_is_cached_and_P_lin_memoizes_vector() -> None:
    """Tests that linpk caches per-k and P_lin memoizes the full vector."""
    calc = _make_calc()
    k0 = float(calc.k[0])

    p1 = calc.linpk(k0)
    p2 = calc.linpk(k0)
    assert p1 == p2
    k0 = float(_kgrid()[0])
    assert any(
        key in calc._linpk_cache  # noqa: SLF001
        for key in (k0, round(k0, 12))
    )

    v1 = calc.P_lin()
    v2 = calc.P_lin()
    assert v1 is v2
    assert isinstance(v1, np.ndarray) and v1.shape == calc.k.shape
    assert np.all(np.isfinite(v1))


def test_mass_grid_dndm_and_bias_are_cached_and_validate_shapes() -> None:
    """Tests that _mass_grid/_dndm/_bias are cached and validate callable
    return shapes."""
    calls = {"hmf": 0, "hb": 0}

    def hmf(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        calls["hmf"] += 1
        return np.ones_like(np.asarray(M, float))

    def hb(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        calls["hb"] += 1
        return np.ones_like(np.asarray(M, float))

    calc = _make_calc(mass_function=hmf, halo_bias=hb, n_m=32)
    mmin, mmax = 1.0e13, 1.0e14

    d1 = calc._dndm(mmin, mmax)  # noqa: SLF001
    b1 = calc._bias(mmin, mmax)  # noqa: SLF001
    d2 = calc._dndm(mmin, mmax)  # noqa: SLF001
    b2 = calc._bias(mmin, mmax)  # noqa: SLF001

    assert d1 is d2
    assert b1 is b2
    assert calls["hmf"] == 1
    assert calls["hb"] == 1

    def hmf_bad(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        return np.ones(np.asarray(M).size - 1)

    calc2 = _make_calc(mass_function=hmf_bad, n_m=16)
    with pytest.raises(ValueError, match=r"mass_function"):
        _ = calc2._dndm(mmin, mmax)  # noqa: SLF001

    def hb_bad(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        return np.ones(np.asarray(M).size - 1)

    calc3 = _make_calc(halo_bias=hb_bad, n_m=16)
    with pytest.raises(ValueError, match=r"halo_bias"):
        _ = calc3._bias(mmin, mmax)  # noqa: SLF001


def test_y_grid_mm_validates_profile_shape_and_is_cached() -> None:
    """Tests that _y_grid_mm enforces (nM, nk) output and caches by
    comp/mass/k/a."""
    calc = _make_calc(n_m=16)
    mmin, mmax = 1.0e13, 1.0e14

    y1 = calc._y_grid_mm("gas", mmin, mmax)  # noqa: SLF001
    y2 = calc._y_grid_mm("gas", mmin, mmax)  # noqa: SLF001
    assert y1 is y2
    assert y1.shape == (calc.n_m, calc.k.size)
    assert np.all(np.isfinite(y1))

    def y_bad(M: np.ndarray, k: np.ndarray) -> np.ndarray:
        M = np.asarray(M, float)
        return np.ones((M.shape[0], 1), dtype=float)

    with pytest.raises(ValueError, match=r"must return shape"):
        _ = calc._y_grid_mm(
            "gas", mmin, mmax, y_fn=y_bad)  # noqa: SLF001


def test_Ib_vec_and_I2_vec_validate_and_cache() -> None:
    """Tests that _Ib_vec/_I2_vec return vectors of shape (nk,) and cache
    hits."""
    calc = _make_calc(n_m=32)
    mmin, mmax = 1.0e13, 1.0e14

    Ib1 = calc._Ib_vec(
        "stars", mmin, mmax)  # noqa: SLF001
    Ib2 = calc._Ib_vec(
        "stars", mmin, mmax)  # noqa: SLF001
    assert Ib1 is Ib2
    assert Ib1.shape == calc.k.shape
    assert np.all(np.isfinite(Ib1))

    I21 = calc._I2_vec(
        "dark_matter", "gas", mmin, mmax)  # noqa: SLF001
    I22 = calc._I2_vec(
        "dark_matter", "gas", mmin, mmax)  # noqa: SLF001
    assert I21 is I22
    assert I21.shape == calc.k.shape
    assert np.all(np.isfinite(I21))


def test_pk_halo_pair_rejects_nonpositive_rhos_and_no_overlap() -> None:
    """Tests that pk_halo_pair validates rho inputs and overlap mass ranges."""
    rho = _densities_full(matter=1.0, dark_matter=1.0, gas=1.0, stars=1.0)
    calc = _make_calc(densities=rho)

    with pytest.raises(ValueError, match=r"rho1 and rho2 must be > 0"):
        _ = calc.pk_halo_pair(comp1="gas", comp2="gas", rho1=0.0, rho2=1.0)

    mr = _mass_ranges()
    mr["dark_matter"] = {"min": 1.0e10, "max": 1.0e11}
    mr["gas"] = {"min": 1.0e12, "max": 1.0e13}
    calc2 = _make_calc(mass_ranges=mr, densities=rho)

    with pytest.raises(ValueError, match=r"No overlap mass range"):
        _ = calc2.pk_halo_pair(
            comp1="dark_matter", comp2="gas", rho1=1.0, rho2=1.0)


def test_weights_have_expected_keys_and_are_finite() -> None:
    """Tests that weights returns the expected keys and finite, non-negative
    values."""
    rho = _densities_full(matter=10.0, dark_matter=6.0, gas=3.0, stars=1.0)
    calc = _make_calc(densities=rho)

    w = calc.weights()
    assert set(w.keys()) == {"w_dm", "w_g", "w_s", "w_dm_g",
                             "w_g_s", "w_dm_s"}
    for val in w.values():
        assert np.isfinite(float(val))
        assert float(val) >= 0.0


def test_pair_packets_have_expected_shapes_and_finiteness() -> None:
    """Tests that pair packets return pk arrays of shape (nk,) with finite
    values."""
    rho = _densities_full(matter=1.0, dark_matter=1.0, gas=1.0, stars=1.0)
    calc = _make_calc(
        densities=rho, gas_params=_gas_params(Fg=0.7, bd=1.2), n_m=32)

    p = calc.pair_halo_packet(
        comp1="dark_matter", comp2="stars", rho1=1.0, rho2=1.0)
    assert p["pk"].shape == calc.k.shape
    assert np.all(np.isfinite(p["pk"]))
    assert set(p["terms"].keys()) == {"1h", "2h"}

    g = calc.pair_gas_auto_packet()
    assert g["pk"].shape == calc.k.shape
    assert np.all(np.isfinite(g["pk"]))
    assert set(g["terms"].keys()) == {"1h", "2h", "diffuse", "diffuse_halo"}

    x = calc.pair_mixed_cross_packet(comp="dark_matter")
    assert x["pk"].shape == calc.k.shape
    assert np.all(np.isfinite(x["pk"]))
    assert set(x["terms"].keys()) == {"Fg_halo_1h",
                                      "Fg_halo_2h",
                                      "1mFg_diffuse"}


def test_pk_gas_auto_dm_gas_star_gas_return_finite_arrays() -> None:
    """Tests that pk_gas_auto/pk_dm_gas/pk_star_gas return finite arrays with
    correct shape."""
    rho = _densities_full(matter=1.0, dark_matter=1.0, gas=1.0, stars=1.0)
    calc = _make_calc(
        densities=rho, gas_params=_gas_params(Fg=0.7, bd=1.2), n_m=32)

    Pg = calc.pk_gas_auto()
    assert isinstance(Pg, np.ndarray) and Pg.shape == calc.k.shape
    assert np.all(np.isfinite(Pg))

    Pdm_g = calc.pk_dm_gas()
    assert isinstance(Pdm_g, np.ndarray) and Pdm_g.shape == calc.k.shape
    assert np.all(np.isfinite(Pdm_g))

    Ps_g = calc.pk_star_gas()
    assert isinstance(Ps_g, np.ndarray) and Ps_g.shape == calc.k.shape
    assert np.all(np.isfinite(Ps_g))


def test_pk_packet_requires_densities_and_builds_aliases() -> None:
    """Tests that pk_packet requires densities and adds commutative aliases
    for pk and weighted pk."""
    calc = _make_calc(densities={})
    with pytest.raises(KeyError, match=r"Missing density"):
        _ = calc.pk_packet(use_cache=False)

    rho = _densities_full(matter=1.0, dark_matter=1.0, gas=1.0, stars=1.0)
    calc2 = _make_calc(
        densities=rho, gas_params=_gas_params(Fg=0.6, bd=1.1), n_m=16)
    packet = calc2.pk_packet(use_cache=False)

    assert (
        "pk" in packet
        and "pk_ref" in packet
        and "halo_pairs" in packet
        and "meta" in packet
    )

    pk = packet["pk"]

    # base names
    for key in ("total", "dm", "gas", "stars",
                "dm_gas", "dm_stars", "stars_gas"):
        assert key in pk
        assert pk[key].shape == calc2.k.shape
        assert np.all(np.isfinite(pk[key]))

    # aliasing symmetry
    assert np.allclose(pk["dm_gas"], pk["gas_dm"])
    assert np.allclose(pk["dm_stars"], pk["stars_dm"])
    assert np.allclose(pk["stars_gas"], pk["gas_stars"])

    # weighted aliasing symmetry (w_dm_gas etc.)
    assert np.allclose(pk["w_dm_gas"], pk["w_gas_dm"])
    assert np.allclose(pk["w_dm_stars"], pk["w_stars_dm"])
    assert np.allclose(pk["w_stars_gas"], pk["w_gas_stars"])


def test_pk_total_is_packet_total_and_is_cached() -> None:
    """Tests that pk_total returns packet['pk']['total'] and is cached via
    pk_packet."""
    rho = _densities_full(matter=1.0, dark_matter=1.0, gas=1.0, stars=1.0)
    calc = _make_calc(
        densities=rho, gas_params=_gas_params(Fg=0.7, bd=1.2), n_m=16)

    p1 = calc.pk_total()
    p2 = calc.pk_total()
    assert p1.shape == calc.k.shape
    assert np.all(np.isfinite(p1))
    assert np.allclose(p1, p2)

    pkt = calc.pk_packet(use_cache=True)
    assert np.allclose(p1, np.asarray(pkt["pk"]["total"], float))


def test_pk_total_dmo_returns_finite_array_and_uses_dmo_profile_override(

) -> None:
    """Tests that pk_total_dmo returns a finite array and respects
    dmo_dm_u_over_m override."""
    rho = _densities_full(matter=1.0, dark_matter=1.0, gas=1.0, stars=1.0)

    # Baseline
    calc = _make_calc(densities=rho, n_m=16)
    pdmo = calc.pk_total_dmo(use_cache=False)
    assert pdmo.shape == calc.k.shape
    assert np.all(np.isfinite(pdmo))

    # Override DMO profile (different constant should change the result).
    dmo_y = _y_const(2.0)
    calc2 = _make_calc(densities=rho, dmo_dm_u_over_m=dmo_y, n_m=16)
    pdmo2 = calc2.pk_total_dmo(use_cache=False)
    assert pdmo2.shape == calc2.k.shape
    assert np.all(np.isfinite(pdmo2))
    assert not np.allclose(pdmo2, pdmo)


def test_boost_hm_over_hm_is_ratio_of_total_to_dmo() -> None:
    """Tests that boost_hm_over_hm returns pk_total / pk_total_dmo."""
    rho = _densities_full(matter=1.0, dark_matter=1.0, gas=1.0, stars=1.0)
    calc = _make_calc(
        densities=rho, gas_params=_gas_params(Fg=0.7, bd=1.2), n_m=16)

    b = calc.boost_hm_over_hm()
    assert isinstance(b, np.ndarray) and b.shape == calc.k.shape
    assert np.all(np.isfinite(b))

    pb = calc.pk_total()
    pdmo = calc.pk_total_dmo(use_cache=True)
    assert np.allclose(b, pb / pdmo, rtol=0, atol=0)


def test_boost_hm_over_hm_rejects_nonpositive_or_nonfinite_dmo() -> None:
    """Tests that boost_hm_over_hm raises if DMO spectrum is non-positive or
    non-finite."""
    rho = _densities_full(matter=1.0, dark_matter=1.0, gas=1.0, stars=1.0)
    calc = _make_calc(densities=rho, n_m=16)

    # Force a bad cached DMO spectrum.
    calc._pk_dmo = np.array([1.0, 0.0, 1.0], dtype=float)  # noqa: SLF001
    calc._pk_packet_cache = None  # noqa: SLF001

    with pytest.raises(ValueError, match=r"DMO halo-model power"):
        _ = calc.boost_hm_over_hm()
