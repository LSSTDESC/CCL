"""Unit tests for `pyccl.baryons.fedeli14_bhm.power_spectra`."""

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


def _gas_params(
    Fg: float = 0.7,
    bd: float = 1.0
) -> dict[str, float]:
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
        M = np.asarray(M, dtype=float)
        return np.full_like(M, v, dtype=float)

    return hb


def _y_const(
    val: float = 1.0
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return y(M,k) = constant, broadcasting over inputs to (nM, nk)."""
    v = float(val)

    def y(M: np.ndarray, k: np.ndarray) -> np.ndarray:
        M = np.asarray(M, dtype=float)
        k = np.asarray(k, dtype=float)
        M, k = np.broadcast_arrays(M, k)
        return np.full_like(M, v, dtype=float)

    return y


def _profiles() -> dict[str, Any]:
    """Return a valid profiles_u_over_m mapping with constant evaluators."""
    return {
        "dark_matter": _y_const(1.0),
        "gas": _y_const(1.0),
        "stars": _y_const(1.0),
    }


def _make_calc(
    *,
    cosmo: Any | None = None,
    a: float = 1.0,
    k: np.ndarray | None = None,
    profiles_u_over_m: dict[str, Any] | None = None,
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

    with pytest.raises(ValueError, match=r"must have the same shape"):
        _ = _dndm_from_dndlog10m(np.ones(2), m)

    with pytest.raises(ValueError, match=r"m must be > 0"):
        _ = _dndm_from_dndlog10m(d, np.array([1.0, 0.0, 2.0]))


def test_init_validates_k_grid_shape_monotonicity_and_sign() -> None:
    """Tests that FedeliPkCalculator validates k as 1D, finite, increasing,
    and > 0."""
    with pytest.raises(ValueError, match=r"non-empty 1D array"):
        _ = _make_calc(k=np.array([], dtype=float))

    with pytest.raises(ValueError, match=r"non-empty 1D array"):
        _ = _make_calc(k=np.ones((2, 2)))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=r"contain only finite values"):
        _ = _make_calc(k=np.array([0.1, np.nan, 1.0], dtype=float))

    with pytest.raises(ValueError, match=r"strictly increasing"):
        _ = _make_calc(k=np.array([0.1, 0.1, 1.0], dtype=float))

    with pytest.raises(ValueError, match=r"must be > 0"):
        _ = _make_calc(k=np.array([0.0, 0.1, 1.0], dtype=float))


def test_init_requires_callable_mass_function_and_halo_bias() -> None:
    """Tests that FedeliPkCalculator requires callable mass_function and
    halo_bias."""
    with pytest.raises(TypeError, match=r"mass_function must be callable"):
        _ = _make_calc(mass_function=123)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=r"halo_bias must be callable"):
        _ = _make_calc(halo_bias=123)  # type: ignore[arg-type]


def test_init_validates_profiles_mapping_and_components() -> None:
    """Tests that FedeliPkCalculator validates profiles_u_over_m mapping and
    required keys."""
    with pytest.raises(
        TypeError,
        match=r"profiles_u_over_m must be a mapping"
    ):
        _ = _make_calc(profiles_u_over_m=123)  # type: ignore[arg-type]

    with pytest.raises(KeyError, match=r"missing component 'stars'"):
        _ = _make_calc(
            profiles_u_over_m={"dark_matter": _y_const(1.0),
                               "gas": _y_const(1.0)}
        )

    bad = {"dark_matter": _y_const(1.0), "gas": 123, "stars": _y_const(1.0)}
    with pytest.raises(
            TypeError,
            match=r"profiles_u_over_m\['gas'\] must be callable"
    ):
        _ = _make_calc(profiles_u_over_m=bad)


def test_init_validates_mass_ranges_structure_and_values() -> None:
    """Tests that FedeliPkCalculator validates mass_ranges structure and
    positivity."""
    with pytest.raises(
            TypeError,
            match=r"mass_ranges must be a mapping"
    ):
        _ = _make_calc(mass_ranges=123)  # type: ignore[arg-type]

    bad_missing = {
        "dark_matter": {"min": 1.0, "max": 2.0},
        "gas": {"min": 1.0, "max": 2.0},
    }
    with pytest.raises(
            KeyError,
            match=r"mass_ranges missing component 'stars'"
    ):
        _ = _make_calc(mass_ranges=bad_missing)  # type: ignore[arg-type]

    bad_keys = _mass_ranges()
    bad_keys["gas"] = {"min": 1.0}  # missing max
    with pytest.raises(KeyError, match=r"must have keys 'min' and 'max'"):
        _ = _make_calc(mass_ranges=bad_keys)

    bad_vals = _mass_ranges()
    bad_vals["stars"] = {"min": 1.0e15, "max": 1.0e14}
    with pytest.raises(ValueError, match=r"must have max>min"):
        _ = _make_calc(mass_ranges=bad_vals)


def test_init_validates_gas_params_and_densities_types() -> None:
    """Tests that FedeliPkCalculator validates gas_params keys, Fg range,
    bd finiteness, and densities type."""
    with pytest.raises(
            KeyError, match=r"gas_params must contain keys 'Fg' and 'bd'"):
        _ = _make_calc(gas_params={"Fg": 0.5})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=r"Fg must be in \[0, 1\]"):
        _ = _make_calc(gas_params=_gas_params(Fg=1.5, bd=1.0))

    with pytest.raises(TypeError, match=r"densities must be a mapping"):
        _ = _make_calc(densities=123)  # type: ignore[arg-type]


def test_init_validates_n_m() -> None:
    """Tests that FedeliPkCalculator validates n_m >= 2."""
    with pytest.raises(ValueError, match=r"n_m must be > 0"):
        _ = _make_calc(n_m=0)

    with pytest.raises(ValueError, match=r"n_m must be >= 2"):
        _ = _make_calc(n_m=1)


def test_rho_from_fraction_validates_callable_range_and_shape() -> None:
    """Tests that rho_from_fraction validates f_of_m, mass range, and output
    shape/values."""
    calc = _make_calc()

    with pytest.raises(TypeError, match=r"f_of_m must be callable"):
        _ = calc.rho_from_fraction(f_of_m=123, mmin=1.0, mmax=10.0)

    with pytest.raises(ValueError, match=r"Invalid mass range"):
        _ = calc.rho_from_fraction(
            f_of_m=lambda m: np.ones_like(m), mmin=0.0, mmax=10.0
        )

    def bad_shape(m: np.ndarray) -> np.ndarray:
        """Returns an array with wrong shape."""
        return np.ones(m.size - 1)

    with pytest.raises(ValueError, match=r"same shape as m"):
        _ = calc.rho_from_fraction(
            f_of_m=bad_shape, mmin=1.0e13, mmax=1.0e14, n_m=16
        )

    def bad_nan(m: np.ndarray) -> np.ndarray:
        """Returns an array with a NaN value."""
        out = np.ones_like(m)
        out[0] = np.nan
        return out

    with pytest.raises(ValueError, match=r"must be finite"):
        _ = calc.rho_from_fraction(
            f_of_m=bad_nan, mmin=1.0e13, mmax=1.0e14, n_m=16
        )


def test_rho_from_fraction_caches_when_cache_key_given() -> None:
    """Tests that rho_from_fraction caches results when cache_key is
    provided."""
    calls = {"n": 0}

    def hmf_count(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        """Mock halo mass function that counts calls."""
        _ = a
        calls["n"] += 1
        return np.ones_like(np.asarray(M, float))

    calc = _make_calc(mass_function=hmf_count)

    def f(m: np.ndarray) -> np.ndarray:
        """Mock fraction function that returns constant density."""
        return np.ones_like(m)

    r1 = calc.rho_from_fraction(
        f_of_m=f, mmin=1.0e13, mmax=1.0e14, n_m=32, cache_key="gas"
    )
    r2 = calc.rho_from_fraction(
        f_of_m=f, mmin=1.0e13, mmax=1.0e14, n_m=32, cache_key="gas"
    )
    assert np.isfinite(r1) and r1 >= 0.0
    assert r2 == r1
    assert calls["n"] == 1


def test_ensure_densities_requires_cosmo_keys_and_fills_missing_entries(

) -> None:
    """Tests that ensure_densities requires Omega_c/Omega_m and fills
    matter/dark_matter/gas/stars if missing."""
    cosmo = _cosmo()
    calc = _make_calc(cosmo=cosmo, densities={})

    def f_g(m):
        """Returns constant gas density for all masses."""
        return np.full_like(m, 0.1, dtype=float)

    def f_s(m):
        """Returns constant star density for all masses."""
        return np.full_like(m, 0.01, dtype=float)

    calc.ensure_densities(
        f_gas=f_g, f_star=f_s, mmin=1.0e13, mmax=1.0e14, n_m=64
    )

    for key in ("matter", "dark_matter", "gas", "stars"):
        assert key in calc.rho
        assert np.isfinite(float(calc.rho[key]))
        assert float(calc.rho[key]) > 0.0

    cosmo_bad = {"Omega_c": 0.25}  # missing Omega_m
    calc2 = _make_calc(cosmo=cosmo_bad, densities={})
    with pytest.raises(KeyError, match=r"cosmo must provide 'Omega_m'"):
        calc2.ensure_densities(
            f_gas=f_g, f_star=f_s, mmin=1.0e13, mmax=1.0e14, n_m=16
        )


def test_linpk_is_cached_and_P_lin_computed_once() -> None:
    """Tests that linpk caches per-k and P_lin memoizes the full vector."""
    calc = _make_calc()
    k0 = float(calc.k[0])

    p1 = calc.linpk(k0)
    p2 = calc.linpk(k0)
    assert p1 == p2
    assert len(calc._linpk_cache) == 1  # noqa: SLF001

    v1 = calc.P_lin()
    v2 = calc.P_lin()
    assert v1 is v2
    assert isinstance(v1, np.ndarray) and v1.shape == calc.k.shape
    assert np.all(np.isfinite(v1))


def test_mass_grid_dndm_and_bias_are_cached_and_validate_shapes() -> None:
    """Tests that mass-grid dependent cached arrays are reused and validate
    callable return shapes."""
    calls = {"hmf": 0, "hb": 0}

    def hmf(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        """Mock halo mass function."""
        _ = a
        calls["hmf"] += 1
        return np.ones_like(np.asarray(M, float))

    def hb(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        """Mock halo bias."""
        _ = a
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
        """Mock halo mass function that returns wrong shape."""
        _ = a
        return np.ones(M.size - 1)

    calc2 = _make_calc(mass_function=hmf_bad, n_m=16)
    with pytest.raises(ValueError, match=r"same shape as M"):
        _ = calc2._dndm(mmin, mmax)  # noqa: SLF001

    def hb_bad(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        """Mock halo bias that returns wrong shape."""
        _ = a
        return np.ones(M.size - 1)

    calc3 = _make_calc(halo_bias=hb_bad, n_m=16)
    with pytest.raises(ValueError, match=r"same shape as M"):
        _ = calc3._bias(mmin, mmax)  # noqa: SLF001


def test_I2_and_Ib_validate_components_and_profile_shapes_and_cache() -> None:
    """Tests that vector I2/Ib validate component keys, profile return shapes,
    and cache results."""
    calc = _make_calc(n_m=32)
    mmin, mmax = 1.0e13, 1.0e14

    I2_1 = calc._I2_vec("dark_matter", "gas", mmin, mmax)  # noqa: SLF001
    I2_2 = calc._I2_vec("dark_matter", "gas", mmin, mmax)  # noqa: SLF001
    assert I2_1 is I2_2
    assert I2_1.shape == calc.k.shape
    assert np.all(np.isfinite(I2_1))

    Ib_1 = calc._Ib_vec("stars", mmin, mmax)  # noqa: SLF001
    Ib_2 = calc._Ib_vec("stars", mmin, mmax)  # noqa: SLF001
    assert Ib_1 is Ib_2
    assert Ib_1.shape == calc.k.shape
    assert np.all(np.isfinite(Ib_1))

    with pytest.raises(KeyError, match=r"Unknown component"):
        _ = calc._I2_vec("nope", "gas", mmin, mmax)

    with pytest.raises(KeyError, match=r"Unknown component"):
        _ = calc._Ib_vec("nope", mmin, mmax)  # noqa: SLF001

    def y_bad(M: np.ndarray, k: np.ndarray) -> np.ndarray:
        M = np.asarray(M, float)
        return np.ones(M.size - 1)

    profiles = _profiles()
    profiles["gas"] = y_bad
    calc2 = _make_calc(profiles_u_over_m=profiles, n_m=16)

    with pytest.raises(ValueError, match=r"must return shape"):
        _ = calc2._I2_vec("dark_matter", "gas", mmin, mmax)


def test_pk_halo_pair_rejects_nonpositive_rhos_and_no_overlap() -> None:
    """Tests that pk_halo_pair validates rho inputs and overlap mass ranges."""
    calc = _make_calc(
        densities={
            "matter": 1.0,
            "dark_matter": 1.0,
            "gas": 1.0,
            "stars": 1.0,
        },
    )

    with pytest.raises(ValueError, match=r"rho1 and rho2 must be > 0"):
        _ = calc.pk_halo_pair(comp1="gas", comp2="gas", rho1=0.0, rho2=1.0)

    mr = _mass_ranges()
    mr["dark_matter"] = {"min": 1.0e10, "max": 1.0e11}
    mr["gas"] = {"min": 1.0e12, "max": 1.0e13}
    calc2 = _make_calc(
        mass_ranges=mr,
        densities={"matter": 1.0,
                   "dark_matter": 1.0,
                   "gas": 1.0,
                   "stars": 1.0},
    )

    with pytest.raises(ValueError, match=r"No overlap mass range"):
        _ = calc2.pk_halo_pair(
            comp1="dark_matter", comp2="gas", rho1=1.0, rho2=1.0
        )


def test_pk_gas_auto_dm_gas_star_gas_return_finite_arrays() -> None:
    """Tests that gas mixing spectra return finite arrays with correct
    shape."""
    rho = {"matter": 1.0, "dark_matter": 1.0, "gas": 1.0, "stars": 1.0}
    calc = _make_calc(
        densities=rho, gas_params=_gas_params(Fg=0.7, bd=1.2), n_m=32
    )

    Pg = calc.pk_gas_auto()
    assert isinstance(Pg, np.ndarray) and Pg.shape == calc.k.shape
    assert np.all(np.isfinite(Pg))

    Pdm_g = calc.pk_dm_gas()
    assert isinstance(Pdm_g, np.ndarray) and Pdm_g.shape == calc.k.shape
    assert np.all(np.isfinite(Pdm_g))

    Ps_g = calc.pk_star_gas()
    assert isinstance(Ps_g, np.ndarray) and Ps_g.shape == calc.k.shape
    assert np.all(np.isfinite(Ps_g))


def test_weights_have_expected_keys_and_are_finite() -> None:
    """Tests that weights returns the expected keys and finite values."""
    rho = {"matter": 10.0, "dark_matter": 6.0, "gas": 3.0, "stars": 1.0}
    calc = _make_calc(densities=rho)

    w = calc.weights()
    assert set(w.keys()) == {"w_dm", "w_g", "w_s", "w_dm_g", "w_g_s", "w_dm_s"}
    for _, val in w.items():
        assert np.isfinite(float(val))
        assert float(val) >= 0.0


def test_pk_total_dmo_adds_1h_and_2h() -> None:
    """Tests that pk_total_dmo returns a finite vector with the correct
    shape."""
    rho = {"matter": 1.0, "dark_matter": 1.0, "gas": 1.0, "stars": 1.0}
    calc = _make_calc(densities=rho, n_m=16)

    pdmo = calc.pk_total_dmo(use_cache=False)
    assert isinstance(pdmo, np.ndarray) and pdmo.shape == calc.k.shape
    assert np.all(np.isfinite(pdmo))


def test_boost_hm_over_hm_is_ratio_of_total_to_dmo() -> None:
    """Tests that boost_hm_over_hm returns pk_total / pk_total_dmo."""
    rho = {"matter": 1.0, "dark_matter": 1.0, "gas": 1.0, "stars": 1.0}
    calc = _make_calc(densities=rho, n_m=16)

    b = calc.boost_hm_over_hm()
    assert isinstance(b, np.ndarray) and b.shape == calc.k.shape
    assert np.all(np.isfinite(b))

    pb = calc.pk_total()
    pdmo = calc.pk_total_dmo(use_cache=True)
    assert np.allclose(b, pb / pdmo, rtol=0, atol=0)
