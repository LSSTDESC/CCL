"""Unit tests for `pyccl.baryons.fedeli14_bhm.mass_fractions`."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pyccl.baryons.fedeli14_bhm.mass_fractions import mass_fractions


class DummyCosmo(dict):
    """Mock cosmology class."""


def _cosmo(
    h: float = 0.7,
    Omega_b: float = 0.05,
    Omega_m: float = 0.3
) -> DummyCosmo:
    """Mock cosmology object."""
    return DummyCosmo(h=h, Omega_b=Omega_b, Omega_m=Omega_m)


def test_requires_positive_finite_a() -> None:
    """Tests that mass_fractions rejects non-finite or non-positive scale
    factor a."""
    cosmo = _cosmo()

    def mf(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        return np.ones_like(np.atleast_1d(M), dtype=float)

    with pytest.raises(ValueError, match=r"a must be finite and > 0"):
        mass_fractions(cosmo=cosmo, a=0.0, mass_function=mf)

    with pytest.raises(ValueError, match=r"a must be finite and > 0"):
        mass_fractions(cosmo=cosmo, a=float("nan"), mass_function=mf)


def test_requires_callable_mass_function() -> None:
    """Tests that mass_fractions requires a callable mass_function."""
    cosmo = _cosmo()
    with pytest.raises(TypeError, match="mass_function must be callable"):
        mass_fractions(cosmo=cosmo, a=1.0, mass_function=123)


def test_requires_cosmo_keys() -> None:
    """Tests that mass_fractions requires cosmo to provide h, Omega_b, and
    Omega_m."""
    def mf(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        return np.ones_like(np.atleast_1d(M), dtype=float)

    cosmo = DummyCosmo(h=0.7, Omega_b=0.05)  # missing Omega_m
    with pytest.raises(KeyError, match="must provide 'Omega_m'"):
        mass_fractions(cosmo=cosmo, a=1.0, mass_function=mf)


def test_cosmo_param_ranges() -> None:
    """Tests that mass_fractions enforces valid Omega_b/Omega_m ranges."""
    def mf(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        return np.ones_like(np.atleast_1d(M), dtype=float)

    with pytest.raises(ValueError, match=r"Omega_m.*> 0"):
        mass_fractions(cosmo=_cosmo(Omega_m=0.0), a=1.0, mass_function=mf)

    with pytest.raises(ValueError, match=r"Omega_b.*>= 0"):
        mass_fractions(cosmo=_cosmo(Omega_b=-0.01), a=1.0, mass_function=mf)

    with pytest.raises(ValueError, match=r"Omega_b <= Omega_m"):
        mass_fractions(cosmo=_cosmo(Omega_b=0.4, Omega_m=0.3),
                       a=1.0,
                       mass_function=mf)


def test_mass_param_ranges() -> None:
    """Tests that mass_fractions validates mass-scale parameters and
    integration bounds."""
    cosmo = _cosmo()

    def mf(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        return np.ones_like(np.atleast_1d(M), dtype=float)

    with pytest.raises(ValueError, match=r"m0_star.*> 0"):
        mass_fractions(cosmo=cosmo, a=1.0, mass_function=mf, m0_star=0.0)

    with pytest.raises(ValueError, match=r"sigma_gas.*> 0"):
        mass_fractions(cosmo=cosmo, a=1.0, mass_function=mf, sigma_gas=0.0)

    with pytest.raises(ValueError, match=r"mmin_star < mmax_star"):
        mass_fractions(cosmo=cosmo,
                       a=1.0,
                       mass_function=mf,
                       mmin_star=1e15,
                       mmax_star=1e10)


def test_return_types_and_scalar_array_behavior() -> None:
    """Tests that mass fraction callables return floats for scalars and arrays
     for arrays."""
    cosmo = _cosmo()

    def mf(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        return np.ones_like(np.atleast_1d(M), dtype=float)

    f_gas, f_star, f_dm = mass_fractions(cosmo=cosmo, a=1.0, mass_function=mf)

    g0 = f_gas(1.0e12)
    s0 = f_star(1.0e12)
    d0 = f_dm(1.0e12)
    assert isinstance(g0, float)
    assert isinstance(s0, float)
    assert isinstance(d0, float)

    M = np.array([1.0e11, 1.0e12, 1.0e13])
    g = f_gas(M)
    s = f_star(M)
    d = f_dm(M)
    assert isinstance(g, np.ndarray) and g.shape == M.shape
    assert isinstance(s, np.ndarray) and s.shape == M.shape
    assert isinstance(d, np.ndarray) and d.shape == M.shape


def test_component_basic_limits() -> None:
    """Tests that mass fractions have sensible limiting behavior and
    non-negativity."""
    cosmo = _cosmo(Omega_b=0.05, Omega_m=0.30)
    omega_ratio = cosmo["Omega_b"] / cosmo["Omega_m"]

    def mf(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        return np.ones_like(np.atleast_1d(M), dtype=float)

    m0_gas = 1.0e12
    sigma_gas = 3.0
    f_gas, f_star, f_dm = mass_fractions(
        cosmo=cosmo,
        a=1.0,
        mass_function=mf,
        m0_gas=m0_gas,
        sigma_gas=sigma_gas
    )

    M = np.geomspace(1e10, 1e15, 10)
    dm = f_dm(M)
    assert np.allclose(dm, 1.0 - omega_ratio)

    assert f_gas(m0_gas * 0.99) == 0.0
    assert f_gas(m0_gas * 1.01) >= 0.0

    high = f_gas(1.0e30)
    assert high == pytest.approx(float(omega_ratio), rel=1e-6, abs=1e-12)

    assert np.all(np.asarray(f_star(M)) >= 0.0)


def test_mass_inputs_must_be_positive_finite() -> None:
    """Tests that each returned mass-fraction callable rejects invalid halo
    masses."""
    cosmo = _cosmo()

    def mf(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        return np.ones_like(np.atleast_1d(M), dtype=float)

    f_gas, f_star, f_dm = mass_fractions(cosmo=cosmo, a=1.0, mass_function=mf)

    for fn in (f_gas, f_star, f_dm):
        with pytest.raises(ValueError, match=r"M must be finite and > 0"):
            fn(0.0)
        with pytest.raises(ValueError, match=r"M must be finite and > 0"):
            fn(-1.0)
        with pytest.raises(ValueError, match=r"M must be finite and > 0"):
            fn(np.array([1.0, float("nan")]))


def test_msunh_detection_path_works() -> None:
    """Tests that the Msun vs Msun/h detection branch produces finite
    results."""
    cosmo = _cosmo(h=0.7)
    m0_star = 5.0e12

    def mf_triggers_detection(_cosmo: Any,
                              M: np.ndarray,
                              a: float) -> np.ndarray:
        _ = a
        M = np.atleast_1d(M).astype(float)
        lo = m0_star * 1.0001
        hi = (m0_star / float(_cosmo["h"])) * 0.9999
        # Ensure lo < hi for h<1 and choose a threshold between them
        thresh = 0.5 * (lo + hi)
        return np.where(M >= thresh, 1.0, 0.0)

    f_gas, f_star, f_dm = mass_fractions(
        cosmo=cosmo,
        a=1.0,
        mass_function=mf_triggers_detection,
        m0_star=m0_star
    )

    assert math.isfinite(float(f_star(m0_star)))
    assert math.isfinite(float(f_gas(m0_star)))
    assert math.isfinite(float(f_dm(m0_star)))


def test_stellar_normalization_is_cached() -> None:
    """Tests that the stellar normalization factor A is cached after first
    evaluation."""
    cosmo = _cosmo()
    calls = {"n": 0}

    def mf_counting(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        _ = a
        calls["n"] += 1
        return np.ones_like(np.atleast_1d(M), dtype=float)

    _f_gas, f_star, _f_dm = mass_fractions(
        cosmo=cosmo, a=1.0, mass_function=mf_counting)

    _ = f_star(1.0e12)
    n1 = calls["n"]
    assert n1 > 0

    _ = f_star(2.0e12)
    n2 = calls["n"]
    assert n2 == n1


def test_rho_star_must_be_positive_finite() -> None:
    """Tests that mass_fractions rejects non-finite or non-positive
    rho_star."""
    cosmo = _cosmo()

    def mf(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        return np.ones_like(np.atleast_1d(M), dtype=float)

    with pytest.raises(ValueError, match=r"rho_star must be finite and > 0"):
        mass_fractions(cosmo=cosmo, a=1.0, mass_function=mf, rho_star=0.0)

    with pytest.raises(ValueError, match=r"rho_star must be finite and > 0"):
        mass_fractions(
            cosmo=cosmo, a=1.0, mass_function=mf, rho_star=float("nan"))


def test_rho_star_none_default_is_finite() -> None:
    """Tests that mass_fractions with rho_star=None uses the default and
    returns finite values."""
    cosmo = _cosmo()

    def mf(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        _ = a
        return np.ones_like(np.atleast_1d(M), dtype=float)

    f_gas, f_star, f_dm = mass_fractions(
        cosmo=cosmo, a=1.0, mass_function=mf, rho_star=None)
    assert math.isfinite(float(f_star(1.0e12)))
    assert math.isfinite(float(f_gas(1.0e12)))
    assert math.isfinite(float(f_dm(1.0e12)))


def test_mass_function_nonfinite_in_unit_probe_raises() -> None:
    """Tests that non-finite mass_function outputs in the unit-probe call
    raise ValueError."""
    cosmo = _cosmo()

    def mf_nan(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        _, _ = M, a
        return np.array([np.nan], dtype=float)

    with pytest.raises(
            ValueError, match=r"returned non-finite values at the test mass"):
        mass_fractions(cosmo=cosmo, a=1.0, mass_function=mf_nan)


def test_mass_function_nonfinite_inside_integrand_raises() -> None:
    """Tests that non-finite dn/dlog10M during the stellar integral raises
    ValueError."""
    cosmo = _cosmo()
    m0_star = 5.0e12
    h = float(cosmo["h"])

    def mf_nan_in_range(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        _ = a
        M0 = float(np.atleast_1d(M)[0])
        if np.isclose(M0, m0_star) or np.isclose(M0, m0_star / h):
            return np.array([1.0], dtype=float)
        return np.array([np.nan], dtype=float)

    _f_gas, f_star, _f_dm = mass_fractions(
        cosmo=cosmo, a=1.0, mass_function=mf_nan_in_range)
    with pytest.raises(ValueError, match=r"returned non-finite dn/dlog10M"):
        _ = f_star(1.0e12)


def test_stellar_normalization_integral_nonpositive_raises() -> None:
    """Tests that a non-positive stellar normalization integral raises
    ValueError with diagnostics."""
    cosmo = _cosmo()

    def mf_zero(_cosmo: Any, M: np.ndarray, a: float) -> np.ndarray:
        return np.zeros_like(np.atleast_1d(M), dtype=float)

    _f_gas, f_star, _f_dm = mass_fractions(
        cosmo=cosmo, a=1.0, mass_function=mf_zero
    )
    with pytest.raises(ValueError, match=r"stellar normalization integral is"):
        _ = f_star(1.0e12)
