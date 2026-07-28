"""Utility functions for numerical operations."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np


def _require_a(a: float) -> float:
    """Check that a is finite and positive."""
    a = float(a)
    if not np.isfinite(a):
        raise ValueError("a must be finite.")
    if a <= 0.0:
        raise ValueError("a must be > 0.")
    return a


def _require_k(k: np.ndarray) -> np.ndarray:
    """Check that k is a non-empty 1D array of finite values strictly
     increasing."""
    k = np.asarray(k, dtype=float)
    if k.ndim != 1 or k.size == 0:
        raise ValueError("k must be a non-empty 1D array.")
    if not np.all(np.isfinite(k)):
        raise ValueError("k must contain only finite values.")
    if np.any(k <= 0.0):
        raise ValueError("k must be > 0.")
    if not np.all(np.diff(k) > 0):
        raise ValueError("k must be strictly increasing.")
    return k


def _require_callable(obj: Any, *, who: str) -> None:
    """Check that an object is callable."""
    if not callable(obj):
        raise TypeError(f"{who} must be callable.")


def _require_array_1d(name: str, x: np.ndarray) -> None:
    """Check that an array is 1D and non-empty."""
    if x.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if x.size == 0:
        raise ValueError(f"{name} must be non-empty.")


def _require_finite_1d(name: str, x: np.ndarray) -> None:
    """Check that a 1D array is finite."""
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must contain only finite values.")


def _as_float(x, name: str) -> float:
    """Return x as a finite float."""
    try:
        y = float(x)
    except Exception as e:  # noqa: BLE001
        raise TypeError(f"{name} must be a real number.") from e
    if not np.isfinite(y):
        raise ValueError(f"{name} must be finite.")
    return y


def _pos_float(x, name: str) -> float:
    """Return x as a finite float > 0."""
    y = _as_float(x, name)
    if y <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return y


def _pos_int(x, name: str) -> int:
    """Return x as an int > 0."""
    try:
        y = int(x)
    except Exception as e:  # noqa: BLE001
        raise TypeError(f"{name} must be an int.") from e
    if y <= 0:
        raise ValueError(f"{name} must be > 0.")
    return y


def _require_attr(obj: Any, attr: str, *, who: str) -> None:
    """Check that an object has a given attribute."""
    if not hasattr(obj, attr):
        raise TypeError(f"{who} must define attribute {attr!r}.")


def _require_mapping(x: Any, *, who: str) -> None:
    """Check that an object is a mapping (dict-like)."""
    if not isinstance(x, Mapping):
        raise TypeError(f"{who} must be a mapping (dict-like).")


def _require_float(name: str, x: Any) -> None:
    """Check that an object is convertible to float."""
    try:
        float(x)
    except Exception as e:  # noqa: BLE001
        raise TypeError(f"{name} must be a real number.") from e


def _require_profiles_u_over_m(
    profiles: Any,
    *,
    components: tuple[str, ...] = ("dark_matter", "gas", "stars"),
) -> dict[str, Callable[[np.ndarray, float], np.ndarray]]:
    """Check profiles_u_over_m is a dict with required callable components."""
    _require_mapping(profiles, who="profiles_u_over_m")

    out: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {}
    for comp in components:
        if comp not in profiles:
            raise KeyError(f"profiles_u_over_m missing component {comp!r}.")
        fn = profiles[comp]
        _require_callable(fn, who=f"profiles_u_over_m[{comp!r}]")
        out[comp] = fn  # type: ignore[assignment]
    return out


def _require_mass_ranges(
    mass_ranges: Any,
    *,
    components: tuple[str, ...] = ("dark_matter", "gas", "stars"),
) -> dict[str, dict[str, float]]:
    """Check mass_ranges[comp] has positive min/max with max>min
     for each component."""
    _require_mapping(mass_ranges, who="mass_ranges")

    out: dict[str, dict[str, float]] = {}
    for comp in components:
        if comp not in mass_ranges:
            raise KeyError(f"mass_ranges missing component {comp!r}.")

        r = mass_ranges[comp]
        _require_mapping(r, who=f"mass_ranges[{comp!r}]")

        if "min" not in r or "max" not in r:
            raise KeyError(f"mass_ranges[{comp!r}]"
                           f" must have keys 'min' and 'max'.")

        mmin = _pos_float(r["min"], f"mass_ranges[{comp!r}]['min']")
        mmax = _pos_float(r["max"], f"mass_ranges[{comp!r}]['max']")
        if mmax <= mmin:
            raise ValueError(
                f"mass_ranges[{comp!r}] must have max>min,"
                f" got min={mmin}, max={mmax}."
            )
        out[comp] = {"min": mmin, "max": mmax}

    return out


def _require_densities(densities: Any) -> dict[str, float]:
    """Check densities is a mapping and coerce values to float."""
    _require_mapping(densities, who="densities")
    return {str(k): float(v) for k, v in densities.items()}


def _require_gas_params(gas_params: Any) -> tuple[float, float]:
    """Check gas_params has finite Fg in [0,1] and finite bd parameter."""
    _require_mapping(gas_params, who="gas_params")

    if "Fg" not in gas_params or "bd" not in gas_params:
        raise KeyError("gas_params must contain keys 'Fg' and 'bd'.")

    Fg = _as_float(gas_params["Fg"], "gas_params['Fg']")
    if not (0.0 <= Fg <= 1.0):
        raise ValueError(f"Fg must be in [0, 1], got {Fg}.")

    bd = _as_float(gas_params["bd"], "gas_params['bd']")
    return Fg, bd


def _add_pair_aliases(d: dict[str, Any], a: str, b: str) -> None:
    """
    Add a commutative alias for a two-component key.

    If exactly one of ``"a_b"`` or ``"b_a"`` exists in the dictionary,
    this function inserts the missing one so that both keys point to
    the same object. No data are copied.

    This allows downstream code to access pair terms without caring
    about the ordering convention used when building the dictionary.
    """
    k1 = f"{a}_{b}"
    k2 = f"{b}_{a}"
    if k1 in d and k2 not in d:
        d[k2] = d[k1]
    elif k2 in d and k1 not in d:
        d[k1] = d[k2]


def _add_weighted_pair_aliases(d: dict[str, Any], a: str, b: str) -> None:
    """
    Add commutative aliases for weighted pair terms.

    Ensures that both ``"w_a_b"`` and ``"w_b_a"`` exist in the dictionary
    if either one is present, pointing to the same object. This mirrors
    the behavior of :func:`_add_pair_aliases` but for weighted
    contributions to the total power spectrum.

    No arrays are copied; only additional dictionary keys are created.
    """
    k1 = f"w_{a}_{b}"
    k2 = f"w_{b}_{a}"
    if k1 in d and k2 not in d:
        d[k2] = d[k1]
    elif k2 in d and k1 not in d:
        d[k1] = d[k2]


def _require_component(
    comp: str,
    *,
    allowed: Mapping[str, Any],
    who: str = "component",
) -> str:
    """Check that `comp` is a known key in `allowed` and return it.

    Raises:
        KeyError: if `comp` is not present in `allowed`.
    """
    if comp not in allowed:
        raise KeyError(f"Unknown component {comp!r}.")
    return comp


def _trapz_compat(
    y: Any,
    x: Any | None = None,
    dx: float = 1.0,
    axis: int = -1
) -> Any:
    """Trapezoidal integration compatible with old/new NumPy.

    Uses np.trapezoid if available, else falls back to np.trapz.
    """
    trap = getattr(np, "trapezoid", None)
    if trap is not None:
        return trap(y, x=x, dx=dx, axis=axis)
    return np.trapz(y, x=x, dx=dx, axis=axis)
