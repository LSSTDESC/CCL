"""Unit tests for `pyccl.baryons.fedeli14_bhm.numerics`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pyccl.baryons.fedeli14_bhm.numerics import (
    _add_pair_aliases,
    _add_weighted_pair_aliases,
    _pos_float,
    _pos_int,
    _require_a,
    _require_gas_params,
    _require_k,
    _require_mass_ranges,
    _require_profiles_u_over_m,
    _require_component,
)


def test_require_a_rejects_nonfinite_or_nonpositive() -> None:
    """Tests that _require_a rejects non-finite and non-positive a."""
    with pytest.raises(ValueError, match=r"a must be finite"):
        _require_a(float("nan"))
    with pytest.raises(ValueError, match=r"a must be > 0"):
        _require_a(0.0)
    assert _require_a(1.0) == 1.0


def test_require_k_validates_shape_finiteness_and_order() -> None:
    """Tests that _require_k enforces 1D, non-empty, finite, >0, strictly
    increasing."""
    with pytest.raises(ValueError, match=r"non-empty 1D"):
        _require_k(np.array([]))
    with pytest.raises(ValueError, match=r"non-empty 1D"):
        _require_k(np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError, match=r"finite"):
        _require_k(np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match=r"k must be > 0"):
        _require_k(np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match=r"strictly increasing"):
        _require_k(np.array([1.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match=r"strictly increasing"):
        _require_k(np.array([2.0, 1.0]))

    k = _require_k(np.array([1.0e-3, 1.0e-2, 1.0e-1]))
    assert isinstance(k, np.ndarray) and k.ndim == 1 and k.size == 3


def test_pos_float_and_pos_int_validation() -> None:
    """Tests that _pos_float/_pos_int enforce type coercion and strict
    positivity."""
    assert _pos_float(2.5, "x") == 2.5
    with pytest.raises(ValueError, match=r"x must be > 0"):
        _pos_float(0.0, "x")
    with pytest.raises(TypeError, match=r"x must be a real number"):
        _pos_float(object(), "x")  # type: ignore[arg-type]

    assert _pos_int(3, "n") == 3
    with pytest.raises(ValueError, match=r"n must be > 0"):
        _pos_int(0, "n")
    with pytest.raises(TypeError, match=r"n must be an int"):
        _pos_int("nope", "n")  # type: ignore[arg-type]


def test_require_profiles_u_over_m_requires_components_and_callables() -> None:
    """Tests that _require_profiles_u_over_m enforces required keys and
    callables."""
    def ok(_k: np.ndarray, _a: float) -> np.ndarray:
        return np.ones_like(_k)

    profiles = {"dark_matter": ok, "gas": ok, "stars": ok}
    out = _require_profiles_u_over_m(profiles)
    assert set(out.keys()) == {"dark_matter", "gas", "stars"}

    with pytest.raises(KeyError, match=r"missing component 'gas'"):
        _require_profiles_u_over_m({"dark_matter": ok, "stars": ok})

    with pytest.raises(TypeError, match=r"must be callable"):
        _require_profiles_u_over_m({"dark_matter": ok,
                                    "gas": 123,
                                    "stars": ok})


def test_require_mass_ranges_validates_min_max_and_order() -> None:
    """Tests that _require_mass_ranges requires min/max keys, positivity,
    and max>min."""
    mr = {
        "dark_matter": {"min": 1.0e10, "max": 1.0e15},
        "gas": {"min": 1.0e10, "max": 1.0e15},
        "stars": {"min": 1.0e10, "max": 1.0e15},
    }
    out = _require_mass_ranges(mr)
    assert out["gas"]["min"] == pytest.approx(1.0e10)
    assert out["gas"]["max"] == pytest.approx(1.0e15)

    with pytest.raises(KeyError, match=r"mass_ranges missing component 'gas'"):
        _require_mass_ranges({"dark_matter": {"min": 1.0, "max": 2.0},
                              "stars": {"min": 1.0, "max": 2.0}})

    with pytest.raises(KeyError, match=r"must have keys 'min' and 'max'"):
        _require_mass_ranges({**mr, "gas": {"min": 1.0}})

    with pytest.raises(ValueError, match=r"max>min"):
        _require_mass_ranges({**mr, "gas": {"min": 2.0, "max": 1.0}})


def test_require_gas_params_validates_keys_and_range() -> None:
    """Tests that _require_gas_params enforces required keys and Fg in
    [0,1]."""
    Fg, bd = _require_gas_params({"Fg": 0.7, "bd": 0.2})
    assert Fg == pytest.approx(0.7)
    assert bd == pytest.approx(0.2)

    with pytest.raises(KeyError, match=r"must contain keys 'Fg' and 'bd'"):
        _require_gas_params({"Fg": 0.7})

    with pytest.raises(ValueError, match=r"Fg must be in \[0, 1\]"):
        _require_gas_params({"Fg": 1.1, "bd": 0.2})


def test_add_pair_aliases_adds_missing_key_only() -> None:
    """Tests that _add_pair_aliases adds the missing commutative key without
    overwriting."""
    obj: Any = object()

    d = {"a_b": obj}
    _add_pair_aliases(d, "a", "b")
    assert d["b_a"] is obj

    d2 = {"b_a": obj}
    _add_pair_aliases(d2, "a", "b")
    assert d2["a_b"] is obj

    d3 = {"a_b": "x", "b_a": "y"}
    _add_pair_aliases(d3, "a", "b")
    assert d3["a_b"] == "x" and d3["b_a"] == "y"


def test_add_weighted_pair_aliases_adds_missing_key_only() -> None:
    """Tests that _add_weighted_pair_aliases adds the missing weighted
    commutative key."""
    obj: Any = object()

    d = {"w_a_b": obj}
    _add_weighted_pair_aliases(d, "a", "b")
    assert d["w_b_a"] is obj

    d2 = {"w_b_a": obj}
    _add_weighted_pair_aliases(d2, "a", "b")
    assert d2["w_a_b"] is obj

    d3 = {"w_a_b": "x", "w_b_a": "y"}
    _add_weighted_pair_aliases(d3, "a", "b")
    assert d3["w_a_b"] == "x" and d3["w_b_a"] == "y"


def test_require_component_accepts_known_and_raises_for_unknown() -> None:
    """Tests that _require_component returns comp for known keys and raises
    for unknown."""
    allowed = {"dark_matter": 1, "gas": 2}

    assert _require_component("gas", allowed=allowed) == "gas"

    with pytest.raises(KeyError, match=r"Unknown component"):
        _require_component("stars", allowed=allowed)
