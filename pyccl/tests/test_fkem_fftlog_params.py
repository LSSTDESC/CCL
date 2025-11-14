"""Unit tests for get_fftlog_params function in nonlimber_fkem.params module."""

from __future__ import annotations

import math
import numpy as np
import pytest

from pyccl.nonlimber_fkem.params import get_fftlog_params


def test_get_fftlog_params_j_zero_uses_defaults():
    """Tests that for j=0, the default nu and plaw are used."""
    nu, deriv, plaw = get_fftlog_params(0, nu_default=1.5, nu_low=0.4, plaw_default=0.7)
    assert nu == pytest.approx(1.5)
    assert deriv == 0.0
    assert plaw == pytest.approx(0.7)


def test_get_fftlog_params_positive_integer():
    """Tests that for positive integer j, the default nu is used, deriv=j, plaw=0."""
    nu, deriv, plaw = get_fftlog_params(3, nu_default=1.5, nu_low=0.4, plaw_default=0.7)
    assert nu == pytest.approx(1.5)
    assert deriv == 3.0
    assert plaw == pytest.approx(0.0)


def test_get_fftlog_params_negative_integer():
    """Tests that for negative integer j, nu=nu_low, deriv=0, plaw=-2."""
    nu, deriv, plaw = get_fftlog_params(-2, nu_default=1.5, nu_low=0.4, plaw_default=0.7)
    assert nu == pytest.approx(0.4)
    assert deriv == 0.0
    assert plaw == -2.0


@pytest.mark.parametrize("j", [0.1, 1.5, np.pi])
def test_get_fftlog_params_rejects_non_integer_orders(j):
    """Tests that non-integer j values raise ValueError."""
    with pytest.raises(ValueError, match="integer order"):
        get_fftlog_params(j)


@pytest.mark.parametrize("j", [math.nan, math.inf, -math.inf])
def test_get_fftlog_params_rejects_non_finite(j):
    """Tests that non-finite j values raise ValueError."""
    with pytest.raises(ValueError, match="must be finite"):
        get_fftlog_params(j)


@pytest.mark.parametrize("bad", ["foo", object(), [1, 2]])
def test_get_fftlog_params_type_error_for_non_numeric(bad):
    """Tests that non-numeric j values raise TypeError."""
    with pytest.raises(TypeError, match="real number"):
        get_fftlog_params(bad)


@pytest.mark.parametrize("name, kwargs", [
    ("nu_default", {"nu_default": math.nan}),
    ("nu_low", {"nu_low": math.inf}),
    ("plaw_default", {"plaw_default": -math.inf}),
])
def test_get_fftlog_params_rejects_non_finite_defaults(name, kwargs):
    """Tests that non-finite default parameters raise ValueError."""
    with pytest.raises(ValueError, match=name):
        get_fftlog_params(0, **kwargs)
