"""Unit tests for prepare_power_spectra function in Non-Limber FKEM module."""

from __future__ import annotations

from functools import partial
import pytest

import pyccl as ccl
from pyccl.nonlimber_fkem.power_spectra import prepare_power_spectra


class _FakeCosmoNoMethods:
    """A fake cosmology object that lacks required methods."""

    pass


def simple_pk(k, a, is_linear: bool) -> float:
    """A simple power spectrum function."""
    return k * 0 + a * 0 + (1.0 if is_linear else 2.0)


class _FakeCosmo:
    """A fake cosmology object that provides required methods."""

    def __init__(self):
        self._parsed: list[tuple[object, bool]] = []

    def parse_pk2d(self, p, is_linear: bool):
        """Parses a power spectrum name into a Pk2D object."""
        pk_func = partial(simple_pk, is_linear=is_linear)
        pk2d = ccl.Pk2D.from_function(pk_func)
        self._parsed.append((p, is_linear))
        return pk2d

    def get_linear_power(self, name: str):
        """Returns a linear Pk2D object for the given name."""
        pk_func = partial(simple_pk, is_linear=True)
        return ccl.Pk2D.from_function(pk_func)


def _constant_pk2d(value: float) -> ccl.Pk2D:
    """Helper to build a constant Pk2D without nested functions."""

    def _pk(k, a):
        return k * 0 + a * 0 + value

    return ccl.Pk2D.from_function(_pk)


def test_prepare_power_spectra_type_mismatch_returns_none_and_warns(recwarn):
    """Test prepare_power_spectra returns None and warns on type mismatch."""
    cosmo = _FakeCosmo()
    p_nonlin = "delta_matter:delta_matter"
    p_lin = _constant_pk2d(1.0)

    psp_lin, psp_nonlin, pk_1d = prepare_power_spectra(cosmo, p_nonlin, p_lin)

    assert psp_lin is None
    assert psp_nonlin is None
    assert pk_1d is None

    w = recwarn.pop()
    assert "same type" in str(w.message)


def test_prepare_power_spectra_requires_methods():
    """Tests that prepare_power_spectra raises if methods are missing."""
    cosmo = _FakeCosmoNoMethods()
    with pytest.raises(TypeError, match="parse_pk2d"):
        prepare_power_spectra(
            cosmo,
            "delta_matter:delta_matter",
            "delta_matter:delta_matter",
        )


def test_prepare_power_spectra_success_for_string_names():
    """Tests that prepare_power_spectra works for string names."""
    cosmo = _FakeCosmo()
    psp_lin, psp_nonlin, pk_1d = prepare_power_spectra(
        cosmo,
        p_nonlin="delta_matter:delta_matter",
        p_lin="delta_matter:delta_matter",
    )

    assert isinstance(psp_lin, ccl.Pk2D)
    assert isinstance(psp_nonlin, ccl.Pk2D)
    assert isinstance(pk_1d, ccl.Pk2D)
