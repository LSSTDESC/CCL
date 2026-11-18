"""Tests for the base Baryons class API."""

from __future__ import annotations

import pytest

import pyccl as ccl
from pyccl.baryons import Baryons


class _DummyBaryons(Baryons):
    """A dummy subclass of Baryons for testing."""

    # CCLNamedClass expects subclasses to have a name
    name = "DummyBaryons"

    def __init__(self):
        """Initializes the base class and a list to record calls."""
        self.calls: list[tuple[object, object]] = []

    def _include_baryonic_effects(self, cosmo, pk):
        """Records the inputs and returns the pk unchanged."""
        self.calls.append((cosmo, pk))
        return pk


def test_baryons_is_abstract():
    """Tests that Baryons cannot be instantiated directly."""
    with pytest.raises(TypeError, match="abstract"):
        Baryons()  # type: ignore[abstract]


def test_include_baryonic_effects_delegates_to_impl():
    """Tests that include_baryonic_effects calls the subclass method."""
    cosmo = ccl.CosmologyVanillaLCDM()
    pk = cosmo.get_nonlin_power()

    bar = _DummyBaryons()
    pk_out = bar.include_baryonic_effects(cosmo, pk)

    # Retruned object is whatever the subclass returns
    assert pk_out is pk

    # And the subclass saw exactly the inputs we passed in
    assert len(bar.calls) == 1
    call_cosmo, call_pk = bar.calls[0]
    assert call_cosmo is cosmo
    assert call_pk is pk


class _NoneBaryons(Baryons):
    """A subclass of Baryons that returns None."""
    name = "NoneBaryons"

    def _include_baryonic_effects(self, cosmo, pk):
        """A helper that returns None."""
        return None


def test_include_baryonic_effects_allows_none():
    """Tests that include_baryonic_effects can return None."""
    cosmo = ccl.CosmologyVanillaLCDM()
    pk = cosmo.get_nonlin_power()

    bar = _NoneBaryons()
    pk_out = bar.include_baryonic_effects(cosmo, pk)
    assert pk_out is None
