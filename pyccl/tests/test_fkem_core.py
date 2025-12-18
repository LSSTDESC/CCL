"""Tests for nonlimber_fkem core functionality."""

from __future__ import annotations

import pytest

import numpy as np

import pyccl as ccl
from pyccl.nonlimber_fkem.core import nonlimber_fkem


def pk_constant_one(k, a):
    """Returns a constant power spectrum P(k, a) = 1 everywhere."""
    # k and a can be arrays; we just return ones like k
    return np.ones_like(k, dtype=float)


def build_constant_pk2d():
    """Returns a Pk2D with P(k, a) = 1 using from_function."""
    # is_logp=False so we store the actual value, not log(P)
    return ccl.Pk2D.from_function(pk_constant_one, is_logp=False)


@pytest.fixture
def cosmo():
    """Simple LCDM cosmology for FKEM interface tests."""
    return ccl.CosmologyVanillaLCDM()


@pytest.fixture
def tracer1(cosmo):
    """Basic number-counts tracer for FKEM interface tests."""
    z = np.linspace(0.01, 2.0, 5)
    n = np.ones_like(z)
    b = np.ones_like(z)  # constant bias = 1
    return ccl.NumberCountsTracer(
        cosmo,
        dndz=(z, n),
        bias=(z, b),
        has_rsd=False,
    )


@pytest.fixture
def tracer2(tracer1):
    """Use the same tracer for both legs."""
    return tracer1


@pytest.fixture
def pk_nonlin():
    """Non-linear P(k) spec understood by Cosmology.parse_pk2d."""
    return "delta_matter:delta_matter"


@pytest.fixture
def pk_lin():
    """Linear P(k) spec for pk_linear argument."""
    return "delta_matter:delta_matter"


class _DummyTracer:
    """A dummy tracer for testing purposes."""

    def __init__(self):
        """Initializes the dummy tracer."""
        self._trc = (
            []
        )  # Just something so identity matters, not used internally

    def get_kernel(self):
        """Returns a dummy kernel and chi array."""
        chis = np.linspace(10.0, 1000.0, 5)
        kernels = np.ones_like(chis)
        return [kernels], [chis]

    def get_bessel_derivative(self):
        """Returns a dummy Bessel derivative (0)."""
        return [0]

    def get_f_ell(self, ells):
        """Returns a dummy f_ell (ones)."""
        return [np.ones_like(ells, dtype=float)]

    def get_avg_weighted_a(self):
        """Returns a dummy average scale factor (1.0)."""
        return [1.0]


class _DummyCosmo:
    """A dummy cosmology for testing purposes."""

    def __init__(self):
        """Initializes the dummy cosmology."""
        # just something so code that expects `cosmo.cosmo` doesn't blow up
        self.cosmo = (
            object()
        )

    def parse_pk2d(self, p, is_linear):
        """Parses a power spectrum spec into a constant Pk2D.

        We ignore `p` and `is_linear` and always return a constant Pk2D.
        This avoids any real Cosmology/parse_pk2d logic.
        """
        return build_constant_pk2d()

    def get_linear_power(self, name):
        """Returns a constant linear Pk2D."""
        return build_constant_pk2d()


def _setup_dummy():
    """Sets up a dummy cosmology and tracers for testing."""
    cosmo = _DummyCosmo()
    t = _DummyTracer()
    return cosmo, t, t  # use same tracer as both tracer1 and tracer2


# rel_diff pattern for the auto-transition test:
# [0.2, 0.05, 0.05, 0.05, 0.05]
# With limber_max_error = 0.1 and n_consec_ell = 3,
# we should stop at ell=40 (the 4th ells entry).
REL_DIFFS = [0.2, 0.05, 0.05, 0.05, 0.05]


def fake_single_ell(*args, **kwargs):
    """Fake compute_single_ell that returns controlled rel_diffs.

    We only care about the ell index (args[1]) here and ignore the rest
    of the arguments. It returns (cl_limber, cl_nonlimber, rel_diff).
    """
    ell_idx = args[1]  # by construction in nonlimber_fkem
    rel = REL_DIFFS[ell_idx]
    return 1.0, 1.0, rel


class DummyTracerCollection:
    """Minimal dummy tracer-collection object.

    nonlimber_fkem only passes these into compute_single_ell. Since
    we monkeypatch compute_single_ell, the internal structure doesn't
    matter for this test.
    """

    def __init__(self):
        """Initializes the dummy tracer collection."""
        self.tracers = []


def fake_build_tracer_collections(tracer1, tracer2):
    """Fake build_tracer_collections that bypasses the C layer entirely."""
    return DummyTracerCollection(), DummyTracerCollection()


@pytest.fixture
def dummy_cosmo():
    """Fixture returning a dummy cosmology."""
    cosmo, _, _ = _setup_dummy()
    return cosmo


@pytest.fixture
def dummy_tracer():
    """Fixture returning a dummy tracer."""
    _, t1, _ = _setup_dummy()
    return t1


def test_nonlimber_auto_transition_uses_n_consec(monkeypatch):
    """Tests that nonlimber_fkem auto transition respects n_consec_ell."""
    cosmo, tracer1, tracer2 = _setup_dummy()
    ells = np.array([10, 20, 30, 40, 50], dtype=float)

    # Monkeypatch compute_single_ell to use our fake with controlled rel_diffs
    monkeypatch.setattr(
        "pyccl.nonlimber_fkem.core.compute_single_ell",
        fake_single_ell,
    )

    # Monkeypatch build_tracer_collections so we never touch the C API
    monkeypatch.setattr(
        "pyccl.nonlimber_fkem.core.build_tracer_collections",
        fake_build_tracer_collections,
    )

    ell_limber, cells, status = nonlimber_fkem(
        cosmo,
        tracer1,
        tracer2,
        p_of_k_a="delta_matter:delta_matter",  # ignored by _DummyCosmo
        ell=ells,
        ell_limber="auto",
        pk_linear="delta_matter:delta_matter",  # ignored by _DummyCosmo
        limber_max_error=0.1,
        n_chi_fkem=5,
        chi_min_fkem=1.0,
        n_consec_ell=3,
    )

    # With REL_DIFFS and n_consec_ell=3, we should stop at ell=40
    assert ell_limber == 40.0
    # We should have computed up to ell=40: indices 0,1,2,3 -> 4 entries
    assert cells.shape[0] == 4
    assert status == 0


def test_nonlimber_requires_increasing_ells_in_auto(dummy_cosmo, dummy_tracer):
    """Tests that nonlimber_fkem raises error for non-increasing ells."""
    with pytest.raises(ValueError, match="strictly increasing"):
        nonlimber_fkem(
            dummy_cosmo,
            dummy_tracer,
            dummy_tracer,
            p_of_k_a="delta_matter:delta_matter",
            ell=[10, 5, 20],
            ell_limber="auto",
            pk_linear="delta_matter:delta_matter",
            limber_max_error=0.1,
            n_chi_fkem=5,
            chi_min_fkem=1.0,
        )


def test_fkem_ls_deprecated_matches_ell(
        cosmo,
        tracer1,
        tracer2,
        pk_nonlin,
        pk_lin):
    """Tests that ls and ell arguments give the same result."""
    ells = np.array([10, 20, 30])

    with pytest.warns(FutureWarning, match="`ls` is deprecated"):
        ell_limber_old, cells_old, status_old = nonlimber_fkem(
            cosmo,
            tracer1,
            tracer2,
            p_of_k_a=pk_nonlin,
            ls=ells,
            l_limber=30,
            pk_linear=pk_lin,
            limber_max_error=0.01,
            fkem_Nchi=64,
            chi_min_fkem=None,
        )

    ell_limber_new, cells_new, status_new = nonlimber_fkem(
        cosmo,
        tracer1,
        tracer2,
        p_of_k_a=pk_nonlin,
        ell=ells,
        ell_limber=30,
        pk_linear=pk_lin,
        limber_max_error=0.01,
        n_chi_fkem=64,
        chi_min_fkem=None,
    )

    np.testing.assert_allclose(cells_old, cells_new)
    assert status_old == status_new
    assert ell_limber_old == ell_limber_new


def test_fkem_ls_and_ell_together_raises(
        cosmo,
        tracer1,
        tracer2,
        pk_nonlin,
        pk_lin):
    """Tests that using both ls and ell raises an error."""
    ells = np.array([10, 20])

    with pytest.raises(ValueError, match="only one of `ls`"):
        nonlimber_fkem(
            cosmo,
            tracer1,
            tracer2,
            p_of_k_a=pk_nonlin,
            ls=ells,
            ell=ells,
            pk_linear=pk_lin,
            limber_max_error=0.01,
        )
