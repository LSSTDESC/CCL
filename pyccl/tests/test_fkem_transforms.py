"""Unit tests for pyccl.nonlimber_fkem.transforms."""

import pytest

import numpy as np

from pyccl.nonlimber_fkem.transforms import compute_collection_fft


class FakeCollection:
    """A fake tracer collection for testing purposes."""
    def __init__(self, n_tracers, transfer_shape="scalar"):
        """Initializes the fake tracer collection."""
        self._trc = list(range(n_tracers))
        self._cache = {}
        self.transfer_shape = transfer_shape
        self.transfer_low_val = 2.0
        self.transfer_avg_val = 4.0

    def get_transfer(self, logk_or_chi, a):
        """Return transfer functions with shapes similar to the real code."""
        n_tracers = len(self._trc)

        # For scalar k: return one value per tracer, shape (n_tracers,)
        if np.ndim(logk_or_chi) == 0:
            if self.transfer_shape == "scalar":
                return np.full(n_tracers, self.transfer_low_val)
            if self.transfer_shape == "per_tracer":
                return np.full(n_tracers, self.transfer_low_val)
            if self.transfer_shape == "per_tracer_chi":
                # degenerate radial dimension of size 1
                return np.full((n_tracers, 1), self.transfer_low_val)
        else:
            # For array k (e.g. k_out): return shape (n_tracers, n_k)
            n_k = np.asarray(logk_or_chi).size
            if self.transfer_shape == "scalar":
                return np.full((n_tracers, n_k), self.transfer_avg_val)
            if self.transfer_shape == "per_tracer":
                return np.full((n_tracers, n_k), self.transfer_avg_val)
            if self.transfer_shape == "per_tracer_chi":
                return np.full((n_tracers, n_k), self.transfer_low_val)

        raise RuntimeError("Unknown transfer_shape")

    def _get_fkem_fft(self, trc, n_chi, chi_min, chi_max, ell):
        """Retrieves cached FFTLog results if available."""
        key = (trc, n_chi, chi_min, chi_max, ell)
        return self._cache.get(key, (None, None))

    def _set_fkem_fft(self, trc, n_chi, chi_min, chi_max, ell, k_fft, fk):
        """Caches FFTLog results."""
        key = (trc, n_chi, chi_min, chi_max, ell)
        self._cache[key] = (k_fft, fk)


def test_compute_collection_fft_basic_scalar():
    """Tests that compute_collection_fft returns arrays
    with correct shapes for scalar transfers."""
    n_tracers = 2
    clt = FakeCollection(n_tracers, transfer_shape="scalar")

    n_chi = 5
    chi_min = 10.0
    chi_max = 100.0
    chi_logspace_arr = np.geomspace(chi_min, chi_max, n_chi)

    # simple monotonic chis and kernels
    chis = [np.linspace(chi_min, chi_max, n_chi) for _ in range(n_tracers)]
    kernels = [np.ones(n_chi), 2 * np.ones(n_chi)]
    bessels = [0, 0]
    avg_as = [0.5, 0.6]
    a_arr = np.linspace(0.5, 1.0, n_chi)
    growfac_arr = np.ones_like(chi_logspace_arr)
    ell = 10.0
    k_low = 1e-3

    k_out, fks, transfers = compute_collection_fft(
        clt,
        kernels,
        chis,
        bessels,
        avg_as,
        n_chi,
        chi_min,
        chi_max,
        ell,
        chi_logspace_arr,
        a_arr,
        growfac_arr,
        k_low,
    )

    assert k_out.ndim == 1
    assert k_out.size == n_chi
    assert fks.shape == (n_tracers, n_chi)
    assert transfers.shape[0] == n_tracers
    assert np.all(np.isfinite(k_out))
    assert np.all(np.isfinite(fks))
    assert np.all(np.isfinite(transfers))


def test_compute_collection_fft_uses_cache():
    """Tests that compute_collection_fft uses caching correctly."""
    n_tracers = 1
    clt = FakeCollection(n_tracers, transfer_shape="scalar")

    n_chi = 4
    chi_min = 1.0
    chi_max = 10.0
    chi_logspace_arr = np.geomspace(chi_min, chi_max, n_chi)
    chis = [np.linspace(chi_min, chi_max, n_chi)]
    kernels = [np.ones(n_chi)]
    bessels = [0]
    avg_as = [0.8]
    a_arr = np.linspace(0.5, 1.0, n_chi)
    growfac_arr = np.ones_like(chi_logspace_arr)
    ell = 5.0
    k_low = 1e-3

    # First call populates cache
    k_out1, fks1, transfers1 = compute_collection_fft(
        clt,
        kernels,
        chis,
        bessels,
        avg_as,
        n_chi,
        chi_min,
        chi_max,
        ell,
        chi_logspace_arr,
        a_arr,
        growfac_arr,
        k_low,
    )

    # Second call should retrieve from cache
    k_out2, fks2, transfers2 = compute_collection_fft(
        clt,
        kernels,
        chis,
        bessels,
        avg_as,
        n_chi,
        chi_min,
        chi_max,
        ell,
        chi_logspace_arr,
        a_arr,
        growfac_arr,
        k_low,
    )

    assert np.allclose(k_out1, k_out2)
    assert np.allclose(fks1, fks2)
    assert np.allclose(transfers1, transfers2)


def test_compute_collection_fft_rejects_empty_kernels():
    """Tests that compute_collection_fft rejects empty kernel lists."""
    clt = FakeCollection(n_tracers=0)
    with pytest.raises(ValueError, match="must contain at least one tracer"):
        compute_collection_fft(
            clt,
            kernels=[],
            chis=[],
            bessels=[],
            avg_as=[],
            n_chi=5,
            chi_min=1.0,
            chi_max=10.0,
            ell=5.0,
            chi_logspace_arr=np.geomspace(1.0, 10.0, 5),
            a_arr=np.ones(5),
            growfac_arr=np.ones(5),
            k_low=1e-3,
        )


def test_compute_collection_fft_rejects_length_mismatch():
    """Tests that compute_collection_fft rejects length mismatches."""
    clt = FakeCollection(n_tracers=1)
    with pytest.raises(ValueError, match="length mismatch"):
        compute_collection_fft(
            clt,
            kernels=[np.ones(3)],
            chis=[np.ones(3)],
            bessels=[0, 1],    # mismatch
            avg_as=[0.5],
            n_chi=3,
            chi_min=1.0,
            chi_max=10.0,
            ell=5.0,
            chi_logspace_arr=np.geomspace(1.0, 10.0, 3),
            a_arr=np.ones(3),
            growfac_arr=np.ones(3),
            k_low=1e-3,
        )


def test_compute_collection_fft_rejects_nonmonotonic_chi_logspace():
    """Tests that compute_collection_fft rejects
     non-monotonic chi_logspace_arr."""
    clt = FakeCollection(n_tracers=1)
    chi_logspace_arr = np.array([1.0, 2.0, 1.5])  # not strictly increasing
    with pytest.raises(ValueError, match="must be strictly increasing"):
        compute_collection_fft(
            clt,
            kernels=[np.ones(3)],
            chis=[np.linspace(1.0, 3.0, 3)],
            bessels=[0],
            avg_as=[0.5],
            n_chi=3,
            chi_min=1.0,
            chi_max=3.0,
            ell=5.0,
            chi_logspace_arr=chi_logspace_arr,
            a_arr=np.ones(3),
            growfac_arr=np.ones(3),
            k_low=1e-3,
        )


def test_compute_collection_fft_rejects_bad_k_low():
    """Tests that compute_collection_fft rejects non-positive k_low."""
    clt = FakeCollection(n_tracers=1)
    chi_logspace_arr = np.geomspace(1.0, 3.0, 3)
    with pytest.raises(ValueError, match="k_low.*positive and finite"):
        compute_collection_fft(
            clt,
            kernels=[np.ones(3)],
            chis=[np.linspace(1.0, 3.0, 3)],
            bessels=[0],
            avg_as=[0.5],
            n_chi=3,
            chi_min=1.0,
            chi_max=3.0,
            ell=5.0,
            chi_logspace_arr=chi_logspace_arr,
            a_arr=np.ones(3),
            growfac_arr=np.ones(3),
            k_low=0.0,
        )


def test_compute_collection_fft_rejects_nonfinite_transfer_low():
    """Tests that compute_collection_fft rejects non-finite transfer_low."""
    n_tracers = 1

    class BadTransferCollection(FakeCollection):
        """A fake tracer collection that returns non-finite transfer_low."""
        def get_transfer(self, logk_or_chi, a):
            """Returns non-finite transfer_low for testing."""
            if np.ndim(logk_or_chi) == 0:
                return np.array(np.nan)
            return np.array(1.0)

    clt = BadTransferCollection(n_tracers, transfer_shape="scalar")

    chi_logspace_arr = np.geomspace(1.0, 3.0, 3)
    with pytest.raises(RuntimeError,
                       match="non-finite values in 'transfer_low'"):
        compute_collection_fft(
            clt,
            kernels=[np.ones(3)],
            chis=[np.linspace(1.0, 3.0, 3)],
            bessels=[0],
            avg_as=[0.5],
            n_chi=3,
            chi_min=1.0,
            chi_max=3.0,
            ell=5.0,
            chi_logspace_arr=chi_logspace_arr,
            a_arr=np.ones(3),
            growfac_arr=np.ones(3),
            k_low=1e-3,
        )
