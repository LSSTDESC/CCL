from __future__ import annotations

import numpy as np

from pyccl.nonlimber_fkem.transforms import compute_collection_fft


class _FakeTracerCollection:
    """A fake tracer collection for testing purposes."""

    def __init__(self, ntrc):
        """Initializes the fake tracer collection."""
        self._trc = list(range(ntrc))
        self._cache = {}

    def _get_fkem_fft(self, trc, n_chi, chi_min, chi_max, ell):
        """Retrieves cached FFTLog results if available."""
        key = (trc, n_chi, chi_min, chi_max, ell)
        return self._cache.get(key, (None, None))

    def _set_fkem_fft(self, trc, n_chi, chi_min, chi_max, ell, k, fk):
        """Caches FFTLog results."""
        key = (trc, n_chi, chi_min, chi_max, ell)
        self._cache[key] = (k, fk)

    def get_transfer(self, logk, a):
        """Returns fake transfer functions for testing."""
        # shape (ntrc, len(logk)) when a is array; (ntrc,) when scalar
        if np.ndim(a) == 0:
            return np.ones(len(self._trc))
        else:
            return np.ones((len(self._trc), np.size(logk)))


def test_compute_collection_fft_basic_shape():
    """Test compute_collection_fft returns arrays with correct shapes."""
    ntrc = 2
    n_chi = 5
    clt = _FakeTracerCollection(ntrc)

    chis = [np.linspace(10, 20, n_chi), np.linspace(15, 25, n_chi)]
    kernels = [np.ones(n_chi), 2.0 * np.ones(n_chi)]
    bessels = [0, 0]
    avg_as = [0.5, 0.5]
    chi_min = 10.0
    chi_max = 25.0
    ell = 100.0
    chi_log = np.logspace(np.log10(chi_min), np.log10(chi_max), n_chi)
    a_arr = np.linspace(0.5, 0.6, n_chi)
    growfac_arr = np.ones_like(a_arr)
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
        chi_log,
        a_arr,
        growfac_arr,
        k_low,
    )

    assert k_out.ndim == 1
    assert fks.shape == (ntrc, n_chi)
    assert transfers.shape[0] == ntrc


class _FakeClt:
    """A fake tracer collection that counts FFTLog calls."""

    def __init__(self):
        """Initializes the fake tracer collection."""
        self._trc = ["t0"]
        self._cache = {}
        self.calls = 0

    def _get_fkem_fft(self, tr, n_chi, chi_min, chi_max, ell):
        """Retrieves cached FFTLog results if available, counting calls."""
        self.calls += 1
        return self._cache.get((n_chi, chi_min, chi_max, ell), (None, None))

    def _set_fkem_fft(self, tr, n_chi, chi_min, chi_max, ell, k_fft, fk):
        """Caches FFTLog results."""
        self._cache[(n_chi, chi_min, chi_max, ell)] = (k_fft, fk)

    def get_transfer(self, ln_k, a):
        """Returns fake transfer functions for testing."""
        # 1 tracer → array of shape (1,)
        return np.ones(1)


def test_compute_collection_fft_uses_cache(monkeypatch):
    """Tests that compute_collection_fft uses caching correctly."""
    clt = _FakeClt()
    kernels = [np.ones(5)]
    chis = [np.linspace(1, 10, 5)]
    bessels = [0]
    avg_as = [1.0]
    chi_logspace = np.linspace(1, 10, 5)
    a_arr = np.ones_like(chi_logspace)
    growfac_arr = np.ones_like(chi_logspace)

    # Fake FFTLog that returns deterministic output
    def _fake_fftlog(chi_arr, fchi_arr, ell, nu, *args):
        """Returns fake FFTLog outputs for testing."""
        k = np.linspace(0.1, 1.0, chi_arr.size)
        fk = np.full_like(chi_arr, 2.0)
        return k, fk

    monkeypatch.setattr(
        "pyccl.nonlimber_fkem.transforms._fftlog_transform_general",
        _fake_fftlog,
    )

    # First call: should hit fake FFTLog
    k1, fks1, tr1 = compute_collection_fft(
        clt,
        kernels,
        chis,
        bessels,
        avg_as,
        n_chi=5,
        chi_min=1.0,
        chi_max=10.0,
        ell=10.0,
        chi_logspace_arr=chi_logspace,
        a_arr=a_arr,
        growfac_arr=growfac_arr,
        k_low=1e-3,
    )

    # Second call: should use cache, NOT our fake FFTLog (could check via
    # counter)
    k2, fks2, tr2 = compute_collection_fft(
        clt,
        kernels,
        chis,
        bessels,
        avg_as,
        n_chi=5,
        chi_min=1.0,
        chi_max=10.0,
        ell=10.0,
        chi_logspace_arr=chi_logspace,
        a_arr=a_arr,
        growfac_arr=growfac_arr,
        k_low=1e-3,
    )

    assert np.allclose(k1, k2)
    assert np.allclose(fks1, fks2)
