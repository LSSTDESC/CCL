"""Unit tests for compute_single_ell in the FKEM non-Limber module."""

from __future__ import annotations

import pytest

import numpy as np

from pyccl.nonlimber_fkem.single_ell import compute_single_ell


class _DummyCosmo:
    """Dummy cosmology object; only identity matters."""

    def __init__(self):
        self.cosmo = object()


class _DummyTracerCollection:
    """Minimal tracer collection object; only identity matters."""

    pass


def _fake_scale_factor_of_chi(_cosmo, chi_arr):
    """Returns scale factor = 1 for all chi."""
    return np.ones_like(chi_arr, dtype=float)


def _fake_growth_factor(_cosmo, a_arr):
    """Returns growth factor = 1 for all a."""
    return np.ones_like(a_arr, dtype=float)


def _fake_angular_cl_vec_limber(
    _cosmo_cosmo, _t1, _t2, pk_id, ells, _integ_type, _n_ell, status
):
    """Returns deterministic Limber C_ell values for testing."""
    if pk_id == "LIN":
        cl_val = 1.0
    elif pk_id == "NL":
        cl_val = 3.0
    else:
        cl_val = 0.0
    # Return a length-1 array, like the real function
    return np.array([cl_val], dtype=float), status


def _fake_compute_collection_fft(
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
):
    """Returns fake FFTLog outputs for testing."""
    n_tracers = len(kernels)
    k = np.full(n_chi, 0.5, dtype=float)
    fks = np.full((n_tracers, n_chi), 2.0, dtype=float)
    transfers = np.full(n_tracers, 0.5, dtype=float)
    return k, fks, transfers


def _pk_const(k, a, cosmo=None):
    """Simple constant power spectrum P(k) = 4."""
    return np.full_like(k, 4.0, dtype=float)


def test_compute_single_ell_basic_fk_em_correction(monkeypatch):
    """Tests that compute_single_ell computes the FKEM correction correctly."""

    # Need to monkey patch some stuff to control the test environment:
    # first we patch scale factor and growth factor
    monkeypatch.setattr(
        "pyccl.nonlimber_fkem.single_ell.ccl.scale_factor_of_chi",
        _fake_scale_factor_of_chi,
    )
    monkeypatch.setattr(
        "pyccl.nonlimber_fkem.single_ell.ccl.growth_factor",
        _fake_growth_factor,
    )

    # we then patch Limber integrals
    monkeypatch.setattr(
        "pyccl.nonlimber_fkem.single_ell.lib.angular_cl_vec_limber",
        _fake_angular_cl_vec_limber,
    )

    # and then patch FFTLog collection
    monkeypatch.setattr(
        "pyccl.nonlimber_fkem.single_ell.compute_collection_fft",
        _fake_compute_collection_fft,
    )

    # her I set fake cosmology and tracer collections
    cosmo = _DummyCosmo()
    clt = _DummyTracerCollection()

    # Here i set up FKEM grid and kernels
    n_chi = 5
    chi_min = 1.0
    chi_max = 10.0
    chi_log = np.linspace(chi_min, chi_max, n_chi)
    dlnr = 0.1  # arbitrary but fixed which is important for the test

    kernels = [np.ones(n_chi, dtype=float)]
    chis_list = [np.linspace(chi_min, chi_max, n_chi)]
    bessels = [0]
    fll = [np.array([1.0], dtype=float)]  # f_ell = 1 for ell_idx = 0

    # Call the function under test
    cl_out, limber_ref, rel_diff = compute_single_ell(
        cosmo=cosmo,
        ell_idx=0,
        ell=10.0,
        t1="T1",
        t2="T2",
        psp_lin="LIN",
        psp_nonlin="NL",
        pk=_pk_const,
        clt1=clt,
        clt2=clt,  # same object → clt1 is clt2 branch
        kernels_t1=kernels,
        kernels_t2=kernels,
        chis_t1=chis_list,
        chis_t2=chis_list,
        bessels_t1=bessels,
        bessels_t2=bessels,
        fll_t1=fll,
        fll_t2=fll,
        chi_logspace_arr=chi_log,
        chi_min=chi_min,
        chi_max=chi_max,
        n_chi=n_chi,
        dlnr=dlnr,
        avg_a1s=[1.0],
        avg_a2s=[1.0],
        k_low=1e-3,
        kpow=0,  # so k^kpow = 1
    )

    # Limber pieces:
    c_lin = 1.0
    c_nl = 3.0

    per_sample = 4.0
    total_sum = n_chi * per_sample  # 5 * 4 = 20
    cls_lin_fkem = total_sum * dlnr * (2.0 / np.pi) * 1.0 * 1.0

    expected_cl_out = c_nl - c_lin + cls_lin_fkem
    expected_limber_ref = c_nl
    expected_rel_diff = abs(expected_cl_out / expected_limber_ref - 1.0)

    assert limber_ref == pytest.approx(expected_limber_ref)
    assert cl_out == pytest.approx(expected_cl_out)
    assert rel_diff == pytest.approx(expected_rel_diff)
    assert rel_diff >= 0.0
