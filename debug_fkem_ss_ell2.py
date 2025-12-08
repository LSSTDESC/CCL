#!/usr/bin/env python
import numpy as np

import pyccl as ccl

import benchmarks.test_nonlimber as nonlimber_bench

from pyccl.nonlimber_fkem.legacy_fkem import legacy_nonlimber_fkem
from pyccl.nonlimber_fkem.chi_grid import build_chi_grid
from pyccl.nonlimber_fkem.single_ell import compute_single_ell


def main():
    # Use the same setup as the benchmark
    # Use the same setup as the benchmark, but unwrap the pytest fixture
    set_up_fixture = nonlimber_bench.set_up

    try:
        # Standard @pytest.fixture case: decorator keeps the original function in __wrapped__
        raw_set_up = set_up_fixture.__wrapped__
    except AttributeError:
        # Fallback for safety: grab the underlying function via pytest’s internals
        raw_set_up = set_up_fixture._pytestfixturefunction.func

    cosmo, ells, tracers1, tracers2, truth, errors, indices = raw_set_up()

    # We look at the first shear–shear pair
    i1, i2 = indices["ss"][0]
    t1 = tracers1["ss"][i1]
    t2 = tracers2["ss"][i2]

    ell = 2.0
    ell_idx = np.where(ells == ell)[0][0]

    # ------------------------------------------------------------------
    # 1) NEW FKEM path: reproduce the driver logic just for ell = 2
    # ------------------------------------------------------------------
    print("=== NEW FKEM (compute_single_ell) ===")

    # Power spectra: match test_nonlimber default
    p_of_k_a = p_of_k_a_lin = ccl.DEFAULT_POWER_SPECTRUM
    psp_lin = cosmo.parse_pk2d(p_of_k_a_lin, is_linear=True)
    psp_nonlin = cosmo.parse_pk2d(p_of_k_a, is_linear=False)

    if isinstance(p_of_k_a_lin, ccl.Pk2D):
        pk = p_of_k_a_lin
    else:
        pk = cosmo.get_linear_power(name=p_of_k_a_lin)

    # Tracer kernels, bessel orders, f_ell, etc.
    kernels_t1, chis_t1 = t1.get_kernel()
    kernels_t2, chis_t2 = t2.get_kernel()
    bessels_t1 = t1.get_bessel_derivative()
    bessels_t2 = t2.get_bessel_derivative()
    fll_t1 = t1.get_f_ell(ells)
    fll_t2 = t2.get_f_ell(ells)
    avg_a1s = t1.get_avg_weighted_a()
    avg_a2s = t2.get_avg_weighted_a()

    # FKEM chi-grid: use the same builder as in the new code
    chi_log, dlnr, chi_min_eff, chi_max_eff, n_chi_eff = build_chi_grid(
        chis_t1, chis_t2, chi_min=None, n_chi=None, warn=False
    )

    k_low = 1.0e-5
    kpow = 3

    # Build C-level tracer collections (same as in legacy)
    status = 0
    t1_coll, status = ccl.lib.cl_tracer_collection_t_new(status)
    ccl.check(status)
    t2_coll, status = ccl.lib.cl_tracer_collection_t_new(status)
    ccl.check(status)

    for tr in t1._trc:
        status = ccl.lib.add_cl_tracer_to_collection(t1_coll, tr, status)
        ccl.check(status)
    for tr in t2._trc:
        status = ccl.lib.add_cl_tracer_to_collection(t2_coll, tr, status)
        ccl.check(status)

    # Call the new single-ell FKEM (this will print cls_lin_fkem_new)
    cl_new, limber_ref_new, rel_diff_new = compute_single_ell(
        cosmo=cosmo,
        ell_idx=ell_idx,
        ell=ell,
        t1=t1_coll,
        t2=t2_coll,
        psp_lin=psp_lin,
        psp_nonlin=psp_nonlin,
        pk=pk,
        clt1=t1,
        clt2=t2,
        kernels_t1=kernels_t1,
        kernels_t2=kernels_t2,
        chis_t1=chis_t1,
        chis_t2=chis_t2,
        bessels_t1=bessels_t1,
        bessels_t2=bessels_t2,
        fll_t1=fll_t1,
        fll_t2=fll_t2,
        chi_logspace_arr=chi_log,
        chi_min=chi_min_eff,
        chi_max=chi_max_eff,
        n_chi=n_chi_eff,
        dlnr=dlnr,
        avg_a1s=avg_a1s,
        avg_a2s=avg_a2s,
        k_low=k_low,
        kpow=kpow,
    )

    print("  cl_new(ell=2)         =", cl_new)
    print("  limber_ref_new(ell=2) =", limber_ref_new)
    print("  rel_diff_new          =", rel_diff_new)

    # ------------------------------------------------------------------
    # 2) LEGACY FKEM path: same tracers, same cosmology, ell=[2]
    # ------------------------------------------------------------------
    print("\n=== LEGACY FKEM (legacy_nonlimber_fkem) ===")

    ls = np.array([ell])
    # These params should match the defaults used in the driver
    params = dict(
        pk_linear=p_of_k_a_lin,
        limber_max_error=0.01,
        Nchi=None,
        chi_min=None,
    )

    l_limber_legacy, cells_legacy, status_legacy = legacy_nonlimber_fkem(
        cosmo=cosmo,
        clt1=t1,
        clt2=t2,
        p_of_k_a=p_of_k_a,
        ls=ls,
        l_limber="auto",
        **params,
    )

    print("  l_limber_legacy       =", l_limber_legacy)
    print("  cells_legacy[0]       =", cells_legacy[0])
    print("  status_legacy         =", status_legacy)


if __name__ == "__main__":
    main()
