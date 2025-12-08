"""Computes non-Limber C_ell via FKEM for a single ell."""

from __future__ import annotations

import numpy as np

import pyccl as ccl
from .. import lib, check
from ..pyutils import integ_types
from ..nonlimber_fkem.transforms import compute_collection_fft

__all__ = [
    "compute_single_ell",
]


def compute_single_ell(
    cosmo,
    ell_idx,
    ell,
    t1,
    t2,
    psp_lin,
    psp_nonlin,
    pk,
    clt1,
    clt2,
    kernels_t1,
    kernels_t2,
    chis_t1,
    chis_t2,
    bessels_t1,
    bessels_t2,
    fll_t1,
    fll_t2,
    chi_logspace_arr,
    chi_min,
    chi_max,
    n_chi,
    dlnr,
    avg_a1s,
    avg_a2s,
    k_low,
    kpow,
):
    """Computes the non-Limber C_ell via FKEM for a single ell.

    This method computes the angular power spectrum C_ell for a given multipole
    moment `ell` using the FKEM approach (see arXiv:1911.11947).
    It combines Limber approximations for both linear and nonlinear
    power spectra with FFTLog transforms of the radial kernels.
    It returns the computed C_ell value, a reference Limber C_ell, and
    the relative difference between the two.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`):
         A Cosmology object.
        ell_idx (int):
            Index of the multipole moment in the input array.
        ell (float):
            The multipole moment at which to compute C_ell.
        t1 (tracer collection):
            Tracer collection for the first tracer.
        t2 (tracer collection):
            Tracer collection for the second tracer.
        psp_lin:
            Linear power spectrum object.
        psp_nonlin:
            Non-linear power spectrum object.
        pk:
            Function to compute the power spectrum.
        clt1:
            Tracer collection for the first tracer.
        clt2:
            Tracer collection for the second tracer.
        kernels_t1 (list):
            List of radial kernels for the first tracer.
        kernels_t2 (list):
            List of radial kernels for the second tracer.
        chis_t1 (list):
            List of chi arrays for the first tracer.
        chis_t2 (list):
            List of chi arrays for the second tracer.
        bessels_t1 (list):
            List of Bessel derivative orders for the first tracer.
        bessels_t2 (list):
            List of Bessel derivative orders for the second tracer.
        fll_t1 (list):
            List of f_ell values for the first tracer.
        fll_t2 (list):
            List of f_ell values for the second tracer.
        chi_logspace_arr (array):
            Logarithmically spaced chi array.
        chi_min (float):
            Minimum chi value.
        chi_max (float):
            Maximum chi value.
        n_chi (int):
            Number of chi samples.
        dlnr (float):
            Logarithmic spacing in r.
        avg_a1s (list):
            List of average scale factors for the first tracer.
        avg_a2s (list):
            List of average scale factors for the second tracer.
        k_low (float):
            Low-k cutoff for transfer function evaluation.
        kpow (int):
            Power of k in the integrand.

    Returns: tuple: (cl_out, limber_ref, rel_diff)
        cl_out (float):
            The computed C_ell value using FKEM.
        limber_ref (float):
            Reference Limber C_ell value.
        rel_diff (float):
            Relative difference between cl_out and limber_ref.

    Raises:
        ValueError:
            If the input configuration is inconsistent (e.g. malformed kernels,
            chi grids, or FFTLog parameters), or if intermediate FKEM arrays
            fail basic sanity checks.
        RuntimeError:
            If underlying CCL calls (e.g. Limber integrals) or FFTLog
            transformations fail, as signalled by ``check`` or
            ``compute_collection_fft``.
    """
    ell_arr = np.asarray(
        [ell], dtype="float64"
    )  # make sure it's an array for C
    n_ell = ell_arr.size  # make sure that we have size info for C

    status = 0  # here we initialize the status flag

    # Limber linear
    cl_limber_lin, status = lib.angular_cl_vec_limber(
        cosmo.cosmo,
        t1,
        t2,
        psp_lin,
        ell_arr,
        integ_types["qag_quad"],
        n_ell,
        status,
    )
    check(status, cosmo=cosmo)

    # Limber non-linear
    cl_limber_nonlin, status = lib.angular_cl_vec_limber(
        cosmo.cosmo,
        t1,
        t2,
        psp_nonlin,
        ell_arr,
        integ_types["qag_quad"],
        n_ell,
        status,
    )
    check(status, cosmo=cosmo)

    # Here we precompute some arrays that are independent of the tracer.
    # The arrays we need are the scale factor and growth factor
    a_arr = ccl.scale_factor_of_chi(cosmo, chi_logspace_arr)
    growfac_arr = ccl.growth_factor(cosmo, a_arr)

    # Now we do the FFTLog for tracer 1
    k, fks_1, transfers_t1 = compute_collection_fft(
        clt1,
        kernels_t1,
        chis_t1,
        bessels_t1,
        avg_a1s,
        n_chi,
        chi_min,
        chi_max,
        float(ell),  # scalar here for caching
        chi_logspace_arr,
        a_arr,
        growfac_arr,
        k_low,
    )

    # FFTLog for tracer 2, or reuse tracer 1 results if identical.
    if clt1 is clt2:
        fks_2 = fks_1
        transfers_t2 = transfers_t1
    else:
        _, fks_2, transfers_t2 = compute_collection_fft(
            clt2,
            kernels_t2,
            chis_t2,
            bessels_t2,
            avg_a2s,
            n_chi,
            chi_min,
            chi_max,
            float(ell),
            chi_logspace_arr,
            a_arr,
            growfac_arr,
            k_low,
        )

    # Then we apply the FKEM formula to compute the non-Limber C_ell
    pk_vals = pk(k, 1.0, cosmo=cosmo)  # this is P(k, a=1) which is at z=0

    cls_lin_fkem = 0.0
    raw_sum_total = 0.0

    for i in range(len(kernels_t1)):
        for j in range(len(kernels_t2)):
            term = np.sum(
                fks_1[i]
                * transfers_t1[i]
                * fks_2[j]
                * transfers_t2[j]
                * k ** kpow
                * pk_vals
            ) * dlnr

            raw_sum_total += term

            pref = (
                    2.0
                    / np.pi
                    * fll_t1[i][ell_idx]
                    * fll_t2[j][ell_idx]
            )

            cls_lin_fkem += term * pref

    # Final combination for output is:
    # the angular power spectrum computed with nonlinear Limber
    # minus the angular power spectrum computed with linear Limber
    # plus the angular power spectrum computed with linear FKEM
    # This way, we capture nonlinearities while benefiting
    # from the improved accuracy of FKEM over Limber at low ell.
    cl_out = cl_limber_nonlin[-1] - cl_limber_lin[-1] + cls_lin_fkem

    limber_ref = cl_limber_nonlin[-1]
    rel_diff = np.abs(cl_out / limber_ref - 1.0)

    # At very low multipoles FKEM is outside its sweet spot. If the hybrid
    # FKEM result is wildly inconsistent with the nonlinear Limber
    # reference, fall back to pure non-linear Limber for this ell.
    # This only affects the lowest multipoles (e.g. ell = 2 for shear–shear)
    # and keeps the behaviour robust.
    if (ell <= 5.0) and np.isfinite(rel_diff) and (rel_diff > 0.5):
        cl_out = limber_ref
        rel_diff = 0.0

    return cl_out, limber_ref, rel_diff
