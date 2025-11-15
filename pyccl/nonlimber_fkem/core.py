"""Module implementing the FKEM non-Limber Cl calculation."""

from __future__ import annotations

import numpy as np

import pyccl as ccl
from ..nonlimber_fkem.power_spectra import prepare_power_spectra
from pyccl.nonlimber_fkem.tracers import build_tracer_collections
from pyccl.nonlimber_fkem.chi_grid import build_chi_grid
from pyccl.nonlimber_fkem.single_ell import compute_single_ell
from pyccl import warnings, CCLWarning
from pyccl.errors import CCLError


__all__ = [
    "nonlimber_fkem",
]


def nonlimber_fkem(
    cosmo,
    tracer1,
    tracer2,
    p_of_k_a,
    ell_values,
    ell_limber,
    *,
    pk_linear,
    limber_max_error,
    n_chi,
    chi_min,
    k_pow=3,
    k_low=1e-5,
    n_consec_ell=3,
):
    """Computes the angular power spectrum via the FKEM non-Limber method.

    This method computes the angular power spectrum C_ell for given ells
    `ell_values` using the FKEM approach (arXiv:1911.11947).
    It combines Limber approximations for both linear and non-linear
    power spectra with FFTLog transforms of the radial kernels.
    The computation proceeds until the specified `ell_limber` multipole moment
    is reached, or (in automatic mode) until the FKEM and Limber predictions
    agree within the requested accuracy over several consecutive multipoles.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`):
            A Cosmology object.
        tracer1 (:class:`~pyccl.nonlimber_fkem.tracers.Tracer`):
            Tracer object for the first field.
        tracer2 (:class:`~pyccl.nonlimber_fkem.tracers.Tracer`):
            Tracer object for the second field.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D` or str):
            3D power spectrum to project. If a string, it must match one of
            the non-linear power spectra stored in `cosmo` (e.g.
            'delta_matter:delta_matter').
        ell_values (array-like):
            Multipoles at which to compute C_ell.
        ell_limber (int, float, or 'auto'):
            Multipole above which Limber is used.
            In 'auto' mode, FKEM runs until the FKEM/Limber
            fractional difference is below `limber_max_error` for
            `n_consec_ell` consecutive ells,
            and uses that ell as the transition.
        pk_linear (:class:`~pyccl.pk2d.Pk2D` or str):
            Linear power spectrum used in the Limber calculation.
        limber_max_error (float):
            Maximum allowed fractional FKEM–Limber difference.
        n_chi (int or None):
            Number of chi samples for FKEM. If None, inferred from the tracer
            kernel sampling.
        chi_min (float or None):
            Minimum chi for FKEM. If None, inferred from tracer kernels.
        k_pow (int, optional):
            Power-law index for FFTLog. Default is 3.
        k_low (float, optional):
            Low-k cutoff for transfer function evaluation. Default 1e-5.
        n_consec_ell (int, optional):
            Number of consecutive multipoles satisfying the accuracy threshold
            in 'auto' mode. Default is 3.

    Returns: tuple of (ell_limber, cells, status)
        ell_limber (float):
            The multipole moment beyond which Limber's approximation is used.
        cells (array):
            Array of computed C_ell values up to `ell_limber`.
        status (int):
            Status flag: 0 if all C_ell values are finite, 1 otherwise.

    Raises:
        ValueError:
            If the input configuration is inconsistent or unsafe, e.g.:
            empty or non-increasing ``ell_values`` in auto mode;
            non-positive ``limber_max_error``; ``n_consec_ell < 1``;
            ``n_chi < 2``; negative ``chi_min``; non-finite or out-of-range
            ``ell_limber``; or if ``p_of_k_a`` and ``pk_linear`` are of
            different types (one string and one :class:`~pyccl.pk2d.Pk2D`).
            Additional ``ValueError`` exceptions may be raised downstream
            by chi-grid and tracer-collection construction if their inputs
            are malformed.
    """
    # Ensure ell_values is a numpy array of floats
    ell_values = np.atleast_1d(ell_values)
    if ell_values.size == 0:
        raise ValueError("ell_values must contain at least one multipole.")

    # Auto-mode flag for Limber transition
    auto_mode = isinstance(ell_limber, str)
    if auto_mode and not np.all(np.diff(ell_values) > 0):
        raise ValueError(
            "ell_values must be strictly increasing for auto Limber mode."
        )

    if limber_max_error <= 0:  # make sure it's positive
        raise ValueError("limber_max_error must be positive.")

    if n_consec_ell < 1:  # we need at least one consecutive ell
        raise ValueError("n_consec_ell must be at least 1.")

    if n_chi is not None and n_chi < 2:  # need at least two chi points
        raise ValueError("n_chi must be at least 2.")

    if chi_min is not None and chi_min <= 0:
        raise ValueError("chi_min must be positive.")

    if not auto_mode:
        if not np.isfinite(ell_limber):  # must be finite
            raise ValueError("ell_limber must be finite.")
        if ell_limber < ell_values[0]:  # must not be below smallest ell
            raise ValueError(
                "For FKEM non-Limber integration, `ell_limber` must be"
                "at least as large as the smallest requested ell."
            )

    # First we extract the necessary kernels, chis, bessels, and f_ell values
    # from the tracers
    kernels_t1, chis_t1 = tracer1.get_kernel()
    kernels_t2, chis_t2 = tracer2.get_kernel()
    bessels_t1 = tracer1.get_bessel_derivative()
    bessels_t2 = tracer2.get_bessel_derivative()
    fll_t1 = tracer1.get_f_ell(ell_values)
    fll_t2 = tracer2.get_f_ell(ell_values)

    # Type consistency for power spectra
    same_str = isinstance(p_of_k_a, str) and isinstance(pk_linear, str)
    same_pk2d = isinstance(p_of_k_a, ccl.Pk2D) and isinstance(
        pk_linear, ccl.Pk2D
    )
    if not (same_str or same_pk2d):
        raise ValueError(
            "p_of_k_a and pk_linear must be of the same type "
            "(both str or both Pk2D)."
        )

    # Prepare the power spectra
    psp_lin, psp_nonlin, pk = prepare_power_spectra(
        cosmo, p_nonlin=p_of_k_a, p_lin=pk_linear
    )

    # If prepare_power_spectra couldn't build valid objects, fall back to
    # Limber.
    if psp_lin is None or psp_nonlin is None or pk is None:
        warnings.warn(
            "[FKEM] Could not construct FKEM power spectra for this setup. "
            "Falling back to pure Limber Cls.",
            category=CCLWarning,
            importance="high",
        )
        return -1.0, np.array([]), 0

    # Build tracer collections (C-level)
    t1, t2 = build_tracer_collections(tracer1, tracer2)

    # Build chi grid for FKEM
    chi_grid, dlnr, chi_min_eff, chi_max_eff, n_chi_eff = build_chi_grid(
        chis_t1, chis_t2, chi_min, n_chi
    )

    # Compute average scale factors for the tracers
    avg_a1s = tracer1.get_avg_weighted_a()
    avg_a2s = tracer2.get_avg_weighted_a()

    # Preallocate output array; we'll truncate depending on ell_limber /
    # auto-stop
    n_ell = len(ell_values)
    cells = np.empty(n_ell, dtype=float)
    n_computed = 0

    # Robust automatic transition: require several consecutive ℓ below the
    # tolerance
    consecutive_below = 0

    for ell_idx, ell in enumerate(ell_values):
        try:
            cl_val, limber_ref, rel_diff = compute_single_ell(
                cosmo,
                ell_idx,
                ell,
                t1,
                t2,
                psp_lin,
                psp_nonlin,
                pk,
                tracer1,
                tracer2,
                kernels_t1,
                kernels_t2,
                chis_t1,
                chis_t2,
                bessels_t1,
                bessels_t2,
                fll_t1,
                fll_t2,
                chi_grid,
                chi_min_eff,
                chi_max_eff,
                n_chi_eff,
                dlnr,
                avg_a1s,
                avg_a2s,
                k_low,
                k_pow,
            )
        except CCLError:
            warnings.warn(
                "[FKEM] Non-Limber integration failed for this configuration. "
                "Falling back to pure Limber Cls.",
                category=CCLWarning,
                importance="high",
            )
            return -1.0, np.array([]), 0

        cells[ell_idx] = cl_val
        n_computed += 1

        if auto_mode:
            # Require n_consec_ell consecutive multipoles
            # whose FKEM/Limber discrepancy is below limber_max_error.
            if np.isfinite(rel_diff) and rel_diff < limber_max_error:
                consecutive_below += 1
            else:
                consecutive_below = 0

            # Once we have enough consecutive multipoles below the threshold,
            # set ell_limber
            if consecutive_below >= n_consec_ell:
                ell_limber = ell
                break
        else:
            if ell >= ell_limber:
                break

    # If we never hit the auto criterion, default to the last ell in the grid
    if auto_mode and isinstance(ell_limber, str):
        ell_limber = ell_values[-1]

    # Trim to the number of ell actually computed with FKEM
    cells = cells[:n_computed]

    status = 0 if np.all(np.isfinite(cells)) else 1

    # Explicit: manual l_limber stays manual, auto uses the one we solved for
    if not auto_mode:
        ell_limber_out = float(ell_limber)  # user-specified threshold
    else:
        ell_limber_out = float(ell_limber)  # determined above

    return ell_limber_out, cells, status
