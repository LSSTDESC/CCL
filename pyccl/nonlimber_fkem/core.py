"""Module implementing the FKEM non-Limber Cl calculation."""

from __future__ import annotations

import warnings as _warnings

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
    ls=None,  # OLD name (deprecated)
    l_limber=None,  # OLD name (deprecated)
    fkem_Nchi=None,  # OLD name (deprecated)
    *,
    ell=None,  # NEW name
    ell_limber=None,  # NEW name
    n_chi_fkem=None,  # NEW name
    pk_linear=None,
    limber_max_error=None,
    chi_min_fkem=None,
    k_pow=3,
    k_low=1e-5,
    n_consec_ell=3,
):
    """Computes the angular power spectrum via the FKEM non-Limber method.

    This method computes the angular power spectrum C_ell for given ells
    `ell` using the FKEM approach (arXiv:1911.11947).
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
        ell (array-like):
            Multipoles at which to compute C_ell.
        ls : array_like, optional, deprecated
            Deprecated alias for `ell`. Will be removed in CCL v4.
        ell_limber (int, float, or 'auto'):
            Multipole above which pure non-linear Limber is used.
            For ell ≤ ell_limber, this function returns FKEM-based hybrid
            C_ell values combining linear FKEM with linear and non-linear
            Limber (see `compute_single_ell`).
            In 'auto' mode, FKEM runs until the FKEM / non-linear Limber
            fractional difference (``rel_diff``) is below
            ``limber_max_error`` for ``n_consec_ell`` consecutive multipoles;
            that ell is then used as the Limber transition scale.
        l_limber : int, float, or 'auto', deprecated
            Deprecated alias for `ell_limber`. Will be removed in CCL v4.
        pk_linear (:class:`~pyccl.pk2d.Pk2D` or str):
            Linear power spectrum used in the Limber calculation.
        limber_max_error (float):
            Maximum allowed fractional FKEM–Limber difference.
        n_chi_fkem (int or None):
            Number of chi samples for FKEM. If None, inferred from the tracer
            kernel sampling.
        fkem_Nchi : int or None, deprecated
            Deprecated alias for `n_chi_fkem`. Will be removed in CCL v4.
        chi_min_fkem (float or None):
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
            empty or non-increasing ``ell`` in auto mode;
            non-positive ``limber_max_error``; ``n_consec_ell < 1``;
            ``n_chi_fkem < 2``; negative ``chi_min_fkem``; non-finite or out-of-range
            ``ell_limber``; or if ``p_of_k_a`` and ``pk_linear`` are of
            different types (one string and one :class:`~pyccl.pk2d.Pk2D`).
            Additional ``ValueError`` exceptions may be raised downstream
            by chi-grid and tracer-collection construction if their inputs
            are malformed.
    """
    if ls is not None and ell is not None:
        raise ValueError("Pass only one of `ls` (deprecated) or `ell`.")

    if ell is None and ls is not None:
        _warnings.warn(
            "`ls` is deprecated and will be removed in CCL v4. "
            "Use `ell` instead.",
            FutureWarning,
            stacklevel=2,
        )
        ell = np.asarray(ls)

    if l_limber is not None and ell_limber is not None:
        raise ValueError(
            "Pass only one of `l_limber` (deprecated) or `ell_limber`."
        )

    if ell_limber is None and l_limber is not None:
        _warnings.warn(
            "`l_limber` is deprecated and will be removed in CCL v4. "
            "Use `ell_limber` instead.",
            FutureWarning,
            stacklevel=2,
        )
        ell_limber = l_limber

    if fkem_Nchi is not None and n_chi_fkem is not None:
        raise ValueError(
            "Pass only one of `fkem_Nchi` (deprecated) or `n_chi_fkem`."
        )

    if n_chi_fkem is None and fkem_Nchi is not None:
        _warnings.warn(
            "`fkem_Nchi` is deprecated and will be removed in CCL v4. "
            "Use `n_chi_fkem` instead.",
            FutureWarning,
            stacklevel=2,
        )
        n_chi_fkem = fkem_Nchi

    # Ensure ell is a numpy array of floats
    ell = np.atleast_1d(ell)
    if ell.size == 0:
        raise ValueError("ell must contain at least one multipole.")

    # Auto-mode flag for Limber transition
    auto_mode = isinstance(ell_limber, str)
    if auto_mode and not np.all(np.diff(ell) > 0):
        raise ValueError(
            "ell must be strictly increasing for auto Limber mode."
        )

    if limber_max_error <= 0:  # make sure it's positive
        raise ValueError("limber_max_error must be positive.")

    if n_consec_ell < 1:  # we need at least one consecutive ell
        raise ValueError("n_consec_ell must be at least 1.")

    if n_chi_fkem is not None and n_chi_fkem < 2:  # need at least two chi points
        raise ValueError("n_chi_fkem must be at least 2.")

    if chi_min_fkem is not None and chi_min_fkem <= 0:
        raise ValueError("chi_min_fkem must be positive.")

    if not auto_mode:
        if not np.isfinite(ell_limber):  # must be finite
            raise ValueError("ell_limber must be finite.")
        if ell_limber < ell[0]:  # must not be below smallest ell
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
    fll_t1 = tracer1.get_f_ell(ell)
    fll_t2 = tracer2.get_f_ell(ell)

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

    # If FKEM power spectra cannot be constructed safely (e.g. unsupported
    # p(k, a) choice or numerical issues), we *do not* attempt a partial
    # FKEM run. Instead we return (ell_limber = -1, empty cells) and let
    # the caller compute all C_ell with pure nonlinear Limber.
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
    chi_grid, dlnr, chi_min_fkem_eff, chi_max_eff, n_chi_fkem_eff = build_chi_grid(
        chis_t1, chis_t2, chi_min_fkem, n_chi_fkem
    )

    # Compute average scale factors for the tracers
    avg_a1s = tracer1.get_avg_weighted_a()
    avg_a2s = tracer2.get_avg_weighted_a()

    # Preallocate output array; we'll truncate depending on ell_limber /
    # auto-stop
    n_ell = len(ell)
    cells = np.empty(n_ell, dtype=float)
    n_computed = 0

    # Robust automatic transition: require several consecutive ell below the
    # tolerance
    consecutive_below = 0

    for ell_idx, ell in enumerate(ell):
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
                chi_min_fkem_eff,
                chi_max_eff,
                n_chi_fkem_eff,
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
            # In auto mode we decide where to stop using FKEM.
            # For each ell we compare the hybrid FKEM result to the
            # non-linear Limber reference (rel_diff).
            # If rel_diff stays below `limber_max_error` for
            # `n_consec_ell` consecutive multipoles, we record that ell
            # as the Limber transition scale and stop calling FKEM.
            # The caller then uses pure non-linear Limber for all
            # higher multipoles.
            if np.isfinite(rel_diff) and rel_diff < limber_max_error:
                consecutive_below += 1
            else:
                consecutive_below = 0

            if consecutive_below >= n_consec_ell:
                ell_limber = ell
                break
        else:
            # In manual mode we run FKEM only up to the user-specified
            # `ell_limber` and stop there; higher multipoles are
            # computed with pure non-linear Limber by the caller.
            if ell >= ell_limber:
                break

    # If we never hit the auto criterion, default to the last ell in the grid
    if auto_mode and isinstance(ell_limber, str):
        ell_limber = ell[-1]

    # Trim to the number of ell actually computed with FKEM
    cells = cells[:n_computed]

    status = 0 if np.all(np.isfinite(cells)) else 1

    # Explicit: manual l_limber stays manual, auto uses the one we solved for
    if not auto_mode:
        ell_limber_out = float(ell_limber)  # user-specified threshold
    else:
        ell_limber_out = float(ell_limber)  # determined above

    return ell_limber_out, cells, status
