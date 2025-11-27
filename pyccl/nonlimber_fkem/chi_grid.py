"""Constructs the chi grid for FKEM with sanity checks."""

from __future__ import annotations

import numpy as np

from .. import CCLWarning, warnings

__all__ = ["build_chi_grid"]


def build_chi_grid(chis_t1, chis_t2, chi_min, n_chi, *, warn: bool = True):
    """Constructs the FKEM chi grid from tracer kernel chi arrays.

        This function infers chi_min/chi_max/n_chi from the tracer kernels
        when not provided, and applies conservative guards to prevent FKEM
        from using a chi grid that extends far below the actual kernel
        support, which can cause instabilities at very low redshift.

    Args:
        chis_t1 (list of arrays):
            List of 1D chi arrays for tracer collection 1.
        chis_t2 (list of arrays):
            List of 1D chi arrays for tracer collection 2.
        chi_min (float or None):
            User-specified minimum chi for the FKEM grid (fkem_chi_min).
            If None, it is inferred from the tracer kernels as the minimum
            sampled chi. If set to a non-positive value it is reset to 1e-6.
            If it is much smaller than the actual kernel support, it is
            automatically clamped to ~0.1 times the smallest positive chi
            in the kernels to avoid low-z numerical instabilities.
        n_chi (int or None):
            Number of chi samples for FKEM (fkem_nchi). If None, it is
            inferred from the tracer kernel sampling.
        warn (bool):
            If True, emit a CCLWarning when chi_min is clamped or sampling
            is suspicious.

    Returns:
        chi_log (1D array):
            Logarithmically spaced chi grid for FKEM.
        dlnr (float):
            Logarithmic spacing between chi samples.
        chi_min (float):
            Effective minimum chi used.
        chi_max (float):
            Effective maximum chi used.
        n_chi (int):
            Effective number of chi samples used.

    Raises:
        ValueError:
            If any input chi arrays are invalid (non-1D, too few, non-finite,
            or containing negative values); if chi_min or chi_max are
            non-finite or inconsistent (chi_max ≤ chi_min); if n_chi
            is missing, non-finite, or < 2; or if the inferred
            FKEM grid cannot be constructed (e.g. non-monotonic or
            containing non-finite values).
    """
    # Validate & summarize each tracer collection in helpers
    min1, max1, support1, nmin1 = _validate_and_summarize_chis(
        chis_t1, label="tracer1", warn=warn
    )
    min2, max2, support2, nmin2 = _validate_and_summarize_chis(
        chis_t2, label="tracer2", warn=warn
    )

    chi_log, dlnr, chi_min_eff, chi_max_eff, n_chi_eff = (
        _build_grid_from_stats(
            min1=min1,
            min2=min2,
            max1=max1,
            max2=max2,
            support1=support1,
            support2=support2,
            chi_min=chi_min,
            n_chi=n_chi,
            nmin1=nmin1,
            nmin2=nmin2,
            warn=warn,
        )
    )

    return chi_log, dlnr, chi_min_eff, chi_max_eff, n_chi_eff


def _validate_and_summarize_chis(chis_list, label: str, warn: bool):
    """Validate chi arrays for FKEM and return summary statistics.

    This helper function ensures that each chi array is 1D, has at least
    2 points, contains some finite values, and has no negative values.
    It also returns summary statistics needed to build the FKEM chi grid.

    Args:
        chis_list (list of arrays):
            List of chi arrays to validate.
        label (str):
            Label to identify the tracer collection in error/warning messages.
        warn (bool):
            If True, emit warnings for non-finite entries.

    Returns:
        chi_min (float):
            Minimum finite chi across all arrays.
        chi_max (float):
            Maximum finite chi across all arrays.
        support_min (float):
            Smallest finite chi (effective minimum support).
        n_chi_min (int):
            Minimum array length across all arrays.

    Raises:
        ValueError:
            If any chi array is invalid.
    """
    if chis_list is None or len(chis_list) == 0:
        raise ValueError(f"[FKEM] {label}: no chi arrays provided.")

    any_finite = False
    chi_min = np.inf
    chi_max = -np.inf
    support_min = np.inf
    n_chi_min = np.inf

    for i, chi_arr in enumerate(chis_list):
        chi = np.asarray(chi_arr)

        if chi.ndim != 1:
            raise ValueError(
                f"[FKEM] {label}: chi array #{i} is not 1D "
                f"(ndim={chi.ndim}). FKEM expects 1D radial kernels."
            )

        if chi.size < 2:
            raise ValueError(
                f"[FKEM] {label}: chi array #{i} has only {chi.size}"
                f"point(s). Need at least 2 samples to build a meaningful"
                f"FKEM grid."
            )

        n_chi_min = min(n_chi_min, chi.size)

        finite_mask = np.isfinite(chi)
        if not np.any(finite_mask):
            # All entries non-finite → useless array
            raise ValueError(
                f"[FKEM] {label}: chi array #{i} contains no finite values."
            )

        any_finite = True

        if np.any(~finite_mask) and warn:
            warnings.warn(
                f"[FKEM] {label}: chi array #{i} contains non-finite values; "
                "these will be ignored when constructing the chi grid.",
                category=CCLWarning,
                importance="low",
            )

        # Negative chi is unphysical here
        if np.nanmin(chi) < 0:
            raise ValueError(
                f"[FKEM] {label}: chi array #{i} contains negative values. "
                "Comoving distances must be non-negative."
            )

        chi_finite = chi[finite_mask]
        chi_min = min(chi_min, np.min(chi_finite))
        chi_max = max(chi_max, np.max(chi_finite))

        # Use the smallest *positive* chi as "support" where the kernel actually lives.
        # In other words: do not start the chi grid far below the kernel support,
        # which can cause numerical instabilities for low-z bins.
        pos = chi_finite[chi_finite > 0]
        if pos.size > 0:
            support_candidate = np.min(pos)
        else:
            # All finite values are zero – degenerate but handle gracefully
            support_candidate = np.min(chi_finite)

        support_min = min(support_min, support_candidate)

    if not any_finite:
        raise ValueError(
            f"[FKEM] {label}: all chi arrays are non-finite. "
            "Cannot construct FKEM chi grid."
        )

    return chi_min, chi_max, support_min, int(n_chi_min)


def _build_grid_from_stats(
    *,
    min1,
    min2,
    max1,
    max2,
    support1,
    support2,
    chi_min,
    n_chi,
    nmin1,
    nmin2,
    warn: bool,
):
    """Builds FKEM chi grid from both tracer collections.

    This helper takes the per-collection minima, maxima, effective support
    and sampling, applies FKEM-specific sanity checks, and constructs the
    logarithmic chi grid.

    Returns:
        chi_log (1D array):
            Logarithmically spaced chi grid.
        dlnr (float):
            Logarithmic spacing between chi samples.
        chi_min_eff (float):
            Effective minimum chi used.
        chi_max_eff (float):
            Effective maximum chi used.
        n_chi_eff (int):
            Effective number of chi samples.
    """
    # Min / max over both tracer collections
    min_chi = min(min1, min2)
    max_chi = max(max1, max2)

    if not np.isfinite(min1) or not np.isfinite(min2):
        raise ValueError(
            "[FKEM] Non-finite chi minima encountered in tracer kernels."
        )
    if not np.isfinite(max1) or not np.isfinite(max2):
        raise ValueError(
            "[FKEM] Non-finite chi maxima encountered in tracer kernels."
        )

    # Effective minimum "support" of the kernels (smallest sampled finite chi)
    support_min = min(support1, support2)

    # Infer chi_min if needed
    if chi_min is None:
        chi_min = min_chi

    if not np.isfinite(chi_min):
        raise ValueError(f"[FKEM] fkem_chi_min={chi_min!r} is not finite.")

    # Non-positive chi_min is incompatible with a log grid
    if chi_min <= 0:
        if warn:
            warnings.warn(
                f"[FKEM] Requested fkem_chi_min={chi_min:.3e} "
                f"is non-positive. Resetting to 1e-6.",
                category=CCLWarning,
                importance="low",
            )
        chi_min = 1e-6

    # Avoid chi_min far below the kernel support (can cause low-z
    # instabilities)
    if (
        np.isfinite(support_min)
        and support_min > 0
        and chi_min < 0.1 * support_min
    ):
        chi_new = 0.1 * support_min
        if warn:
            warnings.warn(
                f"[FKEM] Requested fkem_chi_min={chi_min:.3e} is much smaller "
                f"than the tracer kernel support (~{support_min:.3e}). "
                f"Clamping chi_min to {chi_new:.3e} to avoid numerical "
                "instabilities for very low-z bins.",
                category=CCLWarning,
                importance="low",
            )
        chi_min = chi_new

    chi_max = max_chi
    if not np.isfinite(chi_max):
        raise ValueError(f"[FKEM] chi_max={chi_max!r} is not finite.")

    if chi_max <= chi_min:
        raise ValueError(
            f"[FKEM] chi_max <= chi_min (chi_min={chi_min:.3e}, "
            f"chi_max={chi_max:.3e}). Check tracer kernels and fkem_chi_min."
        )

    # Infer n_chi if needed: use the minimum sampling across both tracer
    # collections
    if n_chi is None:
        n_chi = min(nmin1, nmin2)

    if not np.isfinite(n_chi):
        raise ValueError(f"[FKEM] Nchi={n_chi!r} is not finite.")
    n_chi = int(n_chi)

    if n_chi < 2:
        raise ValueError(
            f"[FKEM] Nchi={n_chi} is too small. Need at least 2 chi samples."
        )
    elif warn and n_chi < 10:
        warnings.warn(
            f"[FKEM] Using Nchi={n_chi} samples. This is very coarse and may "
            "lead to inaccurate FKEM results.",
            category=CCLWarning,
            importance="low",
        )

    chi_log = np.logspace(np.log10(chi_min), np.log10(chi_max), n_chi)

    if not np.all(np.isfinite(chi_log)):
        raise RuntimeError("[FKEM] Non-finite values in chi_log grid.")
    if not np.all(np.diff(chi_log) > 0):
        raise RuntimeError("[FKEM] chi_log grid is not strictly increasing.")

    dlnr = np.log(chi_max / chi_min) / (n_chi - 1)

    return chi_log, dlnr, chi_min, chi_max, n_chi
