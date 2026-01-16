"""FKEM FFTLog transforms for tracer collections."""

from __future__ import annotations

import numpy as np

from ..pyutils import _fftlog_transform_general
from ..nonlimber_fkem.params import get_fftlog_params

__all__ = ["compute_collection_fft"]


def compute_collection_fft(
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
    """Computes the FFTLog transforms for a tracer collection.

    This function computes the FFTLog transforms of the radial kernels
    for each tracer in the provided tracer collection. It utilizes caching
    to avoid redundant computations for previously evaluated configurations.

    Args:
        clt (tracer collection):
            The tracer collection.
        kernels (list of arrays):
            List of radial kernels for each tracer in the collection.
        chis (list of arrays):
            List of chi arrays corresponding to each kernel.
        bessels (list of ints):
            List of Bessel derivative orders for each tracer.
        avg_as (list of floats):
            List of average scale factors for each tracer.
        n_chi (int):
            Number of chi samples.
        chi_min (float):
            Minimum chi value.
        chi_max (float):
            Maximum chi value.
        ell (float):
            The multipole moment.
        chi_logspace_arr (array):
            Logarithmically spaced chi array.
        a_arr (array):
            Scale factor array corresponding to chi_logspace_arr.
        growfac_arr (array):
            Growth factor array corresponding to a_arr.
        k_low (float):
            Low-k cutoff for transfer function evaluation.

    Returns: A tuple containing:
        k_out (array):
            The k values corresponding to the FFTLog output.
        fks (2D array):
            2D array of FFTLog transformed functions for each tracer.
        transfers (2D array):
            2D array of transfer function values for each tracer at k_out.

    Raises:
        ValueError:
            If the input configuration is inconsistent or malformed, e.g.:
            empty ``kernels``; mismatched lengths of ``kernels``, ``chis``,
            ``bessels`` and ``avg_as``; non-1D or size-mismatched chi/kernel
            arrays; invalid chi grid (non-1D, wrong length, or non-monotonic);
            non-finite or inconsistent ``chi_min``, ``chi_max``, or
            non-positive/ non-finite ``k_low``.
        RuntimeError:
            If transfer functions contain non-finite values or vanish in the
            rescaling; if the constructed FFTLog input/output arrays have
            inconsistent shapes or non-finite values; if no valid ``k_out`` is
            produced; or if the final transfer matrix at ``k_out`` has the
            wrong shape or contains non-finite values.
    """
    n = len(kernels)

    if n == 0:  # if no tracers provided raise error
        raise ValueError("FKEM: 'kernels' must contain at least one tracer.")

    if not (
        len(chis) == len(bessels) == len(avg_as) == n
    ):  # if lengths mismatch raise error
        raise ValueError(
            "FKEM: kernels/chis/bessels/avg_as length mismatch."
        )

    # Basic checks on chi grid
    chi_logspace_arr = np.asarray(chi_logspace_arr, dtype=float)
    if chi_logspace_arr.ndim != 1:  # this must be 1D
        raise ValueError("[FKEM]: 'chi_logspace_arr' must be 1D.")
    if chi_logspace_arr.size != n_chi:  # this must match n_chi in size
        raise ValueError(
            f"[FKEM]: 'chi_logspace_arr' size ({chi_logspace_arr.size}) "
            f"does not match n_chi ({n_chi})."
        )
    if not np.all(
        np.diff(chi_logspace_arr) > 0
    ):  # this must be strictly increasing
        raise ValueError(
            "[FKEM]: 'chi_logspace_arr' must be strictly increasing."
        )

    if not np.isfinite(chi_min) or not np.isfinite(
        chi_max
    ):  # chi_min and chi_max must be finite
        raise ValueError("[FKEM]: 'chi_min' and 'chi_max' must be finite.")
    if (
        chi_min < 0 or chi_min >= chi_max
    ):  # chi_min must be non-negative and less than chi_max
        raise ValueError("[FKEM]: require 0 <= chi_min < chi_max.")

    if k_low <= 0 or not np.isfinite(
        k_low
    ):  # k_low must be positive and finite
        raise ValueError("[FKEM] 'k_low' must be positive and finite.")

    fks = np.zeros((n, n_chi), dtype=float)
    k_out = None

    logk_low = np.log(k_low)
    transfer_low_all = np.asarray(clt.get_transfer(logk_low, a_arr))

    for i in range(n):
        # Per-tracer chi/kernel shape checks
        ki = np.asarray(kernels[i], dtype=float)
        chi_i = np.asarray(chis[i], dtype=float)
        if chi_i.ndim != 1 or ki.ndim != 1:
            raise ValueError(
                "[FKEM]: each entry in 'kernels' and 'chis' must be 1D."
            )
        if chi_i.size != ki.size:
            raise ValueError(
                f"[FKEM] length mismatch: chis[{i}] vs "
                f"kernels[{i}] ({chi_i.size} vs {ki.size})."
            )
        # Try to read cached FFTLog result for this tracer piece
        k_cached, fk_cached = clt._get_fkem_fft(
            clt._trc[i], n_chi, chi_min, chi_max, ell
        )

        if k_cached is None or fk_cached is None:
            transfer_avg_all = np.asarray(
                clt.get_transfer(logk_low, avg_as[i])
            )

            # here we resolve transfer_low_i per tracer
            if transfer_low_all.ndim == 0:
                # Single scalar for all tracers
                transfer_low_i = float(transfer_low_all)
            elif transfer_low_all.ndim == 1:
                # One value per tracer
                transfer_low_i = float(transfer_low_all[i])
            else:
                # e.g. shape (n_tracers, n_chi) or similar
                row = np.asarray(transfer_low_all[i])
                if row.shape == chi_logspace_arr.shape:
                    # Full radial dependence matches chi grid
                    transfer_low_i = row
                else:
                    # Fallback: use a scalar factor (mean of row)
                    transfer_low_i = float(np.mean(row))

            # resolve transfer_avg as a scalar
            if transfer_avg_all.ndim == 0:
                transfer_avg = float(transfer_avg_all)
            elif transfer_avg_all.ndim == 1:
                # One value per tracer
                transfer_avg = float(transfer_avg_all[i])
            else:
                # Multi-dim, just take the first element for simplicity
                arr = np.asarray(transfer_avg_all).ravel()
                transfer_avg = float(arr[0])

            if not np.all(np.isfinite(transfer_low_i)):
                raise RuntimeError(
                    "[FKEM]: non-finite values in 'transfer_low'."
                )
            if not np.isfinite(transfer_avg):
                raise RuntimeError(
                    "[FKEM]: non-finite value in 'transfer_avg'."
                )
            if transfer_avg == 0.0:
                raise RuntimeError(
                    "[FKEM]: 'transfer_avg' is zero, cannot rescale."
                )

            # Prepare chi/kernel arrays (1D, finite, sorted)
            chi_i = np.asarray(chi_i, dtype=float)
            ki = np.asarray(ki, dtype=float)

            mask = np.isfinite(chi_i) & np.isfinite(ki)
            if not np.any(mask):
                raise ValueError(
                    f"[FKEM]: tracer {i}: no finite chi/kernel samples "
                    "for interpolation."
                )

            chi_i = chi_i[mask]
            ki = ki[mask]

            order = np.argsort(chi_i)
            chi_sorted = chi_i[order]
            ki_sorted = ki[order]

            if chi_sorted.size < 2:
                raise ValueError(
                    f"[FKEM]: tracer {i}: need at least 2 finite chi samples "
                    "for FKEM interpolation."
                )

            # Linear interpolation with zero outside the sampled chi-range
            fchi_vals = np.interp(
                chi_logspace_arr,
                chi_sorted,
                ki_sorted,
                left=0.0,
                right=0.0,
            )

            fchi_arr = (
                fchi_vals
                * chi_logspace_arr
                * growfac_arr
                * transfer_low_i
                / transfer_avg
            )

            if fchi_arr.shape != chi_logspace_arr.shape:
                raise RuntimeError(
                    "[FKEM]: 'fchi_arr' has inconsistent shape "
                    f"(expected {chi_logspace_arr.shape}, "
                    f"got {fchi_arr.shape})."
                )
            if not np.all(np.isfinite(fchi_arr)):
                raise RuntimeError("[FKEM]: non-finite values in 'fchi_arr'.")

            nu, deriv, plaw = get_fftlog_params(bessels[i])

            k_fft, fk = _fftlog_transform_general(
                chi_logspace_arr,
                fchi_arr.flatten(),
                float(ell),
                nu,
                1,
                float(deriv),
                float(plaw),
            )

            if fk.shape[-1] != n_chi:
                raise RuntimeError(
                    "[FKEM]: FFTLog output length does not match n_chi "
                    f"(got {fk.shape[-1]}, expected {n_chi})."
                )

            if not (np.all(np.isfinite(k_fft)) and np.all(np.isfinite(fk))):
                raise RuntimeError(
                    "[FKEM]: non-finite values in FFTLog output."
                )

            clt._set_fkem_fft(
                clt._trc[i], n_chi, chi_min, chi_max, ell, k_fft, fk
            )
            k_out = k_fft
        else:
            k_out = np.asarray(k_cached, dtype=float)
            fk = np.asarray(fk_cached, dtype=float)

            if fk.shape[-1] != n_chi:
                raise RuntimeError(
                    "[FKEM]: cached FFTLog length does not match n_chi "
                    f"(got {fk.shape[-1]}, expected {n_chi})."
                )

        fks[i] = fk

    if k_out is None:
        raise RuntimeError("[FKEM]: k_out not set in compute_collection_fft.")

    # NOTE:
    # Here we evaluate the transfer functions at k_out using a single
    # representative scale factor (avg_as[-1]) for the whole collection,
    # NOT per tracer. This matches the original FKEM implementation, where
    # the final transfer factor does not vary with tracer-specific avg_a.
    # If in future we want fully per-tracer scale-factor dependence here,
    # this is the place to change it (e.g. by passing avg_as instead).
    transfers = np.asarray(clt.get_transfer(np.log(k_out), avg_as[-1]))

    if transfers.shape[0] != n:
        raise RuntimeError(
            "[FKEM]: 'transfers' first dimension must match number of tracers "
            f"(got {transfers.shape[0]}, expected {n})."
        )

    if not np.all(np.isfinite(transfers)):
        raise RuntimeError("[FKEM]: non-finite values in 'transfers'.")

    return k_out, fks, transfers
