"""Utilities for handling power spectra in FKEM."""

from __future__ import annotations

import pyccl as ccl
from pyccl import CCLWarning, warnings

__all__ = ["prepare_power_spectra"]


def prepare_power_spectra(cosmo, p_nonlin, p_lin):
    """Validates and prepares power spectra for FKEM.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        p_nonlin (str or :class:`~pyccl.pk2d.Pk2D`): Non-linear power spectrum.
        p_lin (str or :class:`~pyccl.pk2d.Pk2D`): Linear power spectrum.

    Returns:
        tuple: (psp_lin, psp_nonlin, pk_1d) where
            psp_lin (:class:`~pyccl.pk2d.Pk2D`): Parsed linear power spectrum.
            psp_nonlin (:class:`~pyccl.pk2d.Pk2D`): Parsed non-linear power spectrum.
            pk_1d (:class:`~pyccl.pk2d.Pk2D`): Linear power spectrum for 1D calculations.

        If the configuration is unsafe for FKEM (e.g. type mismatch), all three
        entries are returned as ``None`` and the caller should fall back to Limber.

    Raises:
        TypeError:
            If ``cosmo`` does not provide the required
            ``parse_pk2d`` and ``get_linear_power`` methods.
    """
    # Check matching types for linear and non-linear inputs
    same_str = isinstance(p_nonlin, str) and isinstance(p_lin, str)
    same_pk2d = isinstance(p_nonlin, ccl.Pk2D) and isinstance(p_lin, ccl.Pk2D)

    # If types are inconsistent, we can't safely run FKEM → fall back to Limber.
    if not (same_str or same_pk2d):
        warnings.warn(
            "p_nonlin and p_lin must be of the same type "
            "(both str or both Pk2D). Falling back to Limber.",
            category=CCLWarning,
            importance="high",
        )
        return None, None, None

    # At this point we know the types are consistent. We now require the
    # cosmology-like object to provide the CCL-style helpers.
    for name in ("parse_pk2d", "get_linear_power"):
        if not hasattr(cosmo, name) or not callable(getattr(cosmo, name)):
            # Tests expect the message to mention 'parse_pk2d'
            raise TypeError(
                "prepare_power_spectra requires 'parse_pk2d' and "
                "'get_linear_power' methods on 'cosmo'; "
                f"missing '{name}'."
            )

    # Case 1: both are strings – standard CCL usage.
    if same_str:
        psp_lin = cosmo.parse_pk2d(p_lin, is_linear=True)
        psp_nonlin = cosmo.parse_pk2d(p_nonlin, is_linear=False)
        # For the FKEM correction we need a 'pk' callable – use the linear spectrum.
        pk = cosmo.get_linear_power(name=p_lin)
        return psp_lin, psp_nonlin, pk

    # Case 2: both are Pk2D objects supplied directly.
    # parse_pk2d will wrap/convert them into what the C backend expects.
    psp_lin = cosmo.parse_pk2d(p_lin, is_linear=True)
    psp_nonlin = cosmo.parse_pk2d(p_nonlin, is_linear=False)
    pk = p_lin  # use the linear Pk2D itself as the callable
    return psp_lin, psp_nonlin, pk
