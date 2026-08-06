"""Mead et al. 2020 HMCode baryonic feedback model."""

from __future__ import annotations

__all__ = ("BaryonsMead20",)

from copy import deepcopy
from typing import Any
from numpy.typing import NDArray

import numpy as np

from ..cosmology import Cosmology
from ..pk2d import Pk2D
from .baryons_base import Baryons
from .mead20_hmcode.mead20_utils import (
    _cosmo_signature,
    _make_internal_camb_cosmology,
    _raise_if_already_hmcode_feedback,
    _raise_if_baryons_already_attached,
    _resolve_pk_is_log,
    _apply_boost_to_pk_arr,
    _to_2d_na_nk,
    _safe_pos_ratio,
    _restore_input_ndim,
)

FloatArray = NDArray[np.floating[Any]]
CosmoCacheKey = tuple[
    int, str, bool, float, int | None, str | None, float, float | None,
    float | None, str
]


class BaryonsMead20(Baryons):
    """Mead20 HMCode baryonic feedback model using CAMB as the backend.

    This model constructs a scale- and redshift-dependent boost factor from
    CAMB's HMCode implementation, and applies it multiplicatively to an input
    nonlinear matter power spectrum.

    The boost is defined as the ratio of a feedback run to a dark-matter-only
    HMCode run evaluated with consistent CAMB settings.

    Notes
    -----
    Do not attach this model if the input cosmology already requests HMCode
    feedback through CAMB, or if baryonic effects are already attached.

    CAMB defaults for HMCode parameters are:

    - ``HMCode_logT_AGN = 7.8`` (used only when
      ``halofit_version="mead2020_feedback"``),
    - ``HMCode_A_baryon = 3.13`` and ``HMCode_eta_baryon = 0.603``
      (used only for ``"mead"``/``"mead2015"``/``"mead2016"``).

    For HMCode-2020 (``"mead2020"`` / ``"mead2020_feedback"``), CAMB ignores
    ``HMCode_A_baryon`` and ``HMCode_eta_baryon``; they are kept as optional
    inputs for API completeness.

    Args:
        HMCode_logT_AGN (:obj:`float`): HMCode feedback strength parameter.
        HMCode_A_baryon (:obj:`float` or :obj:`None`): Optional HMCode
            parameter.
        HMCode_eta_baryon (:obj:`float` or :obj:`None`): Optional HMCode
            parameter.
        kmax (:obj:`float`): Maximum k used in the internal CAMB evaluation
            (in units of Mpc^-1).
        lmax (:obj:`int` or :obj:`None`): Optional CAMB ell limit for internal
            calculations.
        dark_energy_model (:obj:`str` or :obj:`None`): Optional CAMB dark
            energy model name for internal calculations.
        halofit_version_feedback (:obj:`str`): CAMB halofit/HMCode version used
            for the feedback spectrum.
        halofit_version_dmo (:obj:`str`): CAMB halofit/HMCode version used for
            the dark-matter-only spectrum.
        camb_overrides (:obj:`dict` or :obj:`None`): Extra CAMB settings to
            apply to the internal cosmologies.
        cache_cosmologies (:obj:`bool`): If True, reuse internal cosmologies
            across calls when inputs and settings match.
    """
    name = "mead20_hmcode"
    __repr_attrs__ = __eq_attrs__ = (
        "HMCode_logT_AGN",
        "HMCode_A_baryon",
        "HMCode_eta_baryon",
        "kmax",
        "lmax",
        "dark_energy_model",
        "halofit_version_feedback",
        "halofit_version_dmo",
        "camb_overrides",
        "cache_cosmologies",
    )

    _cosmo_cache: dict[CosmoCacheKey, Cosmology]

    def __init__(
        self,
        *,
        HMCode_logT_AGN: float = 7.8,
        HMCode_A_baryon: float | None = None,
        HMCode_eta_baryon: float | None = None,
        kmax: float = 20.0,
        lmax: int | None = None,
        dark_energy_model: str | None = None,
        halofit_version_feedback: str = "mead2020_feedback",
        halofit_version_dmo: str = "mead2020",
        camb_overrides: dict[str, Any] | None = None,
        cache_cosmologies: bool = True,
    ) -> None:
        """Initialize the Mead20 HMCode baryonic feedback model."""
        self.HMCode_logT_AGN = float(HMCode_logT_AGN)
        self.HMCode_A_baryon = (
            None if HMCode_A_baryon is None else float(HMCode_A_baryon)
        )
        self.HMCode_eta_baryon = (
            None if HMCode_eta_baryon is None else float(HMCode_eta_baryon)
        )
        self.kmax = float(kmax)
        self.lmax = None if lmax is None else int(lmax)
        self.dark_energy_model = (
            None if dark_energy_model is None else str(dark_energy_model)
        )
        self.halofit_version_feedback = str(halofit_version_feedback)
        self.halofit_version_dmo = str(halofit_version_dmo)
        self.camb_overrides = (
            deepcopy(camb_overrides) if camb_overrides else {}
        )
        self.cache_cosmologies = bool(cache_cosmologies)

        # {(sig, halofit_version, include_feedback): Cosmology}
        self._cosmo_cache = {}

    def _get_internal_cosmo(
        self,
        cosmo: Cosmology,
        *,
        halofit_version: str,
        include_feedback: bool,
    ) -> Cosmology:
        """Return an internal CAMB-backed Cosmology for the requested settings.

        This helper builds a CAMB-based cosmology that matches the input
        parameters  but forces a consistent CAMB pipeline. When caching is
        enabled, previously constructed internal cosmologies are reused.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): Baseline cosmology.
            halofit_version (:obj:`str`): CAMB halofit/HMCode version string.
            include_feedback (:obj:`bool`): Whether to include HMCode feedback.

        Returns:
            :class:`~pyccl.cosmology.Cosmology`: Internal CAMB-backed
                cosmology.

        """
        if not self.cache_cosmologies:
            return _make_internal_camb_cosmology(
                cosmo,
                halofit_version=halofit_version,
                include_feedback=include_feedback,
                kmax=self.kmax,
                HMCode_logT_AGN=self.HMCode_logT_AGN,
                HMCode_A_baryon=self.HMCode_A_baryon,
                HMCode_eta_baryon=self.HMCode_eta_baryon,
                lmax=self.lmax,
                dark_energy_model=self.dark_energy_model,
                camb_overrides=self.camb_overrides,
            )

        key: CosmoCacheKey = (
            _cosmo_signature(cosmo),
            str(halofit_version),
            bool(include_feedback),
            float(self.kmax),
            int(self.lmax) if self.lmax is not None else None,
            (
                str(self.dark_energy_model)
                if self.dark_energy_model is not None
                else None
            ),
            float(self.HMCode_logT_AGN),
            (
                float(self.HMCode_A_baryon)
                if self.HMCode_A_baryon is not None
                else None
            ),
            (
                float(self.HMCode_eta_baryon)
                if self.HMCode_eta_baryon is not None
                else None
            ),
            repr(self.camb_overrides),
        )

        cached = self._cosmo_cache.get(key)
        if cached is None:
            cached = _make_internal_camb_cosmology(
                cosmo,
                halofit_version=halofit_version,
                include_feedback=include_feedback,
                kmax=self.kmax,
                HMCode_logT_AGN=self.HMCode_logT_AGN,
                HMCode_A_baryon=self.HMCode_A_baryon,
                HMCode_eta_baryon=self.HMCode_eta_baryon,
                lmax=self.lmax,
                dark_energy_model=self.dark_energy_model,
                camb_overrides=self.camb_overrides,
            )
            self._cosmo_cache[key] = cached
        return cached

    def boost_factor(
        self,
        cosmo: Cosmology,
        k: float | FloatArray,
        a: float | FloatArray,
    ) -> float | FloatArray:
        """Compute the Mead20 HMCode boost factor B(k, a).

        The boost is defined as the ratio of the nonlinear matter power
        spectrum computed with HMCode feedback enabled to the corresponding
        dark-matter-only HMCode spectrum, evaluated on the requested
        (k,a) grid.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): Baseline cosmology.
            k (:obj:`float` or `array`): Wavenumber in Mpc^-1.
            a (:obj:`float` or `array`): Scale factor.

        Returns:
            :obj:`float` or `array`: Boost factor with shape matching inputs.

        Raises:
            ValueError: If baryonic effects are already attached to ``cosmo``
                or if CAMB HMCode feedback is already enabled on the input
                cosmology.
            FloatingPointError: If the computed boost is not finite and
                positive.
        """
        _raise_if_baryons_already_attached(cosmo)
        _raise_if_already_hmcode_feedback(cosmo)

        a_use = np.atleast_1d(a).astype(float)  # (na,)
        k_use = np.atleast_1d(k).astype(float)  # (nk,)
        na, nk = a_use.size, k_use.size

        cosmo_fb = self._get_internal_cosmo(
            cosmo,
            halofit_version=self.halofit_version_feedback,
            include_feedback=True,
        )
        cosmo_dmo = self._get_internal_cosmo(
            cosmo,
            halofit_version=self.halofit_version_dmo,
            include_feedback=False,
        )

        p_fb = _to_2d_na_nk(
            cosmo_fb.nonlin_matter_power(k_use, a_use), na, nk, name="P_fb"
        )
        p_dmo = _to_2d_na_nk(
            cosmo_dmo.nonlin_matter_power(k_use, a_use), na, nk, name="P_dmo"
        )

        boost2d = _safe_pos_ratio(p_fb, p_dmo, name="BaryonsMead20")
        return _restore_input_ndim(boost2d, a=a, k=k)

    def _include_baryonic_effects(
        self,
        cosmo: Cosmology,
        pk: Pk2D,
    ) -> Pk2D:
        """Apply the Mead20 boost to a Pk2D instance.

        This modifies the tabulated values of the input power spectrum by
        applying the boost factor on the same (a, k) grid, preserving the
        log or linear representation of the input Pk2D.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmology used to
                compute the boost factor on the Pk2D grid.
            pk (:class:`~pyccl.Pk2D`): Input power spectrum to be modified.

        Returns:
            :class:`~pyccl.Pk2D`: New Pk2D with baryonic effects applied.

        Raises:
            RuntimeError: If the boost cannot be matched to the Pk2D grid.
        """
        a_arr, lk_arr, pk_arr_in = pk.get_spline_arrays()
        k_arr = np.exp(lk_arr)

        boost = self.boost_factor(cosmo, k_arr, a_arr)
        if boost.shape != pk_arr_in.shape:
            raise RuntimeError(
                f"Boost shape {boost.shape} does not match "
                f"pk shape {pk_arr_in.shape}."
            )

        boost_safe = np.where(np.isfinite(boost) & (boost > 0.0), boost, 1.0)
        boost_safe = np.clip(boost_safe, 1e-6, 1e6)

        is_log = _resolve_pk_is_log(pk)
        pk_out = _apply_boost_to_pk_arr(pk_arr_in, boost_safe, is_log=is_log)

        return Pk2D(
            a_arr=a_arr,
            lk_arr=lk_arr,
            pk_arr=pk_out,
            is_logp=is_log,
            extrap_order_lok=pk.extrap_order_lok,
            extrap_order_hik=pk.extrap_order_hik,
        )

    def include_baryonic_effects(
        self,
        cosmo: Cosmology,
        pk: Pk2D,
    ) -> Pk2D:
        """Return a baryon-modified Pk2D using a CAMB HMCode baseline.

        This method constructs an internal CAMB HMCode dark-matter-only
        baseline and applies the Mead20 boost to that baseline. The ``pk``
        argument is accepted for API compatibility but is not used as the
        baseline spectrum.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): Baseline cosmology.
            pk (:class:`~pyccl.Pk2D`): Ignored baseline Pk2D provided for API
                compatibility.

        Returns:
            :class:`~pyccl.Pk2D`: Baryon-modified nonlinear matter power
                spectrum.

        Raises:
            ValueError: If baryonic effects are already attached to ``cosmo``
                or if CAMB HMCode feedback is already enabled on the input
                cosmology.
        """
        _raise_if_baryons_already_attached(cosmo)
        _raise_if_already_hmcode_feedback(cosmo)

        # Build internal CAMB HMCode-DMO cosmology (forced CAMB backend)
        cosmo_dmo = self._get_internal_cosmo(
            cosmo,
            halofit_version=self.halofit_version_dmo,
            include_feedback=False,
        )

        # Always use the internal HMCode DMO Pk2D baseline
        pk_dmo = cosmo_dmo.get_nonlin_power()

        # Apply boost (this preserves *all* your is_log recovery logic)
        return self._include_baryonic_effects(cosmo_dmo, pk_dmo)
