"""Utility functions for Mead20 baryon model."""

from __future__ import annotations

from typing import Any
from copy import deepcopy
from numpy.typing import NDArray

import numpy as np
from ...cosmology import Cosmology

FloatArray = NDArray[np.floating[Any]]

_EPS = 1e-300


def _cosmo_signature(cosmo: Cosmology) -> int:
    """Build a cache key for a Cosmology instance.

    This returns a reproducible integer based on the cosmology parameters, to
    support memoization of internally constructed CAMB cosmologies.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmology to summarize.

    Returns:
        :obj:`int`: Hash-like signature suitable for caching.
    """
    return hash(repr(cosmo.to_dict()))


def _raise_if_already_hmcode_feedback(cosmo: Cosmology) -> None:
    """Raise if the input cosmology already enables HMCode feedback in CAMB.

    BaryonsMead20 builds an internal CAMB HMCode ratio. If the input cosmology
    is already configured to request HMCode feedback, applying this model would
    double-count baryonic effects.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmology to validate.

    Raises:
        ValueError: If CAMB HMCode feedback is already requested.
    """
    cfg = cosmo.to_dict()
    if cfg.get("matter_power_spectrum") != "camb":
        return
    camb = (cfg.get("extra_parameters") or {}).get("camb",
                                                   {}) or {}
    hf = str(camb.get("halofit_version", "")).lower()
    if "feedback" in hf:
        raise ValueError(
            "BaryonsMead20: input cosmology already requests CAMB HMCode "
            "baryonic feedback "
            f"(halofit_version={camb.get('halofit_version')!r}). "
            "Do not also attach BaryonsMead20 as `baryonic_effects`."
        )


def _raise_if_baryons_already_attached(cosmo: Cosmology) -> None:
    """Raise if the input cosmology already has baryonic effects attached.

    This prevents stacking multiple baryonic prescriptions on the same
    cosmology, which is not supported by this model.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmology to validate.

    Raises:
        ValueError: If baryonic effects are already present on ``cosmo``.
    """
    if getattr(cosmo, "baryons", None) is not None:
        raise ValueError(
            "BaryonsMead20: input cosmology already has"
            " baryonic_effects attached. "
            "Do not stack baryonic models."
        )


def _build_internal_camb_kwargs(
    cosmo: Cosmology,
    *,
    halofit_version: str,
    include_feedback: bool,
    kmax: float,
    HMCode_logT_AGN: float,
    HMCode_A_baryon: float | None,
    HMCode_eta_baryon: float | None,
    lmax: int | None,
    dark_energy_model: str | None,
    camb_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build kwargs for an internal CAMB-backed Cosmology.

    The returned dictionary is derived from the input cosmology parameters, but
    is adjusted so an internal CAMB pipeline can be evaluated consistently.
    This is used to compute the HMCode ratio with and without feedback.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Baseline cosmology.
        halofit_version (:obj:`str`): CAMB halofit/HMCode version string.
        include_feedback (:obj:`bool`): Whether to enable baryonic feedback.
        kmax (:obj:`float`): Maximum wavenumber for internal CAMB evaluation.
        HMCode_logT_AGN (:obj:`float`): AGN feedback strength parameter.
        HMCode_A_baryon (:obj:`float` or :obj:`None`): Optional HMCode
            parameter.
        HMCode_eta_baryon (:obj:`float` or :obj:`None`): Optional HMCode
            parameter.
        lmax (:obj:`int` or :obj:`None`): Optional CAMB ell limit.
        dark_energy_model (:obj:`str` or :obj:`None`): Optional CAMB DE model.
        camb_overrides (:obj:`dict` or :obj:`None`): Extra CAMB settings.

    Returns:
        :obj:`dict`: Keyword arguments for constructing a Cosmology.
    """
    kwargs = deepcopy(cosmo.to_dict())

    # Force CAMB pipeline for internal eval
    kwargs["transfer_function"] = "boltzmann_camb"
    kwargs["matter_power_spectrum"] = "camb"

    # Avoid recursion/double-baryonization inside internal cosmology
    kwargs["baryonic_effects"] = None

    extra = deepcopy(kwargs.get("extra_parameters", {})) or {}
    camb = deepcopy(extra.get("camb", {})) or {}

    # Required: choose HMCode variant
    camb["halofit_version"] = str(halofit_version)

    # kmax policy: never reduce user kmax
    if "kmax" in camb:
        camb["kmax"] = float(max(float(camb["kmax"]), float(kmax)))
    else:
        camb["kmax"] = float(kmax)

    # Optional CAMB params: don't stomp user values
    if lmax is not None and "lmax" not in camb:
        camb["lmax"] = int(lmax)
    if dark_energy_model is not None and "dark_energy_model" not in camb:
        camb["dark_energy_model"] = str(dark_energy_model)

    # Optional HMCode params: don't stomp user values
    if HMCode_A_baryon is not None and "HMCode_A_baryon" not in camb:
        camb["HMCode_A_baryon"] = float(HMCode_A_baryon)
    if HMCode_eta_baryon is not None and "HMCode_eta_baryon" not in camb:
        camb["HMCode_eta_baryon"] = float(HMCode_eta_baryon)

    # Feedback strength: override on purpose for feedback; remove for dmo
    if include_feedback:
        camb["HMCode_logT_AGN"] = float(HMCode_logT_AGN)
    else:
        camb.pop("HMCode_logT_AGN", None)

    # Escape hatch wins
    for kk, vv in (camb_overrides or {}).items():
        camb[kk] = vv

    extra["camb"] = camb
    kwargs["extra_parameters"] = extra
    return kwargs


def _make_internal_camb_cosmology(
    cosmo: Cosmology,
    *,
    halofit_version: str,
    include_feedback: bool,
    kmax: float,
    HMCode_logT_AGN: float,
    HMCode_A_baryon: float | None,
    HMCode_eta_baryon: float | None,
    lmax: int | None,
    dark_energy_model: str | None,
    camb_overrides: dict[str, Any] | None,
) -> Cosmology:
    """Construct an internal CAMB-backed Cosmology.

    This helper creates a Cosmology that matches the input parameters but is
    configured to evaluate the CAMB nonlinear power required by HMCode.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Baseline cosmology.
        halofit_version (:obj:`str`): CAMB halofit/HMCode version string.
        include_feedback (:obj:`bool`): Whether to enable baryonic feedback.
        kmax (:obj:`float`): Maximum wavenumber for internal CAMB evaluation.
        HMCode_logT_AGN (:obj:`float`): AGN feedback strength parameter.
        HMCode_A_baryon (:obj:`float` or :obj:`None`): Optional HMCode
            parameter.
        HMCode_eta_baryon (:obj:`float` or :obj:`None`): Optional HMCode
            parameter.
        lmax (:obj:`int` or :obj:`None`): Optional CAMB ell limit.
        dark_energy_model (:obj:`str` or :obj:`None`): Optional CAMB DE model.
        camb_overrides (:obj:`dict` or :obj:`None`): Extra CAMB settings.

    Returns:
        :class:`~pyccl.cosmology.Cosmology`: Internal CAMB-backed cosmology.
    """
    from ...cosmology import Cosmology  # local import to avoid cycles

    kwargs = _build_internal_camb_kwargs(
        cosmo,
        halofit_version=halofit_version,
        include_feedback=include_feedback,
        kmax=kmax,
        HMCode_logT_AGN=HMCode_logT_AGN,
        HMCode_A_baryon=HMCode_A_baryon,
        HMCode_eta_baryon=HMCode_eta_baryon,
        lmax=lmax,
        dark_energy_model=dark_energy_model,
        camb_overrides=camb_overrides,
    )
    return Cosmology(**kwargs)


def _resolve_pk_is_log(pk: Any) -> bool:
    """Decide whether a P(k) object should be treated as log(P).

    This inspects common Pk2D-like attributes to infer whether the stored power
    spectrum values are in log space or linear space.

    Args:
        pk: Power spectrum container (e.g. :class:`~pyccl.Pk2D` or similar).

    Returns:
        :obj:`bool`: True if values should be interpreted as log(P).
    """
    psp_is_log = bool(getattr(getattr(pk, "psp", None), "is_log", False))
    pk_is_logp = getattr(pk, "is_logp", None)
    return bool(pk_is_logp) if pk_is_logp is not None else psp_is_log


def _ensure_log_pk_arr(
    pk_arr: FloatArray,
    *,
    absurd_log_threshold: float = 200.0,
) -> FloatArray:
    """Return an array that can be safely interpreted as log(P).

    This is used to robustly handle inputs where the log/linear convention may
    be inconsistent, so downstream operations that assume log(P) remain stable.

    Args:
        pk_arr (:obj:`array`): Power spectrum values.
        absurd_log_threshold (:obj:`float`): Heuristic threshold used to
            identify inconsistent log labeling.

    Returns:
        :obj:`array`: Values suitable for log-space operations.
    """
    # Heuristic: true log(P) rarely has max >> O(10..100) so
    # if it does, it's likely linear.
    if float(np.nanmax(pk_arr)) > absurd_log_threshold:
        return np.log(np.maximum(pk_arr, _EPS))
    return pk_arr


def _apply_boost_to_pk_arr(
    pk_arr_in: FloatArray,
    boost: FloatArray,
    *,
    is_log: bool,
) -> FloatArray:
    """Apply a multiplicative boost to a power spectrum array.

    The boost is applied in a way that respects whether the input power values
    are stored in linear space or log space.

    Args:
        pk_arr_in (:obj:`array`): Input power spectrum values.
        boost (:obj:`array`): Multiplicative correction factor.
        is_log (:obj:`bool`): Whether ``pk_arr_in`` is in log space.

    Returns:
        :obj:`array`: Boosted power spectrum values.
    """
    pk_arr = np.array(pk_arr_in, copy=True)

    if is_log:
        pk_arr = _ensure_log_pk_arr(pk_arr)
        return pk_arr + np.log(boost)
    return pk_arr * boost


def _to_2d_na_nk(
    p: FloatArray,
    na: int,
    nk: int,
    *,
    name: str,
) -> FloatArray:
    """Normalize a power spectrum grid to shape (na, nk).

    This enforces the expected 2D layout for arrays tabulated on (a, k) grids
    and provides a clear error if the input is incompatible.

    Args:
        p (:obj:`array`): Input array.
        na (:obj:`int`): Expected number of scale-factor samples.
        nk (:obj:`int`): Expected number of k samples.
        name (:obj:`str`): Label used in error messages.

    Returns:
        :obj:`array`: View or copy with shape (na, nk).

    Raises:
        RuntimeError: If the input array cannot be reshaped or transposed
            into (na, nk).
    """
    p = np.asarray(p)
    if p.shape == (na, nk):
        return p
    if p.shape == (nk, na):
        return p.T
    raise RuntimeError(f"Unexpected {name} shape {p.shape};"
                       f" expected {(na, nk)} or {(nk, na)}")


def _safe_pos_ratio(
    num: FloatArray,
    den: FloatArray,
    *,
    name: str,
) -> FloatArray:
    """Compute a positive, finite ratio with basic validation.

    This helper divides ``num`` by ``den`` while guarding against tiny
    denominators. It also checks that the result is finite and strictly
    positive, which is required when building multiplicative boosts.

    Args:
        num (:obj:`array`): Numerator values.
        den (:obj:`array`): Denominator values.
        name (:obj:`str`): Label used in error messages.

    Returns:
        :obj:`array`: Elementwise ratio ``num/den``.

    Raises:
        FloatingPointError: If the ratio contains non-finite values or is not
            strictly positive.
    """
    den = np.asarray(den)
    num = np.asarray(num)

    abs_den = np.abs(den)
    sgn_den = np.sign(den)
    sgn_den = np.where(sgn_den == 0.0, 1.0, sgn_den)

    den_safe = np.where(abs_den > _EPS, den, sgn_den * _EPS)
    out = num / den_safe

    if not np.all(np.isfinite(out)):
        raise FloatingPointError(f"{name}: non-finite ratio encountered.")
    if np.any(out <= 0):
        raise FloatingPointError(f"{name}: non-positive ratio encountered.")
    return out


def _restore_input_ndim(
    out2d: FloatArray,
    *,
    a: float | FloatArray,
    k: float | FloatArray,
) -> float | FloatArray:
    """Restore output shape to match scalar and 1D input conventions.

    Many public CCL functions accept ``a`` and ``k`` as scalars or arrays. This
    helper converts a 2D (na, nk) output back to the corresponding scalar, 1D,
    or 2D shape implied by the original inputs.

    Args:
        out2d (:obj:`array`): Output tabulated on an (a, k) grid.
        a (:obj:`float` or `array`): Original scale-factor input.
        k (:obj:`float` or `array`): Original wavenumber input.

    Returns:
        :obj:`float` or `array`: Output with dimensions matching inputs.
    """
    a_is_scalar = np.ndim(a) == 0
    k_is_scalar = np.ndim(k) == 0

    if a_is_scalar and k_is_scalar:
        return float(out2d[0, 0])
    if a_is_scalar:
        return out2d[0, :]
    if k_is_scalar:
        return out2d[:, 0]
    return out2d
