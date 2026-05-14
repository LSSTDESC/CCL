"""SP(k) baryonic suppression model for pyccl.

References:
        - Salcido et al. 2023, MNRAS 523, 2247
            (https://doi.org/10.1093/mnras/stad1474)
        - pyspk package: https://github.com/jemme07/pyspk
"""

__all__ = ("BaryonsSPK",)

import importlib
from collections import OrderedDict
from typing import Any, Callable, cast
import warnings as warnings_builtin

import numpy as np

from .. import CCLWarning, Pk2D, warnings
from . import Baryons


def _warn_ccl(*args: Any, **kwargs: Any) -> None:
    """Forward warnings through CCL's warning utility."""
    cast(Any, warnings).warn(*args, **kwargs)


_SUPPORTED_RELATION_KINDS = (
    "power_law",
    "cosmo_power_law",
    "double_power_law",
    "binned",
)

_SUPPORTED_OUT_OF_RANGE = (
    "raise",
    "unity",
    "nan",
)

_RELATION_PARAMS = {
    "power_law": {
        "required": ("fb_a", "fb_pow"),
        "optional": {"fb_pivot": 1.0},
    },
    "cosmo_power_law": {
        "required": ("alpha", "beta", "gamma"),
        "optional": {},
    },
    "double_power_law": {
        "required": ("epsilon", "alpha", "beta", "gamma", "m_pivot"),
        "optional": {},
    },
    "binned": {
        "required": ("M_halo", "fb"),
        "optional": {"extrapolate": False},
    },
}


def _arraylike_to_float_list(values: Any, *, name: str) -> list[float]:
    """Convert array-like to a list of finite floats."""
    arr = np.atleast_1d(values).astype(float)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"`{name}` must contain finite values.")
    return arr.tolist()


def _normalize_relation_parameters(
    relation_kind: str, relation_params: dict[str, Any]
) -> dict[str, Any]:
    """Validate and fill defaults for relation-specific parameters."""
    if relation_kind not in _SUPPORTED_RELATION_KINDS:
        raise ValueError(
            f"`relation_kind` must be one of {_SUPPORTED_RELATION_KINDS}."
        )

    relation_cfg = _RELATION_PARAMS[relation_kind]
    required = set(relation_cfg["required"])
    optional = relation_cfg["optional"]

    unknown = set(relation_params) - required - set(optional)
    if unknown:
        raise ValueError(
            f"Unknown parameters for relation_kind='{relation_kind}': "
            f"{tuple(sorted(unknown))}."
        )

    missing = required - set(relation_params)
    if missing:
        missing_keys = tuple(sorted(missing))
        raise ValueError(
            "Missing required parameters for "
            f"relation_kind='{relation_kind}': {missing_keys}."
        )

    normalized = {}
    for key in relation_cfg["required"]:
        normalized[key] = relation_params[key]
    for key, default in optional.items():
        normalized[key] = relation_params.get(key, default)

    if relation_kind == "binned":
        normalized["M_halo"] = _arraylike_to_float_list(
            normalized["M_halo"], name="M_halo"
        )
        normalized["fb"] = _arraylike_to_float_list(
            normalized["fb"], name="fb"
        )
        if len(normalized["M_halo"]) != len(normalized["fb"]):
            raise ValueError("`M_halo` and `fb` must have the same length.")
        normalized["extrapolate"] = bool(normalized["extrapolate"])

    if "fb_pivot" in normalized and normalized["fb_pivot"] <= 0:
        raise ValueError("`fb_pivot` must be strictly positive.")
    if "m_pivot" in normalized and normalized["m_pivot"] <= 0:
        raise ValueError("`m_pivot` must be strictly positive.")

    return normalized


class BaryonsSPK(Baryons):
    """SP(k) baryonic suppression model (Salcido et al. 2023).

    Applies a multiplicative correction to the matter power spectrum:
    ``P_bar(k, a) = P_DMO(k, a) * f_SPk(k, a)``.

    Wavenumbers are passed in ``Mpc^-1`` (CCL convention); the
    conversion to ``h/Mpc`` (pyspk convention) is handled internally.

    Args:
        SO (int): Spherical overdensity, ``200`` or ``500``.
        relation_kind (str): Baryon-fraction relation. One of
            ``power_law``, ``cosmo_power_law``, ``double_power_law``,
            or ``binned``.
        max_evaluator_cache_size (int): LRU cache size for pyspk
            evaluator objects.
        k_out_of_range (str): Out-of-range-k policy: ``raise``,
            ``unity``, or ``nan``. Default ``raise``.
        z_out_of_range (str): Out-of-range-z policy: ``raise``,
            ``unity``, or ``nan``. Default ``unity``.
        **relation_params: Keyword arguments for the chosen
            ``relation_kind``:

            - ``power_law``: ``fb_a``, ``fb_pow``,
              optional ``fb_pivot`` (default 1.0).
            - ``cosmo_power_law``: ``alpha``, ``beta``, ``gamma``.
            - ``double_power_law``: ``epsilon``, ``alpha``, ``beta``,
              ``gamma``, ``m_pivot``.
            - ``binned``: ``M_halo`` (array), ``fb`` (array),
              optional ``extrapolate`` (default False).
    """

    name = "SPK"  # pyright: ignore[reportAssignmentType]
    __repr_attrs__ = __eq_attrs__ = (
        "SO",
        "relation_kind",
        "relation_params",
        "max_evaluator_cache_size",
        "k_out_of_range",
        "z_out_of_range",
    )

    def __init__(
        self,
        *,
        SO=200,
        relation_kind="power_law",
        max_evaluator_cache_size=8,
        k_out_of_range="raise",
        z_out_of_range="unity",
        **relation_params,
    ):
        self.SO = SO
        self.relation_kind = relation_kind
        self.max_evaluator_cache_size = int(max_evaluator_cache_size)
        self.k_out_of_range = k_out_of_range
        self.z_out_of_range = z_out_of_range

        self._pyspk = None
        self._forwarded_warning_messages = set()
        self._evaluator_cache: OrderedDict[
            tuple[Any, ...], Callable[..., Any]
        ] = OrderedDict()

        self._validate_settings()
        self.relation_params = _normalize_relation_parameters(
            self.relation_kind, relation_params
        )
        self._import_pyspk()

    def _validate_settings(self) -> None:
        """Check SO, relation_kind, cache size, and OOR policies."""
        if self.SO not in (200, 500):
            raise ValueError("`SO` must be either 200 or 500.")
        if self.relation_kind not in _SUPPORTED_RELATION_KINDS:
            raise ValueError(
                f"`relation_kind` must be one of {_SUPPORTED_RELATION_KINDS}."
            )
        if self.max_evaluator_cache_size < 1:
            raise ValueError("`max_evaluator_cache_size` must be >= 1.")
        if self.k_out_of_range not in _SUPPORTED_OUT_OF_RANGE:
            raise ValueError(
                f"`k_out_of_range` must be one of {_SUPPORTED_OUT_OF_RANGE}."
            )
        if self.z_out_of_range not in _SUPPORTED_OUT_OF_RANGE:
            raise ValueError(
                f"`z_out_of_range` must be one of {_SUPPORTED_OUT_OF_RANGE}."
            )

    def _import_pyspk(self) -> Any:
        """Lazily import and cache the pyspk module."""
        if self._pyspk is None:
            try:
                self._pyspk = importlib.import_module("pyspk")
            except ModuleNotFoundError as err:
                raise ModuleNotFoundError(
                    "BaryonsSPK requires the optional dependency "
                    "`pyspk>=2.0.0`. "
                    "Install it in your environment to use this model."
                ) from err
        return self._pyspk

    @staticmethod
    def _k_mpc_to_hmpc(cosmo: Any, k_mpc: Any) -> np.ndarray:
        """Convert k from Mpc^-1 to h/Mpc."""
        return (
            np.asarray(np.atleast_1d(k_mpc), dtype=float)
            / float(cosmo["h"])
        )

    def _evaluator_cache_key(
        self, cosmo: Any, k_hmpc: np.ndarray
    ) -> tuple[Any, ...]:
        """Build a hashable key for the evaluator LRU cache."""
        return (
            float(cosmo["h"]),
            tuple(np.asarray(k_hmpc, dtype=np.float64).tolist()),
        )

    def _build_evaluator(self, k_hmpc: np.ndarray) -> Callable[..., Any]:
        """Construct a pyspk fast evaluator for the given k-grid."""
        pyspk = self._import_pyspk()
        with warnings_builtin.catch_warnings(record=True) as caught:
            warnings_builtin.simplefilter("always")
            evaluator = pyspk.build_sup_model_evaluator(
                SO=self.SO,
                relation_kind=self.relation_kind,
                k_array=np.asarray(k_hmpc, dtype=float),
            )
        self._forward_pyspk_warnings(caught)
        return evaluator

    def _get_cached_evaluator(
        self, cosmo: Any, k_hmpc: np.ndarray
    ) -> Callable[..., Any]:
        """Return a cached evaluator, building one if needed."""
        key = self._evaluator_cache_key(cosmo, k_hmpc)
        cached = self._evaluator_cache.get(key)
        if cached is not None:
            self._evaluator_cache.move_to_end(key)
            return cached

        evaluator = self._build_evaluator(k_hmpc)
        self._evaluator_cache[key] = evaluator
        self._evaluator_cache.move_to_end(key)
        if len(self._evaluator_cache) > self.max_evaluator_cache_size:
            self._evaluator_cache.popitem(last=False)
        return evaluator

    @staticmethod
    def _make_efunc(cosmo: Any) -> Callable[[float], Any]:
        """Bridge CCL's h_over_h0(a) to pyspk's efunc(z)."""
        return lambda z: cosmo.h_over_h0(1.0 / (1.0 + z))

    def _forward_pyspk_warnings(self, caught_warnings: list[Any]) -> None:
        """Re-emit pyspk warnings via CCL, deduplicating per instance."""
        for caught in caught_warnings:
            msg = str(caught.message)
            if msg in self._forwarded_warning_messages:
                continue
            if len(self._forwarded_warning_messages) >= 10:
                self._forwarded_warning_messages.clear()
            self._forwarded_warning_messages.add(msg)
            _warn_ccl(
                msg,
                category=CCLWarning,
                importance="low",
                stacklevel=3,
            )

    def _evaluate_suppression(
        self, z: float, evaluator: Callable[..., Any], kwargs: dict[str, Any]
    ) -> np.ndarray:
        """Run the evaluator at a single redshift, forwarding warnings."""
        with warnings_builtin.catch_warnings(record=True) as caught:
            warnings_builtin.simplefilter("always")
            _, sup = evaluator(z=float(z), **kwargs)
        self._forward_pyspk_warnings(caught)
        return np.asarray(sup, dtype=float)

    def _compute_suppression_grid(
        self, cosmo: Any, k: Any, a: Any
    ) -> np.ndarray:
        """Evaluate f_SPk(k, a) over a 2-D grid of (a, k).

        The output array has shape ``(len(a), len(k))`` and is filled
        according to the ``k_out_of_range`` and ``z_out_of_range`` policies:

        - The grid is initialised to 1.0 (no baryonic effect).
        - If ``k_out_of_range="nan"``, columns beyond calibration are set
          to NaN before any redshift evaluation.
        - For each scale factor whose redshift is within calibration, the
          valid-k columns are overwritten with pyspk suppression values.
        - For redshifts beyond calibration:
            - ``"unity"``: row keeps its initial fill (1.0 for valid-k,
              NaN for invalid-k if that policy is "nan").
            - ``"nan"``: entire row is overwritten with NaN.
        """
        a_use = np.atleast_1d(a).astype(float)
        k_use = np.atleast_1d(k).astype(float)

        if np.any(~np.isfinite(k_use)) or np.any(k_use <= 0):
            raise ValueError(
                "`k` must contain finite strictly positive values."
            )
        if np.any(~np.isfinite(a_use)) or np.any(a_use <= 0):
            raise ValueError(
                "`a` must contain finite strictly positive values."
            )

        k_hmpc = self._k_mpc_to_hmpc(cosmo, k_use)

        kwargs = dict(self.relation_params)
        if self.relation_kind in ("cosmo_power_law", "double_power_law"):
            kwargs["efunc"] = self._make_efunc(cosmo)

        pyspk = self._import_pyspk()
        z_max_cal = pyspk.constants.CALIBRATED_Z_MAX
        k_max_cal = pyspk.constants.CALIBRATED_K_MAX

        valid_k = k_hmpc <= k_max_cal
        if self.k_out_of_range == "raise" and not np.all(valid_k):
            raise ValueError(
                "Requested k exceeds pyspk calibration range: "
                f"k_hmpc_max={float(np.max(k_hmpc)):.6g} > "
                f"{float(k_max_cal):.6g}."
            )

        if np.any(valid_k):
            evaluator = self._get_cached_evaluator(cosmo, k_hmpc[valid_k])
        else:
            evaluator = None

        fka = np.ones((a_use.size, k_use.size), dtype=float)
        if self.k_out_of_range == "nan":
            fka[:, ~valid_k] = np.nan

        z_use = 1.0 / a_use - 1.0
        valid_z = z_use <= z_max_cal
        if self.z_out_of_range == "raise" and not np.all(valid_z):
            raise ValueError(
                "Requested z exceeds pyspk calibration range: "
                f"z_max={float(np.max(z_use)):.6g} > "
                f"{float(z_max_cal):.6g}."
            )

        for ia, aval in enumerate(a_use):
            if not valid_z[ia]:
                if self.z_out_of_range == "nan":
                    fka[ia, :] = np.nan
                # "unity": leave row as initial fill (1.0 / NaN per k policy).
                continue

            z = float(z_use[ia])
            if evaluator is not None:
                sup = self._evaluate_suppression(z, evaluator, kwargs)
                fka[ia, valid_k] = sup

        return fka

    def boost_factor(self, cosmo: Any, k: Any, a: Any) -> Any:
        """SP(k) multiplicative suppression factor.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological
                parameters.
            k (float or array): Wavenumber in Mpc^-1.
            a (float or array): Scale factor.

        Returns:
            float or array: Suppression factor f_SP(k, a).
        """
        fka = self._compute_suppression_grid(cosmo, k, a)
        if np.ndim(k) == 0:
            fka = np.squeeze(fka, axis=-1)
        if np.ndim(a) == 0:
            fka = np.squeeze(fka, axis=0)
        return fka

    def update_parameters(
        self,
        *,
        SO=None,
        relation_kind=None,
        max_evaluator_cache_size=None,
        k_out_of_range=None,
        z_out_of_range=None,
        **relation_params,
    ):
        """Update SP(k) parameters. ``None`` values are left unchanged.

        Args:
            SO (int): Spherical overdensity.
            relation_kind (str): Baryon-fraction relation kind.
            max_evaluator_cache_size (int): LRU cache size.
            k_out_of_range (str): Out-of-range-k policy.
            z_out_of_range (str): Out-of-range-z policy.
            **relation_params: Relation-specific parameters to update
                (e.g. ``fb_a``, ``fb_pow`` for ``power_law``).
                Only supplied keys are changed; others are preserved.
        """
        if SO is not None:
            self.SO = SO
        if max_evaluator_cache_size is not None:
            self.max_evaluator_cache_size = int(max_evaluator_cache_size)
        if k_out_of_range is not None:
            self.k_out_of_range = k_out_of_range
        if z_out_of_range is not None:
            self.z_out_of_range = z_out_of_range

        new_kind = (
            self.relation_kind if relation_kind is None else relation_kind
        )
        if relation_kind is None or new_kind == self.relation_kind:
            merged_relation_params = dict(self.relation_params)
        else:
            merged_relation_params = {}
        merged_relation_params.update(relation_params)

        self.relation_kind = new_kind
        self._validate_settings()
        self.relation_params = _normalize_relation_parameters(
            self.relation_kind, merged_relation_params
        )
        self._evaluator_cache.clear()

    def _include_baryonic_effects(self, cosmo: Any, pk: Pk2D) -> Pk2D:
        """Apply SP(k) suppression to a Pk2D power spectrum.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological
                parameters.
            pk (:class:`~pyccl.pk2d.Pk2D`): Input (DMO) power spectrum.

        Returns:
            :class:`~pyccl.pk2d.Pk2D`: Baryonic-corrected power spectrum.
        """
        a_arr, lk_arr, pk_arr = pk.get_spline_arrays()
        k_arr = np.exp(lk_arr)

        fka = self._compute_suppression_grid(cosmo, k_arr, a_arr)

        # Preserve raw pyspk non-finite outputs in boost_factor, but avoid
        # contaminating the internal 2D spline representation with NaNs.
        # Policy fills produce all-NaN rows (z) or all-NaN columns (k).
        # Identify these independently to avoid cross-contamination, then
        # clean up any residual partial NaN from pyspk itself.
        nan_rows = np.all(~np.isfinite(fka), axis=1)
        nan_cols = np.all(~np.isfinite(fka), axis=0)

        if np.any(nan_rows):
            keep = ~nan_rows
            a_arr = a_arr[keep]
            pk_arr = pk_arr[keep, :]
            fka = fka[keep, :]

        if np.any(nan_cols):
            keep = ~nan_cols
            lk_arr = lk_arr[keep]
            pk_arr = pk_arr[:, keep]
            fka = fka[:, keep]

        # Handle any residual non-finite values (e.g. from pyspk internals).
        finite_cols = np.all(np.isfinite(fka), axis=0)
        if not np.all(finite_cols):
            lk_arr = lk_arr[finite_cols]
            pk_arr = pk_arr[:, finite_cols]
            fka = fka[:, finite_cols]

        finite_rows = np.all(np.isfinite(fka), axis=1)
        if not np.all(finite_rows):
            a_arr = a_arr[finite_rows]
            pk_arr = pk_arr[finite_rows, :]
            fka = fka[finite_rows, :]

        if (
            np.any(nan_rows)
            or np.any(nan_cols)
            or not np.all(finite_cols)
            or not np.all(finite_rows)
        ):
            _warn_ccl(
                "SP(k) returned non-finite values on part of the Pk2D grid; "
                "dropping non-finite a-rows/k-columns when building the "
                "baryonic Pk2D.",
                category=CCLWarning,
                importance="high",
                stacklevel=3,
            )

        pk_arr *= fka

        if pk.psp.is_log:
            np.log(pk_arr, out=pk_arr)  # in-place log

        return Pk2D(
            a_arr=a_arr,
            lk_arr=lk_arr,
            pk_arr=pk_arr,
            is_logp=pk.psp.is_log,
            extrap_order_lok=pk.extrap_order_lok or 1,
            extrap_order_hik=pk.extrap_order_hik or 2,
        )
