"""SP(k)-based baryonic suppression model integration for pyccl.

References:
        - Salcido et al. 2023, MNRAS 523, 2247
            (https://doi.org/10.1093/mnras/stad1474)
        - arXiv preprint: https://arxiv.org/abs/2305.09710
        - pyspk package: https://github.com/jemme07/pyspk
"""

__all__ = ("BaryonsSPK",)

import hashlib
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
    """Normalize array-like input into a finite list of floats."""
    arr = np.atleast_1d(values).astype(float)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"`{name}` must contain finite values.")
    return arr.tolist()


def _normalize_relation_parameters(
        relation_kind: str,
        relation_params: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize relation-specific SP(k) parameters."""
    if relation_kind not in _SUPPORTED_RELATION_KINDS:
        raise ValueError(
            f"`relation_kind` must be one of {_SUPPORTED_RELATION_KINDS}.")

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
            normalized["M_halo"], name="M_halo")
        normalized["fb"] = _arraylike_to_float_list(
            normalized["fb"], name="fb")
        if len(normalized["M_halo"]) != len(normalized["fb"]):
            raise ValueError("`M_halo` and `fb` must have the same length.")
        normalized["extrapolate"] = bool(normalized["extrapolate"])

    if "fb_pivot" in normalized and normalized["fb_pivot"] <= 0:
        raise ValueError("`fb_pivot` must be strictly positive.")
    if "m_pivot" in normalized and normalized["m_pivot"] <= 0:
        raise ValueError("`m_pivot` must be strictly positive.")

    return normalized


class BaryonsSPK(Baryons):
    """SP(k) baryonic suppression model backed by ``pyspk``.

    The correction is applied multiplicatively:
    ``P_bar(k, a) = P_DMO(k, a) * f_SPk(k, a)``.

    CCL follows non-h-inverse conventions for user-facing wavenumbers
    (``k`` in ``Mpc^-1``). ``pyspk`` expects ``h/Mpc``. This wrapper performs
    that conversion internally and transparently.

    Notes:
        This implementation computes suppression on the exact requested k-grid.
        It avoids wrapper-side interpolation and shares one suppression engine
        across ``boost_factor`` and ``include_baryonic_effects``.

    Args:
        SO: Spherical overdensity. Supported values are ``200`` and ``500``.
        relation_kind: One of ``power_law``, ``cosmo_power_law``,
            ``double_power_law``, or ``binned``.
        max_evaluator_cache_size: Max number of evaluator objects cached for
            exact k-grids. Least-recently-used eviction is applied.
        **relation_params: Parameters required by ``relation_kind``.

    Raises:
        ModuleNotFoundError: If ``pyspk`` is not installed.
        ValueError: If settings or relation parameters are invalid.
    """
    name = "SPK"  # pyright: ignore[reportAssignmentType]
    __repr_attrs__ = __eq_attrs__ = (
        "SO",
        "relation_kind",
        "relation_params",
        "max_evaluator_cache_size",
    )

    def __init__(self, *, SO=200, relation_kind="power_law",
                 max_evaluator_cache_size=8, **relation_params):
        self.SO = SO
        self.relation_kind = relation_kind
        self.max_evaluator_cache_size = int(max_evaluator_cache_size)

        self._pyspk = None
        self._forwarded_warning_messages = set()
        self._evaluator_cache: OrderedDict[tuple[Any, ...], Callable[..., Any]] = (
            OrderedDict())

        self._validate_settings()
        self.relation_params = _normalize_relation_parameters(
            self.relation_kind, relation_params)
        self._import_pyspk()

    def _validate_settings(self) -> None:
        if self.SO not in (200, 500):
            raise ValueError("`SO` must be either 200 or 500.")
        if self.relation_kind not in _SUPPORTED_RELATION_KINDS:
            raise ValueError(
                f"`relation_kind` must be one of {_SUPPORTED_RELATION_KINDS}.")
        if self.max_evaluator_cache_size < 1:
            raise ValueError("`max_evaluator_cache_size` must be >= 1.")

    def _import_pyspk(self) -> Any:
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
        return np.asarray(np.atleast_1d(k_mpc), dtype=float) / float(cosmo["h"])

    @staticmethod
    def _k_digest(k_hmpc: np.ndarray) -> str:
        k_view = np.ascontiguousarray(k_hmpc, dtype=np.float64)
        return hashlib.blake2b(k_view.tobytes(), digest_size=16).hexdigest()

    def _evaluator_cache_key(self, cosmo: Any, k_hmpc: np.ndarray) -> tuple[Any, ...]:
        return (
            self.SO,
            self.relation_kind,
            float(cosmo["h"]),
            int(k_hmpc.size),
            self._k_digest(k_hmpc),
        )

    def _build_evaluator(self, k_hmpc: np.ndarray) -> Callable[..., Any]:
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
            self,
            cosmo: Any,
            k_hmpc: np.ndarray) -> Callable[..., Any]:
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
        return lambda z: cosmo.h_over_h0(1.0 / (1.0 + z))

    def _forward_pyspk_warnings(self, caught_warnings: list[Any]) -> None:
        for caught in caught_warnings:
            msg = str(caught.message)
            if msg in self._forwarded_warning_messages:
                continue
            self._forwarded_warning_messages.add(msg)
            _warn_ccl(
                msg,
                category=CCLWarning,
                importance="low",
                stacklevel=3,
            )

    def _evaluate_suppression(
            self,
            z: float,
            evaluator: Callable[..., Any],
            kwargs: dict[str, Any]) -> np.ndarray:
        with warnings_builtin.catch_warnings(record=True) as caught:
            warnings_builtin.simplefilter("always")
            _, sup = evaluator(z=float(z), **kwargs)
        self._forward_pyspk_warnings(caught)
        return np.asarray(sup, dtype=float)

    def _compute_suppression_grid(
            self,
            cosmo: Any,
            k: Any,
            a: Any,
            *,
            high_k_unity: bool = False) -> np.ndarray:
        a_use = np.atleast_1d(a).astype(float)
        k_use = np.atleast_1d(k).astype(float)

        if np.any(~np.isfinite(k_use)) or np.any(k_use <= 0):
            raise ValueError("`k` must contain finite strictly positive values.")
        if np.any(~np.isfinite(a_use)) or np.any(a_use <= 0):
            raise ValueError("`a` must contain finite strictly positive values.")

        k_hmpc = self._k_mpc_to_hmpc(cosmo, k_use)

        kwargs = dict(self.relation_params)
        if self.relation_kind in ("cosmo_power_law", "double_power_law"):
            kwargs["efunc"] = self._make_efunc(cosmo)

        pyspk = self._import_pyspk()
        z_max_cal = pyspk.constants.CALIBRATED_Z_MAX
        k_max_cal = pyspk.constants.CALIBRATED_K_MAX

        valid_k = k_hmpc <= k_max_cal
        if not high_k_unity and not np.all(valid_k):
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
        for ia, aval in enumerate(a_use):
            z = 1.0 / aval - 1.0
            if z > z_max_cal:
                continue
            if evaluator is not None:
                sup = self._evaluate_suppression(z, evaluator, kwargs)
                sup[~np.isfinite(sup)] = 1.0
                fka[ia, valid_k] = sup

        return fka

    def boost_factor(self, cosmo: Any, k: Any, a: Any) -> Any:
        fka = self._compute_suppression_grid(cosmo, k, a)
        if np.ndim(k) == 0:
            fka = np.squeeze(fka, axis=-1)
        if np.ndim(a) == 0:
            fka = np.squeeze(fka, axis=0)
        return fka

    def update_parameters(self, *, SO=None, relation_kind=None,
                          max_evaluator_cache_size=None, **relation_params):
        if SO is not None:
            self.SO = SO
        if max_evaluator_cache_size is not None:
            self.max_evaluator_cache_size = int(max_evaluator_cache_size)

        new_kind = (
            self.relation_kind if relation_kind is None else relation_kind)
        if relation_kind is None or new_kind == self.relation_kind:
            merged_relation_params = dict(self.relation_params)
        else:
            merged_relation_params = {}
        merged_relation_params.update(relation_params)

        self.relation_kind = new_kind
        self._validate_settings()
        self.relation_params = _normalize_relation_parameters(
            self.relation_kind, merged_relation_params)
        self._evaluator_cache.clear()

    def _include_baryonic_effects(self, cosmo: Any, pk: Pk2D) -> Pk2D:
        a_arr, lk_arr, pk_arr = pk.get_spline_arrays()
        k_arr = np.exp(lk_arr)

        fka = self._compute_suppression_grid(
            cosmo, k_arr, a_arr, high_k_unity=True)
        pk_arr *= fka

        if pk.psp.is_log:
            np.log(pk_arr, out=pk_arr)  # in-place log

        extrap_order_lok = (
            1 if pk.extrap_order_lok is None else pk.extrap_order_lok)
        extrap_order_hik = (
            2 if pk.extrap_order_hik is None else pk.extrap_order_hik)

        return Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr,
                    is_logp=pk.psp.is_log,
                    extrap_order_lok=extrap_order_lok,
                    extrap_order_hik=extrap_order_hik)
