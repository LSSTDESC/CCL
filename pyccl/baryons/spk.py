"""SP(k)-based baryonic suppression model integration for pyccl.

References:
        - Salcido et al. 2023, MNRAS 523, 2247
            (https://doi.org/10.1093/mnras/stad1474)
        - arXiv preprint: https://arxiv.org/abs/2305.09710
        - pyspk package: https://github.com/jemme07/pyspk
"""

__all__ = ("BaryonsSPK",)

import importlib
from typing import Any, Callable, cast
import warnings as warnings_builtin

import numpy as np

from .. import CCLWarning, Pk2D, warnings
from . import Baryons


def _warn_ccl(*args: Any, **kwargs: Any) -> None:
    """Forward warnings through CCL's warning utility.

    Args:
        *args: Positional arguments for ``warnings.warn``.
        **kwargs: Keyword arguments for ``warnings.warn``.
    """
    cast(Any, warnings).warn(*args, **kwargs)


_SUPPORTED_RELATION_KINDS = ("power_law", "cosmo_power_law",
                             "double_power_law", "binned")
_SUPPORTED_OUT_OF_BOUNDS_POLICIES = ("error", "unity", "nan")
_DEFAULT_K_MIN_HMPC = 0.005
_DEFAULT_K_MAX_HMPC = 8.0

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
    """Normalize array-like input into a finite list of floats.

    Args:
        values: Scalar or array-like values to normalize.
        name: Parameter name used in error messages.

    Returns:
        Normalized list of floats.

    Raises:
        ValueError: If any value is not finite.
    """
    arr = np.atleast_1d(values).astype(float)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"`{name}` must contain finite values.")
    return arr.tolist()


def _normalize_relation_parameters(
        relation_kind: str,
        relation_params: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize relation-specific SP(k) parameters.

    Args:
        relation_kind: SP(k) relation mode.
        relation_params: Raw relation parameters.

    Returns:
        A normalized parameter dictionary with defaults applied.

    Raises:
        ValueError: If relation kind is unsupported or parameters are invalid.
    """
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

    Reference:
        Salcido et al. 2023, MNRAS 523, 2247.
        https://doi.org/10.1093/mnras/stad1474

    Relation modes:
        - ``power_law``: ``fb_a``, ``fb_pow``, optional ``fb_pivot``
        - ``cosmo_power_law``: ``alpha``, ``beta``, ``gamma``
        - ``double_power_law``: ``epsilon``, ``alpha``, ``beta``, ``gamma``,
          ``m_pivot``
        - ``binned``: ``M_halo``, ``fb``, optional ``extrapolate``

    Args:
        SO: Spherical overdensity. Supported values are ``200`` and ``500``.
        relation_kind: One of ``power_law``, ``cosmo_power_law``,
            ``double_power_law``, or ``binned``.
        k_min_mpc: Minimum requested CCL scale in ``Mpc^-1``.
        k_max_mpc: Maximum requested CCL scale in ``Mpc^-1``.
        n_k: Number of logarithmic points in the internal SP(k) grid.
        out_of_bounds_policy: Behavior for ``k > k_max_mpc``.
            Supported values are ``error``, ``unity``, and ``nan``.
            For policy ``error``, requests above ``k_max_mpc`` raise
            ``ValueError``.
        **relation_params: Parameters required by ``relation_kind``.

    Raises:
        ModuleNotFoundError: If ``pyspk`` is not installed.
        ValueError: If settings or relation parameters are invalid.
    """
    name = "SPK"  # pyright: ignore[reportAssignmentType]
    __repr_attrs__ = __eq_attrs__ = (
        "SO",
        "relation_kind",
        "k_min_mpc",
        "k_max_mpc",
        "n_k",
        "out_of_bounds_policy",
        "relation_params",
    )

    def __init__(self, *, SO=200, relation_kind="power_law",
                 k_min_mpc=None, k_max_mpc=None, n_k=128,
                 out_of_bounds_policy="error", **relation_params):
        """Initialize a BaryonsSPK model instance.

        Args:
            SO: Spherical overdensity, either 200 or 500.
            relation_kind: Relation mode used by ``pyspk``.
            k_min_mpc: Minimum requested CCL scale in ``Mpc^-1``.
                If ``None``, defaults to ``0.005 * h``.
            k_max_mpc: Maximum requested CCL scale in ``Mpc^-1``.
                If ``None``, defaults to ``8.0 * h``.
            n_k: Number of logarithmic samples in SP(k) internal grid.
            out_of_bounds_policy: Policy for ``k > k_max_mpc``.
            **relation_params: Parameters for the selected relation mode.
        """
        self.SO = SO
        self.relation_kind = relation_kind
        self.k_min_mpc = (None if k_min_mpc is None else float(k_min_mpc))
        self.k_max_mpc = (None if k_max_mpc is None else float(k_max_mpc))
        self.n_k = int(n_k)
        self.out_of_bounds_policy = out_of_bounds_policy
        self._pyspk = None
        self._forwarded_warning_messages = set()
        self._cached_k_grid_hmpc: np.ndarray | None = None
        self._cached_evaluator: Callable[..., Any] | None = None
        self._cached_evaluator_key: tuple[Any, ...] | None = None

        self._validate_settings()
        self.relation_params = _normalize_relation_parameters(
            self.relation_kind, relation_params)
        self._import_pyspk()

    def _validate_settings(self) -> None:
        """Validate global SP(k) model settings.

        Raises:
            ValueError: If any configuration option is invalid.
        """
        if self.SO not in (200, 500):
            raise ValueError("`SO` must be either 200 or 500.")
        if self.relation_kind not in _SUPPORTED_RELATION_KINDS:
            raise ValueError(
                f"`relation_kind` must be one of {_SUPPORTED_RELATION_KINDS}.")
        if self.k_min_mpc is not None and self.k_min_mpc <= 0:
            raise ValueError("`k_min_mpc` must be > 0.")
        if self.k_max_mpc is not None and self.k_max_mpc <= 0:
            raise ValueError("`k_max_mpc` must be > 0.")
        if (self.k_min_mpc is not None and self.k_max_mpc is not None
                and self.k_min_mpc >= self.k_max_mpc):
            raise ValueError(
                "`k_min_mpc` must be strictly smaller than `k_max_mpc`.")
        if self.n_k < 2:
            raise ValueError("`n_k` must be >= 2.")
        if self.out_of_bounds_policy not in _SUPPORTED_OUT_OF_BOUNDS_POLICIES:
            raise ValueError(
                "`out_of_bounds_policy` must be one of "
                f"{_SUPPORTED_OUT_OF_BOUNDS_POLICIES}."
            )

    def _import_pyspk(self) -> Any:
        """Import and cache ``pyspk`` lazily.

        Returns:
            Imported ``pyspk`` module.

        Raises:
            ModuleNotFoundError: If ``pyspk`` is unavailable.
        """
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

    def _build_evaluator(self, k_hmpc: np.ndarray) -> Callable[..., Any]:
        """Build a ``pyspk`` evaluator on a fixed ``h/Mpc`` grid."""
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

    def _evaluator_cache_key(self) -> tuple[Any, ...]:
        """Return a hashable key for evaluator-cache invalidation."""
        return (
            self.SO,
            self.relation_kind,
            self.k_min_mpc,
            self.k_max_mpc,
            self.n_k,
        )

    def _effective_k_bounds_mpc(self, cosmo: Any) -> tuple[float, float]:
        """Return effective CCL-facing k-range bounds in ``Mpc^-1``."""
        h = float(cosmo["h"])
        k_min_mpc = self.k_min_mpc
        k_max_mpc = self.k_max_mpc
        if k_min_mpc is None:
            k_min_mpc = _DEFAULT_K_MIN_HMPC * h
        if k_max_mpc is None:
            k_max_mpc = _DEFAULT_K_MAX_HMPC * h
        if k_min_mpc >= k_max_mpc:
            raise ValueError(
                "Effective k bounds are invalid: "
                f"k_min_mpc={k_min_mpc} must be smaller than "
                f"k_max_mpc={k_max_mpc}.")
        return k_min_mpc, k_max_mpc

    def _get_cached_evaluator(
            self,
            cosmo: Any) -> tuple[np.ndarray, Callable[..., Any]]:
        """Return a cached evaluator and its fixed internal SP(k) grid."""
        h = float(cosmo["h"])
        k_min_mpc, k_max_mpc = self._effective_k_bounds_mpc(cosmo)
        key = self._evaluator_cache_key() + (h, k_min_mpc, k_max_mpc)
        if self._cached_evaluator is None or self._cached_evaluator_key != key:
            k_grid_mpc = np.geomspace(k_min_mpc, k_max_mpc, self.n_k)
            k_grid_hmpc = self._k_mpc_to_hmpc(cosmo, k_grid_mpc)
            self._cached_k_grid_hmpc = k_grid_hmpc
            self._cached_evaluator = self._build_evaluator(k_grid_hmpc)
            self._cached_evaluator_key = key
        return cast(np.ndarray, self._cached_k_grid_hmpc), cast(
            Callable[..., Any], self._cached_evaluator)

    def _invalidate_evaluator_cache(self) -> None:
        """Reset cached evaluator and grid."""
        self._cached_k_grid_hmpc = None
        self._cached_evaluator = None
        self._cached_evaluator_key = None

    @staticmethod
    def _make_efunc(cosmo: Any) -> Callable[[float], Any]:
        """Create an ``E(z)`` callable compatible with ``pyspk``."""
        return lambda z: cosmo.h_over_h0(1.0 / (1.0 + z))

    def _forward_pyspk_warnings(self, caught_warnings: list[Any]) -> None:
        """Forward unique warnings emitted by ``pyspk`` via CCL warnings."""
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
            cosmo: Any,
            z: float,
            evaluator: Callable[..., Any]) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate SP(k) suppression with a provided ``pyspk`` evaluator."""
        kwargs = dict(self.relation_params)
        if self.relation_kind in ("cosmo_power_law", "double_power_law"):
            kwargs["efunc"] = self._make_efunc(cosmo)

        with warnings_builtin.catch_warnings(record=True) as caught:
            warnings_builtin.simplefilter("always")
            k_hmpc, sup = evaluator(z=float(z), **kwargs)
        self._forward_pyspk_warnings(caught)
        return np.asarray(k_hmpc), np.asarray(sup)

    def _apply_out_of_bounds_policy(
            self,
            cosmo: Any,
            k_mpc: np.ndarray,
            fka: np.ndarray) -> np.ndarray:
        """Apply configured high-k policy to suppression factors.

        Args:
            k_mpc: Wavenumbers in ``Mpc^-1``.
            fka: Suppression factors matching ``k_mpc``.

        Returns:
            Policy-adjusted suppression factors.

        Raises:
            ValueError: For out-of-range ``k`` when policy is ``error``.
        """
        _, k_max_mpc = self._effective_k_bounds_mpc(cosmo)
        out_hi = k_mpc > k_max_mpc
        if not np.any(out_hi):
            return fka

        if self.out_of_bounds_policy == "error":
            raise ValueError(
                "Requested k values exceed the configured SP(k) limit "
                f"k_max_mpc={k_max_mpc}. Set out_of_bounds_policy to "
                "'unity' or 'nan' to override this behavior."
            )

        fka = np.array(fka, copy=True)
        if self.out_of_bounds_policy == "unity":
            fka[out_hi] = 1.0
        else:
            fka[out_hi] = np.nan
        return fka

    def _map_k_to_domain(self, cosmo: Any, k_mpc: np.ndarray) -> np.ndarray:
        """Map requested k values onto SP(k) calibrated grid domain."""
        k_min_mpc, k_max_mpc = self._effective_k_bounds_mpc(cosmo)
        return np.clip(k_mpc, k_min_mpc, k_max_mpc)

    def _interpolate_suppression(
            self,
            cosmo: Any,
            k_mpc: np.ndarray,
            k_grid_hmpc: np.ndarray,
            sup_grid: np.ndarray) -> np.ndarray:
        """Interpolate suppression from cached SP(k) grid to target k."""
        k_mapped_hmpc = self._k_mpc_to_hmpc(
            cosmo, self._map_k_to_domain(cosmo, k_mpc))
        return np.interp(k_mapped_hmpc, k_grid_hmpc, sup_grid)

    @staticmethod
    def _k_mpc_to_hmpc(cosmo: Any, k_mpc: Any) -> np.ndarray:
        """Convert CCL wavenumbers from ``Mpc^-1`` to ``h/Mpc``."""
        return np.asarray(np.atleast_1d(k_mpc), dtype=float) / float(cosmo["h"])

    def boost_factor(self, cosmo: Any, k: Any, a: Any) -> Any:
        """Compute the SP(k) baryonic boost factor.

        Args:
            cosmo: CCL cosmology object.
            k: Wavenumber(s) in ``Mpc^-1``.
            a: Scale factor(s).

        Returns:
            Scalar or array correction factor matching the shapes of ``k`` and
            ``a``.

        Raises:
            ValueError: If ``k`` or ``a`` contains non-positive values.
        """
        a_use = np.atleast_1d(a)
        k_use = np.atleast_1d(k)
        if np.any(k_use <= 0):
            raise ValueError("`k` must contain strictly positive values.")
        if np.any(a_use <= 0):
            raise ValueError("`a` must contain strictly positive values.")

        k_grid_hmpc, evaluator = self._get_cached_evaluator(cosmo)
        fka = np.empty((a_use.size, k_use.size))

        for ia, aval in enumerate(a_use):
            z = 1.0 / aval - 1.0
            _, sup_grid = self._evaluate_suppression(cosmo, z, evaluator)
            fka_row = self._interpolate_suppression(
                cosmo, k_use, k_grid_hmpc, np.asarray(sup_grid, dtype=float))
            fka[ia, :] = self._apply_out_of_bounds_policy(cosmo, k_use, fka_row)

        if np.ndim(k) == 0:
            fka = np.squeeze(fka, axis=-1)
        if np.ndim(a) == 0:
            fka = np.squeeze(fka, axis=0)
        return fka

    def update_parameters(self, *, SO=None, relation_kind=None,
                          k_min_mpc=None, k_max_mpc=None, n_k=None,
                          out_of_bounds_policy=None, **relation_params):
        """Update SP(k) model configuration in place.

        Args:
            SO: Optional new spherical overdensity.
            relation_kind: Optional new relation mode.
            k_min_mpc: Optional new minimum ``Mpc^-1`` scale.
            k_max_mpc: Optional new maximum ``Mpc^-1`` scale.
            n_k: Optional new internal grid sample count.
            out_of_bounds_policy: Optional new out-of-bounds policy.
            **relation_params: Relation parameters to replace or update.

        All arguments set to ``None`` will be left untouched.
        """
        if SO is not None:
            self.SO = SO
        if k_min_mpc is not None:
            self.k_min_mpc = float(k_min_mpc)
        if k_max_mpc is not None:
            self.k_max_mpc = float(k_max_mpc)
        if n_k is not None:
            self.n_k = int(n_k)
        if out_of_bounds_policy is not None:
            self.out_of_bounds_policy = out_of_bounds_policy

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
        self._invalidate_evaluator_cache()

    def _include_baryonic_effects(self, cosmo: Any, pk: Pk2D) -> Pk2D:
        """Apply SP(k) baryonic suppression to a ``Pk2D`` power spectrum.

        SP(k) is evaluated on a fixed cached internal grid and interpolated
        to the ``Pk2D`` k-grid. Values above ``k_max_mpc`` are handled by
        ``out_of_bounds_policy``.

        Args:
            cosmo: CCL cosmology object.
            pk: Input dark-matter-only power spectrum.

        Returns:
            New ``Pk2D`` including baryonic suppression.
        """
        a_arr, lk_arr, pk_arr = pk.get_spline_arrays()
        k_arr = np.exp(lk_arr)
        k_grid_hmpc, evaluator = self._get_cached_evaluator(cosmo)

        # Restrict to pyspk's calibrated redshift range
        # (z <= CALIBRATED_Z_MAX).
        pyspk = self._import_pyspk()
        z_max_cal = pyspk.constants.CALIBRATED_Z_MAX
        a_min_cal = 1.0 / (1.0 + z_max_cal)

        fka = np.ones((a_arr.size, k_arr.size))
        for ia, aval in enumerate(a_arr):
            if aval < a_min_cal:
                continue  # z > z_max_cal: baryons negligible, leave unity
            z = 1.0 / aval - 1.0
            _, sup_grid = self._evaluate_suppression(cosmo, z, evaluator)
            fka[ia, :] = self._interpolate_suppression(
                cosmo, k_arr, k_grid_hmpc, np.asarray(sup_grid, dtype=float))

        for ia in range(a_arr.size):
            fka[ia, :] = self._apply_out_of_bounds_policy(cosmo, k_arr, fka[ia, :])

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
