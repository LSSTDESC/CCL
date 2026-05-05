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
    """The SP(k) baryonic suppression model from `pyspk`.

    The boost factor is applied multiplicatively:
    :math:`P_{\\rm bar.}(k, a) = P_{\\rm DMO}(k, a)\\, f_{\\rm SPk}(k, a)`.

    Args:
        SO (:obj:`int`): Spherical overdensity.
            Supported values are 200 and 500.
        relation_kind (:obj:`str`): One of ``power_law``, ``cosmo_power_law``,
            ``double_power_law`` or ``binned``.
        k_min_hmpc (:obj:`float`): Minimum internal SP(k) grid scale in
            :math:`h\\,{\\rm Mpc}^{-1}`.
        k_max_hmpc (:obj:`float`): Maximum internal SP(k) grid scale in
            :math:`h\\,{\\rm Mpc}^{-1}`.
        n_k (:obj:`int`): Number of logarithmic points in the internal
            SP(k) grid.
        out_of_bounds_policy (:obj:`str`): Behavior for requests above
            ``k_max_hmpc``.
            Supported values are ``error``, ``unity`` and ``nan``.
            For :meth:`boost_factor`, ``error`` raises a :class:`ValueError`.
            For :meth:`include_baryonic_effects`, ``error`` emits a
            :class:`~pyccl.CCLWarning` and uses unity above ``k_max_hmpc``
            because CCL's internal spline grid extends beyond the model domain.
        **relation_params: Parameters required by ``relation_kind`` and passed
            to the cached ``pyspk`` evaluator.

    Raises:
        ModuleNotFoundError: If ``pyspk`` is not installed.
        ValueError: If settings or relation parameters are invalid.
    """
    name = "SPK"  # pyright: ignore[reportAssignmentType]
    __repr_attrs__ = __eq_attrs__ = (
        "SO",
        "relation_kind",
        "k_min_hmpc",
        "k_max_hmpc",
        "n_k",
        "out_of_bounds_policy",
        "relation_params",
    )

    def __init__(self, *, SO=200, relation_kind="power_law",
                 k_min_hmpc=0.005, k_max_hmpc=8.0, n_k=128,
                 out_of_bounds_policy="error", **relation_params):
        """Initialize a BaryonsSPK model instance.

        Args:
            SO: Spherical overdensity, either 200 or 500.
            relation_kind: Relation mode used by ``pyspk``.
            k_min_hmpc: Minimum SP(k) internal grid scale in ``h/Mpc``.
            k_max_hmpc: Maximum SP(k) internal grid scale in ``h/Mpc``.
            n_k: Number of logarithmic samples in SP(k) internal grid.
            out_of_bounds_policy: Policy for ``k > k_max_hmpc``.
            **relation_params: Parameters for the selected relation mode.
        """
        self.SO = SO
        self.relation_kind = relation_kind
        self.k_min_hmpc = float(k_min_hmpc)
        self.k_max_hmpc = float(k_max_hmpc)
        self.n_k = int(n_k)
        self.out_of_bounds_policy = out_of_bounds_policy
        self._evaluator = None
        self._pyspk = None
        self._forwarded_warning_messages = set()
        self._warned_oob_on_spline_grid = False

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
        if self.k_min_hmpc <= 0 or self.k_max_hmpc <= 0:
            raise ValueError("`k_min_hmpc` and `k_max_hmpc` must be > 0.")
        if self.k_min_hmpc >= self.k_max_hmpc:
            raise ValueError(
                "`k_min_hmpc` must be strictly smaller than `k_max_hmpc`.")
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

    def _build_evaluator(self) -> None:
        """Build and cache the internal ``pyspk`` evaluator."""
        pyspk = self._import_pyspk()
        self._evaluator = pyspk.build_sup_model_evaluator(
            SO=self.SO,
            relation_kind=self.relation_kind,
            k_min=self.k_min_hmpc,
            k_max=self.k_max_hmpc,
            n=self.n_k,
        )

    def _get_evaluator(self) -> Callable[..., Any]:
        """Return the cached evaluator, creating it if needed."""
        if self._evaluator is None:
            self._build_evaluator()
        assert self._evaluator is not None
        return self._evaluator

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

    def _evaluate_on_internal_grid(
            self, cosmo: Any, z: float) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate suppression on SP(k)'s internal ``h/Mpc`` grid.

        Args:
            cosmo: CCL cosmology object.
            z: Redshift.

        Returns:
            Tuple ``(k_hmpc, suppression)`` as numpy arrays.
        """
        evaluator = self._get_evaluator()
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
            k_hmpc: np.ndarray,
            fka: np.ndarray,
            *,
            for_spline_grid: bool = False) -> np.ndarray:
        """Apply configured high-k policy to suppression factors.

        Args:
            k_hmpc: Wavenumbers in ``h/Mpc``.
            fka: Suppression factors matching ``k_hmpc``.
            for_spline_grid: Whether this is for CCL's internal spline grid.

        Returns:
            Policy-adjusted suppression factors.

        Raises:
            ValueError: For out-of-range ``k`` when policy is ``error`` and
                ``for_spline_grid`` is ``False``.
        """
        out_hi = k_hmpc > self.k_max_hmpc
        if not np.any(out_hi):
            return fka

        if self.out_of_bounds_policy == "error":
            if for_spline_grid:
                if not self._warned_oob_on_spline_grid:
                    self._warned_oob_on_spline_grid = True
                    _warn_ccl(
                        "CCL internal Pk2D grids extend above the configured "
                        f"SP(k) limit k_max_hmpc={self.k_max_hmpc}. "
                        "Falling back to unity above k_max_hmpc for "
                        "include_baryonic_effects(). Set out_of_bounds_policy "
                        "to 'nan' to propagate NaNs instead.",
                        category=CCLWarning,
                        importance="low",
                        stacklevel=3,
                    )
                return fka
            raise ValueError(
                "Requested k values exceed the configured SP(k) limit "
                f"k_max_hmpc={self.k_max_hmpc}. Set out_of_bounds_policy to "
                "'unity' or 'nan' to override this behavior."
            )

        fka = np.array(fka, copy=True)
        if self.out_of_bounds_policy == "unity":
            fka[out_hi] = 1.0
        else:
            fka[out_hi] = np.nan
        return fka

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
        a_use, k_use = map(np.atleast_1d, [a, k])
        if np.any(k_use <= 0):
            raise ValueError("`k` must contain strictly positive values.")
        if np.any(a_use <= 0):
            raise ValueError("`a` must contain strictly positive values.")

        k_hmpc = k_use / cosmo["h"]
        fka = np.empty((a_use.size, k_use.size))

        for ia, aval in enumerate(a_use):
            z = 1.0 / aval - 1.0
            k_spk, sup_spk = self._evaluate_on_internal_grid(cosmo, z)
            fka_row = np.interp(k_hmpc, k_spk, sup_spk,
                                left=sup_spk[0], right=sup_spk[-1])
            fka[ia, :] = self._apply_out_of_bounds_policy(k_hmpc, fka_row)

        if np.ndim(k) == 0:
            fka = np.squeeze(fka, axis=-1)
        if np.ndim(a) == 0:
            fka = np.squeeze(fka, axis=0)
        return fka

    def update_parameters(self, *, SO=None, relation_kind=None,
                          k_min_hmpc=None, k_max_hmpc=None, n_k=None,
                          out_of_bounds_policy=None, **relation_params):
        """Update SP(k) model configuration in place.

        Args:
            SO: Optional new spherical overdensity.
            relation_kind: Optional new relation mode.
            k_min_hmpc: Optional new minimum ``h/Mpc`` scale.
            k_max_hmpc: Optional new maximum ``h/Mpc`` scale.
            n_k: Optional new internal grid sample count.
            out_of_bounds_policy: Optional new out-of-bounds policy.
            **relation_params: Relation parameters to replace or update.

        All arguments set to ``None`` will be left untouched.
        """
        if SO is not None:
            self.SO = SO
        if k_min_hmpc is not None:
            self.k_min_hmpc = float(k_min_hmpc)
        if k_max_hmpc is not None:
            self.k_max_hmpc = float(k_max_hmpc)
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
        self._evaluator = None

    def _include_baryonic_effects(self, cosmo: Any, pk: Pk2D) -> Pk2D:
        """Apply SP(k) baryonic suppression to a ``Pk2D`` power spectrum.

        SP(k) is evaluated only inside calibrated ``k`` and ``z`` ranges.
        Internal CCL spline-grid values outside ``k_max_hmpc`` are handled by
        ``out_of_bounds_policy`` through :meth:`_apply_out_of_bounds_policy`.

        Args:
            cosmo: CCL cosmology object.
            pk: Input dark-matter-only power spectrum.

        Returns:
            New ``Pk2D`` including baryonic suppression.
        """
        a_arr, lk_arr, pk_arr = pk.get_spline_arrays()
        k_arr = np.exp(lk_arr)
        k_hmpc = k_arr / cosmo["h"]
        in_k_range = (k_hmpc >= self.k_min_hmpc) & (k_hmpc <= self.k_max_hmpc)

        # Restrict to pyspk's calibrated redshift range
        # (z <= CALIBRATED_Z_MAX).
        pyspk = self._import_pyspk()
        z_max_cal = pyspk.constants.CALIBRATED_Z_MAX
        a_min_cal = 1.0 / (1.0 + z_max_cal)

        fka = np.ones((a_arr.size, k_arr.size))
        if np.any(in_k_range):
            for ia, aval in enumerate(a_arr):
                if aval < a_min_cal:
                    continue  # z > z_max_cal: baryons negligible, leave unity
                z = 1.0 / aval - 1.0
                k_spk, sup_spk = self._evaluate_on_internal_grid(cosmo, z)
                fka[ia, in_k_range] = np.interp(
                    k_hmpc[in_k_range], k_spk, sup_spk,
                    left=sup_spk[0], right=sup_spk[-1])

        for ia in range(a_arr.size):
            fka[ia, :] = self._apply_out_of_bounds_policy(
                k_hmpc, fka[ia, :], for_spline_grid=True)

        pk_arr *= fka

        if pk.psp.is_log:
            np.log(pk_arr, out=pk_arr)  # in-place log

        extrap_order_lok = 1 if pk.extrap_order_lok is None else pk.extrap_order_lok
        extrap_order_hik = 2 if pk.extrap_order_hik is None else pk.extrap_order_hik

        return Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr,
                    is_logp=pk.psp.is_log,
                    extrap_order_lok=extrap_order_lok,
                    extrap_order_hik=extrap_order_hik)
