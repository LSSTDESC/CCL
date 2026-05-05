__all__ = ("BaryonsSPK",)

import importlib
import warnings as warnings_builtin

import numpy as np

from .. import CCLWarning, Pk2D, warnings
from . import Baryons


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


def _arraylike_to_float_list(values, *, name):
    arr = np.atleast_1d(values).astype(float)
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"`{name}` must contain finite values.")
    return arr.tolist()


def _normalize_relation_parameters(relation_kind, relation_params):
    if relation_kind not in _SUPPORTED_RELATION_KINDS:
        raise ValueError(f"`relation_kind` must be one of {_SUPPORTED_RELATION_KINDS}.")

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
        raise ValueError(
            f"Missing required parameters for relation_kind='{relation_kind}': "
            f"{tuple(sorted(missing))}."
        )

    normalized = {}
    for key in relation_cfg["required"]:
        normalized[key] = relation_params[key]
    for key, default in optional.items():
        normalized[key] = relation_params.get(key, default)

    if relation_kind == "binned":
        normalized["M_halo"] = _arraylike_to_float_list(
            normalized["M_halo"], name="M_halo")
        normalized["fb"] = _arraylike_to_float_list(normalized["fb"], name="fb")
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
        SO (:obj:`int`): Spherical overdensity. Supported values are 200 and 500.
        relation_kind (:obj:`str`): One of ``power_law``, ``cosmo_power_law``,
            ``double_power_law`` or ``binned``.
        k_min_hmpc (:obj:`float`): Minimum internal SP(k) grid scale in
            :math:`h\\,{\\rm Mpc}^{-1}`.
        k_max_hmpc (:obj:`float`): Maximum internal SP(k) grid scale in
            :math:`h\\,{\\rm Mpc}^{-1}`.
        n_k (:obj:`int`): Number of logarithmic points in the internal SP(k) grid.
        out_of_bounds_policy (:obj:`str`): Behavior for requests above
            ``k_max_hmpc``. Supported values are ``error``, ``unity`` and ``nan``.
        **relation_params: Parameters required by ``relation_kind`` and passed
            to the cached ``pyspk`` evaluator.
    """
    name = "SPK"
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
                 k_min_hmpc=0.1, k_max_hmpc=8.0, n_k=128,
                 out_of_bounds_policy="error", **relation_params):
        self.SO = SO
        self.relation_kind = relation_kind
        self.k_min_hmpc = float(k_min_hmpc)
        self.k_max_hmpc = float(k_max_hmpc)
        self.n_k = int(n_k)
        self.out_of_bounds_policy = out_of_bounds_policy
        self._evaluator = None
        self._pyspk = None
        self._forwarded_warning_messages = set()

        self._validate_settings()
        self.relation_params = _normalize_relation_parameters(
            self.relation_kind, relation_params)
        self._import_pyspk()

    def _validate_settings(self):
        if self.SO not in (200, 500):
            raise ValueError("`SO` must be either 200 or 500.")
        if self.relation_kind not in _SUPPORTED_RELATION_KINDS:
            raise ValueError(
                f"`relation_kind` must be one of {_SUPPORTED_RELATION_KINDS}.")
        if self.k_min_hmpc <= 0 or self.k_max_hmpc <= 0:
            raise ValueError("`k_min_hmpc` and `k_max_hmpc` must be > 0.")
        if self.k_min_hmpc >= self.k_max_hmpc:
            raise ValueError("`k_min_hmpc` must be strictly smaller than `k_max_hmpc`.")
        if self.n_k < 2:
            raise ValueError("`n_k` must be >= 2.")
        if self.out_of_bounds_policy not in _SUPPORTED_OUT_OF_BOUNDS_POLICIES:
            raise ValueError(
                "`out_of_bounds_policy` must be one of "
                f"{_SUPPORTED_OUT_OF_BOUNDS_POLICIES}."
            )

    def _import_pyspk(self):
        if self._pyspk is None:
            try:
                self._pyspk = importlib.import_module("pyspk")
            except ModuleNotFoundError as err:
                raise ModuleNotFoundError(
                    "BaryonsSPK requires the optional dependency `pyspk>=2.0.0`. "
                    "Install it in your environment to use this model."
                ) from err
        return self._pyspk

    def _build_evaluator(self):
        pyspk = self._import_pyspk()
        self._evaluator = pyspk.build_sup_model_evaluator(
            SO=self.SO,
            relation_kind=self.relation_kind,
            k_min=self.k_min_hmpc,
            k_max=self.k_max_hmpc,
            n=self.n_k,
        )

    def _get_evaluator(self):
        if self._evaluator is None:
            self._build_evaluator()
        return self._evaluator

    @staticmethod
    def _make_efunc(cosmo):
        return lambda z: cosmo.h_over_h0(1.0 / (1.0 + z))

    def _forward_pyspk_warnings(self, caught_warnings):
        for caught in caught_warnings:
            msg = str(caught.message)
            if msg in self._forwarded_warning_messages:
                continue
            self._forwarded_warning_messages.add(msg)
            warnings.warn(
                msg,
                category=CCLWarning,
                importance="low",
                stacklevel=3,
            )

    def _evaluate_on_internal_grid(self, cosmo, z):
        evaluator = self._get_evaluator()
        kwargs = dict(self.relation_params)
        if self.relation_kind in ("cosmo_power_law", "double_power_law"):
            kwargs["efunc"] = self._make_efunc(cosmo)

        with warnings_builtin.catch_warnings(record=True) as caught:
            warnings_builtin.simplefilter("always")
            k_hmpc, sup = evaluator(z=float(z), **kwargs)
        self._forward_pyspk_warnings(caught)
        return np.asarray(k_hmpc), np.asarray(sup)

    def _apply_out_of_bounds_policy(self, k_hmpc, fka):
        out_hi = k_hmpc > self.k_max_hmpc
        if not np.any(out_hi):
            return fka

        if self.out_of_bounds_policy == "error":
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

    def boost_factor(self, cosmo, k, a):
        """The SP(k) boost factor for baryons.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
            k (:obj:`float` or `array`): Wavenumber in :math:`{\\rm Mpc}^{-1}`.
            a (:obj:`float` or `array`): Scale factor.

        Returns:
            :obj:`float` or `array`: Correction factor to apply to the power
            spectrum.
        """  # noqa
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
        """Update SP(k) model configuration.

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

        new_kind = self.relation_kind if relation_kind is None else relation_kind
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

    def _include_baryonic_effects(self, cosmo, pk):
        # Applies boost factor
        a_arr, lk_arr, pk_arr = pk.get_spline_arrays()
        k_arr = np.exp(lk_arr)
        fka = self.boost_factor(cosmo, k_arr, a_arr)
        pk_arr *= fka

        if pk.psp.is_log:
            np.log(pk_arr, out=pk_arr)  # in-place log

        return Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr,
                    is_logp=pk.psp.is_log,
                    extrap_order_lok=pk.extrap_order_lok,
                    extrap_order_hik=pk.extrap_order_hik)
