"""Fedeli14 baryon halo model."""

__all__ = ("BaryonsFedeli14",)

from typing import Any, Mapping
import numpy as np
from numpy.typing import NDArray

from .. import Pk2D
from . import Baryons

from ..baryons.fedeli14_bhm.baryon_halo_model import BaryonHaloModel

FloatArray = NDArray[np.floating[Any]]
ComponentsDict = dict[str, Any]


class BaryonsFedeli14(Baryons):
    r"""Fedeli14 baryon halo model implemented as a multiplicative boost.

    This model modifies an input matter power spectrum by multiplying it by a
    scale- and redshift-dependent boost computed from the Fedeli (2014) baryon
    halo model (BHM) <https://arxiv.org/abs/1401.2997>`.

    The operation applied is:

    .. math::
        P_{\rm out}(k,a) = P_{\rm in}(k,a)\,f_{\rm fedeli}(k,a).

    Notes:
        - CCL ``Pk2D`` uses ``k`` in units of 1/Mpc, and this implementation
          keeps ``k`` in 1/Mpc throughout (no k -> k/h conversion).
        - If ``renormalize_large_scales=True``, the boost is rescaled (per
          scale factor) so that its mean over ``k <= k_renorm_max`` is unity.
        - For ``a < a_min``, the boost is forced to unity. The default
          ``a_min=0.1`` avoids extrapolating the model to very early times.

    See Also:
        BaryonHaloModel: Lower-level wrapper that computes spectra and boosts.
    """
    name = "Fedeli14"

    __repr_attrs__ = __eq_attrs__ = (
        # diffuse mixing (physics)
        "Fg", "bd",
        # mass fractions (physics)
        "rho_star", "m0_star", "sigma_star", "mmin_star", "mmax_star",
        "m0_gas", "sigma_gas",
        # gas profile (physics)
        "gas_beta", "gas_r_co", "gas_r_ej",
        # stellar profile (physics)
        "star_x_delta", "star_alpha",
        # optional DM knob (physics)
        "concentration",
        # reference used inside BHM.boost
        "pk_ref",
        # numerics/plumbing
        "mass_def", "mass_function", "halo_bias",
        "mass_ranges", "interpolation_grid",
        "update_fftlog_precision", "fftlog_kwargs", "rgi_kwargs",
        "n_m", "density_mmin", "density_mmax",
        "a_min",
    )

    def __init__(
        self,
        *,
        Fg: float = 0.9,
        bd: float = 0.6,
        rho_star: float | None = None,
        m0_star: float = 5.0e12,
        sigma_star: float = 1.2,
        mmin_star: float = 1.0e10,
        mmax_star: float = 1.0e15,
        m0_gas: float = 1.0e12,
        sigma_gas: float = 3.0,
        gas_beta: float = 2.9,
        gas_r_co: float = 0.1,
        gas_r_ej: float = 4.5,
        star_x_delta: float = 1.0 / 0.03,
        star_alpha: float = 1.0,
        concentration: Any = None,
        pk_ref: str = "pk_dmo",
        mass_def: Any = None,
        mass_function: Any = None,
        halo_bias: Any = None,
        mass_ranges: Any = None,
        interpolation_grid: Any = None,
        update_fftlog_precision: bool = True,
        fftlog_kwargs: Mapping[str, Any] | None = None,
        rgi_kwargs: Mapping[str, Any] | None = None,
        n_m: int = 512,
        density_mmin: float = 1e6,
        density_mmax: float = 1e16,
        renormalize_large_scales: bool = True,
        k_renorm_max: float = 1e-2,
        a_min: float = 0.1,
    ) -> None:
        """Initialize Fedeli14 baryon halo model."""
        self.Fg = float(Fg)
        self.bd = float(bd)

        self.rho_star = rho_star
        self.m0_star = float(m0_star)
        self.sigma_star = float(sigma_star)
        self.mmin_star = float(mmin_star)
        self.mmax_star = float(mmax_star)
        self.m0_gas = float(m0_gas)
        self.sigma_gas = float(sigma_gas)

        self.gas_beta = float(gas_beta)
        self.gas_r_co = float(gas_r_co)
        self.gas_r_ej = float(gas_r_ej)

        self.star_x_delta = float(star_x_delta)
        self.star_alpha = float(star_alpha)

        self.concentration = concentration
        self.pk_ref = str(pk_ref)

        self.mass_def = mass_def
        self.mass_function = mass_function
        self.halo_bias = halo_bias
        self.mass_ranges = mass_ranges
        self.interpolation_grid = interpolation_grid
        self.update_fftlog_precision = bool(update_fftlog_precision)
        self.fftlog_kwargs = dict(fftlog_kwargs or {})
        self.rgi_kwargs = dict(rgi_kwargs or {})
        self.n_m = int(n_m)
        self.density_mmin = float(density_mmin)
        self.density_mmax = float(density_mmax)

        self.renormalize_large_scales = bool(renormalize_large_scales)
        self.k_renorm_max = float(k_renorm_max)
        self.a_min = float(a_min)

    def update_parameters(self, **kwargs: Any) -> None:
        """Update model parameters on this instance.

        This is a convenience method for interactive work and testing.
        Any keyword matching an existing attribute on this class will be
        updated, with common numeric parameters cast to ``float`` or
        ``int`` as appropriate.

        Passing ``None`` for a parameter leaves the current value unchanged.

        Args:
            **kwargs: Parameter names and values to update.

        Raises:
            AttributeError: If a provided parameter name does not exist.
        """
        for key, val in kwargs.items():
            if val is None:
                continue
            # cast common numeric params
            if key in {
                "Fg", "bd", "m0_star", "sigma_star", "mmin_star", "mmax_star",
                "m0_gas", "sigma_gas", "gas_beta", "gas_r_co", "gas_r_ej",
                "star_x_delta", "star_alpha", "density_mmin", "density_mmax",
                "a_min",
            }:
                val = float(val)
            elif key == "n_m":
                val = int(val)
            elif key in {"update_fftlog_precision",
                         "renormalize_large_scales"}:
                val = bool(val)
            elif key == "pk_ref":
                val = str(val)

            if not hasattr(self, key):
                raise AttributeError(f"Unknown parameter {key!r}")
            setattr(self, key, val)

    def _build_bhm(self, cosmo: Any) -> BaryonHaloModel:
        """Build a ``BaryonHaloModel`` configured with the current parameters.

        This creates the internal Fedeli14 baryon halo model wrapper used to
        compute boosts and related spectra. The returned object is
        configured using the physics and numerical settings stored on this
        instance.

        Args:
            cosmo: Cosmology object passed through to the underlying model.

        Returns:
            A configured ``BaryonHaloModel`` instance.
        """
        return BaryonHaloModel(
            cosmo=cosmo,
            # plumbing
            mass_def=self.mass_def,
            mass_function=self.mass_function,
            halo_bias=self.halo_bias,
            concentration=self.concentration,
            # mass fractions
            rho_star=self.rho_star,
            m0_star=self.m0_star,
            sigma_star=self.sigma_star,
            mmin_star=self.mmin_star,
            mmax_star=self.mmax_star,
            m0_gas=self.m0_gas,
            sigma_gas=self.sigma_gas,
            # profiles
            gas_beta=self.gas_beta,
            gas_r_co=self.gas_r_co,
            gas_r_ej=self.gas_r_ej,
            star_x_delta=self.star_x_delta,
            star_alpha=self.star_alpha,
            # diffuse mixing
            Fg=self.Fg,
            bd=self.bd,
            # numerics
            mass_ranges=self.mass_ranges,
            interpolation_grid=self.interpolation_grid,
            update_fftlog_precision=self.update_fftlog_precision,
            fftlog_kwargs=self.fftlog_kwargs,
            rgi_kwargs=self.rgi_kwargs,
            n_m=self.n_m,
            density_mmin=self.density_mmin,
            density_mmax=self.density_mmax,
        )

    def boost_factor(
        self,
        cosmo: Any,
        k: float | FloatArray,
        a: float | FloatArray
    ) -> float | FloatArray:
        r"""Evaluate the Fedeli14 boost factor ``f(k,a)``.

        The boost is defined as a ratio of the baryonic halo-model matter
        spectrum to a chosen reference spectrum:

        .. math::
            f(k,a) = \frac{P_{\rm bar}^{\rm HM}(k,a)}{P_{\rm ref}(k,a)}.

        The reference is controlled by ``self.pk_ref`` (e.g. "pk_dmo",
        "pk_nlin", "pk_lin") and is interpreted by ``BaryonHaloModel.boost``.

        Input arrays are broadcast to a 2D grid with shape (na, nk), and the
        return value follows the input scalar/array structure:

        - if both ``a`` and ``k`` are arrays: returns (na, nk)
        - if ``a`` is scalar and ``k`` is array: returns (nk,)
        - if ``a`` is array and ``k`` is scalar: returns (na,)
        - if both are scalars: returns scalar

        Notes:
            - ``k`` is interpreted in units of 1/Mpc and is not rescaled by
              ``h``.
            - If ``renormalize_large_scales=True``, the boost is normalized so
              that its mean over ``k <= k_renorm_max`` is unity for each
              scale factor.

        Args:
            cosmo: CCL Cosmology object.
            k: Wavenumber(s) in units of 1/Mpc.
            a: Scale factor(s).

        Returns:
            Boost factor evaluated on the broadcasted (a, k) grid.
        """
        a_use, k_use = map(np.atleast_1d, [a, k])
        a_use, k_use = a_use[:, None], k_use[None, :]  # (na,1), (1,nk)

        # IMPORTANT: we do NOT convert k -> k/h here.
        # We keep k in 1/Mpc everywhere so the pk_ref division uses the same k
        # units.
        k_1d = k_use.ravel()  # 1D array, 1/Mpc

        bhm = self._build_bhm(cosmo)

        out = np.empty((a_use.shape[0], k_use.shape[1]), dtype=float)
        for i, aval in enumerate(a_use[:, 0]):
            out[i, :] = bhm.boost(k=k_1d, a=float(aval), pk_ref=self.pk_ref)

        # We need to enforce f(k->0)=1 by renormalizing on large scales.
        # Otherwise, our boost will not behave at large scales.
        if self.renormalize_large_scales:
            k0 = self.k_renorm_max
            m = k_1d <= k0
            if np.any(m):
                # Rescale the boost per-a so that its mean over a low-k band is
                # unity. This is a pragmatic large-scale renormalization of
                # the *ratio* to pk_ref: it removes small low-k offsets from
                # numerics/pk_ref choice. It does not enforce B(k->0)=1
                # pointwise, only that the average over k<=k_renorm_max is 1.
                norm = np.mean(out[:, m], axis=1)
                norm = np.where(np.isfinite(norm) & (norm > 0.0), norm, 1.0)

                out = out / norm[:, None]
                out = np.where(np.isfinite(out) & (out > 0.0), out, 1.0)
                out = np.clip(out, 1e-6, 1e6)

        if np.ndim(k) == 0:
            out = np.squeeze(out, axis=-1)
        if np.ndim(a) == 0:
            out = np.squeeze(out, axis=0)
        return out

    def _include_baryonic_effects(
        self,
        cosmo: Any,
        pk: Pk2D,
    ) -> Pk2D:
        r"""Return a new ``Pk2D`` with Fedeli14 baryonic effects applied.

        The baryonic correction is multiplicative in the physical spectrum,

        .. math::
            P_{\rm out}(k,a) = P_{\rm in}(k,a)\,f(k,a).

        If the output ``Pk2D`` is stored in log-space, this method converts the
        physical input spectrum to ``log(P)`` before adding ``log(f)``."""
        a_arr, lk_arr, pk_arr = pk.get_spline_arrays()
        pk_arr = pk_arr.copy()
        k_arr = np.exp(lk_arr)

        bhm = self._build_bhm(cosmo)
        k_model = bhm.interpolation_grid["dark_matter"]["k"]
        kmin_model = k_model.min()
        kmax_model = k_model.max()

        rtol = 1e-12
        fka = np.ones((a_arr.size, k_arr.size))

        amask = a_arr >= self.a_min
        kmask = (
            (k_arr >= kmin_model * (1.0 - rtol))
            & (k_arr <= kmax_model * (1.0 + rtol))
        )

        if np.any(amask) and np.any(kmask):
            fka[np.ix_(amask, kmask)] = self.boost_factor(
                cosmo, k_arr[kmask], a_arr[amask]
            )

        pk_arr *= fka

        if pk.psp.is_log:
            np.log(pk_arr, out=pk_arr)

        return Pk2D(
            a_arr=a_arr,
            lk_arr=lk_arr,
            pk_arr=pk_arr,
            is_logp=pk.psp.is_log,
            extrap_order_lok=pk.extrap_order_lok,
            extrap_order_hik=pk.extrap_order_hik,
        )
