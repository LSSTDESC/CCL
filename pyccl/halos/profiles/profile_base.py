from __future__ import annotations

__all__ = ("HaloProfile", "HaloProfileNumberCounts", "HaloProfileMatter",
           "HaloProfilePressure", "HaloProfileCIB",)

import warnings
from numbers import Real
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ... import CCLObject, FFTLogParams
from ... import CCLDeprecationWarning, deprecate_attr, warn_api, mass_def_api
from ... import physical_constants as const
from ...pyutils import resample_array, _fftlog_transform
from .. import Concentration, MassDef

if TYPE_CHECKING:
    from ... import Cosmology
    from .. import HMCalculator


class HaloProfile(CCLObject):
    r"""Abstract base class for halo profiles.

    Contains methods to compute halo profiles in 3-D real and Fourier space, as
    well as the 2-D projected and the cumulative mean surface density. Hankel
    transforms between real- and Fourier-space are performed with `FFTLog
    <https://jila.colorado.edu/~ajsh/FFTLog/>`_. Subclasses may contain
    analytic implementations of any of those methods to bypass the FFTLog
    computation. At the very least, subclasses must contain :meth:`_real` or
    :meth:`_fourier` to enable calculations with the implemented halo profile.

    Parameters
    ----------
    mass_def
        Mass definition.
    concentration
        Mass-concentration relation, used to calculate the scale radius in
        some halo profiles.

    Raises
    ------
    TypeError
        Trying to instantiate without implementing :meth:`_real` or
        :meth:`_fourier`.
    ValueError
        If concentration and mass definition are both specified but with
        different mass definitions.
    """
    __getattr__ = deprecate_attr(pairs=[('cM', 'concentration')]
                                 )(super.__getattribute__)
    mass_def: MassDef
    concentration: Union[Concentration, None]
    precision_fftlog: FFTLogParams
    "FFTLog accuracy parameters."

    def __init__(
            self,
            *,  # TODO: Move mass_def to the beginning of the docs in CCLv3.
            mass_def: Optional[str, MassDef] = None,
            concentration: Optional[str, Concentration] = None
    ):
        # Verify that profile can be initialized.
        if not (hasattr(self, "_real") or hasattr(self, "_fourier")):
            name = type(self).__name__
            raise TypeError(f"Can't instantiate {name} with no "
                            "_real or _fourier implementation.")

        # Initialize FFTLog.
        self.precision_fftlog = FFTLogParams()

        # TODO: Remove for CCLv3.
        self._is_number_counts = isinstance(self, HaloProfileNumberCounts)

        if (mass_def, concentration) == (None, None):
            warnings.warn(
                "mass_def (or concentration where applicable) will become a "
                "required argument for HaloProfile instantiation in CCLv3 "
                "and will be moved from (real, fourier, projected, cumul2d "
                "convergence, shear, reduced_shear, magnification).",
                CCLDeprecationWarning)
            self.mass_def = self.concentration = None
            return

        # Initialize mass_def and concentration.
        self.mass_def, *out = MassDef.from_specs(
            mass_def, concentration=concentration)
        if out:
            self.concentration = out[0]

    @property
    def is_number_counts(self) -> bool:
        # TODO: Remove for CCLv3.
        return self._is_number_counts

    @is_number_counts.setter
    def is_number_counts(self, value):
        # TODO: Remove for CCLv3.
        with self.unlock():
            self._is_number_counts = value

    def get_normalization(
            self,
            cosmo: Optional[Cosmology] = None,
            a: Optional[Real] = None,
            *,
            hmc: Optional[HMCalculator] = None
    ) -> float:
        r"""Compute the normalization of the halo profile.

        Some profiles may be normalized by an overall time- and cosmology-\
        dependent function. It is often computed by integrating certain halo
        properties over mass.

        .. math::

            \rho(k|M) = \frac{f(k|M)}
            {\int {\rm d}M \, n(M) \, g(k \rightarrow 0 | M)},

        where :math:`f(k)` is the halo profile, :math:`n(M)` is the halo mass
        function, and :math:`g(k)` is implemented with this method.

        Example
        -------
        Get the normalized profile in real-space, at `a = 1` (today).

        .. code-block:: python

            import pyccl as ccl

            cosmo = ccl.CosmologyVanillaLCDM()
            hmc = ccl.halos.HMCalculator(mass_function="Tinker10",
                                         halo_bias="Tinker10",
                                         mass_def="200c")
            prof = MyHaloProfile(...)

            rho_unnorm = prof.real(cosmo, k=0.1, M=1e13, a=1)
            rho_norm = rho_unnorm / prof.get_normalization(cosmo, a=1, hmc=hmc)

        Note that the unnormalized profile has to be divided by this function.

        Arguments
        ---------
        hmc
            Halo model workspace.
        cosmo
            Cosmological parameters.
        a
            Scale factor.

        Returns
        -------

            Normalization.
        """
        def integ(M):
            return self.fourier(cosmo, hmc.precision["k_min"], M, a)
        return hmc.integrate_over_massfunc(integ, cosmo, a)
        # TODO: CCLv3 replace by the below in v3 (profiles will all have a
        # default normalization of 1. Normalization will always be applied).
        # return 1.0

    def update_precision_fftlog(self, **kwargs) -> None:
        r"""Update the precision of FFTLog for the Hankel transforms.

        All parameters in :attr:`~precision_fftlog` can be updated.
        """
        self.precision_fftlog.update_parameters(**kwargs)

    def _get_plaw_fourier(self, cosmo: Cosmology, a: Real) -> float:
        r"""Obtain fine control over `plaw_fourier`. See :class:`~FFTLogParams`
        for details.

        :meta public:

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        a
            Scale factor.

        Returns
        -------

            Power law index to use with FFTLog.
        """
        return self.precision_fftlog['plaw_fourier']

    def _get_plaw_projected(self, cosmo: Cosmology, a: Real) -> float:
        r"""Obtain fine control over `plaw_projected`. See
        :class:`~FFTLogParams` for details.

        :meta public:

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        a
            Scale factor.

        Returns
        -------

            Power law index to use with FFTLog.
        """
        return self.precision_fftlog['plaw_projected']

    @mass_def_api
    def real(
            self,
            cosmo: Cosmology,
            r: Union[Real, NDArray[Real]],
            M: Union[Real, NDArray[Real]],
            a: Real
    ) -> Union[float, NDArray[float]]:
        r"""Compute the 3-D real-space profile.

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        r : array_like (nr,)
            Comoving radius in :math:`\rm Mpc`.
        M : array_like (nM,)
            Halo mass in :math:`\rm M_{\odot}`.
        a
            Scale factor.
        mass_def : MassDef
            Mass definition of `M`.

            .. deprecated:: 2.8.0

                Pass `mass_def` to the constructor.

        Returns
        -------
        array_like (nM, nr)
            3-D real-space halo profile.
        """
        if getattr(self, "_real", None):
            return self._real(cosmo, r, M, a)
        return self._fftlog_wrap(cosmo, r, M, a, fourier_out=False)

    _real: Callable
    """Implementation of :meth:`~real`. Methods share common signature.

    :meta public:
    """

    @mass_def_api
    def fourier(
            self,
            cosmo: Cosmology,
            k: Union[Real, NDArray[Real]],
            M: Union[Real, NDArray[Real]],
            a: Real
    ) -> Union[float, NDArray[float]]:
        r"""Compute the 3-D Forier-space profile.

        .. math::

           \rho(k) = \frac{1}{2\pi^2} \int {\rm d}r \, r^2 \,
           \rho(r) \, j_0(k r)

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        k : array_like (nk,)
            Comoving wavenumber in :math:`\rm Mpc^{-1}`.
        M : array_like (nM,)
            Halo mass in :math:`\rm M_{\odot}`.
        a
            Scale factor.
        mass_def : MassDef
            Mass definition of `M`.

            .. deprecated:: 2.8.0

                Pass `mass_def` to the constructor.

        Returns
        -------
        array_like (nM, nk)
            3-D Fourier-space halo profile.
        """
        if getattr(self, "_fourier", None):
            return self._fourier(cosmo, k, M, a)
        return self._fftlog_wrap(cosmo, k, M, a, fourier_out=True)

    _fourier: Callable
    """Implementation of :meth:`~fourier`. Methods share common signature.

    :meta public:
    """

    @mass_def_api
    def projected(
            self,
            cosmo: Cosmology,
            r_t: Union[Real, NDArray[Real]],
            M: Union[Real, NDArray[Real]],
            a: Real
    ) -> Union[float, NDArray[float]]:
        r"""Compute the 2-D projected profile.

        .. math::

           \Sigma(R)= \int dr_{\parallel} \, \rho(\sqrt{r_{\parallel}^2 + R^2})

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        r_t : array_like (nr,)
            Transverse comoving radius in :math:`\rm Mpc`.
        M : array_like (nM,)
            Halo mass in :math:`\rm M_{\odot}`.
        a
            Scale factor.
        mass_def : MassDef
            Mass definition of `M`.

            .. deprecated:: 2.8.0

                Pass `mass_def` to the constructor.

        Returns
        -------
        array_like (nM, nr)
            2-D projected profile.
        """
        if getattr(self, "_projected", None):
            return self._projected(cosmo, r_t, M, a)
        return self._projected_fftlog_wrap(cosmo, r_t, M, a, is_cumul2d=False)

    _projected: Callable
    """Implementation of :meth:`~projected`. Methods share common signature.

    :meta public:
    """

    @mass_def_api
    def cumul2d(
            self,
            cosmo: Cosmology,
            r_t: Union[Real, NDArray[Real]],
            M: Union[Real, NDArray[Real]],
            a: Real
    ) -> Union[float, NDArray[float]]:
        r"""Compute the 2-D cumulative surface mass density.

        .. math::

           \Sigma(<R)= \frac{2}{R^2} \int {\rm d}R' \, R' \, \Sigma(R')

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        r_t : array_like (nr,)
            Transverse comoving radius in :math:`\rm Mpc`.
        M : array_like (nM,)
            Halo mass in :math:`\rm M_{\odot}`.
        a
            Scale factor.
        mass_def : MassDef
            Mass definition of `M`.

            .. deprecated:: 2.8.0

                Pass `mass_def` to the constructor.

        Returns
        -------
        array_like (nM, nr)
            2-D cumulative surface mass density.
        """
        if getattr(self, "_cumul2d", None):
            return self._cumul2d(cosmo, r_t, M, a)
        return self._projected_fftlog_wrap(cosmo, r_t, M, a, is_cumul2d=True)

    _cumul2d: Callable
    """Implementation of :meth:`~cumul2d`. Methods share common signature.

    :meta public:
    """

    @mass_def_api
    @warn_api
    def convergence(
            self,
            cosmo: Cosmology,
            r: Union[Real, NDArray[Real]],
            M: Union[Real, NDArray[Real]],
            *,
            a_lens: Real,
            a_source: Union[Real, NDArray[Real]]
    ) -> Union[float, NDArray[float]]:
        r"""Compute the convergence.

        .. math::

           \kappa(R) = \frac{\Sigma(R)}{\Sigma_{\rm c}},

        where :math:`\Sigma(R)` is the 2-D projected surface mass density.

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        r : array_like (nr,)
            Comoving radius in :math:`\rm Mpc`.
        M : array_like (nM,)
            Halo mass in :math:`\rm M_{\odot}`.
        a_lens
            Scale factor of lens.
        a_source : array_like (nr,)
            Scale factor of source.

            .. note::

                If `a_source` is a sequence, its shape must match that of `r`.

        mass_def : MassDef
            Mass definition of `M`.

            .. deprecated:: 2.8.0

                Pass `mass_def` to the constructor.

        Returns
        -------
        array_like (nM, nr)
            Convergence.
        """
        Sigma = self.projected(cosmo, r, M, a_lens)
        Sigma /= a_lens**2
        Sigma_crit = cosmo.sigma_critical(a_lens=a_lens, a_source=a_source)
        return Sigma / Sigma_crit

    @mass_def_api
    @warn_api
    def shear(
            self,
            cosmo: Cosmology,
            r: Union[Real, NDArray[Real]],
            M: Union[Real, NDArray[Real]],
            *,
            a_lens: Real,
            a_source: Union[Real, NDArray[Real]]
    ) -> Union[float, NDArray[float]]:
        r"""Compute the tangential shear.

        .. math::

           \gamma(R) = \frac{\Delta \Sigma(R)}{\Sigma_{\rm c}} =
           \frac{\bar{\Sigma}(< R) - \Sigma(R)}{\Sigma_{\rm c}},

        where :math:`\bar{\Sigma}(< R)` is the average surface density within
        :math:`R`.

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        r : array_like (nr,)
            Comoving radius in :math:`\rm Mpc`.
        M : array_like (nM,)
            Halo mass in :math:`\rm M_{\odot}`.
        a_lens
            Scale factor of lens.
        a_source : array_like (nr,)
            Scale factor of source.

            .. note::

                If `a_source` is a sequence, its shape must match that of `r`.

        mass_def : MassDef
            Mass definition of `M`.

            .. deprecated:: 2.8.0

                Pass `mass_def` to the constructor.

        Returns
        -------
        array_like (nM, nr)
            Tangential shear.
        """
        Sigma = self.projected(cosmo, r, M, a_lens)
        Sigma_bar = self.cumul2d(cosmo, r, M, a_lens)
        Sigma_crit = cosmo.sigma_critical(a_lens=a_lens, a_source=a_source)
        return (Sigma_bar - Sigma) / (Sigma_crit * a_lens**2)

    @mass_def_api
    @warn_api
    def reduced_shear(
            self,
            cosmo: Cosmology,
            r: Union[Real, NDArray[Real]],
            M: Union[Real, NDArray[Real]],
            *,
            a_lens: Real,
            a_source: Union[Real, NDArray[Real]]
    ) -> Union[float, NDArray[float]]:
        r"""Compute the reduced shear.

        .. math::

           g_t (R) = \frac{\gamma(R)}{(1 - \kappa(R))},

        where :math:`\gamma(R)` is the shear and `\kappa(R)` is the
        convergence.

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        r : array_like (nr,)
            Comoving radius in :math:`\rm Mpc`.
        M : array_like (nM,)
            Halo mass in :math:`\rm M_{\odot}`.
        a_lens
            Scale factor of lens.
        a_source : array_like (nr,)
            Scale factor of source.

            .. note::

                If `a_source` is a sequence, its shape must match that of `r`.

        mass_def : MassDef
            Mass definition of `M`.

            .. deprecated:: 2.8.0

                Pass `mass_def` to the constructor.

        Returns
        -------
        array_like (nM, nr)
            Reduced shear.
        """
        convergence = self.convergence(cosmo, r, M, a_lens=a_lens,
                                       a_source=a_source)
        shear = self.shear(cosmo, r, M, a_lens=a_lens, a_source=a_source)
        return shear / (1.0 - convergence)

    @mass_def_api
    @warn_api
    def magnification(
            self,
            cosmo: Cosmology,
            r: Union[Real, NDArray[Real]],
            M: Union[Real, NDArray[Real]],
            *,
            a_lens: Real,
            a_source: Union[Real, NDArray[Real]]
    ) -> Union[float, NDArray[float]]:
        r"""Compute the magnification.

        .. math::

           \mu (R) = \frac{1}{\left[(1 - \kappa(R))^2 -
           \vert \gamma(R) \vert^2 \right]},

        where :math:`\gamma(R)` is the shear and :math:`\kappa(R)` is the
        convergence.

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        r : array_like (nr,)
            Comoving radius in :math:`\rm Mpc`.
        M : array_like (nM,)
            Halo mass in :math:`\rm M_{\odot}`.
        a_lens
            Scale factor of lens.
        a_source : array_like (nr,)
            Scale factor of source.

            .. note::

                If `a_source` is a sequence, its shape must match that of `r`.

        mass_def : MassDef
            Mass definition of `M`.

            .. deprecated:: 2.8.0

                Pass `mass_def` to the constructor.

        Returns
        -------
        array_like (nM, nr)
            Magnification.
        """
        convergence = self.convergence(cosmo, r, M, a_lens=a_lens,
                                       a_source=a_source)
        shear = self.shear(cosmo, r, M, a_lens=a_lens, a_source=a_source)

        return 1.0 / ((1.0 - convergence)**2 - np.abs(shear)**2)

    def _fftlog_wrap(self, cosmo, k, M, a,
                     fourier_out=False,
                     large_padding=True):
        # This computes the 3D Hankel transform
        #  ρ(k) = 4π ∫ dr r^2 ρ(r) j_0(k r), if fourier_out is False;
        #  ρ(r) = 1/(2π^2) ∫ dk k^2 ρ(k) j_0(k r), if fourier_out is True.

        # Select which profile should be the input
        p_func = self._real if fourier_out else self._fourier

        k_use = np.atleast_1d(k)
        M_use = np.atleast_1d(M)
        lk_use = np.log(k_use)
        nM = len(M_use)

        # k/r ranges to be used with FFTLog and its sampling.
        if large_padding:
            k_min = self.precision_fftlog['padding_lo_fftlog'] * np.amin(k_use)
            k_max = self.precision_fftlog['padding_hi_fftlog'] * np.amax(k_use)
        else:
            k_min = self.precision_fftlog['padding_lo_extra'] * np.amin(k_use)
            k_max = self.precision_fftlog['padding_hi_extra'] * np.amax(k_use)
        n_k = int(np.ceil(np.log10(k_max / k_min)) *
                  self.precision_fftlog['n_per_decade'])
        r_arr = np.geomspace(k_min, k_max, n_k)

        p_k_out = np.zeros([nM, k_use.size])
        # Compute real profile values
        p_real_M = p_func(cosmo, r_arr, M_use, a)
        # Power-law index to pass to FFTLog.
        plaw_index = self._get_plaw_fourier(cosmo, a)

        # Compute Fourier profile through fftlog
        k_arr, p_fourier_M = _fftlog_transform(r_arr, p_real_M,
                                               3, 0, plaw_index)
        lk_arr = np.log(k_arr)

        for im, p_k_arr in enumerate(p_fourier_M):
            # Resample into input k values
            p_fourier = resample_array(lk_arr, p_k_arr, lk_use,
                                       self.precision_fftlog['extrapol'],
                                       self.precision_fftlog['extrapol'],
                                       0, 0)
            p_k_out[im, :] = p_fourier
        if fourier_out:
            p_k_out *= (2 * np.pi)**3

        if np.ndim(k) == 0:
            p_k_out = np.squeeze(p_k_out, axis=-1)
        if np.ndim(M) == 0:
            p_k_out = np.squeeze(p_k_out, axis=0)
        return p_k_out

    def _projected_fftlog_wrap(self, cosmo, r_t, M, a, is_cumul2d=False):
        # This computes Σ(R) from the Fourier-space profile as:
        # Σ(R) = 1/(2π) ∫ dk k J_0(k R) ρ(k)
        r_t_use = np.atleast_1d(r_t)
        M_use = np.atleast_1d(M)
        lr_t_use = np.log(r_t_use)
        nM = len(M_use)

        # k/r range to be used with FFTLog and its sampling.
        r_t_min = self.precision_fftlog['padding_lo_fftlog'] * np.amin(r_t_use)
        r_t_max = self.precision_fftlog['padding_hi_fftlog'] * np.amax(r_t_use)
        n_r_t = (int(np.log10(r_t_max / r_t_min)) *
                 self.precision_fftlog['n_per_decade'])
        k_arr = np.geomspace(r_t_min, r_t_max, n_r_t)

        sig_r_t_out = np.zeros([nM, r_t_use.size])
        # Compute Fourier-space profile
        if getattr(self, "_fourier", None):
            # Compute from `_fourier` if available.
            p_fourier = self._fourier(cosmo, k_arr, M_use, a)
        else:
            # Compute with FFTLog otherwise.
            lpad = self.precision_fftlog['large_padding_2D']
            p_fourier = self._fftlog_wrap(cosmo, k_arr, M_use, a,
                                          fourier_out=True, large_padding=lpad)
        if is_cumul2d:
            # The cumulative profile involves a factor 1/(k R) in
            # the integrand.
            p_fourier *= 2 / k_arr[None, :]

        # Power-law index to pass to FFTLog.
        if is_cumul2d:
            i_bessel = 1
            plaw_index = self._get_plaw_projected(cosmo, a) - 1
        else:
            i_bessel = 0
            plaw_index = self._get_plaw_projected(cosmo, a)

        # Compute projected profile through fftlog
        r_t_arr, sig_r_t_M = _fftlog_transform(k_arr, p_fourier,
                                               2, i_bessel,
                                               plaw_index)
        lr_t_arr = np.log(r_t_arr)

        if is_cumul2d:
            sig_r_t_M /= r_t_arr[None, :]
        for im, sig_r_t_arr in enumerate(sig_r_t_M):
            # Resample into input r_t values
            sig_r_t = resample_array(lr_t_arr, sig_r_t_arr,
                                     lr_t_use,
                                     self.precision_fftlog['extrapol'],
                                     self.precision_fftlog['extrapol'],
                                     0, 0)
            sig_r_t_out[im, :] = sig_r_t

        if np.ndim(r_t) == 0:
            sig_r_t_out = np.squeeze(sig_r_t_out, axis=-1)
        if np.ndim(M) == 0:
            sig_r_t_out = np.squeeze(sig_r_t_out, axis=0)
        return sig_r_t_out


class HaloProfileNumberCounts(HaloProfile):
    """Base for number counts halo profiles."""


class HaloProfileMatter(HaloProfile):
    """Base for matter halo profiles."""

    def get_normalization(
            self,
            cosmo: Cosmology,
            a: Optional[Real] = None,
            *,
            hmc: Optional[HMCalculator] = None
    ) -> float:
        """Compute the normalization for matter overdensity profiles, which is
        the comoving matter density today.
        """
        return const.RHO_CRITICAL * cosmo["Omega_m"] * cosmo["h"]**2


class HaloProfilePressure(HaloProfile):
    """Base for pressure halo profiles."""


class HaloProfileCIB(HaloProfile):
    """Base for CIB halo profiles."""
