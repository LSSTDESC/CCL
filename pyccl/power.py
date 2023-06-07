"""
==========================
Power (:mod:`pyccl.power`)
==========================

Functions related to power spectra: evaluations, sigmas, scales.
"""

from __future__ import annotations

__all__ = ("linear_power", "nonlin_power", "linear_matter_power",
           "nonlin_matter_power", "sigmaM", "sigmaR", "sigmaV", "sigma8",
           "kNL",)

from numbers import Real
from typing import TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray

from . import DEFAULT_POWER_SPECTRUM, lib, warn_api

if TYPE_CHECKING:
    from . import Cosmology, Pk2D


@warn_api
def linear_power(
        cosmo: Cosmology,
        k: Union[Real, NDArray[Real]],
        a: Union[Real, NDArray[Real]],
        *,
        p_of_k_a: str = DEFAULT_POWER_SPECTRUM
) -> Union[float, NDArray[float]]:
    r"""The linear power spectrum.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    k : array_like (nk,)
        Wavenumber in :math:`\rm Mpc^{-1}`.
    a : array_like (na,)
        Scale factor.
    p_of_k_a
        Power spectrum name. Should be stored in `cosmo`.

    Returns
    -------
    array_like (na, nk)
        Linear power spectrum, in units of :math:`\rm Mpc^3`.

    See Also
    --------
    :func:`~linear_matter_power` : Evaluate the linear matter power spectrum.
    :meth:`~Pk2D.__call__` : Evaluate any power spectrum.
    """
    return cosmo.get_linear_power(p_of_k_a)(k, a, cosmo)


@warn_api
def nonlin_power(
        cosmo: Cosmology,
        k: Union[Real, NDArray[Real]],
        a: Union[Real, NDArray[Real]],
        *,
        p_of_k_a: str = DEFAULT_POWER_SPECTRUM
) -> Union[float, NDArray[float]]:
    r"""The non-linear power spectrum, :math:`P(k)`:

    .. math::

        \xi(r) = \int \frac{{\rm d^3}k}{(2\pi)^3} \, P(k) \,
        e^{i {\bf k} \dot ({\bf x} - {\bf x'})},

    where :math:`\xi(r)` is the *autocorrelation function*, defined as

    .. math::

        \xi(r) &\equiv \langle \delta({\bf x}) \delta({\bf x'}) \rangle \\
        &= \frac{1}{V} \int {\rm d^3}{\bf x} \, \delta({\bf x})
        \delta({\bf x} - {\bf r}),

    where :math:`\delta` is the matter overdensity

    .. math::

        \delta({\bf x}) \equiv \frac{\rho({\bf x}) - \bar\rho}{\bar\rho},

    and :math:`\bar\rho` is the average density over all space.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    k : array_like (nk,)
        Wavenumber in :math:`\rm Mpc^{-1}`.
    a : array_like (na,)
        Scale factor.
    p_of_k_a
        Power spectrum name. Should be stored in `cosmo`.

    Returns
    -------
    array_like (na, nk)
        Non-inear power spectrum, in units of :math:`\rm Mpc^3`.

    See Also
    --------
    :func:`~pyccl.correlations.correlation`
        Inverse Fourier transform of the power spectrum.
    :func:`~nonlin_matter_power`
        Evaluate the non-linear matter power spectrum.
    :meth:`~Pk2D.__call__`
        Evaluate any power spectrum.
    """
    return cosmo.get_nonlin_power(p_of_k_a)(k, a, cosmo)


def linear_matter_power(
        cosmo: Cosmology,
        k: Union[Real, NDArray[Real]],
        a: Union[Real, NDArray[Real]],
) -> Union[float, NDArray[float]]:
    r"""The linear matter power spectrum.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    k : array_like (nk,)
        Wavenumber in :math:`\rm Mpc^{-1}`.
    a : array_like (na,)
        Scale factor.

    Returns
    -------
    array_like (na, nk)
        Linear matter power spectrum, in units of :math:`\rm Mpc^3`.

    See Also
    --------
    :func:`~linear_power` : Evaluate any linear power spectrum.
    :meth:`~Pk2D.__call__` : Evaluate any power spectrum.
    """
    return cosmo.linear_power(k, a, p_of_k_a=DEFAULT_POWER_SPECTRUM)


def nonlin_matter_power(
        cosmo: Cosmology,
        k: Union[Real, NDArray[Real]],
        a: Union[Real, NDArray[Real]],
) -> Union[float, NDArray[float]]:
    r"""The non-linear matter power spectrum.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    k : array_like (nk,)
        Wavenumber in :math:`\rm Mpc^{-1}`.
    a : array_like (na,)
        Scale factor.

    Returns
    -------
    array_like (na, nk)
        Non-linear matter power spectrum, in units of :math:`\rm Mpc^3`.

    See Also
    --------
    :func:`~nonlin_power` : Evaluate any non-linear power spectrum.
    :meth:`~Pk2D.__call__` : Evaluate any power spectrum.
    """
    return cosmo.nonlin_power(k, a, p_of_k_a=DEFAULT_POWER_SPECTRUM)


def sigmaM(
        cosmo: Cosmology,
        M: Union[Real, NDArray[Real]],
        a: Union[Real, NDArray[Real]]
) -> Union[float, NDArray[float]]:
    r"""Root mean squared variance of the linear power spectrum.

    Defined via :math:`\sigma_R`, using the Lagrangian scale of the halo

    .. math::

        R = \left( \frac{3M}{4\pi \bar\rho_{\rm m}} \right)^{\frac{1}{3}},

    where :math:`\bar\rho_{\rm m}` is the average matter density.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    M : array_like (nM,)
        Halo mass in :math:`\rm M_{\odot}`.
    a : array_like (na,)
        Scale factor.

    Returns
    -------
    array_like (na, nM)
        RMS variance of halo mass.

    See Also
    --------
    :func:`~sigmaR` : RMS variance in top-hat spheres of radius :math:`R`.
    """
    cosmo.compute_sigma()

    logM = np.log10(np.atleast_1d(M))
    status = 0
    sigM, status = lib.sigM_vec(cosmo.cosmo, a, logM,
                                len(logM), status)
    cosmo.check(status)
    if np.ndim(M) == 0:
        sigM = sigM[0]
    return sigM


@warn_api
def sigmaR(
        cosmo: Cosmology,
        R: Union[Real, NDArray[Real]],
        a: Union[Real, NDArray[Real]] = 1,
        *,
        p_of_k_a: Union[str, Pk2D] = DEFAULT_POWER_SPECTRUM
):
    r"""RMS variance in a top-hat sphere of radius `R` :math:`\rm Mpc`,

    .. math::

        \sigma_R^2 = \frac{1}{2\pi^2} \int {\rm d}k \, k^2 \, P_{\rm L}(k)
        \tilde{W}_R^2(k),

    where :math:`P_{\rm L}` is the linear matter power spectrum and
    :math:`\tilde{W}` is the Fourier transform of the spherical top-hat window

    .. math::

        \tilde{W}_R(k) = \frac{3}{(kR)^3} \left( \sin(kR) - kR \cos(kR)\right).

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    R : array_like (nR,)
        Radius of the top-hat sphere.
    a : array_like (na,)
        Scale factor.
    p_of_k_a
        3-D power spectrum to integrate.
        If a string is passed, a non-linear power spectrum under that name
        must be stored in `cosmo`.

    Returns
    -------
    array_like (na, nR)
        RMS variance in a top-hat sphere of radius `R`.

    See Also
    --------
    :func:`~sigmaM` : RMS variance of the density field smoothed on :math:`M`.
    """
    psp = cosmo.parse_pk2d(p_of_k_a, is_linear=True)
    status = 0
    R_use = np.atleast_1d(R)
    sR, status = lib.sigmaR_vec(cosmo.cosmo, psp, a, R_use, R_use.size, status)
    cosmo.check(status)
    if np.ndim(R) == 0:
        sR = sR[0]
    return sR


@warn_api
def sigmaV(
        cosmo: Cosmology,
        R: Union[Real, NDArray[Real]],
        a: Union[Real, NDArray[Real]] = 1,
        *,
        p_of_k_a: str = DEFAULT_POWER_SPECTRUM
):
    r"""RMS variance in the displacement field in a top-hat sphere of radius
    `R`. The linear displacement field is the gradient of the linear density
    field:

    .. math::

        \sigma_V^2(z) = \frac{1}{6\pi^2} \int {\rm d}k \, P_{\rm L}(k)
        \tilde{W}_R^2(k),

    where :math:`P_{\rm L}` is the linear matter power spectrum and
    :math:`\tilde{W}` is the Fourier transform of the spherical top-hat window

    .. math::

        \tilde{W}_R(k) = \frac{3}{(kR)^3} \left( \sin(kR) - kR \cos(kR)\right).

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    R : array_like (nR,)
        Radius of the top-hat sphere.
    a : array_like (na,)
        Scale factor.
    p_of_k_a
        3-D power spectrum to integrate.
        If a string is passed, a non-linear power spectrum under that name
        must be stored in `cosmo`.

    Returns
    -------
    array_like (na, nR)
        RMS variance of the displacement field in a top-hat sphere of radius
        `R`.
    """
    psp = cosmo.parse_pk2d(p_of_k_a, is_linear=True)
    status = 0
    R_use = np.atleast_1d(R)
    sV, status = lib.sigmaV_vec(cosmo.cosmo, psp, a, R_use, R_use.size, status)
    cosmo.check(status)
    if np.ndim(R) == 0:
        sV = sV[0]
    return sV


@warn_api
def sigma8(
        cosmo: Cosmology,
        *,
        p_of_k_a: str = DEFAULT_POWER_SPECTRUM
) -> float:
    r"""RMS variance in a top-hat sphere of radius :math:`8 {\rm Mpc}/h`.

    .. note::

        :math:`8 {\rm Mpc}/h` is rescaled based on the chosen value of the
        Hubble constant within `cosmo`.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    p_of_k_a
        3-D power spectrum to integrate.
        If a string is passed, a non-linear power spectrum under that name
        must be stored in `cosmo`.

    Returns
    -------

        :math:`\sigma_8` for the input cosmology.
    """
    sig8 = cosmo.sigmaR(8/cosmo["h"], p_of_k_a=p_of_k_a)
    if np.isnan(cosmo["sigma8"]):
        cosmo._params.sigma8 = sig8
    return sig8


@warn_api
def kNL(
        cosmo: Cosmology,
        a: Union[Real, NDArray[Real]],
        *,
        p_of_k_a: str = DEFAULT_POWER_SPECTRUM
) -> Union[float, NDArray[float]]:
    r"""Scale for the non-linear cut.

    `k_{\rm NL}` is calculated based on Lagrangian perturbation theory, as the
    inverse of the variance of the displacement field,

    .. math::

        k_{\rm NL} = \frac{1}{\sigma_\eta}
        = \left( \frac{1}{6\pi^2} \int P_{\rm L}(k) {\rm d}k \right)^{-1/2}.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    a : array_like (na,)
        Scale factor.
    p_of_k_a
        3-D power spectrum to integrate.
        If a string is passed, a non-linear power spectrum under that name
        must be stored in `cosmo`.

    Returns
    -------
    array_like (na,)
        Scale of non-linear cut-off (:math:`\rm Mpc^{-1}`).
    """
    psp = cosmo.parse_pk2d(p_of_k_a, is_linear=True)
    status = 0
    a_use = np.atleast_1d(a)
    knl, status = lib.kNL_vec(cosmo.cosmo, psp, a_use, a_use.size, status)
    cosmo.check(status)
    if np.ndim(a) == 0:
        knl = knl[0]
    return knl
