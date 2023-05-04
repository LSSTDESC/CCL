"""
======================================
Covariances (:mod:`pyccl.covariances`)
======================================

Functions to compute covariances.
"""

from __future__ import annotations

__all__ = ("angular_cl_cov_cNG", "sigma2_B_disc", "sigma2_B_from_mask",
           "angular_cl_cov_SSC",)

from numbers import Real
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from . import DEFAULT_POWER_SPECTRUM, lib, warn_api
from .pyutils import _check_array_params, integ_types

if TYPE_CHECKING:
    from . import Cosmology, Pk2D, Tk3D, Tracer


@warn_api(pairs=[("cltracer1", "tracer1"), ("cltracer2", "tracer2"),
                 ("cltracer3", "tracer3"), ("cltracer4", "tracer4"),
                 ("tkka", "t_of_kk_a")],
          reorder=['fsky', 'tracer3', 'tracer4', 'ell2'])
def angular_cl_cov_cNG(
        cosmo: Cosmology,
        tracer1: Tracer,
        tracer2: Tracer,
        *,
        ell: Union[Real, NDArray[Real]],
        t_of_kk_a: Tk3D,
        tracer3: Optional[Tracer] = None,
        tracer4: Optional[Tracer] = None,
        ell2: Optional[Union[Real, NDArray[Real]]] = None,
        fsky: Real = 1,
        integration_method: str = 'qag_quad'
) -> Union[float, NDArray[float]]:
    r"""Compute the connected non-Gaussian covariance for a pair of power
    spectra :math:`C_{\ell_1}^{ab}` and :math:`C_{\ell_2}^{cd}`, and between
    two pairs of tracers :math:`(a, b)` and :math:`(c, d)`.

    .. math::

        {\rm Cov}_{\rm cNG}(\ell_1,\ell_2) =
        \int \frac{{\rm d}\chi}{\chi^6}
        \tilde{\Delta}^a_{\ell_1}(\chi)
        \tilde{\Delta}^b_{\ell_1}(\chi)
        \tilde{\Delta}^c_{\ell_2}(\chi)
        \tilde{\Delta}^d_{\ell_2}(\chi)\,
        \bar{T}_{abcd}
        \left[\frac{\ell_1+1/2}{\chi},
        \frac{\ell_2+1/2}{\chi}, a(\chi)\right]

    where :math:`\Delta^x_\ell(\chi)` is the transfer function for tracer
    :math:`x` (see Eq. 39 in the CCL note), and
    :math:`\bar{T}_{abcd}(k_1, k_2, a)` is the isotropized connected
    trispectrum of the four tracers. More details in :class:`~Tk3D`.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    tracer1, tracer2
        Tracer.
    ell : ndarray (nell,)
        Multipole to evaluate the first dimension of the angular power spectrum
        covariance.
    t_of_kk_a
        3-D connected trispectrum.
    tracer3
        Tracer. If not provided, `tracer1` is used.
    tracer4
        Tracer. If not provided, `tracer2` is used.
    ell2 : ndarray (nell2,), optional
        Multipole to evaluate the second dimension of the angular power
        spectrum covariance. If not provided, `ell` is used.
    fsky
        Sky fraction.
    integration_method
        Limber integration method. Options in
        :class:`~pyccl.pyutils.IntegrationMethods`.

    Returns
    -------
    ndarray (ell2, ell1)
        Connceted non-Gaussian angular power spectrum covariance.

    Raises
    ------
    ValueError
        If the integration method is invalid.
    """
    if integration_method not in integ_types:
        raise ValueError(f"Unknown integration method {integration_method}.")

    # we need the distances for the integrals
    cosmo.compute_distances()
    tsp = t_of_kk_a.tsp

    ell1_use = np.atleast_1d(ell)
    if ell2 is None:
        ell2 = ell
    ell2_use = np.atleast_1d(ell2)

    tr1, tr2, tr3, tr4, status = _allocate_tracers(tracer1, tracer2,
                                                   tracer3, tracer4)

    cov, status = lib.angular_cov_vec(
        cosmo.cosmo, tr1, tr2, tr3, tr4, tsp,
        ell1_use, ell2_use, integ_types[integration_method],
        6, 1./(4*np.pi*fsky), ell1_use.size*ell2_use.size, status)

    cov = cov.reshape([ell2_use.size, ell1_use.size])
    if np.ndim(ell2) == 0:
        cov = np.squeeze(cov, axis=0)
    if np.ndim(ell) == 0:
        cov = np.squeeze(cov, axis=-1)

    # Free up tracer collections
    lib.cl_tracer_collection_t_free(tr1)
    lib.cl_tracer_collection_t_free(tr2)
    if tracer3 is not None:
        lib.cl_tracer_collection_t_free(tr3)
    if tracer4 is not None:
        lib.cl_tracer_collection_t_free(tr4)

    cosmo.check(status)
    return cov


@warn_api(pairs=[('a', 'a_arr')])
def sigma2_B_disc(
        cosmo: Cosmology,
        a_arr: Optional[Union[Real, NDArray[Real]]] = None,
        *,
        fsky=1,
        p_of_k_a: Union[str, Pk2D] = DEFAULT_POWER_SPECTRUM
):
    r"""Compute the variance of the projected linear density field over a
    circular disc covering a sky fraction `fsky`.

    .. math::

        \sigma^2_B(z) = \int_0^\infty \frac{k \, {\rm d}k}{2\pi}
        P_{\rm L}(k, z) \, \left[ \frac{2 J_1(k R(z))}{k R(z)} \right]^2,

    where :math:`R(z)` is the corresponding radial aperture as a function of
    redshift. This quantity is used to compute the super-sample covariance.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    a_arr
        Scale factor to evaluate the projected variance. If not specified, the
        default sampling from the spline parameters in `cosmo` is used.
    fsky
        Sky fraction.
    p_of_k_a
        Linear power spectrum. The default is the one stored in `cosmo`.

    Returns
    -------
    a_arr : ndarray (na,)
        Scale factor of the projected variance. Only returned if `a_arr` is not
        specified.
    sigma2_B : array_like (na,)
        Projected variance.
    """
    full_output = a_arr is None

    if full_output:
        a_arr = cosmo.get_pk_spline_a()
    else:
        ndim = np.ndim(a_arr)
        a_arr = np.atleast_1d(a_arr)

    chi_arr = cosmo.comoving_radial_distance(a_arr)
    R_arr = chi_arr * np.arccos(1-2*fsky)
    psp = cosmo.parse_pk2d(p_of_k_a, is_linear=True)

    status = 0
    s2B_arr, status = lib.sigma2b_vec(cosmo.cosmo, a_arr, R_arr, psp,
                                      len(a_arr), status)
    cosmo.check(status)

    if full_output:
        return a_arr, s2B_arr
    if ndim == 0:
        return s2B_arr[0]
    return s2B_arr


@warn_api(pairs=[('a', 'a_arr')])
def sigma2_B_from_mask(
        cosmo: Cosmology,
        a_arr: Optional[Union[Real, NDArray[Real]]] = None,
        *,
        mask_wl: NDArray[Real],
        p_of_k_a: Union[str, Pk2D] = DEFAULT_POWER_SPECTRUM
):
    r"""Compute the variance of the projected linear density field, given the
        angular power spectrum of the footprint mask.

    .. math::

        \sigma^2_B(z) = \frac{1}{\chi^2{z}}\sum_\ell
        P_{\rm L}(\frac{ \ell + \frac{1}{2}}{\chi(z)}, z) \,
        (2\ell + 1) \sum_m W^A_{\ell m} {W^B}^*_{\ell m},

    where :math:`W^A_{\ell m}` and :math:`W^B_{\ell m}` are the spherical
    harmonic decompositions of the footprint masks of fields `A` and `B`,
    normalized by their area.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    a_arr
        Scale factor to evaluate the projected variance.
    mask_wl
        Angular power spectrum of the masks. The power spectrum must be given
        at integer multipoles starting at :math:`\ell = 0`. It is normalized
        as :math:`(2\ell + 1) \sum_m W^A_{\ell m} {W^B}^*_{\ell m}`. Make sure
        to provide the mask power to sufficiently high :math:`\ell` for the
        required precision.
    p_of_k_a
        Linear power spectrum. The default is the one stored in `cosmo`.

    Returns
    -------
    a_arr : ndarray (na,)
        Scale factor of the projected variance. Only returned if `a_arr` is not
        specified.
    sigma2_B : array_like (na,)
        Projected variance.
    """
    full_output = a_arr is None

    if full_output:
        a_arr = cosmo.get_pk_spline_a()
    else:
        ndim = np.ndim(a_arr)
        a_arr = np.atleast_1d(a_arr)

    if p_of_k_a is DEFAULT_POWER_SPECTRUM:
        cosmo.compute_linear_power()
        p_of_k_a = cosmo.get_linear_power()

    ell = np.arange(mask_wl.size)

    sigma2_B = np.zeros(a_arr.size)
    for i in range(sigma2_B.size):
        if 1-a_arr[i] < 1e-6:
            # For a=1, the integral becomes independent of the footprint in
            # the flat-sky approximation. So we are just using the method
            # for the disc geometry here
            sigma2_B[i] = sigma2_B_disc(cosmo=cosmo, a_arr=a_arr[i],
                                        p_of_k_a=p_of_k_a)
        else:
            chi = cosmo.comoving_angular_distance(a=a_arr)
            k = (ell+0.5)/chi[i]
            pk = p_of_k_a(k, a_arr[i], cosmo)
            # See eq. E.10 of 2007.01844
            sigma2_B[i] = np.sum(pk * mask_wl)/chi[i]**2

    if full_output:
        return a_arr, sigma2_B
    if ndim == 0:
        return sigma2_B[0]
    return sigma2_B


@warn_api(pairs=[("cltracer1", "tracer1"), ("cltracer2", "tracer2"),
                 ("cltracer3", "tracer3"), ("cltracer4", "tracer4"),
                 ('tkka', 't_of_kk_a')],
          reorder=['sigma2_B', 'fsky', 'tracer3', 'tracer4', 'ell2'])
def angular_cl_cov_SSC(
        cosmo: Cosmology,
        tracer1: Tracer,
        tracer2: Tracer,
        *,
        ell: Union[Real, NDArray[Real]],
        t_of_kk_a: Tk3D,
        tracer3: Optional[Tracer] = None,
        tracer4: Optional[Tracer] = None,
        ell2: Optional[Union[Real, NDArray[Real]]] = None,
        sigma2_B: Optional[Tuple[NDArray[Real], NDArray[Real]]] = None,
        fsky: Real = 1,
        integration_method: str = 'qag_quad'
) -> Union[float, NDArray[float]]:
    r"""Compute the super-sample contribution to the connected non-Gaussian
    covariance for a pair of power spectra :math:`C_{\ell_1}^{ab}` and
    :math:`C_{\ell_2}^{cd}`, and between two pairs of tracers, :math:`(a, b)`
    and :math:`(c, d)`.

    .. math::

        {\rm Cov}_{\rm cNG}(\ell_1,\ell_2) =
        \int \frac{{\rm d}\chi}{\chi^6}
        \tilde{\Delta}^a_{\ell_1}(\chi)
        \tilde{\Delta}^b_{\ell_1}(\chi)
        \tilde{\Delta}^c_{\ell_2}(\chi)
        \tilde{\Delta}^d_{\ell_2}(\chi)\,
        \bar{T}_{abcd}
        \left[\frac{\ell_1+1/2}{\chi},
        \frac{\ell_2+1/2}{\chi}, a(\chi)\right]

    where :math:`\Delta^x_\ell(\chi)` is the transfer function for tracer
    :math:`x` (see Eq. 39 in the CCL note), and
    :math:`\bar{T}_{abcd}(k_1, k_2, a)` is the isotropized connected
    trispectrum of the four tracers. More details in :class:`~Tk3D`.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    tracer1, tracer2
        Tracer.
    ell : ndarray (nell,)
        Multipole to evaluate the first dimension of the angular power spectrum
        covariance.
    t_of_kk_a
        3-D connected trispectrum.
    tracer3
        Tracer. If not provided, `tracer1` is used.
    tracer4
        Tracer. If not provided, `tracer2` is used.
    ell2 : ndarray (nell2,), optional
        Multipole to evaluate the second dimension of the angular power
        spectrum covariance. If not provided, `ell` is used.
    sigma2_B
        Variance of the projected matter overdensity over the footprint,
        ``(a, sigma2_B(a))``. If not provided, assume a compact circular
        footprint covering a sky fraction `fsky`.
    fsky
        Sky fraction.
    integration_method
        Limber integration method. Options in
        :class:`~pyccl.pyutils.IntegrationMethods`.

    Returns
    -------
    ndarray (ell2, ell1)
        Connceted non-Gaussian angular power spectrum covariance.

    Raises
    ------
    ValueError
        If the integration method is invalid.
    """
    if integration_method not in integ_types:
        raise ValueError(f"Unknown integration method {integration_method}.")

    # we need the distances for the integrals
    cosmo.compute_distances()
    tsp = t_of_kk_a.tsp

    ell1_use = np.atleast_1d(ell)
    if ell2 is None:
        ell2 = ell
    ell2_use = np.atleast_1d(ell2)

    if sigma2_B is None:
        a_arr, s2b_arr = sigma2_B_disc(cosmo, fsky=fsky)
    else:
        a_arr, s2b_arr = _check_array_params(sigma2_B, 'sigma2_B')

    tr1, tr2, tr3, tr4, status = _allocate_tracers(tracer1, tracer2,
                                                   tracer3, tracer4)

    cov, status = lib.angular_cov_ssc_vec(
        cosmo.cosmo, tr1, tr2, tr3, tr4, tsp, a_arr, s2b_arr,
        ell1_use, ell2_use, integ_types[integration_method],
        4, 1., ell1_use.size*ell2_use.size, status)

    cov = cov.reshape([ell2_use.size, ell1_use.size])
    if np.ndim(ell2) == 0:
        cov = np.squeeze(cov, axis=0)
    if np.ndim(ell) == 0:
        cov = np.squeeze(cov, axis=-1)

    # Free up tracer collections
    lib.cl_tracer_collection_t_free(tr1)
    lib.cl_tracer_collection_t_free(tr2)
    if tracer3 is not None:
        lib.cl_tracer_collection_t_free(tr3)
    if tracer4 is not None:
        lib.cl_tracer_collection_t_free(tr4)

    cosmo.check(status)
    return cov


def _allocate_tracers(tracer1, tracer2, tracer3, tracer4):
    # Create tracer colections
    status = 0
    tr1, status = lib.cl_tracer_collection_t_new(status)

    for t in tracer1._trc:
        status = lib.add_cl_tracer_to_collection(tr1, t, status)
    tr2, status = lib.cl_tracer_collection_t_new(status)

    for t in tracer2._trc:
        status = lib.add_cl_tracer_to_collection(tr2, t, status)

    if tracer3 is None:
        tr3 = tr1
    else:
        tr3, status = lib.cl_tracer_collection_t_new(status)
        for t in tracer3._trc:
            status = lib.add_cl_tracer_to_collection(tr3, t, status)

    if tracer4 is None:
        tr4 = tr2
    else:
        tr4, status = lib.cl_tracer_collection_t_new(status)
        for t in tracer4._trc:
            status = lib.add_cl_tracer_to_collection(tr4, t, status)

    return tr1, tr2, tr3, tr4, status
