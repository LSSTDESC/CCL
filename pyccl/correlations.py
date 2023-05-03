"""
========================================
Correlations (:mod:`pyccl.correlations`)
========================================

Functionality related to correlation functions.
"""

from __future__ import annotations

__all__ = ("CorrelationMethods", "CorrelationTypes", "correlation",
           "correlation_3d", "correlation3D_RSD", "correlation_multipole",
           "correlation_3dRsd", "correlation_3dRsd_avgmu",
           "correlation_pi_sigma",)

from enum import Enum
import warnings
from numbers import Real
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray

from . import DEFAULT_POWER_SPECTRUM, Pk2D, lib
from . import CCLDeprecationWarning, warn_api

if TYPE_CHECKING:
    from . import Cosmology


class CorrelationMethods(Enum):
    BESSEL = "bessel"
    """Direct integration using Bessel functions."""

    LEGENDRE = "legendre"
    """Sum over Legendre polynomials."""

    FFTLOG = "fftlog"
    """FFTLog algorithm."""


class CorrelationTypes(Enum):
    r"""The numbers in the descriptions refer to the spins of the correlated
    quantities. For more information see Sec 2.4.2 of :footcite:t:`Chisari19`.

    References
    ----------
    .. footbibliography::
    """
    NN = "NN"
    r""":math:`0 \ast 0`"""

    NG = "NG"
    r""":math:`0 \ast 2`"""

    GG_PLUS = "GG+"
    r""":math:`2 \ast 2`, :math:`\xi_+`"""

    GG_MINUS = "GG-"
    r""":math:`2 \ast 2`, :math:`\xi_-`"""


correlation_methods = {
    'fftlog': lib.CCL_CORR_FFTLOG,
    'bessel': lib.CCL_CORR_BESSEL,
    'legendre': lib.CCL_CORR_LGNDRE,
}

correlation_types = {
    'NN': lib.CCL_CORR_GG,
    'NG': lib.CCL_CORR_GL,
    'GG+': lib.CCL_CORR_LP,
    'GG-': lib.CCL_CORR_LM,
}


@warn_api
def correlation(
        cosmo: Cosmology,
        *,
        ell: NDArray[Real],
        C_ell: NDArray[Real],
        theta: Union[Real, NDArray[Real]],
        type: str = 'NN',
        corr_type: Optional[str] = None,
        method: str = 'fftlog'
) -> Union[float, NDArray[float]]:
    r"""Compute the angular correlation function,

    .. math::

        \xi^{ab}_+(\theta) &\equiv \left\langle \tilde{a}(\hat{\bf n}_1)
        \tilde{b}^*\hat{\bf n}_2) \right\rangle, \\
        \xi^{ab}_-(\theta) &\equiv \left\langle \tilde{a}(\hat{\bf n}_1)
        \tilde{b}(\hat{\bf n}_2) \right\rangle,

    where :math:`\hat{\bf n}_1 \cdot \hat{\bf n}_2 \equiv \cos \theta`.
    :math:`\xi_{\pm}` can be related to the angular power spectra as

    .. math::

        \xi^{ab}_\pm(\theta) =
        \sum_\ell\frac{2\ell+1}{4\pi}\,(\pm1)^{s_b} \,
        C^{ab\pm}_\ell \, d^\ell_{s_a, \pm s_b}(\theta)

    where :math:`\theta` is the angle between the two fields :math:`a` and
    :math:`b` with spins :math:`s_a` and :math:`s_b` after alignement of their
    tangential coordinate. :math:`d^\ell_{mm'}` are the Wigner-d matrices and
    we have defined the power spectra

    .. math::

        C^{ab\pm}_\ell \equiv \left( C^{a_E b_E}_\ell \pm C^{a_B b_B}_\ell
        \right) + i \left( C^{a_B b_E}_\ell \mp C^{a_E b_B}_\ell \right)

    which reduces to the :math:`EE` power spectrum when all :math:`B`-modes
    are 0.

    Spin combinations (also listed in :class:`CorrelationTypes`) are:

      * :math:`s_a=s_b=0` e.g. galaxy-galaxy, galaxy-:math:`\kappa`
        and :math:`\kappa`-:math:`\kappa`
      * :math:`s_a=2`, `s_b=0` e.g. galaxy-shear, and :math:`\kappa`-shear
      * :math:`s_a=s_b=2` e.g. shear-shear.

    .. note::

        For scales smaller than :math:`\sim 0.1^{\circ}`, the input power
        spectrum must be sampled to sufficienly high :math:`\ell` to ensure
        the Hankel transform is well-behaved. The relevant spline parameters
        may be adjusted to improve accuracy:

            - :attr:`ccl.spline_params.ELL_MIN_CORR`,
            - :attr:`ccl.spline_params.ELL_MAX_CORR`,
            - :attr:`ccl.spline_params.N_ELL_CORR`.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    ell
        Multipoles of the angular power spectrum.
    C_ell
        Angular power spectrum.
    theta : array_like (ntheta,)
        Angular separation to calculate the angular correlation function
        (in degrees).
    type
        Correlation function type. Available options in
        :class:`~CorrelationTypes`.
    method
        Method to compute the correlation function. Available options in
        :class:`~CorrelationMethods`.
    corr_type
        Deprecated alias of `type`.

        .. deprecated:: 2.1.0

    Returns
    -------
    array_like (ntheta,)
        Evaluated correlation function.

    Raises
    ------
    ValueError
        If the correlation type or method is invalid.

    See Also
    --------
    :func:`~pyccl.power.nonlin_power`
        Fourier transform of the correlation.
    """
    if corr_type is not None:  # TODO: Remove for CCLv3.
        # Convert to lower case
        corr_type = corr_type.lower()
        if corr_type == 'gg':
            type = 'NN'
        elif corr_type == 'gl':
            type = 'NG'
        elif corr_type == 'l+':
            type = 'GG+'
        elif corr_type == 'l-':
            type = 'GG-'
        else:
            raise ValueError("Unknown corr_type " + corr_type)
        warnings.warn("corr_type is deprecated. Use type = {}".format(type),
                      CCLDeprecationWarning)
    method = method.lower()

    if type not in correlation_types:
        raise ValueError(f"Invalud correlation type {type}.")
    if method not in correlation_methods.keys():
        raise ValueError(f"Invalid correlation method {method}.")

    if scalar := isinstance(theta, (int, float)):
        theta = np.array([theta, ])

    if (np.asarray(C_ell) == 0).all():
        return np.zeros_like(theta)[()]  # shortcut to avoid integration errors

    status = 0
    wth, status = lib.correlation_vec(cosmo.cosmo, ell, C_ell, theta,
                                      correlation_types[type],
                                      correlation_methods[method],
                                      len(theta), status)
    cosmo.check(status)

    if scalar:
        return wth[0]
    return wth


@warn_api(reorder=['a', 'r'])
def correlation_3d(
        cosmo: Cosmology,
        *,
        r: Union[Real, NDArray[Real]],
        a: Real,
        p_of_k_a: Union[str, Pk2D] = DEFAULT_POWER_SPECTRUM
) -> Union[float, NDArray[float]]:
    r"""Compute the 3D correlation function,

    .. math::

        \xi(r) = \frac{1}{2\pi} \int_0^\infty {\rm d}k \, k^2 \, P(k)
        \frac{\sin(kr)}{kr},

    from the power spectrum.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    r : array_like (nr,)
        Distance to compute the correlation function (in :math:`\rm Mpc`).
    a
        Scale factor.
    p_of_k_a
        3-D power spectrum to integrate. String input must be one of the names
        of non-linear power spectra stored in `cosmo`.

    Returns
    -------
    array_like (nr,)
        Evaluated correlation function.
    """
    cosmo.compute_nonlin_power()
    psp = cosmo.parse_pk2d(p_of_k_a)
    # Convert scalar input into an array
    if scalar := isinstance(r, (int, float)):
        r = np.array([r, ])

    status = 0
    xi, status = lib.correlation_3d_vec(cosmo.cosmo, psp, a, r, len(r), status)
    cosmo.check(status)

    if scalar:
        return xi[0]
    return xi


@warn_api(pairs=[('s', 'r'), ('l', 'ell')],
          reorder=['a', 'beta', 'ell', 'r'])
def correlation_multipole(
        cosmo: Cosmology,
        *,
        r: Union[Real, NDArray[Real]],
        a: Real,
        beta: Real,
        ell: int,
        p_of_k_a: Union[str, Pk2D] = DEFAULT_POWER_SPECTRUM
):
    r"""
    .. deprecated:: 2.8.0

        Use general functionality of :func:`~correlation3D_RSD`
    """
    return correlation3D_RSD(cosmo=cosmo, r=r, a=a, beta=beta, ell=ell,
                             p_of_k_a=p_of_k_a)


@warn_api(pairs=[('s', 'r')],
          reorder=['a', 'r', 'mu', 'beta', 'use_spline', 'p_of_k_a'])
def correlation_3dRsd(cosmo, *, r, a, mu, beta,
                      p_of_k_a=DEFAULT_POWER_SPECTRUM, use_spline=True):
    r"""
    .. deprecated:: 2.8.0

        Use general functionality of :func:`~correlation3D_RSD`
    """
    return correlation3D_RSD(cosmo, r=r, mu=mu, a=a, beta=beta,
                             p_of_k_a=p_of_k_a, use_spline=use_spline)


@warn_api(pairs=[('s', 'r')], reorder=['a', 'r'])
def correlation_3dRsd_avgmu(cosmo, *, r, a, beta,
                            p_of_k_a=DEFAULT_POWER_SPECTRUM):
    r"""
    .. deprecated:: 2.8.0

        Use general functionality of :func:`~correlation3D_RSD`
    """
    return correlation3D_RSD(cosmo, r=r, a=a, beta=beta, p_of_k_a=p_of_k_a)


@warn_api(pairs=[("sig", "sigma")],
          reorder=['a', 'beta', 'pi', 'sigma'])
def correlation_pi_sigma(cosmo, *, pi, sigma, a, beta,
                         use_spline=True, p_of_k_a=DEFAULT_POWER_SPECTRUM):
    r"""
    .. deprecated:: 2.8.0

        Use general functionality of :func:`~correlation3D_RSD`
    """
    return correlation3D_RSD(cosmo, pi=pi, sigma=sigma, a=a, beta=beta,
                             p_of_k_a=p_of_k_a, use_spline=use_spline)


def correlation3D_RSD(
        cosmo: Cosmology,
        *,
        r: Optional[Union[Real, NDArray[Real]]] = None,
        mu: Optional[Union[Real, NDArray[Real]]] = None,
        pi: Optional[Union[Real, NDArray[Real]]] = None,
        sigma: Optional[Union[Real, NDArray[Real]]] = None,
        a: Real,
        beta: Real,
        ell: Literal[0, 2, 4] = 0,
        p_of_k_a: Union[str, Pk2D] = DEFAULT_POWER_SPECTRUM,
        use_spline: bool = True
) -> Union[float, NDArray[float]]:
    r"""Compute the 2-point correlation function in redshift space, using
    linear theory :footcite:p:`Kaiser87`,

    .. math::

        \xi_{\langle r \rangle}(r) &\equiv \frac{1}{2}
        \int_{-1}^{+1} {\rm d}\mu \, \xi_r(r, \mu) \\
        &= \left(1 + \frac{2}{3}\Omega^{0.6} + \frac{1}{5}\Omega^{1.2}\right)
        \, \xi(r),

    where :math:`r` is the correlation separation distance.

    This function provides the choice of working in :math:`r`/:math:`\mu` or
    :math:`\pi`/:math:`\sigma` coordinates.

        * For :math:`r`/:math:`\mu`, `mu` is optional; omitting it will compute
          the average correlation function.
        * For :math:`\pi`/:math:`\sigma`, both `pi` and `sigma` are required
          (and the average cannot be computed since `mu` will be nonzero).
        * The shapes of the input coordinate arrays must be broadcastable.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    r : array_like (ns,), optional
        Separation distance to compute the correlation function
        (in :math:`\rm Mpc`). Ignored if `pi` and `sigma` are provided.
    mu : array_like (ns,), optional
        :math:`\cos \theta`, where :math:`\theta` is the angle
        (in :math:`\rm rad`) to compute the correlation function. If not
        provided, and :math:`r`/:math:`\mu` coordinates are used, compute the
        multipole of the correlation function using FFTLog. See `ell`.
    pi : array_like (ns,), optional
        Projected separation (in :math:`\rm Mpc`), :math:`r \, \cos\theta`.
        If not provided, use :math:`r`/:math:`\mu` coordinates.
    sigma : array_like (ns,), optional
        Projected separation (in :math:`\rm Mpc`), :math:`r \, \sin\theta`.
        If not provided, use :math:`r`/:math:`\mu` coordinates.
    a
        Scale factor.
    beta
        Growth rate divided by galaxy bias, :math:`\frac{f(a)}{b_g(a)}`
    ell
        Multipole. Ignored if `mu` or (`pi`, `sigma`) are provided.
        :math:`\ell = 0` computes the average correlation over :math:`\mu` at
        constant separation (higher multipoles integrate to zero).
    p_of_k_a
        3-D power spectrum to integrate. String input must be one of the names
        of non-linear power spectra stored in `cosmo`.
    use_spline
        Whether to compute the correlation function using spline integration.
        Not available when computing the multipole correlation.

    Returns
    -------
    array_like (ns,)
        Evaluated correlation function.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        import pyccl as ccl

        cosmo = ccl.CosmologyVanillaLCDM()
        r = np.geomspace(10, 100, 5)

        # 3-D RSD full correlation function; single `mu`
        cosmo.correlation3D_RSD(r=r, mu=0.7, a=1, beta=0.5)

        # Pairwise `r` and `mu`
        mu = np.linspace(0.4, 0.9, 5)
        cosmo.correlation3D_RSD(r=r, mu=mu, a=1, beta=0.5)

        # Compute single multipole
        cosmo.correlation3D_RSD(r=r, a=1, beta=0.5, ell=2)

        # Averaging over `mu` (single multiple at default ``ell = 0``)
        cosmo.correlation3D_RSD(r=r, a=1, beta=0.5)

        # Working in pi/sigma coordinates
        sigma, pi = r, np.ones_like(r)
        cosmo.correlation3D_RSD(pi=1, sigma=sigma, a=1, beta=0.5)
        cosmo.correlation3D_RSD(pi=pi, sigma=sigma, a=1, beta=0.5)  # pairwise

    References
    ----------
    .. footbibliography::
    """
    if not isinstance(p_of_k_a, Pk2D):
        cosmo.compute_nonlin_power()
    psp = cosmo.parse_pk2d(p_of_k_a, is_linear=False)

    if sigma is not None:
        # transforming π/σ to r/μ
        r = np.hypot(pi, sigma)
        mu = np.divide(pi, r)

    r_use = np.atleast_1d(r)
    rarrs = [r_use, len(r_use)]

    if mu is None:
        # averaging over `μ`
        func = lib.correlation_multipole_vec
        args = [ell] + rarrs
    else:
        func = lib.correlation_3dRsd_vec
        mu = np.broadcast_to(mu, r_use.shape)
        args = [mu] + rarrs + [int(use_spline)]

    status = 0
    xi, status = func(cosmo.cosmo, psp, a, beta, *args, status)
    cosmo.check(status)

    if np.ndim(r) == 0:
        return xi[0]
    return xi
