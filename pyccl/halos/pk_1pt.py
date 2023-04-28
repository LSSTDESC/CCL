"""
===================================================
Fourier 1-point moments (:mod:`pyccl.halos.pk_1pt`)
===================================================

Functions that compute the Fourier mean profile and the profile bias.
"""

from __future__ import annotations

__all__ = ("halomod_mean_profile_1pt", "halomod_bias_1pt",)

from numbers import Real
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .. import warn_api

if TYPE_CHECKING:
    from .. import Cosmology
    from . import HMCalculator, HaloProfile


# I_X_1 dispatcher for internal use
def _Ix1(func, cosmo, hmc, k, a, prof, normprof):
    r"""
    Arguments
    ---------
    cosmo
        Cosmological parameters.
    hmc
        Halo model workspace.
    k : array_like (nk,)
        Comoving wavenumber, in :math:`\rm Mpc^{-1}`.
    a : array_like (na,)
        Scale factor.
    prof
        Halo profile.
    normprof
        If True, normalize by :math:`I^0_1(k\rightarrow 0,a|u)`
        (see :meth:`~HMCalculator.I_0_1`), where :math:`u` is the profile
        represented by `prof`.

        .. deprecated:: 2.8.0

            Halo profiles normalized with
            :meth:`~HaloProfile.get_normalization`.

    Returns
    -------
    array_like (na, nk)
        Value of the integral.
    """
    func = getattr(hmc, func)

    a_use = np.atleast_1d(a).astype(float)
    k_use = np.atleast_1d(k).astype(float)

    na = len(a_use)
    nk = len(k_use)
    out = np.zeros([na, nk])
    for ia, aa in enumerate(a_use):
        i11 = func(cosmo, k_use, aa, prof)
        norm = prof.get_normalization(cosmo, aa, hmc=hmc) if normprof else 1
        # TODO: CCLv3 remove if
        out[ia] = i11 / norm

    if np.ndim(a) == 0:
        out = np.squeeze(out, axis=0)
    if np.ndim(k) == 0:
        out = np.squeeze(out, axis=-1)
    return out


@warn_api
def halomod_mean_profile_1pt(
        cosmo: Cosmology,
        hmc: HMCalculator,
        k: Union[Real, NDArray[Real]],
        a: Union[Real, NDArray[Real]],
        prof: HaloProfile,
        *,
        normprof: Optional[bool] = None
) -> Union[float, NDArray[float]]:
    r"""Compute the mass-weighted mean halo profile.

    .. math::

        I^0_1(k,a|u) = \int {\rm d}M \, n(M,a) \, \langle u(k,a|M) \rangle,

    where :math:`n(M,a)` is the halo mass function, and
    :math:`\langle u(k,a|M) \rangle` is the halo profile as a function of
    scale, scale factor and halo mass.
    """
    return _Ix1("I_0_1", cosmo, hmc, k, a, prof, normprof)


@warn_api
def halomod_bias_1pt(
        cosmo: Cosmology,
        hmc: HMCalculator,
        k: Union[Real, NDArray[Real]],
        a: Union[Real, NDArray[Real]],
        prof: HaloProfile,
        *,
        normprof: Optional[bool] = None
) -> Union[float, NDArray[float]]:
    r"""Compute the mass-and-bias-weighted mean halo profile.

    .. math::

        I^1_1(k,a|u) = \int {\rm d}M \, n(M,a) \, b(M,a) \,
        \langle u(k,a|M) \rangle,

    where :math:`n(M,a)` is the halo mass function, :math:`b(M,a)` is the halo
    bias, and :math:`\langle u(k,a|M) \rangle` is the halo profile as a
    function of scale, scale factor and halo mass.
    """
    return _Ix1("I_1_1", cosmo, hmc, k, a, prof, normprof)


halomod_mean_profile_1pt.__doc__ += _Ix1.__doc__
halomod_bias_1pt.__doc__ += _Ix1.__doc__
