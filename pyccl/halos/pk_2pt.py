"""
=====================================================
Halo model power spectrum (:mod:`pyccl.halos.pk_2pt`)
=====================================================

Functions that compute the halo model power spectrum.
"""

from __future__ import annotations

__all__ = ("halomod_power_spectrum", "halomod_Pk2D",)

import warnings
from numbers import Real
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
import numpy.typing as npt

from .. import CCLWarning, Pk2D, warn_api
from . import Profile2pt

if TYPE_CHECKING:
    from .. import Cosmology
    from . import HMCalculator, HaloProfile


@warn_api(pairs=[("supress_1h", "suppress_1h")],
          reorder=["prof_2pt", "prof2", "p_of_k_a", "normprof1", "normprof2"])
def halomod_power_spectrum(
        cosmo: Cosmology,
        hmc: HMCalculator,
        k: Union[Real, npt.NDArray],
        a: Union[Real, npt.NDArray],
        prof: HaloProfile,
        *,
        prof2: Optional[HaloProfile] = None,
        prof_2pt: Optional[Profile2pt] = None,
        normprof1: Optional[bool] = None,
        normprof2: Optional[bool] = None,
        p_of_k_a: Union[str, Pk2D] = "linear",
        get_1h: bool = True,
        get_2h: bool = True,
        smooth_transition: Optional[Callable[[Real], Real]] = None,
        suppress_1h: Optional[Callable[[Real], Real]] = None,
        extrap_pk: bool = False
) -> Union[Real, npt.NDArray]:
    r"""Compute the halo model power spectrum:

    .. math::

        P_{u,v}(k,a) = I^0_2(k,a|u,v) + I^1_1(k,a|u) \, I^1_1(k,a|v)
        \,P_{\rm L}(k,a),

    where :math:`P_{\rm L}(k,a)` is the linear matter power spectrum, and
    :math:`I^1_1`, :math:`I^0_2` are defined in
    :class:`~pyccl.halos.HMCalculator` as :meth:`I_1_1` and :meth:`I_0_2`,
    respectively.

    Arguments
    ---------
    cosmo : :obj:`~pyccl.Cosmology`
        Cosmological parameters.
    hmc : :obj:`~pyccl.halos.HMCalculator`
        Halo model workspace.
    k : int, float or (nk,) array_like
        Comoving wavenumber, in :math:`\rm Mpc^{-1}`.
    a : int, float or (na,) array_like
        Scale factor.
    prof, prof2 : :obj:`~pyccl.halos.HaloProfile`, required, optional
        Halo profiles. If ``prof2`` is None, ``prof`` is used.
    normprof1, normprof2 : bool, optional - Deprecated, do not use.
        If True, normalize by :math:`I^0_1(k\rightarrow 0,a|u)`
        (see :meth:`~HMCalculator.I_0_1`), where :math:`u` is the profile
        represented by ``prof`` and ``prof2``, respectively.
    prof_2pt : :obj:`~pyccl.halos.Profile2pt`, optional
        Profile covariance. The default is :obj:`pyccl.halos.Profile2pt`.
    p_of_k_a : :obj:`~pyccl.Pk2D` or 'linear',  optional
        Linear power spectrum to integrate. ``'linear'`` gets the linear matter
        power spectrum stored in ``cosmo``. The default is ``'linear'``.
    get_1h, get_2h : bool, optional
        Whether to compute the 1-halo term and the 2-halo term, respectively.
        The defaults are True.
    smooth_transition, suppress_1h : callable, optional
        Functions to (i) smooth the 1-halo/2-halo transition region (ii)
        suppress the 1-halo large-scale contribution, as defined in HMCODE-2020
        (`Mead et al., 2020 <https://arxiv.org/abs/2009.01858>`_). These are
        time-dependent and modify the power spectrum as

        .. math::

            P(k,a) &= \left(P_{\rm 1h}^{\alpha(a)}(k)
            + P_{\rm 2h}^{\alpha(a)}(k) \right)^{1 / \alpha} \\
            P_{\rm 1h} &\rightarrow \frac{(k / k_*(a))^4}{1+(k / k_*(a))^4}

        By default these modifications to the power spectrum are not imposed.

    extrap_pk : bool
        Whether to extrapolate ``p_of_k_a`` in case ``a`` is out of its
        support. If False, and the queried values are out of bounds, an
        exception is raised. The default is False.

    Returns
    -------
    pka : float or (na, nk) numpy.ndarray
        Halo model power spectrum.
    """
    a_use = np.atleast_1d(a).astype(float)
    k_use = np.atleast_1d(k).astype(float)

    # Check inputs
    if smooth_transition is not None:
        if not (get_1h and get_2h):
            raise ValueError("Transition region can only be modified "
                             "when both 1-halo and 2-halo terms are queried.")
    if suppress_1h is not None:
        if not get_1h:
            raise ValueError("Can't suppress the 1-halo term "
                             "when get_1h is False.")

    if prof2 is None:
        prof2 = prof
    if prof_2pt is None:
        prof_2pt = Profile2pt()

    pk2d = cosmo.parse_pk(p_of_k_a)
    extrap = cosmo if extrap_pk else None  # extrapolation rule for pk2d

    na = len(a_use)
    nk = len(k_use)
    out = np.zeros([na, nk])
    for ia, aa in enumerate(a_use):
        # normalizations
        norm1 = prof.get_normalization(cosmo, aa, hmc=hmc) if normprof1 else 1
        # TODO: CCLv3, remove if

        if prof2 == prof:
            norm2 = norm1
        else:
            norm2 = prof2.get_normalization(cosmo, aa,
                                            hmc=hmc) if normprof2 else 1
            # TODO: CCLv3, remove if

        if get_2h:
            # bias factors
            i11_1 = hmc.I_1_1(cosmo, k_use, aa, prof)

            if prof2 == prof:
                i11_2 = i11_1
            else:
                i11_2 = hmc.I_1_1(cosmo, k_use, aa, prof2)

            pk_2h = pk2d(k_use, aa, cosmo=extrap) * i11_1 * i11_2  # 2h term
        else:
            pk_2h = 0

        if get_1h:
            pk_1h = hmc.I_0_2(cosmo, k_use, aa, prof,
                              prof2=prof2, prof_2pt=prof_2pt)  # 1h term

            if suppress_1h is not None:
                # large-scale damping of 1-halo term
                ks = suppress_1h(aa)
                pk_1h *= (k_use / ks)**4 / (1 + (k_use / ks)**4)
        else:
            pk_1h = 0

        # smooth 1h/2h transition region
        if smooth_transition is None:
            out[ia] = (pk_1h + pk_2h) / (norm1 * norm2)
        else:
            alpha = smooth_transition(aa)
            out[ia] = (pk_1h**alpha + pk_2h**alpha)**(1/alpha) / (norm1*norm2)

    if np.ndim(a) == 0:
        out = np.squeeze(out, axis=0)
    if np.ndim(k) == 0:
        out = np.squeeze(out, axis=-1)
    return out


@warn_api(pairs=[("supress_1h", "suppress_1h")],
          reorder=["prof_2pt", "prof2", "p_of_k_a", "normprof1", "normprof2"])
def halomod_Pk2D(
        cosmo: Cosmology,
        hmc: HMCalculator,
        prof: HaloProfile,
        *,
        prof2: Optional[HaloProfile] = None,
        prof_2pt: Optional[Profile2pt] = None,
        normprof1: Optional[bool] = None,
        normprof2: Optional[bool] = None,
        p_of_k_a: Union[str, Pk2D] = "linear",
        get_1h: bool = True,
        get_2h: bool = True,
        lk_arr: Optional[npt.NDArray] = None,
        a_arr: Optional[npt.NDArray] = None,
        extrap_order_lok: int = 1,
        extrap_order_hik: int = 2,
        smooth_transition: Optional[Callable[[Real], Real]] = None,
        suppress_1h: Optional[Callable[[Real], Real]] = None,
        extrap_pk: bool = False,
        use_log: bool = True
) -> Pk2D:
    """Get the halo model power spectrum.

    Create a :obj:`~pyccl.Pk2D` container of the power spectrum.

    * Information on the arguments is in :func:`halomod_power_spectrum`.
    * Arguments ``(a_arr, lk_arr, extrap_order_lok, extrap_order_hik)`` are
      passed to the :class:`~pyccl.Pk2D` constructor.
    * If ``lk_arr`` or ``a_arr`` are not specified, the sampling arrays are
      computed from the spline parameters stored in ``cosmo``.
    * If ``use_log`` is True, the power spectrum is interpolated in log-space.
    """
    if lk_arr is None:
        lk_arr = cosmo.get_pk_spline_lk()
    if a_arr is None:
        a_arr = cosmo.get_pk_spline_a()

    pk_arr = halomod_power_spectrum(
        cosmo, hmc, np.exp(lk_arr), a_arr,
        prof, prof2=prof2, prof_2pt=prof_2pt, p_of_k_a=p_of_k_a,
        normprof1=normprof1, normprof2=normprof2,  # TODO: remove for CCLv3
        get_1h=get_1h, get_2h=get_2h,
        smooth_transition=smooth_transition, suppress_1h=suppress_1h,
        extrap_pk=extrap_pk)

    pk_arr, use_log = _logged_output(pk_arr, log=use_log)
    return Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik,
                is_logp=use_log)


def _logged_output(*arrs, log):
    """Helper that logs the output if needed."""
    if not log:
        return *arrs, log
    is_negative = [(arr <= 0).any() for arr in arrs]
    if any(is_negative):
        warnings.warn("Some values were non-positive. "
                      "Interpolating linearly.", CCLWarning)
        return *arrs, False
    return *[np.log(arr) for arr in arrs], log
