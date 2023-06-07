"""
==================================================
Halo model trispectrum (:mod:`pyccl.halos.pk_4pt`)
==================================================

Functions that compute the halo model trispectrum.
"""

from __future__ import annotations

__all__ = ("halomod_trispectrum_1h", "halomod_Tk3D_1h",
           "halomod_Tk3D_SSC_linear_bias", "halomod_Tk3D_SSC",)

from numbers import Real
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .. import Tk3D, warn_api
from . import HaloProfileNFW, Profile2pt
from .pk_2pt import _logged_output

if TYPE_CHECKING:
    from .. import Cosmology, Pk2D
    from . import HMCalculator, HaloProfile


@warn_api(pairs=[("prof1", "prof")], reorder=["prof12_2pt", "prof3", "prof4"])
def halomod_trispectrum_1h(
        cosmo: Cosmology,
        hmc: HMCalculator,
        k: Union[Real, NDArray[Real]],
        a: Union[Real, NDArray[Real]],
        prof: HaloProfile,
        *,
        prof2: Optional[HaloProfile] = None,
        prof3: Optional[HaloProfile] = None,
        prof4: Optional[HaloProfile] = None,
        prof12_2pt: Optional[Profile2pt] = None,
        prof34_2pt: Optional[Profile2pt] = None,
        normprof1: Optional[bool] = None,
        normprof2: Optional[bool] = None,
        normprof3: Optional[bool] = None,
        normprof4: Optional[bool] = None
) -> NDArray[float]:
    r"""Compute the halo model 1-halo trispectrum:

    .. math::

        T_{u_1,u_2; v_1,v_2}(k_u,k_v,a) = I^0_{2,2}(k_u,k_v,a|u_{1,2},v_{1,2})

    where :math:`I^0_{2,2}` is defined in :class:`~pyccl.halos.HMCalculator` as
    :meth:`I_0_22`.

    .. note::

        The approximation currently assumes that the 4-point cumulant is given
        by the product of the 2-point cumulants.

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
        First halo profile.
    prof2
        Second halo profile. If None, `prof` is used.
    prof3
        Third halo profile. If None, `prof` is used.
    prof4
        Fourth halo profile. If None, `prof2` is used.
    prof12_2pt
        Covariance of `prof` and `prof2`.  The default is :obj:`~Profile2pt()`.
    prof34_2pt
        Covariance of `prof3` and `prof4` If None, `prof12_2pt` is used.
    normprof1, normprof2, normprof3, normprof4
        If True, normalize by :math:`I^0_1(k\rightarrow 0,a|u)`
        (see :meth:`~HMCalculator.I_0_1`), where :math:`u` is the profile
        represented by `prof`, `prof2`, `prof3`, and `prof4`, respectively.

        .. deprecated:: 2.8.0

            Halo profiles normalized with
            :meth:`~HaloProfile.get_normalization`.

    Returns
    -------
    array_like (na, nk, nk)
        Halo model 1-halo trispectrum.
    """
    a_use = np.atleast_1d(a).astype(float)
    k_use = np.atleast_1d(k).astype(float)

    # define all the profiles
    prof, prof2, prof3, prof4, prof12_2pt, prof34_2pt = \
        _allocate_profiles(prof, prof2, prof3, prof4, prof12_2pt, prof34_2pt)
    hmc._fix_profile_mass_def(prof)
    hmc._fix_profile_mass_def(prof2)
    hmc._fix_profile_mass_def(prof3)
    hmc._fix_profile_mass_def(prof4)

    na = len(a_use)
    nk = len(k_use)
    out = np.zeros([na, nk, nk])
    for ia, aa in enumerate(a_use):
        # normalizations
        norm1 = prof.get_normalization(cosmo, aa,
                                       hmc=hmc) if normprof1 else 1
        # TODO: CCLv3 remove if

        if prof2 == prof:
            norm2 = norm1
        else:
            norm2 = prof2.get_normalization(cosmo, aa,
                                            hmc=hmc) if normprof2 else 1
            # TODO: CCLv3 remove if

        if prof3 == prof:
            norm3 = norm1
        else:
            norm3 = prof3.get_normalization(cosmo, aa,
                                            hmc=hmc) if normprof3 else 1
            # TODO: CCLv3 remove if

        if prof4 == prof2:
            norm4 = norm2
        else:
            norm4 = prof4.get_normalization(cosmo, aa,
                                            hmc=hmc) if normprof4 else 1
            # TODO: CCLv3 remove if

        # trispectrum
        tk_1h = hmc.I_0_22(cosmo, k_use, aa,
                           prof=prof, prof2=prof2,
                           prof4=prof4, prof3=prof3,
                           prof12_2pt=prof12_2pt,
                           prof34_2pt=prof34_2pt)

        out[ia] = tk_1h / (norm1 * norm2 * norm3 * norm4)  # assign

    if np.ndim(a) == 0:
        out = np.squeeze(out, axis=0)
    if np.ndim(k) == 0:
        out = np.squeeze(out, axis=-1)
        out = np.squeeze(out, axis=-1)
    return out


@warn_api(pairs=[("prof1", "prof")],
          reorder=["prof12_2pt", "prof3", "prof4"])
def halomod_Tk3D_1h(
        cosmo: Cosmology,
        hmc: HMCalculator,
        prof: HaloProfile,
        *,
        prof2: Optional[HaloProfile] = None,
        prof3: Optional[HaloProfile] = None,
        prof4: Optional[HaloProfile] = None,
        prof12_2pt: Optional[Profile2pt] = None,
        prof34_2pt: Optional[Profile2pt] = None,
        normprof1: Optional[bool] = None,
        normprof2: Optional[bool] = None,
        normprof3: Optional[bool] = None,
        normprof4: Optional[bool] = None,
        lk_arr: Optional[NDArray[Real]] = None,
        a_arr: Optional[NDArray[Real]] = None,
        extrap_order_lok: int = 1,
        extrap_order_hik: int = 1,
        use_log: bool = False
) -> Tk3D:
    """Get the halo model 1-halo trispectrum.

    Create a :class:`~Tk3D` container of the trispectrum.

    * Information on the arguments is in :func:`~halomod_trispectrum_1h`.
    * If `lk_arr` or `a_arr` are not specified, the sampling arrays are
      computed from the spline parameters stored in `cosmo`.
    * If `use_log` is True, the trispectrum is interpolated in log-space.
    * Arguments `(a_arr, lk_arr, extrap_order_lok, extrap_order_hik)` are
      passed to :class:`~Tk3D`.

    Returns
    -------

        Halo model 1-halo trispectrum.
    """
    if lk_arr is None:
        lk_arr = cosmo.get_pk_spline_lk()
    if a_arr is None:
        a_arr = cosmo.get_pk_spline_a()

    tkk = halomod_trispectrum_1h(
        cosmo, hmc, np.exp(lk_arr), a_arr,
        prof, prof2=prof2, prof3=prof3, prof4=prof4,
        prof12_2pt=prof12_2pt, prof34_2pt=prof34_2pt,
        normprof1=normprof1, normprof2=normprof2,
        normprof3=normprof3, normprof4=normprof4)

    tkk, use_log = _logged_output(tkk, log=use_log)

    return Tk3D(a_arr=a_arr, lk_arr=lk_arr, tkk_arr=tkk,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik, is_logt=use_log)


@warn_api
def halomod_Tk3D_SSC_linear_bias(
        cosmo: Cosmology,
        hmc: HMCalculator,
        *,
        prof: HaloProfileNFW,
        bias1: Union[Real, NDArray[Real]] = 1,
        bias2: Union[Real, NDArray[Real]] = 1,
        bias3: Union[Real, NDArray[Real]] = 1,
        bias4: Union[Real, NDArray[Real]] = 1,
        is_number_counts1: bool = False,
        is_number_counts2: bool = False,
        is_number_counts3: bool = False,
        is_number_counts4: bool = False,
        p_of_k_a: Union[Literal["linear", "nonlinear"], Pk2D] = "linear",
        lk_arr: Optional[NDArray[Real]] = None,
        a_arr: Optional[NDArray[Real]] = None,
        extrap_order_lok: int = 1,
        extrap_order_hik: int = 1,
        use_log: bool = False,
        extrap_pk: bool = False
) -> Tk3D:
    r"""Compute the super-sample covariance (SSC) trispectrum.

    This is equal to the tensor product of the power specturm responses
    associated with the two pairs of correlated quantities. Each response is
    calculated as

    .. math::

        \frac{\partial P_{u,v}(k)}{\partial\delta_L} = b_u b_v \left(
        \left(\frac{68}{21} - \frac{{\rm d}\log k^3 P_L(k)}{{\rm d}\log k}
        \right) P_L(k) + I^1_2(k|u,v) - (b_{u} + b_{v}) P_{u,v}(k) \right)

    where :math:`I^1_2` is defined in :class:`~pyccl.halos.HMCalculator` as
    :meth:`I_1_2`, and :math:`b_u`, :math:`b_v` are the linear halo biases for
    :math:`u` and :math:`v`, respectively, which are only nonzero in
    clustering.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    hmc
        Halo model workspace.
    prof
        NFW profile.
    bias1, bias2, bias3, bias4 : array_like (na,)
        Linear galaxy bias. Shape of array-like input must match `a_arr`.
    is_number_counts1, is_number_counts2

        Whether to compute the clustering counter terms for the repsective
        profiles.

        .. deprecated:: 2.8.0

            Number counts profiles inherit from
            :class:`~pycc.halos.profiles.profile_base.HaloProfileNumberCounts`.

    is_number_counts3, is_number_counts4

        Whether to compute the clustering counter terms for the repsective
        profiles.

        .. deprecated:: 2.8.0

            Number counts profiles inherit from
            :class:`~pycc.halos.profiles.profile_base.HaloProfileNumberCounts`.

    p_of_k_a
        Power spectrum to integrate. `'linear'` and `'nonlinear'` get the
        corresponding power stored in `cosmo`.
    a_arr
        Scale factor where the trispectrum is sampled. The default retrieves
        the sampling rate from `cosmo`.
    lk_arr
        :math:`\ln k`, where :math:`k` is the wavenumber where the trispectrum
        is sampled (in :math:`rm Mpc^{-1}`). The default is retrieves the
        sampling rate from `cosmo`.
    extrap_order_lok, extrap_order_hik
        Spline extrapolation order passed to :class:`~Tk3D`.
    use_log
        Whether to interpolate in log-space.
    extrap_pk
        Whether to extrapolate `p_of_k_a` in case `a` is out of its
        support. If False, and the queried values are out of bounds,
        an error is raised.

        .. versionadded:: 2.8.0

    Returns
    -------

        SSC effective trispectrum.
    """
    if lk_arr is None:
        lk_arr = cosmo.get_pk_spline_lk()
    if a_arr is None:
        a_arr = cosmo.get_pk_spline_a()

    if not isinstance(prof, HaloProfileNFW):
        raise TypeError("prof should be HaloProfileNFW.")
    hmc._fix_profile_mass_def(prof)

    # Make sure biases are of the form number of a x number of k
    ones = np.ones_like(a_arr)
    bias1 *= ones
    bias2 *= ones
    bias3 *= ones
    bias4 *= ones

    k_use = np.exp(lk_arr)
    prof_2pt = Profile2pt()

    pk2d = cosmo.parse_pk(p_of_k_a)
    extrap = cosmo if extrap_pk else None  # extrapolation rule for pk2d

    na = len(a_arr)
    nk = len(k_use)
    dpk12, dpk34 = [np.zeros([na, nk]) for _ in range(2)]
    for ia, aa in enumerate(a_arr):
        norm = prof.get_normalization(cosmo, aa, hmc=hmc)**2
        # TODO: CCLv3 remove if
        i12 = hmc.I_1_2(cosmo, k_use, aa, prof, prof2=prof, prof_2pt=prof_2pt)

        pk = pk2d(k_use, aa, cosmo=extrap)
        dpk = pk2d(k_use, aa, derivative=True, cosmo=extrap)

        # ~ (47/21 - 1/3 dlogPk/dlogk) * Pk + I12
        dpk12[ia] = (47/21 - dpk/3)*pk + i12 / norm
        dpk34[ia] = dpk12[ia].copy()

        # Counter terms for clustering (i.e. - (bA + bB) * PAB)
        if any([is_number_counts1, is_number_counts2,
                is_number_counts3, is_number_counts4]):
            b1 = b2 = b3 = b4 = 0
            i02 = hmc.I_0_2(cosmo, k_use, aa, prof,
                            prof2=prof, prof_2pt=prof_2pt)

            P_12 = P_34 = pk + i02 / norm

            if is_number_counts1:
                b1 = bias1[ia]
            if is_number_counts2:
                b2 = bias2[ia]
            if is_number_counts3:
                b3 = bias3[ia]
            if is_number_counts4:
                b4 = bias4[ia]

            dpk12[ia] -= (b1 + b2) * P_12
            dpk34[ia] -= (b3 + b4) * P_34

        dpk12[ia] *= bias1[ia] * bias2[ia]
        dpk34[ia] *= bias3[ia] * bias4[ia]

    dpk12, dpk34, use_log = _logged_output(dpk12, dpk34, log=use_log)

    return Tk3D(a_arr=a_arr, lk_arr=lk_arr,
                pk1_arr=dpk12, pk2_arr=dpk34,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik, is_logt=use_log)


@warn_api(pairs=[("prof1", "prof")],
          reorder=["prof12_2pt", "prof3", "prof4"])
def halomod_Tk3D_SSC(
        cosmo: Cosmology,
        hmc: HMCalculator,
        prof: HaloProfile,
        *,
        prof2: Optional[HaloProfile] = None,
        prof3: Optional[HaloProfile] = None,
        prof4: Optional[HaloProfile] = None,
        prof12_2pt: Optional[Profile2pt] = None,
        prof34_2pt: Optional[Profile2pt] = None,
        normprof1: Optional[bool] = None,
        normprof2: Optional[bool] = None,
        normprof3: Optional[bool] = None,
        normprof4: Optional[bool] = None,
        p_of_k_a: Union[Literal["linear", "nonlinear"], Pk2D] = "linear",
        lk_arr: Optional[NDArray[Real]] = None,
        a_arr: Optional[NDArray[Real]] = None,
        extrap_order_lok: int = 1,
        extrap_order_hik: int = 1,
        use_log: bool = False,
        extrap_pk: bool = False
) -> Tk3D:
    r"""Get the super-sample covariance trispectrum.

    The form of the responses is described in
    :func:`~halomod_Tk3D_SSC_linear_bias`.

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    hmc
        Halo model workspace.
    prof
        First halo profile.
    prof2
        Second halo profile. If None, `prof` is used.
    prof3
        Third halo profile. If None, `prof` is used.
    prof4
        Fourth halo profile. If None, `prof2` is used.
    prof12_2pt
        Covariance of `prof` and `prof2`.  The default is :obj:`~Profile2pt()`.
    prof34_2pt
        Covariance of `prof3` and `prof4` If None, `prof12_2pt` is used.
    normprof1, normprof2, normprof3, normprof4
        If True, normalize by :math:`I^0_1(k\rightarrow 0,a|u)`
        (see :meth:`~HMCalculator.I_0_1`), where :math:`u` is the profile
        represented by `prof`, `prof2`, `prof3`, and `prof4`, respectively.

        .. deprecated:: 2.8.0

            Halo profiles normalized with
            :meth:`~HaloProfile.get_normalization`.
    p_of_k_a
        Power spectrum to integrate. `'linear'` and `'nonlinear'` get the
        corresponding power stored in `cosmo`.
    a_arr
        Scale factor where the trispectrum is sampled. The default retrieves
        the sampling rate from `cosmo`.
    lk_arr
        :math:`\ln k`, where :math:`k` is the wavenumber where the trispectrum
        is sampled (in :math:`rm Mpc^{-1}`). The default is retrieves the
        sampling rate from `cosmo`.
    extrap_order_lok, extrap_order_hik
        Spline extrapolation order passed to :class:`~Tk3D`.
    use_log
        Whether to interpolate in log-space.
    extrap_pk
        Whether to extrapolate `p_of_k_a` in case `a` is out of its
        support. If False, and the queried values are out of bounds,
        an error is raised.

        .. versionadded:: 2.8.0

    Returns:

        SSC effective trispectrum.
    """
    if lk_arr is None:
        lk_arr = cosmo.get_pk_spline_lk()
    if a_arr is None:
        a_arr = cosmo.get_pk_spline_a()

    # define all the profiles
    prof, prof2, prof3, prof4, prof12_2pt, prof34_2pt = \
        _allocate_profiles(prof, prof2, prof3, prof4, prof12_2pt, prof34_2pt)
    hmc._fix_profile_mass_def(prof)
    hmc._fix_profile_mass_def(prof2)
    hmc._fix_profile_mass_def(prof3)
    hmc._fix_profile_mass_def(prof4)

    k_use = np.exp(lk_arr)
    pk2d = cosmo.parse_pk(p_of_k_a)
    extrap = cosmo if extrap_pk else None  # extrapolation rule for pk2d

    dpk12, dpk34 = [np.zeros((len(a_arr), len(k_use))) for _ in range(2)]
    for ia, aa in enumerate(a_arr):
        # normalizations & I11 integral
        norm1 = prof.get_normalization(cosmo, aa,
                                       hmc=hmc) if normprof1 else 1
        # TODO: CCLv3 remove if
        i11_1 = hmc.I_1_1(cosmo, k_use, aa, prof)

        if prof2 == prof:
            norm2 = norm1
            i11_2 = i11_1
        else:
            norm2 = prof2.get_normalization(cosmo, aa,
                                            hmc=hmc) if normprof2 else 1
            # TODO: CCLv3 remove if
            i11_2 = hmc.I_1_1(cosmo, k_use, aa, prof2)

        if prof3 == prof:
            norm3 = norm1
            i11_3 = i11_1
        else:
            norm3 = prof3.get_normalization(cosmo, aa,
                                            hmc=hmc) if normprof3 else 1
            # TODO: CCLv3 remove if
            i11_3 = hmc.I_1_1(cosmo, k_use, aa, prof3)

        if prof4 == prof2:
            norm4 = norm2
            i11_4 = i11_2
        else:
            norm4 = prof4.get_normalization(cosmo, aa,
                                            hmc=hmc) if normprof4 else 1
            # TODO: CCLv3 remove if
            i11_4 = hmc.I_1_1(cosmo, k_use, aa, prof4)

        # I12 integral
        i12_12 = hmc.I_1_2(cosmo, k_use, aa, prof,
                           prof2=prof2, prof_2pt=prof12_2pt)
        if (prof, prof2, prof12_2pt) == (prof3, prof4, prof34_2pt):
            i12_34 = i12_12
        else:
            i12_34 = hmc.I_1_2(cosmo, k_use, aa, prof3,
                               prof2=prof4, prof_2pt=prof34_2pt)

        # power spectrum
        pk = pk2d(k_use, aa, cosmo=extrap)
        dpk = pk2d(k_use, aa, derivative=True, cosmo=extrap)

        # (47/21 - 1/3 dlogPk/dlogk) * I11 * I11 * Pk + I12
        dpk12[ia] = ((47/21 - dpk/3)*i11_1*i11_2*pk + i12_12) / (norm1 * norm2)
        dpk34[ia] = ((47/21 - dpk/3)*i11_3*i11_4*pk + i12_34) / (norm3 * norm4)

        # Counter terms for clustering (i.e. - (bA + bB) * PAB)
        def _get_counterterm(pA, pB, p2pt, nA, nB, i11_A, i11_B):
            """Helper to compute counter-terms."""
            # p : profiles | p2pt : 2-point | n : norms | i11 : I_1_1 integral
            bA = i11_A / nA if pA.is_number_counts else np.zeros_like(k_use)
            bB = i11_B / nB if pB.is_number_counts else np.zeros_like(k_use)
            i02 = hmc.I_0_2(cosmo, k_use, aa, pA, prof2=pB, prof_2pt=p2pt)
            P = (pk * i11_A * i11_B + i02) / (nA * nB)
            return (bA + bB) * P

        if prof.is_number_counts or prof2.is_number_counts:
            dpk12[ia] -= _get_counterterm(prof, prof2, prof12_2pt,
                                          norm1, norm2, i11_1, i11_2)

        if prof3.is_number_counts or prof4.is_number_counts:
            if (prof, prof2, prof12_2pt) == (prof3, prof4, prof34_2pt):
                dpk34[ia] = dpk12[ia]
            else:
                dpk34[ia] -= _get_counterterm(prof3, prof4, prof34_2pt,
                                              norm3, norm4, i11_3, i11_4)

    dpk12, dpk34, use_log = _logged_output(dpk12, dpk34, log=use_log)

    return Tk3D(a_arr=a_arr, lk_arr=lk_arr,
                pk1_arr=dpk12, pk2_arr=dpk34,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik, is_logt=use_log)


def _allocate_profiles(prof, prof2, prof3, prof4, prof12_2pt, prof34_2pt):
    """Helper that controls how the undefined profiles are allocated."""
    if prof2 is None:
        prof2 = prof
    if prof3 is None:
        prof3 = prof
    if prof4 is None:
        prof4 = prof2
    if prof12_2pt is None:
        prof12_2pt = Profile2pt()
    if prof34_2pt is None:
        prof34_2pt = prof12_2pt

    return prof, prof2, prof3, prof4, prof12_2pt, prof34_2pt
