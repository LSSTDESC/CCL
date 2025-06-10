__all__ = ("halomod_trispectrum_1h", "halomod_Tk3D_1h",
           "halomod_trispectrum_2h_22", "halomod_trispectrum_2h_13",
           "halomod_trispectrum_3h", "halomod_trispectrum_4h",
           "halomod_Tk3D_2h", "halomod_Tk3D_3h", "halomod_Tk3D_4h",
           "halomod_Tk3D_SSC_linear_bias", "halomod_Tk3D_SSC",
           "halomod_Tk3D_cNG")

import numpy as np
import scipy

from .. import CCLWarning, warnings, Tk3D, Pk2D
from . import HaloProfileNFW, Profile2pt


def halomod_trispectrum_1h(cosmo, hmc, k, a, prof, *,
                           prof2=None, prof3=None, prof4=None,
                           prof12_2pt=None, prof34_2pt=None):
    """ Computes the halo model 1-halo trispectrum for four different
    quantities defined by their respective halo profiles. The 1-halo
    trispectrum for four profiles :math:`u_{1,2}`, :math:`v_{1,2}` is
    calculated as:

    .. math::
        T_{u_1,u_2;v_1,v_2}(k_u,k_v,a) =
        I^0_{2,2}(k_u,k_v,a|u_{1,2},v_{1,2})

    where :math:`I^0_{2,2}` is defined in the documentation
    of :meth:`~pyccl.halos.halo_model.HMCalculator.I_0_22`.

    .. note:: This approximation assumes that the 4-point
              profile cumulant is the same as the product of two
              2-point cumulants. We may relax this assumption in
              future versions of CCL.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        k (:obj:`float` or `array`): comoving wavenumber in
            :math:`{\\rm Mpc}^{-1}`.
        a (:obj:`float` or `array`): scale factor.
        prof (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile (corresponding to :math:`u_1` above).
        prof2 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile (corresponding to :math:`u_2` above). If ``None``,
            ``prof`` will be used as ``prof2``.
        prof12_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object returning the the two-point
            moment of ``prof`` and ``prof2``. If ``None``, the default
            second moment will be used, corresponding to the
            products of the means of both profiles.
        prof3 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile (corresponding to :math:`v_1` above. If ``None``,
            ``prof`` will be used as ``prof3``.
        prof4 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile (corresponding to :math:`v_2` above. If ``None``,
            ``prof2`` will be used as ``prof4``.
        prof34_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as ``prof12_2pt`` for ``prof3`` and ``prof4``.

    Returns:
        (:obj:`float` or `array`): 1-halo trispectrum evaluated at each
        combination of ``k`` and ``a``. The shape of the output will
        be ``(N_a, N_k, N_k)`` where ``N_k`` and ``N_a`` are the sizes of
        ``k`` and ``a`` respectively. The ordering is such that
        ``output[ia, ik2, ik1] = T(k[ik1], k[ik2], a[ia])``
        If ``k`` or ``a`` are scalars, the corresponding dimension will
        be squeezed out on output.
    """
    a_use = np.atleast_1d(a).astype(float)
    k_use = np.atleast_1d(k).astype(float)

    # define all the profiles
    prof, prof2, prof3, prof4, prof12_2pt, prof34_2pt = \
        _allocate_profiles(prof, prof2, prof3, prof4, prof12_2pt, prof34_2pt)

    na = len(a_use)
    nk = len(k_use)
    out = np.zeros([na, nk, nk])
    for ia, aa in enumerate(a_use):
        # normalizations
        norm1 = prof.get_normalization(cosmo, aa, hmc=hmc)

        if prof2 == prof:
            norm2 = norm1
        else:
            norm2 = prof2.get_normalization(cosmo, aa, hmc=hmc)

        if prof3 == prof:
            norm3 = norm1
        else:
            norm3 = prof3.get_normalization(cosmo, aa, hmc=hmc)

        if prof4 == prof2:
            norm4 = norm2
        else:
            norm4 = prof4.get_normalization(cosmo, aa, hmc=hmc)

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


def halomod_Tk3D_1h(cosmo, hmc, prof, *,
                    prof2=None, prof3=None, prof4=None,
                    prof12_2pt=None, prof34_2pt=None,
                    lk_arr=None, a_arr=None,
                    extrap_order_lok=1, extrap_order_hik=1,
                    use_log=False):
    """ Returns a :class:`~pyccl.tk3d.Tk3D` object containing
    the 1-halo trispectrum for four quantities defined by
    their respective halo profiles. See :meth:`halomod_trispectrum_1h`
    for more details about the actual calculation.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        prof (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile (corresponding to :math:`u_1` above.
        prof2 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile. If ``None``, ``prof`` will be used as ``prof2``.
        prof3 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile. If ``None``, ``prof`` will be used as ``prof3``.
        prof4 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile. If ``None``, ``prof2`` will be used as ``prof4``.
        prof12_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object returning the the two-point
            moment of ``prof`` and ``prof2``. If ``None``, the default
            second moment will be used, corresponding to the
            products of the means of both profiles.
        prof34_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as ``prof12_2pt`` for ``prof3`` and ``prof4``.
        a_arr (array): an array holding values of the scale factor
            at which the trispectrum should be calculated for
            interpolation. If ``None``, the internal values used
            by ``cosmo`` will be used.
        lk_arr (array): an array holding values of the natural
            logarithm of the wavenumber (in units of
            :math:`{\\rm Mpc}^{-1}`) at which the trispectrum should
            be calculated for interpolation. If ``None``, the internal
            values used by ``cosmo`` will be used.
        extrap_order_lok (:obj:`int`): extrapolation order to be used on
            k-values below the minimum of the splines. See
            :class:`~pyccl.tk3d.Tk3D`.
        extrap_order_hik (:obj:`int`): extrapolation order to be used on
            k-values above the maximum of the splines. See
            :class:`~pyccl.tk3d.Tk3D`.
        use_log (:obj:`bool`): if ``True``, the trispectrum will be
            interpolated in log-space (unless negative or
            zero values are found).

    Returns:
        :class:`~pyccl.tk3d.Tk3D`: 1-halo trispectrum.
    """
    if lk_arr is None:
        lk_arr = cosmo.get_pk_spline_lk()
    if a_arr is None:
        a_arr = cosmo.get_pk_spline_a()

    tkk = halomod_trispectrum_1h(cosmo, hmc, np.exp(lk_arr), a_arr,
                                 prof, prof2=prof2,
                                 prof12_2pt=prof12_2pt,
                                 prof3=prof3, prof4=prof4,
                                 prof34_2pt=prof34_2pt)

    tkk, use_log = _logged_output(tkk, log=use_log)

    return Tk3D(a_arr=a_arr, lk_arr=lk_arr, tkk_arr=tkk,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik, is_logt=use_log)


def halomod_Tk3D_SSC_linear_bias(cosmo, hmc, *, prof,
                                 bias1=1, bias2=1, bias3=1, bias4=1,
                                 is_number_counts1=False,
                                 is_number_counts2=False,
                                 is_number_counts3=False,
                                 is_number_counts4=False,
                                 p_of_k_a=None, lk_arr=None,
                                 a_arr=None, extrap_order_lok=1,
                                 extrap_order_hik=1, use_log=False,
                                 extrap_pk=False):
    """ Returns a :class:`~pyccl.tk3d.Tk3D` object containing
    the super-sample covariance trispectrum, given by the tensor
    product of the power spectrum responses associated with the
    two pairs of quantities being correlated. Each response is
    calculated as:

    .. math::
        \\frac{\\partial P_{u,v}(k)}{\\partial\\delta_L} = b_u b_v \\left(
        \\left(\\frac{68}{21}-\\frac{1}{3}\\frac{d\\log k^3P_L(k)}{d\\log k}\\right)
        P_L(k)+I^1_2(k|u,v)\\right) - (b_{u} + b_{v}) P_{u,v}(k)

    where the :math:`I^1_2` is defined in the documentation
    :meth:`~pyccl.halos.halo_model.HMCalculator.I_1_2` and :math:`b_{u}`
    and :math:`b_{v}` are the linear halo biases for quantities :math:`u`
    and :math:`v`, respectively. The second term is only included if the
    corresponding profiles do not represent number counts.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        prof (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): a halo
            profile representing the matter overdensity.
        bias1 (:obj:`float` or `array`): linear galaxy bias for quantity 1.
            If an array, it has to have the shape of ``a_arr``.
        bias2 (:obj:`float` or `array`): linear galaxy bias for quantity 2.
        bias3 (:obj:`float` or `array`): linear galaxy bias for quantity 3.
        bias4 (:obj:`float` or `array`): linear galaxy bias for quantity 4.
        is_number_counts1 (:obj:`bool`): If ``True``, quantity 1 will be considered
            number counts and the clustering counter terms computed.
        is_number_counts2 (:obj:`bool`): as ``is_number_counts1`` but for quantity 2.
        is_number_counts3 (:obj:`bool`): as ``is_number_counts1`` but for quantity 3.
        is_number_counts4 (:obj:`bool`): as ``is_number_counts1`` but for quantity 4.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If ``None``,
            the power spectrum stored within ``cosmo`` will be used.
        a_arr (array): an array holding values of the scale factor
            at which the trispectrum should be calculated for
            interpolation. If ``None``, the internal values used
            by ``cosmo`` will be used.
        lk_arr (array): an array holding values of the natural
            logarithm of the wavenumber (in units of
            :math:`{\\rm Mpc}^{-1}`) at which the trispectrum should be
            calculated for interpolation. If ``None``, the internal values
            used by ``cosmo`` will be used.
        extrap_order_lok (:obj:`int`): extrapolation order to be used on
            k-values below the minimum of the splines. See
            :class:`~pyccl.tk3d.Tk3D`.
        extrap_order_hik (:obj:`int`): extrapolation order to be used on
            k-values above the maximum of the splines. See
            :class:`~pyccl.tk3d.Tk3D`.
        use_log (:obj:`bool`): if ``True``, the trispectrum will be
            interpolated in log-space (unless negative or
            zero values are found).
        extrap_pk (:obj:`bool`):
            Whether to extrapolate ``p_of_k_a`` in case ``a`` is out of its
            support. If ``False``, and the queried values are out of bounds,
            an error is raised. The default is ``False``.

    Returns:
        :class:`~pyccl.tk3d.Tk3D`: SSC effective trispectrum.
    """ # noqa
    if lk_arr is None:
        lk_arr = cosmo.get_pk_spline_lk()
    if a_arr is None:
        a_arr = cosmo.get_pk_spline_a()

    if not isinstance(prof, HaloProfileNFW):
        raise TypeError("prof should be HaloProfileNFW.")

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


def halomod_Tk3D_SSC(
        cosmo, hmc, prof, *, prof2=None, prof3=None, prof4=None,
        prof12_2pt=None, prof34_2pt=None,
        p_of_k_a=None, lk_arr=None, a_arr=None,
        extrap_order_lok=1, extrap_order_hik=1, use_log=False,
        extrap_pk=False):
    """ Returns a :class:`~pyccl.tk3d.Tk3D` object containing
    the super-sample covariance trispectrum, given by the tensor
    product of the power spectrum responses associated with the
    two pairs of quantities being correlated. Each response is
    calculated as:

    .. math::
        \\frac{\\partial P_{u,v}(k)}{\\partial\\delta_L} =
        \\left(\\frac{68}{21}-\\frac{1}{3}\\frac{d\\log k^3P_L(k)}{d\\log k}
        \\right)P_L(k)I^1_1(k,|u)I^1_1(k,|v)+I^1_2(k|u,v) - (b_{u} + b_{v})
        P_{u,v}(k)

    where the :math:`I^a_b` are defined in the documentation
    of :meth:`~pyccl.halos.halo_model.HMCalculator.I_1_1` and
    :meth:`~pyccl.halos.halo_model.HMCalculator.I_1_2` and
    :math:`b_{u}` and :math:`b_{v}` are the linear halo biases for
    quantities :math:`u` and :math:`v`, respectively (zero if the
    profiles are not number counts).

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
        hmc (:class:`~pyccl.halos.halo_model.HMCalculator`):
            a halo model calculator.
        prof (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile.
        prof2 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile (corresponding to :math:`u_2` above). If ``None``,
            ``prof`` will be used as ``prof2``.
        prof3 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile (corresponding to :math:`v_1` above). If ``None``,
            ``prof`` will be used as ``prof3``.
        prof4 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile (corresponding to :math:`v_2` above). If ``None``,
            ``prof2`` will be used as ``prof4``.
        prof12_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object returning the the two-point
            moment of ``prof`` and ``prof2``. If ``None``, the default
            second moment will be used, corresponding to the
            products of the means of both profiles.
        prof34_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as ``prof12_2pt`` for ``prof3`` and ``prof4``. If ``None``,
            ``prof12_2pt`` will be used.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If ``None``,
            the power spectrum stored within ``cosmo`` will be used.
        a_arr (array): an array holding values of the scale factor
            at which the trispectrum should be calculated for
            interpolation. If ``None``, the internal values used
            by ``cosmo`` will be used.
        lk_arr (array): an array holding values of the natural
            logarithm of the wavenumber (in units of
            :math:`{\\rm Mpc}^{-1}`) at which the trispectrum should
            be calculated for interpolation. If ``None``, the internal
            values used by ``cosmo`` will be used.
        extrap_order_lok (:obj:`int`): extrapolation order to be used on
            k-values below the minimum of the splines. See
            :class:`~pyccl.tk3d.Tk3D`.
        extrap_order_hik (:obj:`int`): extrapolation order to be used on
            k-values above the maximum of the splines. See
            :class:`~pyccl.tk3d.Tk3D`.
        use_log (:obj:`bool`): if ``True``, the trispectrum will be
            interpolated in log-space (unless negative or
            zero values are found).
        extrap_pk (:obj:`bool`):
            Whether to extrapolate ``p_of_k_a`` in case ``a`` is out of its
            support. If ``False``, and the queried values are out of bounds,
            an error is raised. The default is ``False``.

    Returns:
        :class:`~pyccl.tk3d.Tk3D`: SSC effective trispectrum.
    """
    if lk_arr is None:
        lk_arr = cosmo.get_pk_spline_lk()
    if a_arr is None:
        a_arr = cosmo.get_pk_spline_a()

    # define all the profiles
    prof, prof2, prof3, prof4, prof12_2pt, prof34_2pt = \
        _allocate_profiles(prof, prof2, prof3, prof4, prof12_2pt, prof34_2pt)

    k_use = np.exp(lk_arr)
    pk2d = cosmo.parse_pk(p_of_k_a)
    extrap = cosmo if extrap_pk else None  # extrapolation rule for pk2d

    dpk12, dpk34 = [np.zeros((len(a_arr), len(k_use))) for _ in range(2)]
    for ia, aa in enumerate(a_arr):
        # normalizations & I11 integral
        norm1 = prof.get_normalization(cosmo, aa, hmc=hmc)
        i11_1 = hmc.I_1_1(cosmo, k_use, aa, prof)

        if prof2 == prof:
            norm2 = norm1
            i11_2 = i11_1
        else:
            norm2 = prof2.get_normalization(cosmo, aa, hmc=hmc)
            i11_2 = hmc.I_1_1(cosmo, k_use, aa, prof2)

        if prof3 == prof:
            norm3 = norm1
            i11_3 = i11_1
        else:
            norm3 = prof3.get_normalization(cosmo, aa, hmc=hmc)
            i11_3 = hmc.I_1_1(cosmo, k_use, aa, prof3)

        if prof4 == prof2:
            norm4 = norm2
            i11_4 = i11_2
        else:
            norm4 = prof4.get_normalization(cosmo, aa, hmc=hmc)
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
    prof, prof2, prof3, prof4 = _allocate_profiles1pt(prof, prof2, prof3,
                                                      prof4)
    if prof12_2pt is None:
        prof12_2pt = Profile2pt()
    if prof34_2pt is None:
        prof34_2pt = prof12_2pt

    return prof, prof2, prof3, prof4, prof12_2pt, prof34_2pt


def _allocate_profiles1pt(prof, prof2, prof3, prof4):
    """Helper that controls how the undefined profiles are allocated."""
    if prof2 is None:
        prof2 = prof
    if prof3 is None:
        prof3 = prof
    if prof4 is None:
        prof4 = prof2

    return prof, prof2, prof3, prof4


def _allocate_profiles2(prof, prof2, prof3, prof4, prof13_2pt, prof14_2pt,
                        prof24_2pt, prof32_2pt):
    """Helper that controls how the undefined profiles are allocated."""
    prof, prof2, prof3, prof4 = \
        _allocate_profiles1pt(prof, prof2, prof3, prof4)

    if prof13_2pt is None:
        prof13_2pt = Profile2pt()

    if prof14_2pt is None:
        prof14_2pt = prof13_2pt

    if prof24_2pt is None:
        prof24_2pt = prof13_2pt

    if prof32_2pt is None:
        prof32_2pt = prof13_2pt

    return prof, prof2, prof3, prof4, prof13_2pt, prof14_2pt, prof24_2pt, \
        prof32_2pt


def _get_norms(prof, prof2, prof3, prof4, cosmo, aa, hmc):
    """Helper that returns the profiles normalization."""
    # Compute profile normalizations
    norm1 = prof.get_normalization(cosmo, aa, hmc=hmc)

    if prof2 == prof:
        norm2 = norm1
    else:
        norm2 = prof2.get_normalization(cosmo, aa, hmc=hmc)

    if prof3 == prof:
        norm3 = norm1
    elif prof3 == prof2:
        norm3 = norm2
    else:
        norm3 = prof3.get_normalization(cosmo, aa, hmc=hmc)

    if prof4 == prof:
        norm4 = norm1
    elif prof4 == prof2:
        norm4 = norm2
    elif prof4 == prof3:
        norm4 = norm3
    else:
        norm4 = prof4.get_normalization(cosmo, aa, hmc=hmc)

    return norm1, norm2, norm3, norm4


def _get_pk2d(p_of_k_a, cosmo):
    if isinstance(p_of_k_a, Pk2D):
        pk2d = p_of_k_a
    elif (p_of_k_a is None) or (str(p_of_k_a) == 'linear'):
        pk2d = cosmo.get_linear_power()
    elif str(p_of_k_a) == 'nonlinear':
        pk2d = cosmo.get_nonlin_power()
    else:
        raise TypeError("p_of_k_a must be `None`, \'linear\', "
                        "\'nonlinear\' or a `Pk2D` object")
    return pk2d


def _get_ints_I_1_1(hmc, cosmo, k_use, aa, prof, prof2, prof3, prof4):
    """Helper that returns the I_1_1 integrals for 4 profiles."""
    i1 = hmc.I_1_1(cosmo, k_use, aa, prof)[:, None]

    if prof2 == prof:
        i2 = i1
    else:
        i2 = hmc.I_1_1(cosmo, k_use, aa, prof2)[:, None]

    if prof3 == prof:
        i3 = i1.T
    elif prof3 == prof2:
        i3 = i2.T
    else:
        i3 = hmc.I_1_1(cosmo, k_use, aa, prof3)[None, :]

    if prof4 == prof:
        i4 = i1.T
    elif prof4 == prof2:
        i4 = i2.T
    elif prof4 == prof3:
        i4 = i3
    else:
        i4 = hmc.I_1_1(cosmo, k_use, aa, prof4)[None, :]

    return i1, i2, i3, i4


def _logged_output(*arrs, log):
    """Helper that logs the output if needed."""
    if not log:
        return *arrs, log
    is_negative = [(arr <= 0).any() for arr in arrs]
    if any(is_negative):
        warnings.warn("Some values were non-positive. "
                      "Interpolating linearly.",
                      category=CCLWarning, importance='high')
        return *arrs, False
    return *[np.log(arr) for arr in arrs], log


def halomod_trispectrum_2h_22(cosmo, hmc, k, a, prof, *, prof2=None,
                              prof3=None, prof4=None, prof13_2pt=None,
                              prof14_2pt=None, prof24_2pt=None,
                              prof32_2pt=None, p_of_k_a=None,
                              separable_growth=False):
    """ Computes the "22" term of the isotropized halo model 2-halo trispectrum
    for four profiles :math:`u_{1,2}`, :math:`v_{1,2}` as

    .. math::
        \\bar{T}^{2h}_{22}(k_1, k_2, a) = \\int \\frac{d\\varphi_1}{2\\pi}
        \\int \\frac{d\\varphi_2}{2\\pi}
        T^{2h,(22)}_{u_1,u_2;v_1,v_2}({\\bf k}_1,-{\\bf k}_1,
        {\\bf k}_2,-{\\bf k}_2),

    with

    .. math::
        T^{2h,(22)}_{u_1,u_2;v_1,v_2}(k_u,k_v,a) =
        \\langle P_{\\rm lin}(|{\\bf k}_u + {\\bf k}_v|)\\rangle_{\\varphi}\\,
        I^1_2(k_u, k_v|u_1,v_1)\\,
        I^1_2(k_u, k_v|u_2,v_2) + 1\\,{\\rm perm.}

    where :math:`\\langle\\cdots\\rangle_\\varphi` denotes averaging over the
    relative angle between the two wavevectors, and :math:`I^1_2` is defined in
    the documentation of :meth:`~pyccl.halos.halo_model.HMCalculator.I_1_2`.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        k (float or array_like): comoving wavenumber in Mpc^-1.
        a (float or array_like): scale factor.
        prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_1` above.
        prof2 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_2` above. If `None`,
            `prof` will be used as `prof2`.
        prof12_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object returning the the two-point
            moment of `prof` and `prof2`. If `None`, the default
            second moment will be used, corresponding to the
            products of the means of both profiles.
        prof3 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_1` above. If `None`,
            `prof` will be used as `prof3`.
        prof4 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_2` above. If `None`,
            `prof2` will be used as `prof4`.
        prof13_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof12_2pt` for `prof` and `prof3`.
        prof14_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof14_2pt` for `prof` and `prof4`.
        prof24_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof14_2pt` for `prof2` and `prof4`.
        prof32_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof14_2pt` for `prof3` and `prof2`.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If `None`, the power
            spectrum stored within `cosmo` will be used.
        separable_growth (bool): Indicates whether a separable
            growth function approximation can be used to calculate
            the isotropized power spectrum.

    Returns:
        float or array_like: integral values evaluated at each
        combination of `k` and `a`. The shape of the output will
        be `(N_a, N_k, N_k)` where `N_k` and `N_a` are the sizes of
        `k` and `a` respectively. The ordering is such that
        `output[ia, ik2, ik1] = T(k[ik1], k[ik2], a[ia])`
        If `k` or `a` are scalars, the corresponding dimension will
        be squeezed out on output.
    """
    a_use = np.atleast_1d(a)
    k_use = np.atleast_1d(k)

    # Check inputs
    prof, prof2, prof3, prof4, prof13_2pt, prof14_2pt, \
        prof24_2pt, prof32_2pt = _allocate_profiles2(prof, prof2, prof3, prof4,
                                                     prof13_2pt, prof14_2pt,
                                                     prof24_2pt, prof32_2pt)

    na = len(a_use)
    nk = len(k_use)

    # Power spectrum
    pk2d = _get_pk2d(p_of_k_a, cosmo)

    def get_isotropized_pkr(aa):
        def integ(theta):
            mu = np.cos(theta)
            k = np.sqrt(k_use[:, None]**2+k_use[None, :]**2
                        + 2*k_use[None, :]*k_use[:, None]*mu)
            kk = k.flatten()
            pk = pk2d(kk, aa, cosmo).reshape([nk, nk])
            return pk
        int_pk = scipy.integrate.quad_vec(integ, 0, np.pi)[0]
        return int_pk/np.pi

    out = np.zeros([na, nk, nk])
    if separable_growth:
        p_separable = get_isotropized_pkr(1.0)
    for ia, aa in enumerate(a_use):
        norm1, norm2, norm3, norm4 = _get_norms(prof, prof2, prof3, prof4,
                                                cosmo, aa, hmc)

        norm = norm1 * norm2 * norm3 * norm4
        if separable_growth:
            p = p_separable * (cosmo.growth_factor(aa)) ** 2
        else:
            p = get_isotropized_pkr(aa)

        # Compute trispectrum at this redshift
        # Permutation 0 is 0 due to P(k1 - k1 = 0) = 0

        # Permutation 1
        i13 = hmc.I_1_2(cosmo, k_use, aa, prof, prof2=prof3,
                        prof_2pt=prof13_2pt, diag=False)

        if (([prof2, prof4] == [prof, prof3]) or
           [prof2, prof4] == [prof3, prof]) and \
           (prof24_2pt == prof13_2pt):
            i24 = i13
        else:
            i24 = hmc.I_1_2(cosmo, k_use, aa, prof2, prof2=prof4,
                            prof_2pt=prof24_2pt, diag=False)
        # Permutation 2
        if (prof4 == prof3) and (prof14_2pt == prof13_2pt):
            i14 = i13
        elif (prof == prof2) and (prof14_2pt == prof24_2pt):
            i14 = i24
        else:
            i14 = hmc.I_1_2(cosmo, k_use, aa, prof, prof2=prof4,
                            prof_2pt=prof14_2pt, diag=False)

        if (prof2 == prof) and (prof32_2pt == prof13_2pt):
            i32 = i13.T
        elif (prof3 == prof4) and (prof32_2pt == prof24_2pt):
            i32 = i24.T
        elif (([prof3, prof2] == [prof, prof4]) or
              ([prof3, prof2] == [prof4, prof])) and \
             (prof32_2pt == prof14_2pt):
            i32 = i14.T
        else:
            i32 = hmc.I_1_2(cosmo, k_use, aa, prof3, prof2=prof2,
                            prof_2pt=prof32_2pt, diag=False)

        tk_2h_22 = p * (i13 * i24 + i14 * i32)
        # Normalize
        out[ia, :, :] = tk_2h_22 / norm

    if np.ndim(a) == 0:
        out = np.squeeze(out, axis=0)
    if np.ndim(k) == 0:
        out = np.squeeze(out, axis=-1)
        out = np.squeeze(out, axis=-1)
    return out


def halomod_trispectrum_2h_13(cosmo, hmc, k, a, prof, *,
                              prof2=None, prof3=None, prof4=None,
                              prof12_2pt=None, prof34_2pt=None,
                              p_of_k_a=None):
    """ Computes the "12" term of the isotropized halo model 2-halo trispectrum
    for four profiles :math:`u_{1,2}`, :math:`v_{1,2}` as

    .. math::
        \\bar{T}^{2h}_{13}(k_1, k_2, a) = \\int \\frac{d\\varphi_1}{2\\pi}
        \\int \\frac{d\\varphi_2}{2\\pi}
        T^{2h,(13)}_{u_1,u_2;v_1,v_2}({\\bf k}_1,
        -{\\bf k}_1,{\\bf k}_2,-{\\bf k}_2),

    with

    .. math::
        T^{2h,(13)}_{u_1,u_2;v_1,v_2}(k_u,k_v,a) =
        P_{\\rm lin}(k_u)\\, [I^1_1(k_u|u_1)\\,
        I^1_3(k_u,k_v,k_v|u_2,v_1,v_2)+(u_1\\leftrightarrow u_2)]+
        (u_i\\leftrightarrow v_i)

    where :math:`I^1_1` is defined in the documentation of
    :meth:`~pyccl.halos.halo_model.HMCalculator.I_1_1` and
    :math:`I^1_3` is defined in the documentation of
    :meth:`~pyccl.halos.halo_model.HMCalculator.I_1_3`.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        k (float or array_like): comoving wavenumber in Mpc^-1.
        a (float or array_like): scale factor.
        prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_1` above.
        prof2 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_2` above. If `None`,
            `prof` will be used as `prof2`.
        prof3 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_1` above. If `None`,
            `prof` will be used as `prof3`.
        prof4 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_2` above. If `None`,
            `prof2` will be used as `prof4`.
        prof12_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object returning the 2-point
            moment of `prof`, `prof2`. If `None`, the default second moment
            will be used, corresponding to the products of the means of each
            profile.
        prof34_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof34_2pt` for `prof3` and `prof4`.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If `None`, the power
            spectrum stored within `cosmo` will be used.

    Returns:
        float or array_like: integral values evaluated at each
        combination of `k` and `a`. The shape of the output will
        be `(N_a, N_k, N_k)` where `N_k` and `N_a` are the sizes of
        `k` and `a` respectively. The ordering is such that
        `output[ia, ik2, ik1] = T(k[ik1], k[ik2], a[ia])`
        If `k` or `a` are scalars, the corresponding dimension will
        be squeezed out on output.
    """
    a_use = np.atleast_1d(a)
    k_use = np.atleast_1d(k)

    # Check inputs
    prof, prof2, prof3, prof4, prof12_2pt, prof34_2pt, _, _ = \
        _allocate_profiles2(prof, prof2, prof3, prof4, prof12_2pt, prof34_2pt,
                            None, None)

    # Power spectrum
    pk2d = _get_pk2d(p_of_k_a, cosmo)

    na = len(a_use)
    nk = len(k_use)
    out = np.zeros([na, nk, nk])
    for ia, aa in enumerate(a_use):
        # Compute profile normalizations
        norm1, norm2, norm3, norm4 = _get_norms(prof, prof2, prof3, prof4,
                                                cosmo, aa, hmc)
        norm = norm1 * norm2 * norm3 * norm4

        # Compute trispectrum at this redshift
        p1 = pk2d(k_use, aa, cosmo)[None, :]
        i1 = hmc.I_1_1(cosmo, k_use, aa, prof)[None, :]
        i234 = hmc.I_1_3(cosmo, k_use, aa, prof2, prof2=prof3,
                         prof_2pt=prof34_2pt, prof3=prof4)
        # Permutation 1
        # p2 = p1  # (because k_a = k_b)
        if prof2 == prof:
            i2 = i1
            i134 = i234
        else:
            i2 = hmc.I_1_1(cosmo, k_use, aa, prof2)[None, :]
            i134 = hmc.I_1_3(cosmo, k_use, aa, prof, prof2=prof3,
                             prof_2pt=prof34_2pt, prof3=prof4)
        # Attention to axis order change!
        # Permutation 2
        p3 = p1.T
        if prof3 == prof:
            i3 = i1.T
        elif prof3 == prof2:
            i3 = i2.T
        else:
            i3 = hmc.I_1_1(cosmo, k_use, aa, prof3)[:, None]

        if (([prof, prof2] == [prof3, prof4] or
             [prof, prof2] == [prof4, prof3])) and prof2 == prof4 and \
                prof12_2pt == prof34_2pt:
            i124 = i234.T
        elif (([prof, prof2] == [prof3, prof4] or
               [prof, prof2] == [prof4, prof3]) and prof4 == prof) and \
                prof12_2pt == prof34_2pt:
            i124 = i134.T
        else:
            i124 = hmc.I_1_3(cosmo, k_use, aa, prof4, prof2=prof,
                             prof_2pt=prof12_2pt, prof3=prof2).T
        # Permutation 4
        # p4 = p3  # (because k_c = k_d)
        if prof4 == prof:
            i4 = i1.T
        elif prof4 == prof2:
            i4 = i2.T
        elif prof4 == prof3:
            i4 = i3.T
        else:
            i4 = hmc.I_1_1(cosmo, k_use, aa, prof3)[:, None]

        if prof3 == prof4:
            i123 = i124
        elif (([prof, prof2] == [prof3, prof4]) or
              [prof, prof2] == [prof4, prof3]) and \
             (prof12_2pt == prof34_2pt) and (prof3 == prof):
            i123 = i134.T
        elif (([prof, prof2] == [prof3, prof4]) or
              [prof, prof2] == [prof4, prof3]) and \
             (prof12_2pt == prof34_2pt) and (prof3 == prof2):
            i123 = i234.T
        else:
            i123 = hmc.I_1_3(cosmo, k_use, aa, prof3, prof2=prof,
                             prof_2pt=prof12_2pt, prof3=prof2).T
        ####

        tk_2h_13 = p1 * (i1 * i234 + i2 * i134) + p3 * (i3 * i124 + i4 * i123)

        # Normalize
        out[ia, :, :] = tk_2h_13 / norm

    if np.ndim(a) == 0:
        out = np.squeeze(out, axis=0)
    if np.ndim(k) == 0:
        out = np.squeeze(out, axis=-1)
        out = np.squeeze(out, axis=-1)
    return out


def halomod_trispectrum_3h(cosmo, hmc, k, a, prof, *, prof2=None,
                           prof3=None, prof4=None,
                           prof13_2pt=None, prof14_2pt=None,
                           prof24_2pt=None, prof32_2pt=None,
                           p_of_k_a=None, separable_growth=False):
    """ Computes the isotropized halo model 3-halo trispectrum for four
    profiles :math:`u_{1,2}`, :math:`v_{1,2}` as

    .. math::
        \\bar{T}^{3h}(k_1, k_2, a) = \\int \\frac{d\\varphi_1}{2\\pi}
        \\int \\frac{d\\varphi_2}{2\\pi}
        T^{3h}_{u_1,u_2;v_1,v_2}({\\bf k_1},
        -{\\bf k_1},{\\bf k_2},-{\\bf k_2}),

    with

    .. math::
        T^{3h}_{u_1,u_2;v_1,v_2}({\\bf k}_u,{\\bf k}_v,a) =
        B^{\\rm PT}({\\bf k}_u, -{\\bf k}_v,
                    -{\\bf k}_u+{\\bf k}_v)
        I^1_1(k_u | u_1) I^1_1(k_v | v_1) I^1_2(k_u, k_v|u_2,v_2) \\,
        + 3\\,{\\rm perm.}

    where :math:`I^1_1` and :math:`I^1_2` are defined in the documentation
    of :meth:`~pyccl.halos.halo_model.HMCalculator.I_1_1` and
    :meth:`~pyccl.halos.halo_model.HMCalculator.I_1_2`,
    respectively; and the tree-level bispectrum :math:`B^{PT}` is calculated
    according to Eq. 30 of `Takada et al. 2013
    <https://arxiv.org/abs/1302.6994>`_

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        k (float or array_like): comoving wavenumber in Mpc^-1.
        a (float or array_like): scale factor.
        prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_1` above.
        prof2 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_2` above. If `None`,
            `prof` will be used as `prof2`.
        prof3 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_1` above. If `None`,
            `prof` will be used as `prof3`.
        prof4 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_2` above. If `None`,
            `prof2` will be used as `prof4`.
        prof13_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object returning the the two-point
            moment of `prof` and `prof3`. If `None`, the default
            second moment will be used, corresponding to the
            products of the means of both profiles.
        prof14_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof14_2pt` for `prof` and `prof4`.
        prof24_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof14_2pt` for `prof2` and `prof4`.
        prof32_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof14_2pt` for `prof3` and `prof2`.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If `None`, the power
            spectrum stored within `cosmo` will be used.
        separable_growth (bool): Indicates whether a separable
            growth function approximation can be used to calculate
            the isotropized power spectrum.

    Returns:
        float or array_like: integral values evaluated at each
        combination of `k` and `a`. The shape of the output will
        be `(N_a, N_k, N_k)` where `N_k` and `N_a` are the sizes of
        `k` and `a` respectively. The ordering is such that
        `output[ia, ik2, ik1] = T(k[ik1], k[ik2], a[ia])`
        If `k` or `a` are scalars, the corresponding dimension will
        be squeezed out on output.
    """
    a_use = np.atleast_1d(a)
    k_use = np.atleast_1d(k)

    # Check inputs
    prof, prof2, prof3, prof4, prof13_2pt, prof14_2pt, \
        prof24_2pt, prof32_2pt = _allocate_profiles2(prof, prof2, prof3, prof4,
                                                     prof13_2pt, prof14_2pt,
                                                     prof24_2pt, prof32_2pt)

    # Power spectrum
    pk2d = _get_pk2d(p_of_k_a, cosmo)

    # Compute bispectrum
    # Encapsulate code in a function
    def get_kr_and_f2(theta):
        cth = np.cos(theta)
        kk = k_use[None, :]
        kp = k_use[:, None]
        kr2 = kk ** 2 + kp ** 2 + 2 * kk * kp * cth
        kr = np.sqrt(kr2)

        f2 = 5./7. - 0.5 * (1 + kk ** 2 / kr2) * (1 + kp / kk * cth) + \
            2/7. * kk ** 2 / kr2 * (1 + kp / kk * cth)**2
        # When kr = 0:
        # k^2 / kr^2 (1 + k / kr cos) -> k^2/(2k^2 + 2k^2 cos)*(1 + cos) = 1/2
        # k^2 / kr^2 (1 + k / kr cos)^2 -> (1 + cos)/2 = 0
        f2[np.where(kr == 0)] = 13. / 28

        return kr, f2

    def get_Bpt(a):
        # We only need to compute the independent k * k * cos(theta) since Pk
        # only depends on the module of ki + kj
        pk = pk2d(k_use, a, cosmo)[None, :]

        def integ(theta):
            kr, f2 = get_kr_and_f2(theta)
            pkr = pk2d(kr.flatten(), a, cosmo).reshape(kr.shape)
            return pkr * f2
        P3 = scipy.integrate.quad_vec(integ, 0, np.pi)[0] / np.pi

        Bpt = 6. / 7. * pk * pk.T + 2 * pk * P3
        Bpt += Bpt.T

        return Bpt

    na = len(a_use)
    nk = len(k_use)

    out = np.zeros([na, nk, nk])

    if separable_growth:
        Bpt_separable = get_Bpt(1.0)

    for ia, aa in enumerate(a_use):
        # Compute profile normalizations
        norm1, norm2, norm3, norm4 = _get_norms(prof, prof2, prof3, prof4,
                                                cosmo, aa, hmc)
        norm = norm1 * norm2 * norm3 * norm4

        # Permutation 0 is 0 due to Bpt_1_2_34=0
        i1, i2, i3, i4 = _get_ints_I_1_1(hmc, cosmo, k_use, aa, prof, prof2,
                                         prof3, prof4)

        # Permutation 1: 2 <-> 3
        i24 = hmc.I_1_2(cosmo, k_use, aa, prof2, prof2=prof4,
                        prof_2pt=prof24_2pt, diag=False)
        # Permutation 2: 2 <-> 4
        if (prof3 == prof4) and (prof32_2pt == prof24_2pt):
            i32 = i24.T
        else:
            i32 = hmc.I_1_2(cosmo, k_use, aa, prof3, prof2=prof2,
                            prof_2pt=prof32_2pt, diag=False)
        # Permutation 3: 1 <-> 3
        if (prof == prof2) and (prof14_2pt == prof24_2pt):
            i14 = i24
        elif ([prof, prof4] == [prof3, prof2]) and (prof14_2pt == prof32_2pt):
            i14 = i32
        elif ([prof, prof4] == [prof2, prof3]) and (prof14_2pt == prof32_2pt):
            i14 = i32.T
        else:
            i14 = hmc.I_1_2(cosmo, k_use, aa, prof, prof2=prof4,
                            prof_2pt=prof14_2pt, diag=False)
        # Permutation 4: 1 <-> 4
        if (prof == prof2) and (prof13_2pt == prof32_2pt):
            i31 = i32
        elif prof3 == prof4 and (prof13_2pt == prof32_2pt):
            i31 = i14.T
        elif ([prof3, prof] == [prof2, prof4]) and (prof13_2pt == prof24_2pt):
            i31 = i24
        elif ([prof3, prof] == [prof4, prof2]) and (prof13_2pt == prof24_2pt):
            i31 = i24.T
        else:
            i31 = hmc.I_1_2(cosmo, k_use, aa, prof3, prof2=prof,
                            prof_2pt=prof13_2pt, diag=False)

        # Permutation 5: 12 <-> 34 is 0 due to Bpt_3_4_12=0
        if separable_growth:
            Bpt = Bpt_separable * (cosmo.growth_factor(aa)) ** 4
        else:
            Bpt = get_Bpt(aa)

        tk_3h = Bpt * (i1 * i3 * i24 + i1 * i4 * i32 +
                       i3 * i2 * i14 + i4 * i2 * i31)

        # Normalize
        out[ia, :, :] = tk_3h / norm

    if np.ndim(a) == 0:
        out = np.squeeze(out, axis=0)
    if np.ndim(k) == 0:
        out = np.squeeze(out, axis=-1)
        out = np.squeeze(out, axis=-1)

    return out


def halomod_trispectrum_4h(cosmo, hmc, k, a, prof, prof2=None, prof3=None,
                           prof4=None, p_of_k_a=None, separable_growth=False):
    """ Computes the isotropized halo model 4-halo trispectrum for four
    profiles :math:`u_{1,2}`, :math:`v_{1,2}` as

    .. math::
        \\bar{T}^{4h}(k_1, k_2, a) = \\int \\frac{d\\varphi_1}{2\\pi}
        \\int \\frac{d\\varphi_2}{2\\pi}
        T^{4h}_{u_1,u_2;v_1,v_2}({\\bf k_1},-{\\bf k_1},
        {\\bf k_2},-{\\bf k_2}),

    with

    .. math::
        T^{4h}_{u_1,u_2;v_1,v_2}({\\bf k}_u,{\\bf k}_v,a) =
        T^{PT}({\\bf k}_u, -{\\bf k}_u, {\\bf k}_v, -{\\bf k}_v) \\,
        I^1_1(k_u | u_1) I^1_1(k_u | u_2) I^1_1(k_v | v_1)
        I^1_1(k_v | v_2) \\,

    where :math:`I^1_1` is defined in the documentation
    of :meth:`~pyccl.halos.halo_model.HMCalculator.I_1_1`, and
    the tree-level trispectrum :math:`T^{PT}` is calculated
    according to Eq. 30 of `Takada et al. 2013
    <https://arxiv.org/abs/1302.6994>`_

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        k (float or array_like): comoving wavenumber in Mpc^-1.
        a (float or array_like): scale factor.
        prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_1` above.
        prof2 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_2` above. If `None`,
            `prof` will be used as `prof2`.
        prof3 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_1` above. If `None`,
            `prof` will be used as `prof3`.
        prof4 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_2` above. If `None`,
            `prof2` will be used as `prof4`.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If `None`, the power
            spectrum stored within `cosmo` will be used.
        separable_growth (bool): Indicates whether a separable
            growth function approximation can be used to calculate
            the isotropized power spectrum.

    Returns:
        float or array_like: integral values evaluated at each
        combination of `k` and `a`. The shape of the output will
        be `(N_a, N_k, N_k)` where `N_k` and `N_a` are the sizes of
        `k` and `a` respectively. The ordering is such that
        `output[ia, ik2, ik1] = T(k[ik1], k[ik2], a[ia])`
        If `k` or `a` are scalars, the corresponding dimension will
        be squeezed out on output.
    """
    a_use = np.atleast_1d(a)
    k_use = np.atleast_1d(k)

    # Check inputs
    prof, prof2, prof3, prof4 = \
        _allocate_profiles1pt(prof, prof2, prof3, prof4)

    na = len(a_use)
    nk = len(k_use)

    # Power spectrum
    pk2d = _get_pk2d(p_of_k_a, cosmo)

    kk = k_use[None, :]
    kp = k_use[:, None]

    def get_P4A_P4X(a):
        k = kk

        def integ(theta):
            cth = np.cos(theta)
            kr2 = k ** 2 + kp ** 2 + 2 * k * kp * cth
            kr = np.sqrt(kr2)

            f2 = 5./7. - 0.5 * (1 + k ** 2 / kr2) * (1 + kp / k * cth) + \
                2/7. * k ** 2 / kr2 * (1 + kp / k * cth)**2
            f2[np.where(kr == 0)] = 13. / 28

            pkr = pk2d(kr.flatten(), a, cosmo).reshape((nk, nk))
            return np.array([pkr * f2**2, pkr * f2 * f2.T])
        P4A, P4X = scipy.integrate.quad_vec(integ, 0, np.pi)[0] / np.pi

        return P4A, P4X

    def get_X():
        k = kk
        r = kp / k

        def integ(theta):
            cth = np.cos(theta)
            kr2 = k ** 2 + kp ** 2 + 2 * k * kp * cth
            kr = np.sqrt(kr2)
            intd = (5 * r + (7 - 2*r**2)*cth) / (1 + r**2 + 2*r*cth) * \
                   (3/7. * r + 0.5 * (1 + r**2) * cth + 4/7. * r * cth**2)
            # When kr = 0, r = 1 and intd = 0
            intd[np.where(kr == 0)] = 0
            return intd

        isotropized_integ = \
            scipy.integrate.quad_vec(integ, 0, np.pi)[0] / np.pi

        X = -7./4. * (1 + r**2) + isotropized_integ

        return X

    X = get_X()
    out = np.zeros([na, nk, nk])
    if separable_growth:
        pk_separable = pk2d(k_use, 1.0, cosmo)[None, :]
        P4A_separable, P4X_separable = get_P4A_P4X(1.0)

    for ia, aa in enumerate(a_use):
        # Compute profile normalizations
        norm1, norm2, norm3, norm4 = _get_norms(prof, prof2, prof3, prof4,
                                                cosmo, aa, hmc)
        norm = norm1 * norm2 * norm3 * norm4

        if separable_growth:
            pk = pk_separable * (cosmo.growth_factor(aa)) ** 2
            P4A = P4A_separable * (cosmo.growth_factor(aa)) ** 2
            P4X = P4X_separable * (cosmo.growth_factor(aa)) ** 2
        else:
            pk = pk2d(k_use, aa, cosmo)[None, :]
            P4A, P4X = get_P4A_P4X(aa)

        t1113 = 4/9. * pk**2 * pk.T * X
        t1113 += t1113.T

        t1122 = 8 * (pk**2 * P4A + pk * pk.T * P4X)
        t1122 += t1122.T

        # Now the halo model integrals
        i1, i2, i3, i4 = _get_ints_I_1_1(hmc, cosmo, k_use, aa, prof, prof2,
                                         prof3, prof4)

        tk_4h = i1 * i2 * i3 * i4 * (t1113 + t1122)

        # Normalize
        out[ia, :, :] = tk_4h / norm

    if np.ndim(a) == 0:
        out = np.squeeze(out, axis=0)
    if np.ndim(k) == 0:
        out = np.squeeze(out, axis=-1)
        out = np.squeeze(out, axis=-1)

    return out


def halomod_Tk3D_2h(cosmo, hmc,
                    prof, prof2=None,
                    prof3=None, prof4=None,
                    prof12_2pt=None, prof13_2pt=None, prof14_2pt=None,
                    prof24_2pt=None, prof32_2pt=None, prof34_2pt=None,
                    p_of_k_a=None,
                    lk_arr=None, a_arr=None,
                    extrap_order_lok=1, extrap_order_hik=1, use_log=False,
                    separable_growth=False):
    """ Returns a :class:`~pyccl.tk3d.Tk3D` object containing the 2-halo
    trispectrum for four quantities defined by their respective halo profiles.
    See :meth:`halomod_trispectrum_1h` for more details about the actual
    calculation.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_1` above.
        prof2 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_2` above. If `None`,
            `prof` will be used as `prof2`.
        prof3 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_1` above. If `None`,
            `prof` will be used as `prof3`.
        prof4 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_2` above. If `None`,
            `prof2` will be used as `prof4`.
        prof12_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object returning the the two-point
            moment of `prof` and `prof2`. If `None`, the default
            second moment will be used, corresponding to the
            products of the means of both profiles.
        prof13_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof12_2pt` for `prof` and `prof3`.
        prof14_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof14_2pt` for `prof` and `prof4`.
        prof24_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof14_2pt` for `prof2` and `prof4`.
        prof32_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof14_2pt` for `prof3` and `prof2`.
        prof34_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof34_2pt` for `prof3` and `prof4`.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If `None`, the power
            spectrum stored within `cosmo` will be used.
        a_arr (array): an array holding values of the scale factor
            at which the trispectrum should be calculated for
            interpolation. If `None`, the internal values used
            by `cosmo` will be used.
        lk_arr (array): an array holding values of the natural
            logarithm of the wavenumber (in units of Mpc^-1) at
            which the trispectrum should be calculated for
            interpolation. If `None`, the internal values used
            by `cosmo` will be used.
        extrap_order_lok (int): extrapolation order to be used on
            k-values below the minimum of the splines. See
            :class:`~pyccl.tk3d.Tk3D`.
        extrap_order_hik (int): extrapolation order to be used on
            k-values above the maximum of the splines. See
            :class:`~pyccl.tk3d.Tk3D`.
        use_log (bool): if `True`, the trispectrum will be
            interpolated in log-space (unless negative or
            zero values are found).
        separable_growth (bool): Indicates whether a separable
            growth function approximation can be used to calculate
            the isotropized power spectrum.

    Returns:
        :class:`~pyccl.tk3d.Tk3D`: 2-halo trispectrum.
    """
    if lk_arr is None:
        lk_arr = cosmo.get_pk_spline_lk()
    if a_arr is None:
        a_arr = cosmo.get_pk_spline_a()

    tkk_2h_22 = halomod_trispectrum_2h_22(cosmo, hmc, np.exp(lk_arr), a_arr,
                                          prof, prof2=prof2,
                                          prof3=prof3, prof4=prof4,
                                          prof13_2pt=prof13_2pt,
                                          prof14_2pt=prof14_2pt,
                                          prof24_2pt=prof24_2pt,
                                          prof32_2pt=prof32_2pt,
                                          p_of_k_a=p_of_k_a,
                                          separable_growth=separable_growth)

    tkk_2h_13 = halomod_trispectrum_2h_13(cosmo, hmc, np.exp(lk_arr), a_arr,
                                          prof, prof2=prof2,
                                          prof3=prof3, prof4=prof4,
                                          prof12_2pt=prof12_2pt,
                                          prof34_2pt=prof34_2pt,
                                          p_of_k_a=p_of_k_a)

    tkk = tkk_2h_22 + tkk_2h_13

    tkk, use_log = _logged_output(tkk, log=use_log)

    tk3d = Tk3D(a_arr=a_arr, lk_arr=lk_arr, tkk_arr=tkk,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik, is_logt=use_log)
    return tk3d


def halomod_Tk3D_3h(cosmo, hmc,
                    prof, prof2=None, prof3=None, prof4=None,
                    prof13_2pt=None, prof14_2pt=None, prof24_2pt=None,
                    prof32_2pt=None,
                    lk_arr=None, a_arr=None, p_of_k_a=None,
                    extrap_order_lok=1, extrap_order_hik=1,
                    use_log=False, separable_growth=False):
    """ Returns a :class:`~pyccl.tk3d.Tk3D` object containing
    the 3-halo trispectrum for four quantities defined by
    their respective halo profiles. See :meth:`halomod_trispectrum_3h`
    for more details about the actual calculation.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_1` above.
        prof2 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_2` above. If `None`,
            `prof` will be used as `prof2`.
        prof3 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_1` above. If `None`,
            `prof` will be used as `prof3`.
        prof4 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_2` above. If `None`,
            `prof2` will be used as `prof4`.
        prof13_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object returning the the two-point
            moment of `prof` and `prof3`. If `None`, the default
            second moment will be used, corresponding to the
            products of the means of both profiles.
        prof14_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof14_2pt` for `prof` and `prof4`.
        prof24_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof14_2pt` for `prof2` and `prof4`.
        prof32_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof14_2pt` for `prof3` and `prof2`.
        lk_arr (array): an array holding values of the natural
            logarithm of the wavenumber (in units of Mpc^-1) at
            which the trispectrum should be calculated for
            interpolation. If `None`, the internal values used
            by `cosmo` will be used.
        a_arr (array): an array holding values of the scale factor
            at which the trispectrum should be calculated for
            interpolation. If `None`, the internal values used
            by `cosmo` will be used.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If `None`, the power
            spectrum stored within `cosmo` will be used.
        extrap_order_lok (int): extrapolation order to be used on
            k-values below the minimum of the splines. See
            :class:`~pyccl.tk3d.Tk3D`.
        extrap_order_hik (int): extrapolation order to be used on
            k-values above the maximum of the splines. See
            :class:`~pyccl.tk3d.Tk3D`.
        use_log (bool): if `True`, the trispectrum will be
            interpolated in log-space (unless negative or
            zero values are found).
        separable_growth (bool): Indicates whether a separable
            growth function approximation can be used to calculate
            the isotropized power spectrum.

    Returns:
        :class:`~pyccl.tk3d.Tk3D`: 3-halo trispectrum.
    """
    if lk_arr is None:
        lk_arr = cosmo.get_pk_spline_lk()
    if a_arr is None:
        a_arr = cosmo.get_pk_spline_a()

    tkk = halomod_trispectrum_3h(cosmo, hmc, np.exp(lk_arr), a_arr,
                                 prof=prof,
                                 prof2=prof2,
                                 prof3=prof3,
                                 prof4=prof4,
                                 prof13_2pt=prof13_2pt,
                                 prof14_2pt=prof14_2pt,
                                 prof24_2pt=prof24_2pt,
                                 prof32_2pt=prof32_2pt,
                                 p_of_k_a=p_of_k_a,
                                 separable_growth=separable_growth)

    tkk, use_log = _logged_output(tkk, log=use_log)

    tk3d = Tk3D(a_arr=a_arr, lk_arr=lk_arr, tkk_arr=tkk,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik, is_logt=use_log)
    return tk3d


def halomod_Tk3D_4h(cosmo, hmc,
                    prof, prof2=None, prof3=None, prof4=None,
                    lk_arr=None, a_arr=None, p_of_k_a=None,
                    extrap_order_lok=1, extrap_order_hik=1,
                    use_log=False, separable_growth=False):
    """ Returns a :class:`~pyccl.tk3d.Tk3D` object containing
    the 3-halo trispectrum for four quantities defined by
    their respective halo profiles. See :meth:`halomod_trispectrum_4h`
    for more details about the actual calculation.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_1` above.
        prof2 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_2` above. If `None`,
            `prof` will be used as `prof2`.
        prof3 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_1` above. If `None`,
            `prof` will be used as `prof3`.
        prof4 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_2` above. If `None`,
            `prof2` will be used as `prof4`.
        lk_arr (array): an array holding values of the natural
            logarithm of the wavenumber (in units of Mpc^-1) at
            which the trispectrum should be calculated for
            interpolation. If `None`, the internal values used
            by `cosmo` will be used.
        a_arr (array): an array holding values of the scale factor
            at which the trispectrum should be calculated for
            interpolation. If `None`, the internal values used
            by `cosmo` will be used.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If `None`, the power
            spectrum stored within `cosmo` will be used.
        extrap_order_lok (int): extrapolation order to be used on
            k-values below the minimum of the splines. See
            :class:`~pyccl.tk3d.Tk3D`.
        extrap_order_hik (int): extrapolation order to be used on
            k-values above the maximum of the splines. See
            :class:`~pyccl.tk3d.Tk3D`.
        use_log (bool): if `True`, the trispectrum will be
            interpolated in log-space (unless negative or
            zero values are found).
        separable_growth (bool): Indicates whether a separable
            growth function approximation can be used to calculate
            the isotropized power spectrum.

    Returns:
        :class:`~pyccl.tk3d.Tk3D`: 4-halo trispectrum.
    """
    if lk_arr is None:
        lk_arr = cosmo.get_pk_spline_lk()
    if a_arr is None:
        a_arr = cosmo.get_pk_spline_a()

    tkk = halomod_trispectrum_4h(cosmo, hmc, np.exp(lk_arr), a_arr,
                                 prof=prof,
                                 prof2=prof2,
                                 prof3=prof3,
                                 prof4=prof4,
                                 p_of_k_a=None,
                                 separable_growth=separable_growth)

    tkk, use_log = _logged_output(tkk, log=use_log)

    tk3d = Tk3D(a_arr=a_arr, lk_arr=lk_arr, tkk_arr=tkk,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik, is_logt=use_log)
    return tk3d


def halomod_Tk3D_cNG(cosmo, hmc, prof, prof2=None, prof3=None, prof4=None,
                     prof12_2pt=None, prof13_2pt=None, prof14_2pt=None,
                     prof24_2pt=None, prof32_2pt=None, prof34_2pt=None,
                     p_of_k_a=None,
                     lk_arr=None, a_arr=None, extrap_order_lok=1,
                     extrap_order_hik=1, use_log=False,
                     separable_growth=False):
    """ Returns a :class:`~pyccl.tk3d.Tk3D` object containing the non-Gaussian
    covariance trispectrum for four quantities defined by their respective halo
    profiles. This is the sum of the trispectrum terms 1h + 2h + 3h + 4h.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_1` above.
        prof2 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_2` above. If `None`,
            `prof` will be used as `prof2`.
        prof3 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_1` above. If `None`,
            `prof` will be used as `prof3`.
        prof4 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_2` above. If `None`,
            `prof2` will be used as `prof4`.
        prof12_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object returning the the two-point
            moment of `prof` and `prof2`. If `None`, the default
            second moment will be used, corresponding to the
            products of the means of both profiles.
        prof13_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof12_2pt` for `prof` and `prof3`.
        prof14_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof12_2pt` for `prof` and `prof4`.
        prof24_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof12_2pt` for `prof2` and `prof4`.
        prof32_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof12_2pt` for `prof3` and `prof2`.
        prof34_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof12_2pt` for `prof3` and `prof4`.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If `None`, the power
            spectrum stored within `cosmo` will be used.
        a_arr (array): an array holding values of the scale factor
            at which the trispectrum should be calculated for
            interpolation. If `None`, the internal values used
            by `cosmo` will be used.
        lk_arr (array): an array holding values of the natural
            logarithm of the wavenumber (in units of Mpc^-1) at
            which the trispectrum should be calculated for
            interpolation. If `None`, the internal values used
            by `cosmo` will be used.
        extrap_order_lok (int): extrapolation order to be used on
            k-values below the minimum of the splines. See
            :class:`~pyccl.tk3d.Tk3D`.
        extrap_order_hik (int): extrapolation order to be used on
            k-values above the maximum of the splines. See
            :class:`~pyccl.tk3d.Tk3D`.
        use_log (bool): if `True`, the trispectrum will be
            interpolated in log-space (unless negative or
            zero values are found).
        separable_growth (bool): Indicates whether a separable
            growth function approximation can be used to calculate
            the isotropized power spectrum.

    Returns:
        :class:`~pyccl.tk3d.Tk3D`: 2-halo trispectrum.
    """
    if lk_arr is None:
        lk_arr = cosmo.get_pk_spline_lk()
    if a_arr is None:
        a_arr = cosmo.get_pk_spline_a()

    tkk = halomod_trispectrum_1h(cosmo, hmc, np.exp(lk_arr), a_arr,
                                 prof, prof2=prof2,
                                 prof12_2pt=prof12_2pt,
                                 prof3=prof3, prof4=prof4,
                                 prof34_2pt=prof34_2pt)

    tkk += halomod_trispectrum_2h_22(cosmo, hmc, np.exp(lk_arr), a_arr,
                                     prof, prof2=prof2,
                                     prof3=prof3, prof4=prof4,
                                     prof13_2pt=prof13_2pt,
                                     prof14_2pt=prof14_2pt,
                                     prof24_2pt=prof24_2pt,
                                     prof32_2pt=prof32_2pt,
                                     p_of_k_a=p_of_k_a,
                                     separable_growth=separable_growth)

    tkk += halomod_trispectrum_2h_13(cosmo, hmc, np.exp(lk_arr), a_arr,
                                     prof, prof2=prof2,
                                     prof3=prof3, prof4=prof4,
                                     prof12_2pt=prof12_2pt,
                                     prof34_2pt=prof34_2pt,
                                     p_of_k_a=p_of_k_a)

    tkk += halomod_trispectrum_3h(cosmo, hmc, np.exp(lk_arr), a_arr,
                                  prof=prof,
                                  prof2=prof2,
                                  prof3=prof3,
                                  prof4=prof4,
                                  prof13_2pt=prof13_2pt,
                                  prof14_2pt=prof14_2pt,
                                  prof24_2pt=prof24_2pt,
                                  prof32_2pt=prof32_2pt,
                                  p_of_k_a=p_of_k_a,
                                  separable_growth=separable_growth)

    tkk += halomod_trispectrum_4h(cosmo, hmc, np.exp(lk_arr), a_arr,
                                  prof=prof,
                                  prof2=prof2,
                                  prof3=prof3,
                                  prof4=prof4,
                                  p_of_k_a=p_of_k_a,
                                  separable_growth=separable_growth)

    tkk, use_log = _logged_output(tkk, log=use_log)

    tk3d = Tk3D(a_arr=a_arr, lk_arr=lk_arr, tkk_arr=tkk,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik, is_logt=use_log)
    return tk3d
