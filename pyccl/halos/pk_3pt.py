__all__ = ("halomod_bispectrum_squeezed", "halomod_Pk2D_bispectrum_squeezed",
           "halomod_bispectrum_squeezed_SSC",
           "halomod_Pk2D_bispectrum_squeezed_SSC",)

import numpy as np

from .. import Pk2D
from . import HaloProfileSelection, Profile2pt, Profile2ptSelection


def halomod_bispectrum_squeezed(cosmo, hmc, k, a, prof, *, prof2,
                                prof3=None, prof12_2pt=None, prof23_2pt=None,
                                p_of_k_a=None, normalize_counts=False,
                                get_1h=True, get_2h=True, extrap_pk=False):
    """ Computes the halo model bispectrum between the number counts of
    selected halos :math:`N` (e.g. clusters) and two quantities :math:`u`
    and :math:`v` defined by their respective halo profiles, in the limit
    where the wavenumber of the counts is much smaller than :math:`k`
    (see `Takada & Bridle 2007 <https://arxiv.org/abs/0705.0163>`_):

    .. math::
        B_{N,u,v}(k,a) = I^0_3(k,a|N,u,v) +
        \\left[I^1_1(k,a|u)\\,I^1_2(k,a|N,v) +
        I^1_1(k,a|v)\\,I^1_2(k,a|N,u)\\right]\\,P_{\\rm lin}(k,a),

    where :math:`I^a_b` are defined in the documentation of
    :class:`~pyccl.halos.halo_model.HMCalculator`. The remaining 2- and
    3-halo terms give the super-sample contribution (see
    :meth:`halomod_bispectrum_squeezed_SSC`). The covariance between the
    angular power spectrum of :math:`u` and :math:`v` and the angular
    number density of selected halos is then

    .. math::
        {\\rm Cov}(\\ell) = \\frac{1}{4\\pi f_{\\rm sky}}
        \\int \\frac{d\\chi}{\\chi^2}\\,
        \\Delta^u_\\ell(\\chi)\\,\\Delta^v_\\ell(\\chi)\\,
        B_{N,u,v}\\left(\\frac{\\ell+1/2}{\\chi}, a(\\chi)\\right).

    By default, the counts are not normalized, i.e. :math:`N` is the number
    density of selected halos. If ``normalize_counts`` is ``True``, the
    counts are divided by their mean :math:`\\bar{n}=\\int dM\\,n(M)\\,S(M)`.
    If :math:`u` is also a
    :class:`~pyccl.halos.profiles.selection.HaloProfileSelection` (e.g. for
    cluster lensing), its miscentering is taken into account, while the
    counts themselves are always evaluated at :math:`k\\rightarrow 0`, where
    miscentering is irrelevant.

    .. note:: The squeezed limit assumes that the footprint is much larger
              than :math:`1/k`. Close to the fundamental mode of the
              survey, :math:`\\ell\\sim 2\\pi/\\sqrt{\\Omega_s}`, the split
              into this term and the super-sample term is therefore only
              approximate.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
        hmc (:class:`~pyccl.halos.halo_model.HMCalculator`): a halo model
            calculator.
        k (:obj:`float` or `array`): comoving wavenumber in Mpc^-1.
        a (:obj:`float` or `array`): scale factor.
        prof (:class:`~pyccl.halos.profiles.selection.HaloProfileSelection`):
            selected-halo profile of the counts.
        prof2 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile (corresponding to :math:`u` above).
        prof3 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile (corresponding to :math:`v` above). If ``None``,
            ``prof2`` will be used as ``prof3``.
        prof12_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2ptSelection`):
            two-point moment of ``prof`` and ``prof2`` if the latter is also
            a :class:`~pyccl.halos.profiles.selection.HaloProfileSelection`.
            If ``None``, ``Profile2ptSelection()`` will be used.
        prof23_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object returning the the two-point
            moment of ``prof2`` and ``prof3``. If ``None``, the default
            second moment will be used, corresponding to the
            products of the means of both profiles.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If ``None``,
            the power spectrum stored within `cosmo` will be used.
        normalize_counts (:obj:`bool`): if ``True``, the counts are divided
            by their mean number density.
        get_1h (:obj:`bool`): if ``False``, the 1-halo term (i.e. the first
            term in the first equation above) won't be computed.
        get_2h (:obj:`bool`): if ``False``, the 2-halo term (i.e. the second
            term in the first equation above) won't be computed.
        extrap_pk (:obj:`bool`):
            Whether to extrapolate ``p_of_k_a`` in case ``a`` is out of its
            support. If ```False```, and the queried values are out of bounds,
            an error is raised.

    Returns:
        (:obj:`float` or `array`): squeezed bispectrum evaluated at each
        combination of ``k`` and ``a``. The shape of the output will
        be ``(N_a, N_k)`` where ``N_k`` and ``N_a`` are the sizes of
        ``k`` and ``a`` respectively. If ``k`` or ``a`` are scalars, the
        corresponding dimension will be squeezed out on output.
    """
    a_use = np.atleast_1d(a).astype(float)
    k_use = np.atleast_1d(k).astype(float)

    if not isinstance(prof, HaloProfileSelection):
        raise TypeError("prof must be HaloProfileSelection.")

    if prof.f_off != 0:
        prof = HaloProfileSelection(mass_def=prof.mass_def,
                                    selection=prof.selection,
                                    is_number_counts=prof.is_number_counts)

    if prof3 is None:
        prof3 = prof2

    if isinstance(prof3, HaloProfileSelection):
        prof2, prof3 = prof3, prof2
    if prof12_2pt is None:
        prof12_2pt = Profile2ptSelection()
    if prof23_2pt is None:
        prof23_2pt = Profile2pt()

    sel2 = isinstance(prof2, HaloProfileSelection)
    sel3 = isinstance(prof3, HaloProfileSelection)
    prof12 = prof12_2pt if sel2 else Profile2pt()
    prof13 = prof12_2pt if sel3 else Profile2pt()

    pk2d = cosmo.parse_pk(p_of_k_a)
    extrap = cosmo if extrap_pk else None  

    na = len(a_use)
    nk = len(k_use)
    out = np.zeros([na, nk])
    for ia, aa in enumerate(a_use):
        norm1 = 1.
        if normalize_counts:
            norm1 = prof.get_normalization(cosmo, aa, hmc=hmc)
        norm2 = prof2.get_normalization(cosmo, aa, hmc=hmc)
        if prof3 == prof2:
            norm3 = norm2
        else:
            norm3 = prof3.get_normalization(cosmo, aa, hmc=hmc)

        if get_1h:
            if sel2:
                bk_1h = hmc.I_0_3(cosmo, k_use, aa, prof3, prof2=prof,
                                  prof3=prof2, prof_2pt=prof12_2pt)
            else:
                bk_1h = hmc.I_0_3(cosmo, k_use, aa, prof, prof2=prof2,
                                  prof3=prof3, prof_2pt=prof23_2pt)
        else:
            bk_1h = 0
        if get_2h:
            i11_2 = hmc.I_1_1(cosmo, k_use, aa, prof2)
            i12_2 = hmc.I_1_2(cosmo, k_use, aa, prof, prof2=prof2,
                              prof_2pt=prof12)
            if (prof3, prof13) == (prof2, prof12):
                i11_3 = i11_2
                i12_3 = i12_2
            else:
                i11_3 = hmc.I_1_1(cosmo, k_use, aa, prof3)
                i12_3 = hmc.I_1_2(cosmo, k_use, aa, prof, prof2=prof3,
                                  prof_2pt=prof13)
            pk = pk2d(k_use, aa, cosmo=extrap)
            bk_2h = (i11_3 * i12_2 + i11_2 * i12_3) * pk
        else:
            bk_2h = 0
        out[ia] = (bk_1h + bk_2h) / (norm1 * norm2 * norm3)
    if np.ndim(a) == 0:
        out = np.squeeze(out, axis=0)
    if np.ndim(k) == 0:
        out = np.squeeze(out, axis=-1)
    return out


def halomod_Pk2D_bispectrum_squeezed(cosmo, hmc, prof, *, prof2, prof3=None,
                                     prof12_2pt=None, prof23_2pt=None,
                                     p_of_k_a=None, normalize_counts=False,
                                     get_1h=True, get_2h=True,
                                     lk_arr=None, a_arr=None,
                                     extrap_order_lok=1, extrap_order_hik=2,
                                     extrap_pk=False):
    """ Returns a :class:`~pyccl.pk2d.Pk2D` object containing the squeezed
    halo model bispectrum between the number counts of selected halos and
    two quantities defined by their respective halo profiles. See
    :meth:`halomod_bispectrum_squeezed` for more details about the actual
    calculation.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
        hmc (:class:`~pyccl.halos.halo_model.HMCalculator`): a halo model
            calculator.
        prof (:class:`~pyccl.halos.profiles.selection.HaloProfileSelection`):
            selected-halo profile of the counts.
        prof2 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile.
        prof3 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile. If ``None``, ``prof2`` will be used as ``prof3``.
        prof12_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2ptSelection`):
            two-point moment of ``prof`` and ``prof2`` if the latter is also
            a :class:`~pyccl.halos.profiles.selection.HaloProfileSelection`.
            If ``None``, ``Profile2ptSelection()`` will be used.
        prof23_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object returning the the two-point
            moment of ``prof2`` and ``prof3``. If ``None``, the default
            second moment will be used, corresponding to the
            products of the means of both profiles.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If ``None``,
            the power spectrum stored within `cosmo` will be used.
        normalize_counts (:obj:`bool`): if ``True``, the counts are divided
            by their mean number density.
        get_1h (:obj:`bool`): if ``False``, the 1-halo term won't be
            computed.
        get_2h (:obj:`bool`): if ``False``, the 2-halo term won't be
            computed.
        a_arr (array): an array holding values of the scale factor
            at which the bispectrum should be calculated for
            interpolation. If ``None``, the internal values used
            by `cosmo` will be used.
        lk_arr (array): an array holding values of the natural
            logarithm of the wavenumber (in units of Mpc^-1) at
            which the bispectrum should be calculated for
            interpolation. If ``None``, the internal values used
            by `cosmo` will be used.
        extrap_order_lok (:obj:`int`): extrapolation order to be used on
            k-values below the minimum of the splines. See
            :class:`~pyccl.pk2d.Pk2D`.
        extrap_order_hik (:obj:`int`): extrapolation order to be used on
            k-values above the maximum of the splines. See
            :class:`~pyccl.pk2d.Pk2D`.
        extrap_pk (:obj:`bool`):
            Whether to extrapolate ``p_of_k_a`` in case ``a`` is out of its
            support. If ```False```, and the queried values are out of bounds,
            an error is raised.

    Returns:
        :class:`~pyccl.pk2d.Pk2D`: squeezed halo model bispectrum.
    """
    if lk_arr is None:
        lk_arr = cosmo.get_pk_spline_lk()
    if a_arr is None:
        a_arr = cosmo.get_pk_spline_a()

    bk_arr = halomod_bispectrum_squeezed(
        cosmo, hmc, np.exp(lk_arr), a_arr, prof, prof2=prof2, prof3=prof3,
        prof12_2pt=prof12_2pt, prof23_2pt=prof23_2pt, p_of_k_a=p_of_k_a,
        normalize_counts=normalize_counts, get_1h=get_1h, get_2h=get_2h,
        extrap_pk=extrap_pk)

    return Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=bk_arr,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik,
                is_logp=False)


def halomod_bispectrum_squeezed_SSC(cosmo, hmc, k, a, prof, *, prof2,
                                    prof3=None, prof23_2pt=None,
                                    p_of_k_a=None, normalize_counts=False,
                                    extrap_pk=False):
    """ Computes the super-sample contribution to the squeezed halo model
    bispectrum between the number counts of selected halos :math:`N` and
    two quantities :math:`u` and :math:`v` defined by their respective
    halo profiles (see :meth:`halomod_bispectrum_squeezed`). It is given by
    the response of the number counts times the response of the power
    spectrum to a long-wavelength mode:

    .. math::
        B^{\\rm SSC}_{N,u,v}(k,a) = I^1_1(a|N)\\,
        \\frac{\\partial P_{u,v}(k)}{\\partial\\delta_L},

    where :math:`I^1_1` is defined in the documentation of
    :meth:`~pyccl.halos.halo_model.HMCalculator.I_1_1`, and the response
    :math:`\\partial P_{u,v}/\\partial\\delta_L` is described in
    :meth:`~pyccl.halos.pk_4pt.halomod_Tk3D_SSC`. The super-sample
    covariance between the angular power spectrum of :math:`u` and
    :math:`v` and the angular number density of selected halos is then

    .. math::
        {\\rm Cov}_{\\rm SSC}(\\ell) =
        \\int d\\chi\\,\\sigma^2_B(\\chi)\\,
        \\Delta^u_\\ell(\\chi)\\,\\Delta^v_\\ell(\\chi)\\,
        B^{\\rm SSC}_{N,u,v}\\left(\\frac{\\ell+1/2}{\\chi}, a(\\chi)\\right),

    where :math:`\\sigma^2_B` is the variance of the projected linear
    density field over the survey footprint (see
    :func:`~pyccl.covariances.sigma2_B_disc`).

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
        hmc (:class:`~pyccl.halos.halo_model.HMCalculator`): a halo model
            calculator.
        k (:obj:`float` or `array`): comoving wavenumber in Mpc^-1.
        a (:obj:`float` or `array`): scale factor.
        prof (:class:`~pyccl.halos.profiles.selection.HaloProfileSelection`):
            selected-halo profile of the counts.
        prof2 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile (corresponding to :math:`u` above).
        prof3 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile (corresponding to :math:`v` above). If ``None``,
            ``prof2`` will be used as ``prof3``.
        prof23_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object returning the the two-point
            moment of ``prof2`` and ``prof3``. If ``None``, the default
            second moment will be used, corresponding to the
            products of the means of both profiles.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If ``None``,
            the power spectrum stored within `cosmo` will be used.
        normalize_counts (:obj:`bool`): if ``True``, the counts are divided
            by their mean number density.
        extrap_pk (:obj:`bool`):
            Whether to extrapolate ``p_of_k_a`` in case ``a`` is out of its
            support. If ```False```, and the queried values are out of bounds,
            an error is raised.

    Returns:
        (:obj:`float` or `array`): super-sample bispectrum evaluated at each
        combination of ``k`` and ``a``. The shape of the output will
        be ``(N_a, N_k)`` where ``N_k`` and ``N_a`` are the sizes of
        ``k`` and ``a`` respectively. If ``k`` or ``a`` are scalars, the
        corresponding dimension will be squeezed out on output.
    """ 
    a_use = np.atleast_1d(a).astype(float)
    k_use = np.atleast_1d(k).astype(float)

    if not isinstance(prof, HaloProfileSelection):
        raise TypeError("prof must be HaloProfileSelection.")
    if prof.f_off != 0:
        prof = HaloProfileSelection(mass_def=prof.mass_def,
                                    selection=prof.selection,
                                    is_number_counts=prof.is_number_counts)

    if prof3 is None:
        prof3 = prof2
    if prof23_2pt is None:
        prof23_2pt = Profile2pt()
    pk2d = cosmo.parse_pk(p_of_k_a)
    extrap = cosmo if extrap_pk else None  
    na = len(a_use)
    nk = len(k_use)
    out = np.zeros([na, nk])
    for ia, aa in enumerate(a_use):
        norm1 = 1.
        if normalize_counts:
            norm1 = prof.get_normalization(cosmo, aa, hmc=hmc)
        norm2 = prof2.get_normalization(cosmo, aa, hmc=hmc)
        i11_1 = hmc.I_1_1(cosmo, k_use, aa, prof)
        i11_2 = hmc.I_1_1(cosmo, k_use, aa, prof2)
        if prof3 == prof2:
            norm3 = norm2
            i11_3 = i11_2
        else:
            norm3 = prof3.get_normalization(cosmo, aa, hmc=hmc)
            i11_3 = hmc.I_1_1(cosmo, k_use, aa, prof3)
        i12_23 = hmc.I_1_2(cosmo, k_use, aa, prof2, prof2=prof3,
                           prof_2pt=prof23_2pt)
        pk = pk2d(k_use, aa, cosmo=extrap)
        dpk = pk2d(k_use, aa, derivative=True, cosmo=extrap)
        dpk23 = (47/21 - dpk/3)*i11_2*i11_3*pk + i12_23
        if prof2.is_number_counts or prof3.is_number_counts:
            b2 = i11_2 / norm2 if prof2.is_number_counts else 0
            b3 = i11_3 / norm3 if prof3.is_number_counts else 0
            i02 = hmc.I_0_2(cosmo, k_use, aa, prof2, prof2=prof3,
                            prof_2pt=prof23_2pt)
            dpk23 -= (b2 + b3) * (pk * i11_2 * i11_3 + i02)

        out[ia] = i11_1 * dpk23 / (norm1 * norm2 * norm3)

    if np.ndim(a) == 0:
        out = np.squeeze(out, axis=0)
    if np.ndim(k) == 0:
        out = np.squeeze(out, axis=-1)
    return out


def halomod_Pk2D_bispectrum_squeezed_SSC(cosmo, hmc, prof, *, prof2,
                                         prof3=None, prof23_2pt=None,
                                         p_of_k_a=None, normalize_counts=False,
                                         lk_arr=None, a_arr=None,
                                         extrap_order_lok=1,
                                         extrap_order_hik=2, extrap_pk=False):
    """ Returns a :class:`~pyccl.pk2d.Pk2D` object containing the
    super-sample contribution to the squeezed halo model bispectrum between
    the number counts of selected halos and two quantities defined by their
    respective halo profiles. See :meth:`halomod_bispectrum_squeezed_SSC`
    for more details about the actual calculation.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
        hmc (:class:`~pyccl.halos.halo_model.HMCalculator`): a halo model
            calculator.
        prof (:class:`~pyccl.halos.profiles.selection.HaloProfileSelection`):
            selected-halo profile of the counts.
        prof2 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile.
        prof3 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile. If ``None``, ``prof2`` will be used as ``prof3``.
        prof23_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object returning the the two-point
            moment of ``prof2`` and ``prof3``. If ``None``, the default
            second moment will be used, corresponding to the
            products of the means of both profiles.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If ``None``,
            the power spectrum stored within `cosmo` will be used.
        normalize_counts (:obj:`bool`): if ``True``, the counts are divided
            by their mean number density.
        a_arr (array): an array holding values of the scale factor
            at which the bispectrum should be calculated for
            interpolation. If ``None``, the internal values used
            by `cosmo` will be used.
        lk_arr (array): an array holding values of the natural
            logarithm of the wavenumber (in units of Mpc^-1) at
            which the bispectrum should be calculated for
            interpolation. If ``None``, the internal values used
            by `cosmo` will be used.
        extrap_order_lok (:obj:`int`): extrapolation order to be used on
            k-values below the minimum of the splines. See
            :class:`~pyccl.pk2d.Pk2D`.
        extrap_order_hik (:obj:`int`): extrapolation order to be used on
            k-values above the maximum of the splines. See
            :class:`~pyccl.pk2d.Pk2D`.
        extrap_pk (:obj:`bool`):
            Whether to extrapolate ``p_of_k_a`` in case ``a`` is out of its
            support. If ```False```, and the queried values are out of bounds,
            an error is raised.

    Returns:
        :class:`~pyccl.pk2d.Pk2D`: super-sample bispectrum.
    """ 
    if lk_arr is None:
        lk_arr = cosmo.get_pk_spline_lk()
    if a_arr is None:
        a_arr = cosmo.get_pk_spline_a()

    bk_arr = halomod_bispectrum_squeezed_SSC(
        cosmo, hmc, np.exp(lk_arr), a_arr, prof, prof2=prof2, prof3=prof3,
        prof23_2pt=prof23_2pt, p_of_k_a=p_of_k_a,
        normalize_counts=normalize_counts, extrap_pk=extrap_pk)
    return Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=bk_arr,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik,
                is_logp=False)
