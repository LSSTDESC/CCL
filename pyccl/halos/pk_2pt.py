__all__ = ("halomod_power_spectrum", "halomod_Pk2D",)

import numpy as np

from .. import Pk2D
from . import Profile2pt


def halomod_power_spectrum(cosmo, hmc, k, a, prof, *,
                           prof2=None, prof_2pt=None,
                           p_of_k_a=None,
                           get_1h=True, get_2h=True,
                           smooth_transition=None, suppress_1h=None,
                           extrap_pk=False):
    """ Computes the halo model power spectrum for two
    quantities defined by their respective halo profiles.
    The halo model power spectrum for two profiles
    :math:`u` and :math:`v` is:

    .. math::
        P_{u,v}(k,a) = I^0_2(k,a|u,v) +
        I^1_1(k,a|u)\\,I^1_1(k,a|v)\\,P_{\\rm lin}(k,a)

    where :math:`P_{\\rm lin}(k,a)` is the linear matter
    power spectrum, :math:`I^1_1` is defined in the documentation
    of :meth:`~pyccl.halos.halo_model.HMCalculator.I_1_1`, and :math:`I^0_2`
    is defined in the documentation of
    :meth:`~pyccl.halos.halo_model.HMCalculator.I_0_2`.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
        hmc (:class:`~pyccl.halos.halo_model.HMCalculator`): a halo model calculator.
        k (:obj:`float` or `array`): comoving wavenumber in Mpc^-1.
        a (:obj:`float` or `array`): scale factor.
        prof (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile.
        prof2 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): a
            second halo profile. If ``None``, ``prof`` will be used as
            ``prof2``.
        prof_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object
            returning the the two-point moment of the two profiles
            being correlated. If ``None``, the default second moment
            will be used, corresponding to the products of the means
            of both profiles.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If ``None``,
            the power spectrum stored within `cosmo` will be used.
        get_1h (:obj:`bool`): if ``False``, the 1-halo term (i.e. the first
            term in the first equation above) won't be computed.
        get_2h (:obj:`bool`): if ``False``, the 2-halo term (i.e. the second
            term in the first equation above) won't be computed.
        smooth_transition (:obj:`callable` or :obj:`None`):
            Modify the halo model 1-halo/2-halo transition region
            via a time-dependent function :math:`\\alpha(a)`,
            defined as in `HMCODE-2020 <https://arxiv.org/abs/2009.01858>`_:
            :math:`P(k,a)=
            (P_{1h}^{\\alpha(a)}(k)+P_{2h}^{\\alpha(a)}(k))^{1/\\alpha}`.
            If ``None`` the extra factor is just 1.
        suppress_1h (:obj:`callable` or :obj:`None`):
            Suppress the 1-halo large scale contribution by a
            time- and scale-dependent function :math:`k_*(a)`,
            defined as in `HMCODE-2020 <https://arxiv.org/abs/2009.01858>`_:
            :math:`1/[1+(k_*(a)/k)^4]`.
            If ``None`` the standard 1-halo term is returned with no damping.
        extrap_pk (:obj:`bool`):
            Whether to extrapolate ``p_of_k_a`` in case ``a`` is out of its
            support. If ```False```, and the queried values are out of bounds,
            an error is raised.

    Returns:
        (:obj:`float` or `array`): integral values evaluated at each
        combination of ``k`` and ``a``. The shape of the output will
        be ``(N_a, N_k)`` where ``N_k`` and ``N_a`` are the sizes of
        ``k`` and ``a`` respectively. If ``k`` or ``a`` are scalars, the
        corresponding dimension will be squeezed out on output.
    """ # noqa
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
        norm1 = prof.get_normalization(cosmo, aa, hmc=hmc)

        if prof2 == prof:
            norm2 = norm1
        else:
            norm2 = prof2.get_normalization(cosmo, aa, hmc=hmc)

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


def halomod_Pk2D(cosmo, hmc, prof, *,
                 prof2=None, prof_2pt=None,
                 p_of_k_a=None,
                 get_1h=True, get_2h=True,
                 lk_arr=None, a_arr=None,
                 extrap_order_lok=1, extrap_order_hik=2,
                 smooth_transition=None, suppress_1h=None, extrap_pk=False):
    """ Returns a :class:`~pyccl.pk2d.Pk2D` object containing
    the halo-model power spectrum for two quantities defined by
    their respective halo profiles. See :meth:`halomod_power_spectrum`
    for more details about the actual calculation.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
        hmc (:class:`~pyccl.halos.halo_model.HMCalculator`): a halo model calculator.
        prof (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile.
        prof2 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): a
            second halo profile. If ``None``, ``prof`` will be used as
            ``prof2``.
        prof_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object
            returning the the two-point moment of the two profiles
            being correlated. If ``None``, the default second moment
            will be used, corresponding to the products of the means
            of both profiles.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If ``None``,
            the power spectrum stored within `cosmo` will be used.
        get_1h (:obj:`bool`): if ``False``, the 1-halo term (i.e. the first
            term in the first equation above) won't be computed.
        get_2h (:obj:`bool`): if ``False``, the 2-halo term (i.e. the second
            term in the first equation above) won't be computed.
        a_arr (array): an array holding values of the scale factor
            at which the halo model power spectrum should be
            calculated for interpolation. If ``None``, the internal
            values used by `cosmo` will be used.
        lk_arr (array): an array holding values of the natural
            logarithm of the wavenumber (in units of Mpc^-1) at
            which the halo model power spectrum should be calculated
            for interpolation. If ``None``, the internal values used
            by `cosmo` will be used.
        extrap_order_lok (:obj:`int`): extrapolation order to be used on
            k-values below the minimum of the splines. See
            :class:`~pyccl.pk2d.Pk2D`.
        extrap_order_hik (:obj:`int`): extrapolation order to be used on
            k-values above the maximum of the splines. See
            :class:`~pyccl.pk2d.Pk2D`.
        smooth_transition (:obj:`callable` or :obj:`None`):
            Modify the halo model 1-halo/2-halo transition region
            via a time-dependent function :math:`\\alpha(a)`,
            defined as in `HMCODE-2020 <https://arxiv.org/abs/2009.01858>`_:
            :math:`P(k,a)= (P_{1h}^{\\alpha(a)}(k)+P_{2h}^{\\alpha(a)}(k))^{1/\\alpha}`.
            If ``None`` the extra factor is just 1.
        suppress_1h (:obj:`callable` or :obj:`None`):
            Suppress the 1-halo large scale contribution by a
            time- and scale-dependent function :math:`k_*(a)`,
            defined as in `HMCODE-2020 <https://arxiv.org/abs/2009.01858>`_:
            :math:`1/[1+(k_*(a)/k)^4]`.
            If ``None`` the standard 1-halo term is returned with no damping.
        extrap_pk (:obj:`bool`):
            Whether to extrapolate ``p_of_k_a`` in case ``a`` is out of its
            support. If ```False```, and the queried values are out of bounds,
            an error is raised.

    Returns:
        :class:`~pyccl.pk2d.Pk2D`: halo model power spectrum.
    """ # noqa
    if lk_arr is None:
        lk_arr = cosmo.get_pk_spline_lk()
    if a_arr is None:
        a_arr = cosmo.get_pk_spline_a()

    pk_arr = halomod_power_spectrum(
        cosmo, hmc, np.exp(lk_arr), a_arr,
        prof, prof2=prof2, prof_2pt=prof_2pt, p_of_k_a=p_of_k_a,
        get_1h=get_1h, get_2h=get_2h,
        smooth_transition=smooth_transition, suppress_1h=suppress_1h,
        extrap_pk=extrap_pk)

    return Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik,
                is_logp=False)
