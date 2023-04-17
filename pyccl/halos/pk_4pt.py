from .profiles import HaloProfile, HaloProfileNFW
from ..profiles_2pt import Profile2pt
from ... import ccllib as lib
from ...pyutils import check
from ...base import warn_api
from ...pk2d import Pk2D
from ...tk3d import Tk3D
from ...errors import CCLWarning
import numpy as np
import warnings


__all__ = ("halomod_trispectrum_1h", "halomod_Tk3D_1h",
           "halomod_Tk3D_SSC_linear_bias", "halomod_Tk3D_SSC",)


@warn_api(pairs=[("prof1", "prof"), ("normprof1", "normprof")],
          reorder=["prof12_2pt", "prof3", "prof4"])
def halomod_trispectrum_1h(cosmo, hmc, k, a, prof, *,
                           prof2=None, prof3=None, prof4=None,
                           prof12_2pt=None, prof34_2pt=None,
                           normprof=False, normprof2=False,
                           normprof3=False, normprof4=False):
    """ Computes the halo model 1-halo trispectrum for four different
    quantities defined by their respective halo profiles. The 1-halo
    trispectrum for four profiles :math:`u_{1,2}`, :math:`v_{1,2}` is
    calculated as:

    .. math::
        T_{u_1,u_2;v_1,v_2}(k_u,k_v,a) =
        I^0_{2,2}(k_u,k_v,a|u_{1,2},v_{1,2})

    where :math:`I^0_{2,2}` is defined in the documentation
    of :meth:`~HMCalculator.I_0_22`.

    .. note:: This approximation assumes that the 4-point
              profile cumulant is the same as the product of two
              2-point cumulants. We may relax this assumption in
              future versions of CCL.

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
        prof34_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof12_2pt` for `prof3` and `prof4`.
        normprof (bool): if `True`, this integral will be
            normalized by :math:`I^0_1(k\\rightarrow 0,a|u)`
            (see :meth:`~HMCalculator.I_0_1`), where
            :math:`u` is the profile represented by `prof`.
        normprof2 (bool): same as `normprof` for `prof2`.
        normprof3 (bool): same as `normprof` for `prof3`.
        normprof4 (bool): same as `normprof` for `prof4`.

    Returns:
        float or array_like: integral values evaluated at each
        combination of `k` and `a`. The shape of the output will
        be `(N_a, N_k, N_k)` where `N_k` and `N_a` are the sizes of
        `k` and `a` respectively. The ordering is such that
        `output[ia, ik2, ik1] = T(k[ik1], k[ik2], a[ia])`
        If `k` or `a` are scalars, the corresponding dimension will
        be squeezed out on output.
    """
    a_use = np.atleast_1d(a).astype(float)
    k_use = np.atleast_1d(k).astype(float)

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

    # Check inputs
    if not isinstance(prof, HaloProfile):
        raise TypeError("prof must be of type `HaloProfile`")
    if not isinstance(prof2, HaloProfile):
        raise TypeError("prof2 must be of type `HaloProfile` or `None`")
    if not isinstance(prof3, HaloProfile):
        raise TypeError("prof3 must be of type `HaloProfile` or `None`")
    if not isinstance(prof4, HaloProfile):
        raise TypeError("prof4 must be of type `HaloProfile` or `None`")
    if not isinstance(prof12_2pt, Profile2pt):
        raise TypeError("prof12_2pt must be of type `Profile2pt` or `None`")
    if not isinstance(prof34_2pt, Profile2pt):
        raise TypeError("prof34_2pt must be of type `Profile2pt` or `None`")

    def get_norm(normprof, prof, sf):
        return hmc.profile_norm(cosmo, sf, prof) if normprof else 1

    na = len(a_use)
    nk = len(k_use)
    out = np.zeros([na, nk, nk])
    for ia, aa in enumerate(a_use):
        # Compute profile normalizations
        norm1 = get_norm(normprof, prof, aa)
        # Compute second profile normalization
        if prof2 == prof:
            norm2 = norm1
        else:
            norm2 = get_norm(normprof2, prof2, aa)

        if prof3 == prof:
            norm3 = norm1
        else:
            norm3 = get_norm(normprof3, prof3, aa)

        if prof4 == prof2:
            norm4 = norm2
        else:
            norm4 = get_norm(normprof4, prof4, aa)

        norm = norm1 * norm2 * norm3 * norm4

        # Compute trispectrum at this redshift
        tk_1h = hmc.I_0_22(cosmo, k_use, aa,
                           prof=prof, prof2=prof2,
                           prof4=prof4, prof3=prof3,
                           prof12_2pt=prof12_2pt,
                           prof34_2pt=prof34_2pt)

        # Normalize
        out[ia, :, :] = tk_1h * norm

    if np.ndim(a) == 0:
        out = np.squeeze(out, axis=0)
    if np.ndim(k) == 0:
        out = np.squeeze(out, axis=-1)
        out = np.squeeze(out, axis=-1)
    return out


@warn_api(pairs=[("prof1", "prof"), ("normprof1", "normprof")],
          reorder=["prof12_2pt", "prof3", "prof4"])
def halomod_Tk3D_1h(cosmo, hmc, prof, *,
                    prof2=None, prof3=None, prof4=None,
                    prof12_2pt=None, prof34_2pt=None,
                    normprof=False, normprof2=False,
                    normprof3=False, normprof4=False,
                    lk_arr=None, a_arr=None,
                    extrap_order_lok=1, extrap_order_hik=1,
                    use_log=False):
    """ Returns a :class:`~pyccl.tk3d.Tk3D` object containing
    the 1-halo trispectrum for four quantities defined by
    their respective halo profiles. See :meth:`halomod_trispectrum_1h`
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
        prof12_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object returning the the two-point
            moment of `prof` and `prof2`. If `None`, the default
            second moment will be used, corresponding to the
            products of the means of both profiles.
        prof34_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof12_2pt` for `prof3` and `prof4`.
        normprof (bool): if `True`, this integral will be
            normalized by :math:`I^0_1(k\\rightarrow 0,a|u)`
            (see :meth:`~HMCalculator.I_0_1`), where
            :math:`u` is the profile represented by `prof`.
        normprof2 (bool): same as `normprof` for `prof2`.
        normprof3 (bool): same as `normprof` for `prof3`.
        normprof4 (bool): same as `normprof` for `prof4`.
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

    Returns:
        :class:`~pyccl.tk3d.Tk3D`: 1-halo trispectrum.
    """
    if lk_arr is None:
        status = 0
        nk = lib.get_pk_spline_nk(cosmo.cosmo)
        lk_arr, status = lib.get_pk_spline_lk(cosmo.cosmo, nk, status)
        check(status, cosmo=cosmo)
    if a_arr is None:
        status = 0
        na = lib.get_pk_spline_na(cosmo.cosmo)
        a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, status)
        check(status, cosmo=cosmo)

    tkk = halomod_trispectrum_1h(cosmo, hmc, np.exp(lk_arr), a_arr,
                                 prof, prof2=prof2,
                                 prof12_2pt=prof12_2pt,
                                 prof3=prof3, prof4=prof4,
                                 prof34_2pt=prof34_2pt,
                                 normprof=normprof, normprof2=normprof2,
                                 normprof3=normprof3, normprof4=normprof4)

    if use_log:
        # avoid zeros (this is system-dependent)
        tiny = np.nextafter(0, 1)
        tkk[tkk == 0] = tiny
        if np.any(tkk < 0):
            warnings.warn(
                "Some values were not positive. "
                "Will not interpolate in log-space.",
                category=CCLWarning)
            use_log = False
        else:
            tkk = np.log(tkk)

    tk3d = Tk3D(a_arr=a_arr, lk_arr=lk_arr, tkk_arr=tkk,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik, is_logt=use_log)
    return tk3d


@warn_api
def halomod_Tk3D_SSC_linear_bias(cosmo, hmc, *, prof,
                                 bias1=1, bias2=1, bias3=1, bias4=1,
                                 is_number_counts1=False,
                                 is_number_counts2=False,
                                 is_number_counts3=False,
                                 is_number_counts4=False,
                                 p_of_k_a=None, lk_arr=None,
                                 a_arr=None, extrap_order_lok=1,
                                 extrap_order_hik=1, use_log=False):
    """ Returns a :class:`~pyccl.tk3d.Tk3D` object containing
    the super-sample covariance trispectrum, given by the tensor
    product of the power spectrum responses associated with the
    two pairs of quantities being correlated. Each response is
    calculated as:

    .. math::
        \\frac{\\partial P_{u,v}(k)}{\\partial\\delta_L} = b_u b_v \\left(
        \\left(\\frac{68}{21}-\\frac{d\\log k^3P_L(k)}{d\\log k}\\right)
        P_L(k)+I^1_2(k|u,v) - (b_{u} + b_{v}) P_{u,v}(k) \\right)

    where the :math:`I^1_2` is defined in the documentation
    :meth:`~HMCalculator.I_1_2` and :math:`b_{}` and :math:`b_{vv}` are the
    linear halo biases for quantities :math:`u` and :math:`v`, respectively
    (zero if they are not clustering).

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo NFW
            profile.
        bias1 (float or array): linear galaxy bias for quantity 1. If an array,
        it has to have the shape of `a_arr`.
        bias2 (float or array): linear galaxy bias for quantity 2.
        bias3 (float or array): linear galaxy bias for quantity 3.
        bias4 (float or array): linear galaxy bias for quantity 4.
        is_number_counts1 (bool): If True, quantity 1 will be considered
        number counts and the clustering counter terms computed. Default False.
        is_number_counts2 (bool): as is_number_counts1 but for quantity 2.
        is_number_counts3 (bool): as is_number_counts1 but for quantity 3.
        is_number_counts4 (bool): as is_number_counts1 but for quantity 4.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If `None`,
            the power spectrum stored within `cosmo` will be used.
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

    Returns:
        :class:`~pyccl.tk3d.Tk3D`: SSC effective trispectrum.
    """
    if lk_arr is None:
        status = 0
        nk = lib.get_pk_spline_nk(cosmo.cosmo)
        lk_arr, status = lib.get_pk_spline_lk(cosmo.cosmo, nk, status)
        check(status, cosmo=cosmo)
    if a_arr is None:
        status = 0
        na = lib.get_pk_spline_na(cosmo.cosmo)
        a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, status)
        check(status, cosmo=cosmo)

    # Make sure biases are of the form number of a x number of k
    ones = np.ones_like(a_arr)
    bias1 *= ones
    bias2 *= ones
    bias3 *= ones
    bias4 *= ones

    k_use = np.exp(lk_arr)

    # Check inputs
    if not isinstance(prof, HaloProfileNFW):
        raise TypeError("prof must be of type `HaloProfileNFW`")
    prof_2pt = Profile2pt()

    # Power spectrum
    if isinstance(p_of_k_a, Pk2D):
        pk2d = p_of_k_a
    elif (p_of_k_a is None) or (str(p_of_k_a) == 'linear'):
        pk2d = cosmo.get_linear_power('delta_matter:delta_matter')
    elif str(p_of_k_a) == 'nonlinear':
        pk2d = cosmo.get_nonlin_power('delta_matter:delta_matter')
    else:
        raise TypeError("p_of_k_a must be `None`, \'linear\', "
                        "\'nonlinear\' or a `Pk2D` object")

    na = len(a_arr)
    nk = len(k_use)
    dpk12 = np.zeros([na, nk])
    dpk34 = np.zeros([na, nk])
    for ia, aa in enumerate(a_arr):
        # Compute profile normalizations
        norm = hmc.profile_norm(cosmo, aa, prof) ** 2
        i12 = hmc.I_1_2(cosmo, k_use, aa, prof,
                        prof2=prof, prof_2pt=prof_2pt) * norm

        pk = pk2d.eval(k_use, aa, cosmo)
        dpk = pk2d.eval_dlPk_dlk(k_use, aa, cosmo)
        # ~ [(47/21 - 1/3 dlogPk/dlogk) * Pk+I12]
        dpk12[ia] = ((47/21 - dpk/3)*pk + i12)
        dpk34[ia] = dpk12[ia].copy()  # Avoid surprises

        # Counter terms for clustering (i.e. - (bA + bB) * PAB
        if any([is_number_counts1, is_number_counts2,
                is_number_counts3, is_number_counts4]):
            b1 = b2 = b3 = b4 = 0

            i02 = hmc.I_0_2(cosmo, k_use, aa, prof,
                            prof2=prof, prof_2pt=prof_2pt) * norm
            P_12 = P_34 = pk + i02

            if is_number_counts1:
                b1 = bias1[ia]
            if is_number_counts2:
                b2 = bias2[ia]
            if is_number_counts3:
                b3 = bias3[ia]
            if is_number_counts4:
                b4 = bias4[ia]

            dpk12[ia, :] -= (b1 + b2) * P_12
            dpk34[ia, :] -= (b3 + b4) * P_34

        dpk12[ia] *= bias1[ia] * bias2[ia]
        dpk34[ia] *= bias3[ia] * bias4[ia]

    if use_log:
        if np.any(dpk12 <= 0) or np.any(dpk34 <= 0):
            warnings.warn(
                "Some values were not positive. "
                "Will not interpolate in log-space.",
                category=CCLWarning)
            use_log = False
        else:
            dpk12 = np.log(dpk12)
            dpk34 = np.log(dpk34)

    tk3d = Tk3D(a_arr=a_arr, lk_arr=lk_arr,
                pk1_arr=dpk12, pk2_arr=dpk34,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik, is_logt=use_log)
    return tk3d


@warn_api(pairs=[("prof1", "prof"), ("normprof1", "normprof")],
          reorder=["prof12_2pt", "prof3", "prof4"])
def halomod_Tk3D_SSC(
        cosmo, hmc, prof, *, prof2=None, prof3=None, prof4=None,
        prof12_2pt=None, prof34_2pt=None,
        normprof=False, normprof2=False, normprof3=False, normprof4=False,
        p_of_k_a=None, lk_arr=None, a_arr=None,
        extrap_order_lok=1, extrap_order_hik=1, use_log=False):
    """ Returns a :class:`~pyccl.tk3d.Tk3D` object containing
    the super-sample covariance trispectrum, given by the tensor
    product of the power spectrum responses associated with the
    two pairs of quantities being correlated. Each response is
    calculated as:

    .. math::
        \\frac{\\partial P_{u,v}(k)}{\\partial\\delta_L} =
        \\left(\\frac{68}{21}-\\frac{d\\log k^3P_L(k)}{d\\log k}\\right)
        P_L(k)I^1_1(k,|u)I^1_1(k,|v)+I^1_2(k|u,v) - (b_{u} + b_{v})
        P_{u,v}(k)

    where the :math:`I^a_b` are defined in the documentation
    of :meth:`~HMCalculator.I_1_1` and  :meth:`~HMCalculator.I_1_2` and
    :math:`b_{u}` and :math:`b_{v}` are the linear halo biases for quantities
    :math:`u` and :math:`v`, respectively (zero if they are not clustering).

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
        prof34_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof12_2pt` for `prof3` and `prof4`.
        normprof (bool): if `True`, this integral will be
            normalized by :math:`I^0_1(k\\rightarrow 0,a|u)`
            (see :meth:`~HMCalculator.I_0_1`), where
            :math:`u` is the profile represented by `prof`.
        normprof2 (bool): same as `normprof` for `prof2`.
        normprof3 (bool): same as `normprof` for `prof3`.
        normprof4 (bool): same as `normprof` for `prof4`.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If `None`,
            the power spectrum stored within `cosmo` will be used.
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

    Returns:
        :class:`~pyccl.tk3d.Tk3D`: SSC effective trispectrum.
    """
    if lk_arr is None:
        status = 0
        nk = lib.get_pk_spline_nk(cosmo.cosmo)
        lk_arr, status = lib.get_pk_spline_lk(cosmo.cosmo, nk, status)
        check(status, cosmo=cosmo)
    if a_arr is None:
        status = 0
        na = lib.get_pk_spline_na(cosmo.cosmo)
        a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, status)
        check(status, cosmo=cosmo)

    k_use = np.exp(lk_arr)

    if prof2 is None:
        prof2 = prof
        normprof2 = normprof
    if prof3 is None:
        prof3 = prof
        normprof3 = normprof
    if prof4 is None:
        prof4 = prof2
        normprof4 = normprof2
    if prof12_2pt is None:
        prof12_2pt = Profile2pt()
    if prof34_2pt is None:
        prof34_2pt = prof12_2pt

    # Check inputs
    if not isinstance(prof, HaloProfile):
        raise TypeError("prof must be of type `HaloProfile`")
    if not isinstance(prof2, HaloProfile):
        raise TypeError("prof2 must be of type `HaloProfile` or `None`")
    if not isinstance(prof3, HaloProfile):
        raise TypeError("prof3 must be of type `HaloProfile` or `None`")
    if not isinstance(prof4, HaloProfile):
        raise TypeError("prof4 must be of type `HaloProfile` or `None`")
    if not isinstance(prof12_2pt, Profile2pt):
        raise TypeError("prof12_2pt must be of type `Profile2pt` or `None`")
    if not isinstance(prof34_2pt, Profile2pt):
        raise TypeError("prof34_2pt must be of type `Profile2pt` or `None`")

    # number counts profiles must be normalized
    profs = {prof: normprof, prof2: normprof2,
             prof3: normprof3, prof4: normprof4}

    for i, (p, n) in enumerate(profs.items()):
        if p.is_number_counts and not n:
            raise ValueError(
                f"normprof{i+1} must be True if prof{i+1}.is_number_counts")

    # Power spectrum
    if isinstance(p_of_k_a, Pk2D):
        pk2d = p_of_k_a
    elif (p_of_k_a is None) or str(p_of_k_a) == 'linear':
        pk2d = cosmo.get_linear_power('delta_matter:delta_matter')
    elif str(p_of_k_a) == 'nonlinear':
        pk2d = cosmo.get_nonlin_power('delta_matter:delta_matter')
    else:
        raise ValueError("p_of_k_a must be `None`, 'linear', "
                         "'nonlinear' or a `Pk2D` object")

    def get_norm(normalize, profile, sf):
        return hmc.profile_norm(cosmo, sf, profile) if normalize else 1

    dpk12, dpk34 = [np.zeros((len(a_arr), len(k_use))) for _ in range(2)]
    for ia, aa in enumerate(a_arr):
        # Compute profile normalizations
        norm1 = get_norm(normprof, prof, aa)
        i11_1 = hmc.I_1_1(cosmo, k_use, aa, prof)
        # Compute second profile normalization
        if prof2 == prof:
            norm2 = norm1
            i11_2 = i11_1
        else:
            norm2 = get_norm(normprof2, prof2, aa)
            i11_2 = hmc.I_1_1(cosmo, k_use, aa, prof2)

        if prof3 == prof:
            norm3 = norm1
            i11_3 = i11_1
        else:
            norm3 = get_norm(normprof3, prof3, aa)
            i11_3 = hmc.I_1_1(cosmo, k_use, aa, prof3)

        if prof4 == prof2:
            norm4 = norm2
            i11_4 = i11_2
        else:
            norm4 = get_norm(normprof4, prof4, aa)
            i11_4 = hmc.I_1_1(cosmo, k_use, aa, prof4)

        i12_12 = hmc.I_1_2(cosmo, k_use, aa, prof,
                           prof2=prof2, prof_2pt=prof12_2pt)
        if (prof, prof2) == (prof3, prof4):
            i12_34 = i12_12
        else:
            i12_34 = hmc.I_1_2(cosmo, k_use, aa, prof3,
                               prof2=prof4, prof_2pt=prof34_2pt)

        norm12 = norm1 * norm2
        norm34 = norm3 * norm4

        pk = pk2d.eval(k_use, aa, cosmo)
        dpk = pk2d.eval_dlPk_dlk(k_use, aa, cosmo)
        # (47/21 - 1/3 dlogPk/dlogk) * I11 * I11 * Pk+I12
        dpk12[ia, :] = norm12*((47/21 - dpk/3)*i11_1*i11_2*pk + i12_12)
        dpk34[ia, :] = norm34*((47/21 - dpk/3)*i11_3*i11_4*pk + i12_34)

        # Counter terms for clustering (i.e. - (bA + bB) * PAB
        if prof.is_number_counts or prof2.is_number_counts:
            b1 = b2 = np.zeros_like(k_use)
            i02_12 = hmc.I_0_2(cosmo, k_use, aa, prof, prof2=prof2,
                               prof_2pt=prof12_2pt)
            P_12 = norm12 * (pk * i11_1 * i11_2 + i02_12)

            if prof.is_number_counts:
                b1 = i11_1 * norm1

            if prof2 == prof:
                b2 = b1
            elif prof2.is_number_counts:
                b2 = i11_2 * norm2

            dpk12[ia, :] -= (b1 + b2) * P_12

        if any([p.is_number_counts for p in [prof3, prof4]]):
            b3 = b4 = np.zeros_like(k_use)
            if (prof, prof2, prof12_2pt) == (prof3, prof4, prof34_2pt):
                i02_34 = i02_12
            else:
                i02_34 = hmc.I_0_2(cosmo, k_use, aa, prof3, prof2=prof4,
                                   prof_2pt=prof34_2pt)
            P_34 = norm34 * (pk * i11_3 * i11_4 + i02_34)

            if prof3 == prof:
                b3 = b1
            elif prof3.is_number_counts:
                b3 = i11_3 * norm3

            if prof4 == prof2:
                b4 = b2
            elif prof4.is_number_counts:
                b4 = i11_4 * norm4

            dpk34[ia, :] -= (b3 + b4) * P_34

    if use_log:
        if np.any(dpk12 <= 0) or np.any(dpk34 <= 0):
            warnings.warn(
                "Some values were not positive. "
                "Will not interpolate in log-space.",
                category=CCLWarning)
            use_log = False
        else:
            dpk12 = np.log(dpk12)
            dpk34 = np.log(dpk34)

    tk3d = Tk3D(a_arr=a_arr, lk_arr=lk_arr,
                pk1_arr=dpk12, pk2_arr=dpk34,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik, is_logt=use_log)
    return tk3d
