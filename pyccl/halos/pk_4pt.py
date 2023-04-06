from ..base import UnlockInstance, warn_api
from ..pk2d import parse_pk
from ..tk3d import Tk3D
from ..errors import CCLWarning
from .profiles import HaloProfileNFW
from .profiles import HaloProfileNumberCounts as ProfNC
from .profiles_2pt import Profile2pt
import numpy as np
import warnings
from functools import partial


__all__ = ("halomod_trispectrum_1h", "halomod_Tk3D_1h",
           "halomod_Tk3D_SSC_linear_bias", "halomod_Tk3D_SSC",)


@warn_api(pairs=[("prof1", "prof")], reorder=["prof12_2pt", "prof3", "prof4"])
def halomod_trispectrum_1h(cosmo, hmc, k, a, prof, *,
                           prof2=None, prof3=None, prof4=None,
                           prof12_2pt=None, prof34_2pt=None,
                           normprof1=None, normprof2=None,
                           normprof3=None, normprof4=None):
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
        normprof1 (bool): (Deprecated - do not use)
            if `True`, this integral will be
            normalized by :math:`I^0_1(k\\rightarrow 0,a|u)`
            (see :meth:`~HMCalculator.I_0_1`), where
            :math:`u` is the profile represented by `prof`.
        normprof2 (bool): same as `normprof1` for `prof2`.
        normprof3 (bool): same as `normprof1` for `prof3`.
        normprof4 (bool): same as `normprof1` for `prof4`.

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

    # define all the profiles
    prof, prof2, prof3, prof4, prof12_2pt, prof34_2pt = \
        _allocate_profiles(prof, prof2, prof3, prof4, prof12_2pt, prof34_2pt,
                           normprof1, normprof2, normprof3, normprof4)

    na = len(a_use)
    nk = len(k_use)
    out = np.zeros([na, nk, nk])
    for ia, aa in enumerate(a_use):
        # normalizations
        norm1 = hmc.get_profile_norm(cosmo, aa, prof)

        if prof2 == prof:
            norm2 = norm1
        else:
            norm2 = hmc.get_profile_norm(cosmo, aa, prof2)

        if prof3 == prof:
            norm3 = norm1
        else:
            norm3 = hmc.get_profile_norm(cosmo, aa, prof3)

        if prof4 == prof2:
            norm4 = norm2
        else:
            norm4 = hmc.get_profile_norm(cosmo, aa, prof4)

        # trispectrum
        tk_1h = hmc.I_0_22(cosmo, k_use, aa,
                           prof=prof, prof2=prof2,
                           prof4=prof4, prof3=prof3,
                           prof12_2pt=prof12_2pt,
                           prof34_2pt=prof34_2pt)

        out[ia] = tk_1h * norm1 * norm2 * norm3 * norm4  # assign

    if np.ndim(a) == 0:
        out = np.squeeze(out, axis=0)
    if np.ndim(k) == 0:
        out = np.squeeze(out, axis=-1)
        out = np.squeeze(out, axis=-1)
    return out


@warn_api(pairs=[("prof1", "prof")],
          reorder=["prof12_2pt", "prof3", "prof4"])
def halomod_Tk3D_1h(cosmo, hmc, prof, *,
                    prof2=None, prof3=None, prof4=None,
                    prof12_2pt=None, prof34_2pt=None,
                    normprof1=None, normprof2=None,
                    normprof3=None, normprof4=None,
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
        normprof1 (bool): (Deprecated - do not use)
            if `True`, this integral will be
            normalized by :math:`I^0_1(k\\rightarrow 0,a|u)`
            (see :meth:`~HMCalculator.I_0_1`), where
            :math:`u` is the profile represented by `prof`.
        normprof2 (bool): same as `normprof1` for `prof2`.
        normprof3 (bool): same as `normprof1` for `prof3`.
        normprof4 (bool): same as `normprof1` for `prof4`.
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
        lk_arr = cosmo.get_pk_spline_lk()
    if a_arr is None:
        a_arr = cosmo.get_pk_spline_a()

    tkk = halomod_trispectrum_1h(cosmo, hmc, np.exp(lk_arr), a_arr,
                                 prof, prof2=prof2,
                                 prof12_2pt=prof12_2pt,
                                 prof3=prof3, prof4=prof4,
                                 prof34_2pt=prof34_2pt,
                                 normprof1=normprof1, normprof2=normprof2,
                                 normprof3=normprof3, normprof4=normprof4)

    tkk, use_log = _logged_output(tkk, log=use_log)

    return Tk3D(a_arr=a_arr, lk_arr=lk_arr, tkk_arr=tkk,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik, is_logt=use_log)


@warn_api
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
        extrap_pk (bool):
            Whether to extrapolate ``p_of_k_a`` in case ``a`` is out of its
            support. If False, and the queried values are out of bounds,
            an error is raised. The default is False.

    Returns:
        :class:`~pyccl.tk3d.Tk3D`: SSC effective trispectrum.
    """
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

    pk2d = parse_pk(cosmo, p_of_k_a)
    extrap = cosmo if extrap_pk else None  # extrapolation rule for pk2d

    na = len(a_arr)
    nk = len(k_use)
    dpk12, dpk34 = [np.zeros([na, nk]) for _ in range(2)]
    for ia, aa in enumerate(a_arr):
        norm = hmc.get_profile_norm(cosmo, aa, prof)**2
        i12 = hmc.I_1_2(cosmo, k_use, aa, prof, prof2=prof, prof_2pt=prof_2pt)

        pk = pk2d(k_use, aa, cosmo=extrap)
        dpk = pk2d(k_use, aa, derivative=True, cosmo=extrap)

        # ~ (47/21 - 1/3 dlogPk/dlogk) * Pk + I12
        dpk12[ia] = ((47/21 - dpk/3)*pk + i12 * norm)
        dpk34[ia] = dpk12[ia].copy()

        # Counter terms for clustering (i.e. - (bA + bB) * PAB)
        if any([is_number_counts1, is_number_counts2,
                is_number_counts3, is_number_counts4]):
            b1 = b2 = b3 = b4 = 0
            i02 = hmc.I_0_2(cosmo, k_use, aa, prof,
                            prof2=prof, prof_2pt=prof_2pt)

            P_12 = P_34 = pk + i02 * norm

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
        cosmo, hmc, prof, *, prof2=None, prof3=None, prof4=None,
        prof12_2pt=None, prof34_2pt=None,
        normprof1=None, normprof2=None, normprof3=None, normprof4=None,
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
        normprof1 (bool): (Deprecated - do not use)
            if `True`, this integral will be
            normalized by :math:`I^0_1(k\\rightarrow 0,a|u)`
            (see :meth:`~HMCalculator.I_0_1`), where
            :math:`u` is the profile represented by `prof`.
        normprof2 (bool): same as `normprof1` for `prof2`.
        normprof3 (bool): same as `normprof1` for `prof3`.
        normprof4 (bool): same as `normprof1` for `prof4`.
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
        extrap_pk (bool):
            Whether to extrapolate ``p_of_k_a`` in case ``a`` is out of its
            support. If False, and the queried values are out of bounds,
            an error is raised. The default is False.

    Returns:
        :class:`~pyccl.tk3d.Tk3D`: SSC effective trispectrum.
    """
    if lk_arr is None:
        lk_arr = cosmo.get_pk_spline_lk()
    if a_arr is None:
        a_arr = cosmo.get_pk_splne_a()

    # define all the profiles
    prof, prof2, prof3, prof4, prof12_2pt, prof34_2pt = \
        _allocate_profiles(prof, prof2, prof3, prof4, prof12_2pt, prof34_2pt,
                           normprof1, normprof2, normprof3, normprof4)

    k_use = np.exp(lk_arr)
    pk2d = parse_pk(cosmo, p_of_k_a)
    extrap = cosmo if extrap_pk else None  # extrapolation rule for pk2d

    dpk12, dpk34 = [np.zeros((len(a_arr), len(k_use))) for _ in range(2)]
    for ia, aa in enumerate(a_arr):
        # normalizations & I11 integral
        norm1 = hmc.get_profile_norm(cosmo, aa, prof)
        i11_1 = hmc.I_1_1(cosmo, k_use, aa, prof)

        if prof2 == prof:
            norm2 = norm1
            i11_2 = i11_1
        else:
            norm2 = hmc.get_profile_norm(cosmo, aa, prof2)
            i11_2 = hmc.I_1_1(cosmo, k_use, aa, prof2)

        if prof3 == prof:
            norm3 = norm1
            i11_3 = i11_1
        else:
            norm3 = hmc.get_profile_norm(cosmo, aa, prof3)
            i11_3 = hmc.I_1_1(cosmo, k_use, aa, prof3)

        if prof4 == prof2:
            norm4 = norm2
            i11_4 = i11_2
        else:
            norm4 = hmc.get_profile_norm(cosmo, aa, prof4)
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
        dpk12[ia] = norm1 * norm2 * ((47/21 - dpk/3)*i11_1*i11_2*pk + i12_12)
        dpk34[ia] = norm3 * norm4 * ((47/21 - dpk/3)*i11_3*i11_4*pk + i12_34)

        # Counter terms for clustering (i.e. - (bA + bB) * PAB
        counterterm = partial(_get_counterterm, cosmo=cosmo, hmc=hmc,
                              k=k_use, a=aa, pk=pk)

        if isinstance(prof, ProfNC) or isinstance(prof2, ProfNC):
            dpk12[ia] -= counterterm(prof, prof2, prof12_2pt,
                                     norm1, norm2, i11_1, i11_2)

        if isinstance(prof3, ProfNC) or isinstance(prof4, ProfNC):
            if (prof, prof2, prof12_2pt) == (prof3, prof4, prof34_2pt):
                dpk34[ia] -= dpk12[ia]
            else:
                dpk34[ia] -= counterterm(prof3, prof4, prof34_2pt,
                                         norm3, norm4, i11_3, i11_4)

    dpk12, dpk34, use_log = _logged_output(dpk12, dpk34, log=use_log)

    return Tk3D(a_arr=a_arr, lk_arr=lk_arr,
                pk1_arr=dpk12, pk2_arr=dpk34,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik, is_logt=use_log)


def _allocate_profiles(prof, prof2, prof3, prof4, prof12_2pt, prof34_2pt,
                       normprof1, normprof2, normprof3, normprof4):
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

    # TODO: Remove for CCLv3.
    if normprof1 is not None:
        with UnlockInstance(prof):
            prof.normprof = normprof1
    if normprof2 is not None:
        with UnlockInstance(prof2):
            prof2.normprof = normprof2
    if normprof3 is not None:
        with UnlockInstance(prof3):
            prof3.normprof = normprof3
    if normprof4 is not None:
        with UnlockInstance(prof4):
            prof4.normprof = normprof4

    return prof, prof2, prof3, prof4, prof12_2pt, prof34_2pt


def _get_counterterm(pA, pB, p2pt, nA, nB, i11_A, i11_B, *,
                     cosmo, hmc, k, a, pk):
    """Helper to compute counter-terms."""
    # p : profiles | p2pt : 2-point | n : norms | i11 : I_1_1 integral
    bA = i11_A * nA if isinstance(pA, ProfNC) else np.zeros_like(k)
    bB = i11_B * nB if isinstance(pB, ProfNC) else np.zeros_like(k)
    i02 = hmc.I_0_2(cosmo, k, a, pA, prof2=pB, prof_2pt=p2pt)
    P = nA * nB * (pk * i11_A * i11_B + i02)
    return (bA + bB) * P


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
