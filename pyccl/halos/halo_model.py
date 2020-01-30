from .. import ccllib as lib
from .hmfunc import MassFunc
from .hbias import HaloBias
from .profiles import HaloProfile
from .profiles_2pt import Profile2pt
from ..core import check
from ..pk2d import Pk2D
from ..power import linear_matter_power, nonlin_matter_power
from ..background import rho_x
from ..pyutils import _spline_integrate
import numpy as np


class HMCalculator(object):
    """ This class implements a set of methods that can be used to
    compute various halo model quantities. A lot of these quantities
    will involve integrals of the sort:

    .. math::
       \\int dM\\,n(M,a)\\,f(M,k,a),

    where :math:`n(M,a)` is the halo mass function, and :math:`f` is
    an arbitrary function of mass, scale factor and Fourier scales.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        log10M_min (float): logarithmic mass (in units of solar mass)
            corresponding to the lower bound of the integrals in
            mass. Default: 8.
        log10M_max (float): logarithmic mass (in units of solar mass)
            corresponding to the upper bound of the integrals in
            mass. Default: 16.
        nlog10M (int): number of samples in log(Mass) to be used in
            the mass integrals. Default: 128.
        integration_method_M (string): integration method to use
            in the mass integrals. Options: "simpson" and "spline".
            Default: "simpson".
        k_min (float): some of the integrals solved by this class
            will often be normalized by their value on very large
            scales. This parameter (in units of inverse Mpc)
            determines what is considered a "very large" scale.
            Default: 1E-5.
    """
    def __init__(self, cosmo, log10M_min=8., log10M_max=16.,
                 nlog10M=128, integration_method_M='simpson',
                 k_min=1E-5):
        self.rho0 = rho_x(cosmo, 1., 'matter', is_comoving=True)
        self.precision = {'log10M_min': log10M_min,
                          'log10M_max': log10M_max,
                          'nlog10M': nlog10M,
                          'integration_method_M': integration_method_M,
                          'k_min': k_min}
        self.lmass = np.linspace(self.precision['log10M_min'],
                                 self.precision['log10M_max'],
                                 self.precision['nlog10M'])
        self.mass = 10.**self.lmass
        self.m0 = self.mass[0]

        if self.precision['integration_method_M'] not in ['spline',
                                                          'simpson']:
            raise NotImplementedError("Only \'simpson\' and 'spline' "
                                      "supported as integration methods")
        elif self.precision['integration_method_M'] == 'simpson':
            from scipy.integrate import simps
            self.integrator = simps
        else:
            self.integrator = self._integ_spline

    def _integ_spline(self, fM, lM):
        # Spline integrator
        return _spline_integrate(lM, fM, lM[0], lM[-1])

    def _hmf(self, hmf, cosmo, a, mass_def=None):
        # Evaluates halo mass function at the correct mass
        # values.
        return hmf.get_mass_function(cosmo, self.mass,
                                     a, mdef_other=mass_def)

    def _hbf(self, hbf, cosmo, a, mass_def=None):
        # Evaluates halo bias at the correct mass values.
        return hbf.get_halo_bias(cosmo, self.mass,
                                 a, mdef_other=mass_def)

    def _I_0_1_from_arrays(self, hmf_a, hmf0, uk_a):
        # Solves the integral:
        #   \int dM n(M,a) * u(k,a|M)
        # - hmf_a is an array of halo mass function values.
        # - hmf0 is the value of the halo mass function at
        #       a very low mass.
        # - uk_a is the halo profile evaluated at the same
        #       masses.
        i1 = self.integrator(hmf_a[..., :] * uk_a,
                             self.lmass)
        return i1 + hmf0 * uk_a[..., 0]

    def _I_1_1_from_arrays(self, hmf_a, hbf_a, hmf0, uk_a):
        # Solves the integral:
        #   \int n(M,a) * b(M,a) * u(k,a|M)
        # - hmf_a is an array of halo mass function values.
        # - hbf_a is an array of halo bias values.
        # - hmf0 is the value of the halo mass function at
        #       a very low mass.
        # - uk_a is the halo profile evaluated at the same
        #       masses.
        i1 = self.integrator((hmf_a * hbf_a)[..., :] * uk_a,
                             self.lmass)
        return i1 + hmf0 * uk_a[..., 0]

    def _mf0(self, mf):
        # Returns (rho_M - \int dM * n(M) * M) / M_min
        return (self.rho0 -
                self.integrator(mf * self.mass,
                                self.lmass)) / self.m0

    def _mbf0(self, mf, bf):
        # Returns (rho_M - \int dM * n(M) * b(M) * M) / M_min
        return (self.rho0 -
                self.integrator(mf *bf * self.mass,
                                self.lmass)) / self.m0

    def _profile_norm(self, mf, mf0, cosmo, prof,
                      aa, mass_def, normprof):
        if normprof:
            # Computes [ \int dM * n(M) * <u(k->0|M)> ]^{-1}
            uk01 = self._eval_profile(cosmo, prof,
                                      self.precision['k_min'],
                                      aa, mass_def=mass_def)
            norm = 1. / self._I_0_1_from_arrays(mf, mf0, uk01)
        else:
            norm = 1.
        return norm

    def _eval_profile(self, cosmo, prof, k, a, mass_def):
        return prof.fourier(cosmo, k, self.mass, a, mass_def=mass_def).T

    def _check_massfunc(self, massfunc):
        if not isinstance(massfunc, MassFunc):
            raise TypeError("massfunc must be of type `MassFunc`")

    def _check_hbias(self, hbias):
        if not isinstance(hbias, HaloBias):
            raise TypeError("hbias must be of type `HaloBias`")

    def _check_prof(self, prof):
        if not isinstance(prof, HaloProfile):
            raise TypeError("prof must be of type `HaloProfile`")


def halomod_mean_profile_1pt(cosmo, hmc, k, a, massfunc, prof,
                             normprof=False, mass_def=None):
    """ Solves the integral:

    .. math::
        I^1_0(k,a|u) = \\int dM\\,n(M,a)\\,\\langle u(k,a|M)\\rangle,

    where :math:`n(M,a)` is the halo mass function, and
    :math:`\\langle u(k,a|M)\\rangle` is the halo profile as a
    function of scale, scale factor and halo mass.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        k (float or array_like): comoving wavenumber in Mpc^-1.
        a (float or array_like): scale factor.
        massfunc (:class:`~pyccl.halos.hmfunc.MassFunc`): a mass
            function object.
        prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile.
        normprof (bool): if `True`, this integral will be
            normalized by :math:`I^1_0(k\\rightarrow 0,a|u)`.
        mass_def (:class:`~pyccl.halos.massdef.MassDef`): a mass
            definition object.

    Returns:
        float or array_like: integral values evaluated at each
        combination of `k` and `a`. The shape of the output will
        be `(N_a, N_k)` where `N_k` and `N_a` are the sizes of
        `k` and `a` respectively. If `k` or `a` are scalars, the
        corresponding dimension will be squeezed out on output.
    """
    a_use = np.atleast_1d(a)
    k_use = np.atleast_1d(k)

    # Check inputs
    hmc._check_massfunc(massfunc)
    hmc._check_prof(prof)

    na = len(a_use)
    nk = len(k_use)
    out = np.zeros([na, nk])
    for ia, aa in enumerate(a_use):
        # Evaluate mass function
        mf = hmc._hmf(massfunc, cosmo, aa, mass_def=mass_def)
        # Evaluate offset for mass function integral
        mf0 = hmc._mf0(mf)
        # Evaluate profile
        uk = hmc._eval_profile(cosmo, prof, k_use, aa, mass_def)
        # Compute profile normalization
        norm = hmc._profile_norm(mf, mf0, cosmo, prof, aa,
                                 mass_def, normprof)
        # Compute integral
        out[ia, :] = hmc._I_0_1_from_arrays(mf, mf0, uk) * norm

    if np.ndim(a) == 0:
        out = np.squeeze(out, axis=0)
    if np.ndim(k) == 0:
        out = np.squeeze(out, axis=-1)
    return out


def halomod_bias_1pt(cosmo, hmc, k, a, massfunc, hbias, prof,
                     normprof=False, mass_def=None):
    """ Solves the integral:

    .. math::
        I^1_1(k,a|u) = \\int dM\\,n(M,a)\\,b(M,a)\\,
        \\langle u(k,a|M)\\rangle,

    where :math:`n(M,a)` is the halo mass function,
    :math:`b(M,a)` is the halo bias, and
    :math:`\\langle u(k,a|M)\\rangle` is the halo profile as a
    function of scale, scale factor and halo mass.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        k (float or array_like): comoving wavenumber in Mpc^-1.
        a (float or array_like): scale factor.
        massfunc (:class:`~pyccl.halos.hmfunc.MassFunc`): a mass
            function object.
        hbias (:class:`~pyccl.halos.hbias.HaloBias`): a halo bias
            object.
        prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile.
        normprof (bool): if `True`, this integral will be
            normalized by :math:`I^1_0(k\\rightarrow 0,a|u)`
            (see :meth:`~HMCalculator.mean_profile`).
        mass_def (:class:`~pyccl.halos.massdef.MassDef`): a mass
            definition object.

    Returns:
        float or array_like: integral values evaluated at each
        combination of `k` and `a`. The shape of the output will
        be `(N_a, N_k)` where `N_k` and `N_a` are the sizes of
        `k` and `a` respectively. If `k` or `a` are scalars, the
        corresponding dimension will be squeezed out on output.
    """
    a_use = np.atleast_1d(a)
    k_use = np.atleast_1d(k)

    # Check inputs
    hmc._check_massfunc(massfunc)
    hmc._check_hbias(hbias)
    hmc._check_prof(prof)

    na = len(a_use)
    nk = len(k_use)
    out = np.zeros([na, nk])
    for ia, aa in enumerate(a_use):
        # Evaluate mass function
        mf = hmc._hmf(massfunc, cosmo, aa, mass_def=mass_def)
        # Evaluate halo bias
        bf = hmc._hbf(hbias, cosmo, aa, mass_def=mass_def)
        # Evaluate offset for halo bias integral
        mbf0 = hmc._mbf0(mf, bf)
        # Evaluate profile
        uk = hmc._eval_profile(cosmo, prof, k_use, aa, mass_def)
        # Compute profile normalization
        mf0 = 1
        if normprof:
            mf0 = hmc._mf0(mf)
        norm = hmc._profile_norm(mf, mf0, cosmo, prof,
                                 aa, mass_def, normprof)
        # Compute integral
        out[ia, :] = hmc._I_1_1_from_arrays(mf, bf,
                                            mbf0, uk) * norm

    if np.ndim(a) == 0:
        out = np.squeeze(out, axis=0)
    if np.ndim(k) == 0:
        out = np.squeeze(out, axis=-1)
    return out


def halomod_power_spectrum(cosmo, hmc, k, a, massfunc, hbias, prof,
                           prof_2pt=None, prof2=None, p_of_k_a=None,
                           normprof1=False, normprof2=False,
                           mass_def=None, get_1h=True, get_2h=True):
    """ Computes the halo model power spectrum for two
    quantities defined by their respective halo profiles.
    The halo model power spectrum for two profiles
    :math:`u` and :math:`v` is:

    .. math::
        P_{u,v}(k,a) = I^2_0(k,a|u,v) +
        I^1_1(k,a|u)\\,I^1_1(k,a|v)\\,P_{\\rm lin}(k,a)

    where :math:`P_{\\rm lin}(k,a)` is the linear matter
    power spectrum, :math:`I^1_1` is defined in the documentation
    of :meth:`~HMCalculator.bias`, and

    .. math::
        I^2_0(k,a|u) = \\int dM\\,n(M,a)\\,
        \\langle u(k,a|M) v(k,a|M)\\rangle,

    where :math:`n(M,a)` is the halo mass function, and
    :math:`\\langle u(k,a|M) v(k,a|M)\\rangle` is the two-point
    moment of the two halo profiles.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        k (float or array_like): comoving wavenumber in Mpc^-1.
        a (float or array_like): scale factor.
        massfunc (:class:`~pyccl.halos.hmfunc.MassFunc`): a mass
            function object.
        hbias (:class:`~pyccl.halos.hbias.HaloBias`): a halo bias
            object.
        prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile.
        prof_2pt (:class:`Profile2pt`): a profile covariance object
            returning the the two-point moment of the two profiles
            being correlated. If `None`, the default second moment
            will be used, corresponding to the products of the means
            of both profiles.
        prof2 (:class:`~pyccl.halos.profiles.HaloProfile`): a
            second halo profile. If `None`, `prof` will be used as
            `prof2`.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If `None`,
            the power spectrum stored within `cosmo` will be used.
        normprof1 (bool): if `True`, this integral will be
            normalized by :math:`I^1_0(k\\rightarrow 0,a|u)`
            (see :meth:`~HMCalculator.mean_profile`), where
            :math:`u` is the profile represented by `prof`.
        normprof2 (bool): if `True`, this integral will be
            normalized by :math:`I^1_0(k\\rightarrow 0,a|v)`
            (see :meth:`~HMCalculator.mean_profile`), where
            :math:`v` is the profile represented by `prof2`.
        mass_def (:class:`~pyccl.halos.massdef.MassDef`): a mass
            definition object.
        get_1h (bool): if `False`, the 1-halo term (i.e. the first
            term in the first equation above) won't be computed.
        get_2h (bool): if `False`, the 2-halo term (i.e. the second
            term in the first equation above) won't be computed.

    Returns:
        float or array_like: integral values evaluated at each
        combination of `k` and `a`. The shape of the output will
        be `(N_a, N_k)` where `N_k` and `N_a` are the sizes of
        `k` and `a` respectively. If `k` or `a` are scalars, the
        corresponding dimension will be squeezed out on output.
    """
    a_use = np.atleast_1d(a)
    k_use = np.atleast_1d(k)

    # Check inputs
    hmc._check_massfunc(massfunc)
    hmc._check_hbias(hbias)
    hmc._check_prof(prof)
    if (prof2 is not None) and (not isinstance(prof2, HaloProfile)):
        raise TypeError("prof2 must be of type `HaloProfile` or `None`")
    if prof_2pt is None:
        prof_2pt = Profile2pt()
    elif not isinstance(prof_2pt, Profile2pt):
        raise TypeError("prof_2pt must be of type "
                        "`Profile2pt` or `None`")
    # Power spectrum
    if isinstance(p_of_k_a, Pk2D):
        def pkf(sf): return p_of_k_a.eval(k_use, sf, cosmo)
    elif (p_of_k_a is None) or (p_of_k_a == 'linear'):
        def pkf(sf): return linear_matter_power(cosmo, k_use, sf)
    elif p_of_k_a == 'nonlinear':
        def pkf(sf): return nonlin_matter_power(cosmo, k_use, sf)
    else:
        raise TypeError("p_of_k_a must be `None`, \'linear\', "
                        "\'nonlinear\' or a `Pk2D` object")

    na = len(a_use)
    nk = len(k_use)
    out = np.zeros([na, nk])
    for ia, aa in enumerate(a_use):
        # Evaluate mass function
        mf = hmc._hmf(massfunc, cosmo, aa, mass_def=mass_def)
        # Evaluate offset for mass function integral
        mf0 = hmc._mf0(mf)

        # Compute first profile normalization
        norm1 = hmc._profile_norm(mf, mf0, cosmo, prof,
                                  aa, mass_def, normprof1)
        # Compute second profile normalization
        if prof2 is None:
            norm2 = norm1
        else:
            norm2 = hmc._profile_norm(mf, mf0, cosmo, prof2,
                                       aa, mass_def, normprof2)
        norm = norm1 * norm2

        if get_2h:
            # Evaluate halo bias
            bf = hmc._hbf(hbias, cosmo, aa, mass_def=mass_def)
            # Evaluate offset for halo bias integral
            mbf0 = hmc._mbf0(mf, bf)
            # Evaluate first profile
            uk_1 = hmc._eval_profile(cosmo, prof, k_use, aa,
                                     mass_def)
            # Compute integral
            bk_1 = hmc._I_1_1_from_arrays(mf, bf, mbf0, uk_1)

            # Compute second bias factor
            if prof2 is None:
                bk_2 = bk_1
            else:
                # Evaluate second profile
                uk_2 = hmc._eval_profile(cosmo, prof2, k_use, aa,
                                         mass_def)
                # Compute integral
                bk_2 = hmc._I_1_1_from_arrays(mf, bf, mbf0, uk_2)

            # Compute 2-halo power spectrum
            pk_2h = pkf(aa) * bk_1 * bk_2
        else:
            pk_2h = 0.

        if get_1h:
            # 2-point profile cumulant
            uk2 = prof_2pt.fourier_2pt(prof, cosmo, k_use,
                                       hmc.mass, aa,
                                       prof2=prof2,
                                       mass_def=mass_def).T
            # Compute integral
            pk_1h = hmc._I_0_1_from_arrays(mf, mf0, uk2)
        else:
            pk_1h = 0.

        # Total power spectrum
        out[ia, :] = (pk_1h + pk_2h) * norm

    if np.ndim(a) == 0:
        out = np.squeeze(out, axis=0)
    if np.ndim(k) == 0:
        out = np.squeeze(out, axis=-1)
    return out


def halomod_Pk2D(cosmo, hmc, massfunc, hbias, prof,
                 prof_2pt=None, prof2=None, p_of_k_a=None,
                 normprof1=False, normprof2=False,
                 mass_def=None, get_1h=True, get_2h=True,
                 lk_arr=None, a_arr=None,
                 extrap_order_lok=1, extrap_order_hik=2):
    """ Returns a :class:`~pyccl.pk2d.Pk2D` object containing
    the halo-model power spectrum for two quantities defined by
    their respective halo profiles. See :meth:`~HMCalculator.pk`
    for more details about the actual calculation.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        massfunc (:class:`~pyccl.halos.hmfunc.MassFunc`): a mass
            function object.
        hbias (:class:`~pyccl.halos.hbias.HaloBias`): a halo bias
            object.
        prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile.
        prof_2pt (:class:`Profile2pt`): a profile covariance object
            returning the the two-point moment of the two profiles
            being correlated. If `None`, the default second moment
            will be used, corresponding to the products of the means
            of both profiles.
        prof2 (:class:`~pyccl.halos.profiles.HaloProfile`): a
            second halo profile. If `None`, `prof` will be used as
            `prof2`.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If `None`,
            the power spectrum stored within `cosmo` will be used.
        normprof1 (bool): if `True`, this integral will be
            normalized by :math:`I^1_0(k\\rightarrow 0,a|u)`
            (see :meth:`~HMCalculator.mean_profile`), where
            :math:`u` is the profile represented by `prof`.
        normprof2 (bool): if `True`, this integral will be
            normalized by :math:`I^1_0(k\\rightarrow 0,a|v)`
            (see :meth:`~HMCalculator.mean_profile`), where
            :math:`v` is the profile represented by `prof2`.
        mass_def (:class:`~pyccl.halos.massdef.MassDef`): a mass
            definition object.
        get_1h (bool): if `False`, the 1-halo term (i.e. the first
            term in the first equation above) won't be computed.
        get_2h (bool): if `False`, the 2-halo term (i.e. the second
            term in the first equation above) won't be computed.
        a_arr (array): an array holding values of the scale factor
            at which the halo model power spectrum should be
            calculated for interpolation. If `None`, the internal
            values used by `cosmo` will be used.
        lk_arr (array): an array holding values of the natural
            logarithm of the wavenumber (in units of Mpc^-1) at
            which the halo model power spectrum should be calculated
            for interpolation. If `None`, the internal values used
            by `cosmo` will be used.
        extrap_order_lok (int): extrapolation order to be used on
            k-values below the minimum of the splines. See
            :class:`~pyccl.pk2d.Pk2D`.
        extrap_order_hik (int): extrapolation order to be used on
            k-values above the maximum of the splines. See
            :class:`~pyccl.pk2d.Pk2D`.

    Returns:
        float or array_like: integral values evaluated at each
        combination of `k` and `a`. The shape of the output will
        be `(N_a, N_k)` where `N_k` and `N_a` are the sizes of
        `k` and `a` respectively. If `k` or `a` are scalars, the
        corresponding dimension will be squeezed out on output.
    """
    if lk_arr is None:
        status = 0
        nk = lib.get_pk_spline_nk(cosmo.cosmo)
        lk_arr, status = lib.get_pk_spline_lk(cosmo.cosmo, nk, status)
        check(status)
    if a_arr is None:
        status = 0
        na = lib.get_pk_spline_na(cosmo.cosmo)
        a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, status)
        check(status)

    pk_arr = halomod_power_spectrum(cosmo, hmc, np.exp(lk_arr), a_arr,
                                    massfunc, hbias, prof, prof_2pt=prof_2pt,
                                    prof2=prof2, p_of_k_a=p_of_k_a,
                                    normprof1=normprof1, normprof2=normprof2,
                                    mass_def=mass_def,
                                    get_1h=get_1h, get_2h=get_2h)

    pk2d = Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik,
                cosmo=cosmo, is_logp=False)
    return pk2d
