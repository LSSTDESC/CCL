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
        massfunc (:class:`~pyccl.halos.hmfunc.MassFunc`): a mass
            function object.
        hbias (:class:`~pyccl.halos.hbias.HaloBias`): a halo bias
            object.
        mass_def (:class:`~pyccl.halos.massdef.MassDef`): a mass
            definition object.
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
    def __init__(self, cosmo, massfunc, hbias, mass_def,
                 log10M_min=8., log10M_max=16.,
                 nlog10M=128, integration_method_M='simpson',
                 k_min=1E-5):
        self._rho0 = rho_x(cosmo, 1., 'matter', is_comoving=True)
        if not isinstance(massfunc, MassFunc):
            raise TypeError("massfunc must be of type `MassFunc`")
        self._massfunc = massfunc
        if not isinstance(hbias, HaloBias):
            raise TypeError("hbias must be of type `HaloBias`")
        self._hbias = hbias
        self._mdef = mass_def
        self._prec = {'log10M_min': log10M_min,
                      'log10M_max': log10M_max,
                      'nlog10M': nlog10M,
                      'integration_method_M': integration_method_M,
                      'k_min': k_min}
        self._lmass = np.linspace(self._prec['log10M_min'],
                                  self._prec['log10M_max'],
                                  self._prec['nlog10M'])
        self._mass = 10.**self._lmass
        self._m0 = self._mass[0]

        if self._prec['integration_method_M'] not in ['spline',
                                                      'simpson']:
            raise NotImplementedError("Only \'simpson\' and 'spline' "
                                      "supported as integration methods")
        elif self._prec['integration_method_M'] == 'simpson':
            from scipy.integrate import simps
            self._integrator = simps
        else:
            self._integrator = self._integ_spline

        self._a_current_mf = -1
        self._a_current_bf = -1

    def _integ_spline(self, fM, lM):
        # Spline integrator
        return _spline_integrate(lM, fM, lM[0], lM[-1])

    def _get_ingredients(self, a, cosmo, get_bf):
        # Compute mass function and bias (if needed) at a new
        # value of the scale factor.
        if a != self._a_current_mf:
            self.mf = self._massfunc.get_mass_function(cosmo, self._mass, a,
                                                       mdef_other=self._mdef)
            self.mf0 = (self._rho0 -
                        self._integrator(self.mf * self._mass,
                                         self._lmass)) / self._m0
            self._a_current_mf = a

        if get_bf:
            if a != self._a_current_bf:
                self.bf = self._hbias.get_halo_bias(cosmo, self._mass, a,
                                                    mdef_other=self._mdef)
                self.mbf0 = (self._rho0 -
                             self._integrator(self.mf * self.bf * self._mass,
                                              self._lmass)) / self._m0
            self._a_current_bf = a

    def _integrate_over_mf(self, array_2):
        i1 = self._integrator(self.mf[..., :] * array_2,
                              self._lmass)
        return i1 + self.mf0 * array_2[..., 0]

    def _integrate_over_mbf(self, array_2):
        i1 = self._integrator((self.mf * self.bf)[..., :] * array_2,
                              self._lmass)
        return i1 + self.mbf0 * array_2[..., 0]

    def profile_norm(self, cosmo, a, prof):
        """ Returns :math:`I^0_1(k\\rightarrow0,a|u)`
        (see :meth:`~HMCalculator.I_0_1`).

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
            a (float): scale factor.
            prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
                profile.

        Returns:
            float or array_like: integral value.
        """
        # Compute mass function
        self._get_ingredients(a, cosmo, False)
        uk0 = prof.fourier(cosmo, self._prec['k_min'],
                           self._mass, a, mass_def=self._mdef).T
        norm = 1. / self._integrate_over_mf(uk0)
        return norm

    def I_0_1(self, cosmo, k, a, prof):
        """ Solves the integral:

        .. math::
            I^0_1(k,a|u) = \\int dM\\,n(M,a)\\,\\langle u(k,a|M)\\rangle,

        where :math:`n(M,a)` is the halo mass function, and
        :math:`\\langle u(k,a|M)\\rangle` is the halo profile as a
        function of scale, scale factor and halo mass.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
            k (float or array_like): comoving wavenumber in Mpc^-1.
            a (float): scale factor.
            prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
                profile.

        Returns:
            float or array_like: integral values evaluated at each
            value of `k`.
        """
        # Compute mass function
        self._get_ingredients(a, cosmo, False)
        uk = prof.fourier(cosmo, k, self._mass, a,
                          mass_def=self._mdef).T
        i01 = self._integrate_over_mf(uk)
        return i01

    def I_1_1(self, cosmo, k, a, prof):
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
            k (float or array_like): comoving wavenumber in Mpc^-1.
            a (float): scale factor.
            prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
                profile.

        Returns:
            float or array_like: integral values evaluated at each
            value of `k`.
        """
        # Compute mass function and halo bias
        self._get_ingredients(a, cosmo, True)
        uk = prof.fourier(cosmo, k, self._mass, a,
                          mass_def=self._mdef).T
        i11 = self._integrate_over_mbf(uk)
        return i11

    def I_0_2(self, cosmo, k, a, prof1, prof_2pt, prof2=None):
        """ Solves the integral:

        .. math::
            I^0_2(k,a|u,v) = \\int dM\\,n(M,a)\\,
            \\langle u(k,a|M) v(k,a|M)\\rangle,

        where :math:`n(M,a)` is the halo mass function, and
        :math:`\\langle u(k,a|M) v(k,a|M)\\rangle` is the two-point
        moment of the two halo profiles.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
            k (float or array_like): comoving wavenumber in Mpc^-1.
            a (float): scale factor.
            prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
                profile.
            prof_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
                a profile covariance object
                returning the the two-point moment of the two profiles
                being correlated.
            prof2 (:class:`~pyccl.halos.profiles.HaloProfile`): a
                second halo profile. If `None`, `prof` will be used as
                `prof2`.

        Returns:
             float or array_like: integral values evaluated at each
             value of `k`.
        """
        # Compute mass function
        self._get_ingredients(a, cosmo, False)
        uk = prof_2pt.fourier_2pt(prof1, cosmo, k, self._mass, a,
                                  prof2=prof2,
                                  mass_def=self._mdef).T
        i02 = self._integrate_over_mf(uk)
        return i02


def halomod_mean_profile_1pt(cosmo, hmc, k, a, prof,
                             normprof=False):
    """ Returns the mass-weighted mean halo profile.

    .. math::
        I^0_1(k,a|u) = \\int dM\\,n(M,a)\\,\\langle u(k,a|M)\\rangle,

    where :math:`n(M,a)` is the halo mass function, and
    :math:`\\langle u(k,a|M)\\rangle` is the halo profile as a
    function of scale, scale factor and halo mass.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        k (float or array_like): comoving wavenumber in Mpc^-1.
        a (float or array_like): scale factor.
        prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile.
        normprof (bool): if `True`, this integral will be
            normalized by :math:`I^0_1(k\\rightarrow 0,a|u)`.

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
    if not isinstance(prof, HaloProfile):
        raise TypeError("prof must be of type `HaloProfile`")

    na = len(a_use)
    nk = len(k_use)
    out = np.zeros([na, nk])
    for ia, aa in enumerate(a_use):
        i01 = hmc.I_0_1(cosmo, k_use, aa, prof)
        if normprof:
            norm = hmc.profile_norm(cosmo, aa, prof)
            i01 *= norm
        out[ia, :] = i01

    if np.ndim(a) == 0:
        out = np.squeeze(out, axis=0)
    if np.ndim(k) == 0:
        out = np.squeeze(out, axis=-1)
    return out


def halomod_bias_1pt(cosmo, hmc, k, a, prof, normprof=False):
    """ Returns the mass-and-bias-weighted mean halo profile.

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
        prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile.
        normprof (bool): if `True`, this integral will be
            normalized by :math:`I^0_1(k\\rightarrow 0,a|u)`
            (see :meth:`~HMCalculator.I_0_1`).

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
    if not isinstance(prof, HaloProfile):
        raise TypeError("prof must be of type `HaloProfile`")

    na = len(a_use)
    nk = len(k_use)
    out = np.zeros([na, nk])
    for ia, aa in enumerate(a_use):
        i11 = hmc.I_1_1(cosmo, k_use, aa, prof)
        if normprof:
            norm = hmc.profile_norm(cosmo, aa, prof)
            i11 *= norm
        out[ia, :] = i11

    if np.ndim(a) == 0:
        out = np.squeeze(out, axis=0)
    if np.ndim(k) == 0:
        out = np.squeeze(out, axis=-1)
    return out


def halomod_power_spectrum(cosmo, hmc, k, a, prof,
                           prof_2pt=None, prof2=None, p_of_k_a=None,
                           normprof1=False, normprof2=False,
                           get_1h=True, get_2h=True):
    """ Computes the halo model power spectrum for two
    quantities defined by their respective halo profiles.
    The halo model power spectrum for two profiles
    :math:`u` and :math:`v` is:

    .. math::
        P_{u,v}(k,a) = I^2_0(k,a|u,v) +
        I^1_1(k,a|u)\\,I^1_1(k,a|v)\\,P_{\\rm lin}(k,a)

    where :math:`P_{\\rm lin}(k,a)` is the linear matter
    power spectrum, :math:`I^1_1` is defined in the documentation
    of :meth:`~HMCalculator.I_1_1`, and :math:`I^0_2` is defined
    in the documentation of :meth:`~HMCalculator.I_0_2`.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        k (float or array_like): comoving wavenumber in Mpc^-1.
        a (float or array_like): scale factor.
        prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile.
        prof_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object
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
            normalized by :math:`I^0_1(k\\rightarrow 0,a|u)`
            (see :meth:`~HMCalculator.I_0_1`), where
            :math:`u` is the profile represented by `prof`.
        normprof2 (bool): if `True`, this integral will be
            normalized by :math:`I^0_1(k\\rightarrow 0,a|v)`
            (see :meth:`~HMCalculator.I_0_1`), where
            :math:`v` is the profile represented by `prof2`.
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
    if not isinstance(prof, HaloProfile):
        raise TypeError("prof must be of type `HaloProfile`")
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
        # Compute first profile normalization
        if normprof1:
            norm1 = hmc.profile_norm(cosmo, aa, prof)
        else:
            norm1 = 1
        # Compute second profile normalization
        if prof2 is None:
            norm2 = norm1
        else:
            if normprof2:
                norm2 = hmc.profile_norm(cosmo, aa, prof2)
            else:
                norm2 = 1
        norm = norm1 * norm2

        if get_2h:
            # Compute first bias factor
            i11_1 = hmc.I_1_1(cosmo, k_use, aa, prof)

            # Compute second bias factor
            if prof2 is None:
                i11_2 = i11_1
            else:
                i11_2 = hmc.I_1_1(cosmo, k_use, aa, prof2)

            # Compute 2-halo power spectrum
            pk_2h = pkf(aa) * i11_1 * i11_2
        else:
            pk_2h = 0.

        if get_1h:
            pk_1h = hmc.I_0_2(cosmo, k_use, aa, prof,
                              prof_2pt, prof2)
        else:
            pk_1h = 0.

        # Total power spectrum
        out[ia, :] = (pk_1h + pk_2h) * norm

    if np.ndim(a) == 0:
        out = np.squeeze(out, axis=0)
    if np.ndim(k) == 0:
        out = np.squeeze(out, axis=-1)
    return out


def halomod_Pk2D(cosmo, hmc, prof,
                 prof_2pt=None, prof2=None, p_of_k_a=None,
                 normprof1=False, normprof2=False,
                 get_1h=True, get_2h=True,
                 lk_arr=None, a_arr=None,
                 extrap_order_lok=1, extrap_order_hik=2):
    """ Returns a :class:`~pyccl.pk2d.Pk2D` object containing
    the halo-model power spectrum for two quantities defined by
    their respective halo profiles. See :meth:`~HMCalculator.pk`
    for more details about the actual calculation.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile.
        prof_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object
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
            normalized by :math:`I^0_1(k\\rightarrow 0,a|u)`
            (see :meth:`~HMCalculator.I_0_1`), where
            :math:`u` is the profile represented by `prof`.
        normprof2 (bool): if `True`, this integral will be
            normalized by :math:`I^0_1(k\\rightarrow 0,a|v)`
            (see :meth:`~HMCalculator.I_0_1`), where
            :math:`v` is the profile represented by `prof2`.
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
        :class:`~pyccl.pk2d.Pk2D`: halo model power spectrum.
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
                                    prof, prof_2pt=prof_2pt,
                                    prof2=prof2, p_of_k_a=p_of_k_a,
                                    normprof1=normprof1, normprof2=normprof2,
                                    get_1h=get_1h, get_2h=get_2h)

    pk2d = Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik,
                cosmo=cosmo, is_logp=False)
    return pk2d
