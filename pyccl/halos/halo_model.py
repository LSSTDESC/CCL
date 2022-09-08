import warnings
from .. import ccllib as lib
from .massdef import MassDef
from .hmfunc import MassFunc
from .hbias import HaloBias
from .profiles import HaloProfile, HaloProfileNFW
from .profiles_2pt import Profile2pt
from ..core import check
from ..pk2d import Pk2D
from ..tk3d import Tk3D
from ..power import linear_matter_power, nonlin_matter_power
from ..pyutils import _spline_integrate
from .. import background
from ..errors import CCLWarning
from ..parameters import physical_constants
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
        # halo mass definition
        if isinstance(mass_def, MassDef):
            self._mdef = mass_def
        elif isinstance(mass_def, str):
            self._mdef = MassDef.from_name(mass_def)()
        else:
            raise TypeError("mass_def must be of type `MassDef` "
                            "or a mass definition name string")

        # halo mass function
        if isinstance(massfunc, MassFunc):
            self._massfunc = massfunc
        elif isinstance(massfunc, str):
            nMclass = MassFunc.from_name(massfunc)
            self._massfunc = nMclass(cosmo, mass_def=self._mdef)
        else:
            raise TypeError("mass_function must be of type `MassFunc` "
                            "or a mass function name string")

        # halo bias function
        if isinstance(hbias, HaloBias):
            self._hbias = hbias
        elif isinstance(hbias, str):
            bMclass = HaloBias.from_name(hbias)
            self._hbias = bMclass(cosmo, mass_def=self._mdef)
        else:
            raise TypeError("halo_bias must be of type `HaloBias` "
                            "or a halo bias name string")

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
        rho0 = None
        if a != self._a_current_mf:
            rho0 = cosmo.rho_x(1., "matter", is_comoving=True)
            self.mf = self._massfunc.get_mass_function(cosmo, self._mass, a,
                                                       mdef_other=self._mdef)
            self.mf0 = (rho0 -
                        self._integrator(self.mf * self._mass,
                                         self._lmass)) / self._m0
            self._a_current_mf = a

        if get_bf:
            if a != self._a_current_bf:
                if rho0 is None:
                    rho0 = cosmo.rho_x(1., "matter", is_comoving=True)
                self.bf = self._hbias.get_halo_bias(cosmo, self._mass, a,
                                                    mdef_other=self._mdef)
                self.mbf0 = (rho0 -
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

    def number_counts(self, cosmo, sel, na=128, amin=None, amax=1.0):
        """ Solves the integral:

        .. math::
            nc(sel) = \\int dM\\int da\\,\\frac{dV}{dad\\Omega}\\,n(M,a)\\,sel(M,a)

        where :math:`n(M,a)` is the halo mass function, and
        :math:`sel(M,a)` is the selection function as a function of halo mass
        and scale factor.

        Note that the selection function is normalized to integrate to unity and
        assumed to represent the selection probaility per unit scale factor and
        per unit mass.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
            sel (callable): function of mass and scale factor that returns the
                selection function. This function should take in floats or arrays
                with a signature ``sel(m, a)`` and return an array with shape
                ``(len(m), len(a))`` according to the numpy broadcasting rules.
            na (int): number of samples in scale factor to be used in
                the integrals. Default: 128.
            amin (float): the minimum scale factor at which to start integrals
                over the selection function.
                Default: value of ``cosmo.cosmo.spline_params.A_SPLINE_MIN``
            amax (float): the maximum scale factor at which to end integrals
                over the selection function.
                Default: 1.0

        Returns:
            float: the total number of clusters
        """  # noqa

        # get a values for integral
        if amin is None:
            amin = cosmo.cosmo.spline_params.A_SPLINE_MIN
        a = np.linspace(amin, amax, na)

        # compute the volume element
        abs_dzda = 1 / a / a
        dc = background.comoving_angular_distance(cosmo, a)
        ez = background.h_over_h0(cosmo, a)
        dh = physical_constants.CLIGHT_HMPC / cosmo['h']
        dvdz = dh * dc**2 / ez
        dvda = dvdz * abs_dzda

        # now do m intergrals in a loop
        mint = np.zeros_like(a)
        for i, _a in enumerate(a):
            self._get_ingredients(_a, cosmo, False)
            _selm = np.atleast_2d(sel(self._mass, _a)).T
            mint[i] = self._integrator(
                dvda[i] * self.mf[..., :] * _selm[..., :],
                self._lmass
            )

        # now do scale factor integral
        mtot = self._integrator(mint, a)

        return mtot

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

    def I_1_2(self, cosmo, k, a, prof1, prof_2pt, prof2=None):
        """ Solves the integral:

        .. math::
            I^1_2(k,a|u,v) = \\int dM\\,n(M,a)\\,b(M,a)\\,
            \\langle u(k,a|M) v(k,a|M)\\rangle,

        where :math:`n(M,a)` is the halo mass function,
        :math:`b(M,a)` is the halo bias, and
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
        self._get_ingredients(a, cosmo, True)
        uk = prof_2pt.fourier_2pt(prof1, cosmo, k, self._mass, a,
                                  prof2=prof2,
                                  mass_def=self._mdef).T
        i02 = self._integrate_over_mbf(uk)
        return i02

    def I_0_22(self, cosmo, k, a,
               prof1, prof12_2pt, prof2=None,
               prof3=None, prof34_2pt=None, prof4=None):
        """ Solves the integral:

        .. math::
            I^0_{2,2}(k_u,k_v,a|u_{1,2},v_{1,2}) =
            \\int dM\\,n(M,a)\\,
            \\langle u_1(k_u,a|M) u_2(k_u,a|M)\\rangle
            \\langle v_1(k_v,a|M) v_2(k_v,a|M)\\rangle,

        where :math:`n(M,a)` is the halo mass function, and
        :math:`\\langle u(k,a|M) v(k,a|M)\\rangle` is the
        two-point moment of the two halo profiles.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
            k (float or array_like): comoving wavenumber in Mpc^-1.
            a (float): scale factor.
            prof1 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
                profile.
            prof12_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
                a profile covariance object returning the the
                two-point moment of `prof1` and `prof2`.
            prof2 (:class:`~pyccl.halos.profiles.HaloProfile`): a
                second halo profile. If `None`, `prof1` will be used as
                `prof2`.
            prof3 (:class:`~pyccl.halos.profiles.HaloProfile`): a
                second halo profile. If `None`, `prof1` will be used as
                `prof3`.
            prof34_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
                a profile covariance object returning the the
                two-point moment of `prof3` and `prof4`.
            prof4 (:class:`~pyccl.halos.profiles.HaloProfile`): a
                second halo profile. If `None`, `prof3` will be used as
                `prof4`.

        Returns:
             float or array_like: integral values evaluated at each
             value of `k`.
        """
        if prof3 is None:
            prof3 = prof1
        if prof34_2pt is None:
            prof34_2pt = prof12_2pt

        self._get_ingredients(a, cosmo, False)
        uk12 = prof12_2pt.fourier_2pt(prof1, cosmo, k, self._mass, a,
                                      prof2=prof2, mass_def=self._mdef).T
        uk34 = prof34_2pt.fourier_2pt(prof3, cosmo, k, self._mass, a,
                                      prof2=prof4, mass_def=self._mdef).T
        i04 = self._integrate_over_mf(uk12[None, :, :] * uk34[:, None, :])
        return i04


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
                           get_1h=True, get_2h=True,
                           smooth_transition=None, supress_1h=None):
    """ Computes the halo model power spectrum for two
    quantities defined by their respective halo profiles.
    The halo model power spectrum for two profiles
    :math:`u` and :math:`v` is:

    .. math::
        P_{u,v}(k,a) = I^0_2(k,a|u,v) +
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
        smooth_transition (function or None):
            Modify the halo model 1-halo/2-halo transition region
            via a time-dependent function :math:`\\alpha(a)`,
            defined as in HMCODE-2020 (``arXiv:2009.01858``): :math:`P(k,a)=
            (P_{1h}^{\\alpha(a)}(k)+P_{2h}^{\\alpha(a)}(k))^{1/\\alpha}`.
            If `None` the extra factor is just 1.
        supress_1h (function or None):
            Supress the 1-halo large scale contribution by a
            time- and scale-dependent function :math:`k_*(a)`,
            defined as in HMCODE-2020 (``arXiv:2009.01858``):
            :math:`\\frac{(k/k_*(a))^4}{1+(k/k_*(a))^4}`.
            If `None` the standard 1-halo term is returned with no damping.

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
    if smooth_transition is not None:
        if not (get_1h and get_2h):
            raise ValueError("transition region can only be modified "
                             "when both 1-halo and 2-halo terms are queried")
        if not hasattr(smooth_transition, "__call__"):
            raise TypeError("smooth_transition must be "
                            "a function of `a` or None")
    if supress_1h is not None:
        if not get_1h:
            raise ValueError("can't supress the 1-halo term "
                             "when get_1h is False")
        if not hasattr(supress_1h, "__call__"):
            raise TypeError("supress_1h must be "
                            "a function of `a` or None")

    # Power spectrum
    if isinstance(p_of_k_a, Pk2D):
        def pkf(sf):
            return p_of_k_a.eval(k_use, sf, cosmo)
    elif (p_of_k_a is None) or (str(p_of_k_a) == 'linear'):
        def pkf(sf):
            return linear_matter_power(cosmo, k_use, sf)
    elif str(p_of_k_a) == 'nonlinear':
        def pkf(sf):
            return nonlin_matter_power(cosmo, k_use, sf)
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
            pk_1h = hmc.I_0_2(cosmo, k_use, aa, prof, prof_2pt, prof2)
            if supress_1h is not None:
                ks = supress_1h(aa)
                pk_1h *= (k_use / ks)**4 / (1 + (k_use / ks)**4)
        else:
            pk_1h = 0.

        # Transition region
        if smooth_transition is None:
            out[ia, :] = (pk_1h + pk_2h) * norm
        else:
            alpha = smooth_transition(aa)
            out[ia, :] = (pk_1h**alpha + pk_2h**alpha)**(1/alpha) * norm

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
                 extrap_order_lok=1, extrap_order_hik=2,
                 smooth_transition=None, supress_1h=None):
    """ Returns a :class:`~pyccl.pk2d.Pk2D` object containing
    the halo-model power spectrum for two quantities defined by
    their respective halo profiles. See :meth:`halomod_power_spectrum`
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
        smooth_transition (function or None):
            Modify the halo model 1-halo/2-halo transition region
            via a time-dependent function :math:`\\alpha(a)`,
            defined as in HMCODE-2020 (``arXiv:2009.01858``): :math:`P(k,a)=
            (P_{1h}^{\\alpha(a)}(k)+P_{2h}^{\\alpha(a)}(k))^{1/\\alpha}`.
            If `None` the extra factor is just 1.
        supress_1h (function or None):
            Supress the 1-halo large scale contribution by a
            time- and scale-dependent function :math:`k_*(a)`,
            defined as in HMCODE-2020 (``arXiv:2009.01858``):
            :math:`\\frac{(k/k_*(a))^4}{1+(k/k_*(a))^4}`.
            If `None` the standard 1-halo term is returned with no damping.

    Returns:
        :class:`~pyccl.pk2d.Pk2D`: halo model power spectrum.
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

    pk_arr = halomod_power_spectrum(cosmo, hmc, np.exp(lk_arr), a_arr,
                                    prof, prof_2pt=prof_2pt,
                                    prof2=prof2, p_of_k_a=p_of_k_a,
                                    normprof1=normprof1, normprof2=normprof2,
                                    get_1h=get_1h, get_2h=get_2h,
                                    smooth_transition=smooth_transition,
                                    supress_1h=supress_1h)

    pk2d = Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik,
                cosmo=cosmo, is_logp=False)
    return pk2d


def halomod_trispectrum_1h(cosmo, hmc, k, a,
                           prof1, prof2=None, prof12_2pt=None,
                           prof3=None, prof4=None, prof34_2pt=None,
                           normprof1=False, normprof2=False,
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
        prof1 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_1` above.
        prof2 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_2` above. If `None`,
            `prof1` will be used as `prof2`.
        prof12_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object returning the the two-point
            moment of `prof1` and `prof2`. If `None`, the default
            second moment will be used, corresponding to the
            products of the means of both profiles.
        prof3 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_1` above. If `None`,
            `prof1` will be used as `prof3`.
        prof4 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_2` above. If `None`,
            `prof3` will be used as `prof4`.
        prof34_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof12_2pt` for `prof3` and `prof4`.
        normprof1 (bool): if `True`, this integral will be
            normalized by :math:`I^0_1(k\\rightarrow 0,a|u)`
            (see :meth:`~HMCalculator.I_0_1`), where
            :math:`u` is the profile represented by `prof1`.
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
    a_use = np.atleast_1d(a)
    k_use = np.atleast_1d(k)

    # Check inputs
    if not isinstance(prof1, HaloProfile):
        raise TypeError("prof1 must be of type `HaloProfile`")
    if (prof2 is not None) and (not isinstance(prof2, HaloProfile)):
        raise TypeError("prof2 must be of type `HaloProfile` or `None`")
    if (prof3 is not None) and (not isinstance(prof3, HaloProfile)):
        raise TypeError("prof3 must be of type `HaloProfile` or `None`")
    if (prof4 is not None) and (not isinstance(prof4, HaloProfile)):
        raise TypeError("prof4 must be of type `HaloProfile` or `None`")
    if prof12_2pt is None:
        prof12_2pt = Profile2pt()
    elif not isinstance(prof12_2pt, Profile2pt):
        raise TypeError("prof12_2pt must be of type "
                        "`Profile2pt` or `None`")
    if (prof34_2pt is not None) and (not isinstance(prof34_2pt, Profile2pt)):
        raise TypeError("prof34_2pt must be of type `Profile2pt` or `None`")

    def get_norm(normprof, prof, sf):
        if normprof:
            return hmc.profile_norm(cosmo, sf, prof)
        else:
            return 1

    na = len(a_use)
    nk = len(k_use)
    out = np.zeros([na, nk, nk])
    for ia, aa in enumerate(a_use):
        # Compute profile normalizations
        norm1 = get_norm(normprof1, prof1, aa)
        # Compute second profile normalization
        if prof2 is None:
            norm2 = norm1
        else:
            norm2 = get_norm(normprof2, prof2, aa)
        if prof3 is None:
            norm3 = norm1
        else:
            norm3 = get_norm(normprof3, prof3, aa)
        if prof4 is None:
            norm4 = norm3
        else:
            norm4 = get_norm(normprof4, prof4, aa)

        norm = norm1 * norm2 * norm3 * norm4

        # Compute trispectrum at this redshift
        tk_1h = hmc.I_0_22(cosmo, k_use, aa,
                           prof1, prof12_2pt, prof2=prof2,
                           prof3=prof3, prof34_2pt=prof34_2pt,
                           prof4=prof4)

        # Normalize
        out[ia, :, :] = tk_1h * norm

    if np.ndim(a) == 0:
        out = np.squeeze(out, axis=0)
    if np.ndim(k) == 0:
        out = np.squeeze(out, axis=-1)
        out = np.squeeze(out, axis=-1)
    return out


def halomod_Tk3D_1h(cosmo, hmc,
                    prof1, prof2=None, prof12_2pt=None,
                    prof3=None, prof4=None, prof34_2pt=None,
                    normprof1=False, normprof2=False,
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
        prof1 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_1` above.
        prof2 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_2` above. If `None`,
            `prof1` will be used as `prof2`.
        prof12_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object returning the the two-point
            moment of `prof1` and `prof2`. If `None`, the default
            second moment will be used, corresponding to the
            products of the means of both profiles.
        prof3 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_1` above. If `None`,
            `prof1` will be used as `prof3`.
        prof4 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_2` above. If `None`,
            `prof3` will be used as `prof4`.
        prof34_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof12_2pt` for `prof3` and `prof4`.
        normprof1 (bool): if `True`, this integral will be
            normalized by :math:`I^0_1(k\\rightarrow 0,a|u)`
            (see :meth:`~HMCalculator.I_0_1`), where
            :math:`u` is the profile represented by `prof1`.
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
                                 prof1, prof2=prof2,
                                 prof12_2pt=prof12_2pt,
                                 prof3=prof3, prof4=prof4,
                                 prof34_2pt=prof34_2pt,
                                 normprof1=normprof1, normprof2=normprof2,
                                 normprof3=normprof3, normprof4=normprof4)
    if use_log:
        if np.any(tkk <= 0):
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


def halomod_Tk3D_SSC_linear_bias(cosmo, hmc, prof, bias1=1, bias2=1, bias3=1,
                                 bias4=1,
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
        i12 = hmc.I_1_2(cosmo, k_use, aa, prof, prof_2pt, prof) * norm

        pk = pk2d.eval(k_use, aa, cosmo)
        dpk = pk2d.eval_dlogpk_dlogk(k_use, aa, cosmo)
        # ~ [(47/21 - 1/3 dlogPk/dlogk) * Pk+I12]
        dpk12[ia] = ((2.2380952381-dpk/3)*pk + i12)
        dpk34[ia] = dpk12[ia].copy()  # Avoid surprises

        # Counter terms for clustering (i.e. - (bA + bB) * PAB
        if is_number_counts1 or is_number_counts2 or is_number_counts3 or \
           is_number_counts4:
            b1 = b2 = b3 = b4 = 0

            i02 = hmc.I_0_2(cosmo, k_use, aa, prof, prof_2pt, prof) * norm
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


def halomod_Tk3D_SSC(cosmo, hmc,
                     prof1, prof2=None, prof12_2pt=None,
                     prof3=None, prof4=None, prof34_2pt=None,
                     normprof1=False, normprof2=False,
                     normprof3=False, normprof4=False,
                     p_of_k_a=None, lk_arr=None, a_arr=None,
                     extrap_order_lok=1, extrap_order_hik=1,
                     use_log=False):
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
        prof1 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_1` above.
        prof2 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_2` above. If `None`,
            `prof1` will be used as `prof2`.
        prof12_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object returning the the two-point
            moment of `prof1` and `prof2`. If `None`, the default
            second moment will be used, corresponding to the
            products of the means of both profiles.
        prof3 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_1` above. If `None`,
            `prof1` will be used as `prof3`.
        prof4 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_2` above. If `None`,
            `prof3` will be used as `prof4`.
        prof34_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof12_2pt` for `prof3` and `prof4`.
        normprof1 (bool): if `True`, this integral will be
            normalized by :math:`I^0_1(k\\rightarrow 0,a|u)`
            (see :meth:`~HMCalculator.I_0_1`), where
            :math:`u` is the profile represented by `prof1`.
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

    # Check inputs
    if not isinstance(prof1, HaloProfile):
        raise TypeError("prof1 must be of type `HaloProfile`")
    if (prof2 is not None) and (not isinstance(prof2, HaloProfile)):
        raise TypeError("prof2 must be of type `HaloProfile` or `None`")
    if (prof3 is not None) and (not isinstance(prof3, HaloProfile)):
        raise TypeError("prof3 must be of type `HaloProfile` or `None`")
    if (prof4 is not None) and (not isinstance(prof4, HaloProfile)):
        raise TypeError("prof4 must be of type `HaloProfile` or `None`")
    if prof12_2pt is None:
        prof12_2pt = Profile2pt()
    elif not isinstance(prof12_2pt, Profile2pt):
        raise TypeError("prof12_2pt must be of type "
                        "`Profile2pt` or `None`")
    if (prof34_2pt is not None) and (not isinstance(prof34_2pt, Profile2pt)):
        raise TypeError("prof34_2pt must be of type `Profile2pt` or `None`")

    # number counts profiles must be normalized
    profs = {prof1: normprof1, prof2: normprof2,
             prof3: normprof3, prof4: normprof4}

    for i, (profile, normalization) in enumerate(profs.items()):
        if (profile is not None
                and profile.is_number_counts
                and not normalization):
            raise ValueError(
                f"normprof{i+1} must be True if prof{i+1} is number counts")

    if prof3 is None:
        prof3_bak = prof1
    else:
        prof3_bak = prof3
    if prof34_2pt is None:
        prof34_2pt_bak = prof12_2pt
    else:
        prof34_2pt_bak = prof34_2pt

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

    def get_norm(normprof, prof, sf):
        if normprof:
            return hmc.profile_norm(cosmo, sf, prof)
        else:
            return 1

    na = len(a_arr)
    nk = len(k_use)
    dpk12 = np.zeros([na, nk])
    dpk34 = np.zeros([na, nk])
    for ia, aa in enumerate(a_arr):
        # Compute profile normalizations
        norm1 = get_norm(normprof1, prof1, aa)
        i11_1 = hmc.I_1_1(cosmo, k_use, aa, prof1)
        # Compute second profile normalization
        if prof2 is None:
            norm2 = norm1
            i11_2 = i11_1
        else:
            norm2 = get_norm(normprof2, prof2, aa)
            i11_2 = hmc.I_1_1(cosmo, k_use, aa, prof2)
        if prof3 is None:
            norm3 = norm1
            i11_3 = i11_1
        else:
            norm3 = get_norm(normprof3, prof3, aa)
            i11_3 = hmc.I_1_1(cosmo, k_use, aa, prof3)
        if prof4 is None:
            norm4 = norm3
            i11_4 = i11_3
        else:
            norm4 = get_norm(normprof4, prof4, aa)
            i11_4 = hmc.I_1_1(cosmo, k_use, aa, prof4)

        i12_12 = hmc.I_1_2(cosmo, k_use, aa, prof1,
                           prof12_2pt, prof2)
        if (prof3 is None) and (prof4 is None) and (prof34_2pt is None):
            i12_34 = i12_12
        else:
            i12_34 = hmc.I_1_2(cosmo, k_use, aa, prof3_bak,
                               prof34_2pt_bak, prof4)
        norm12 = norm1 * norm2
        norm34 = norm3 * norm4

        pk = pk2d.eval(k_use, aa, cosmo)
        dpk = pk2d.eval_dlogpk_dlogk(k_use, aa, cosmo)
        # (47/21 - 1/3 dlogPk/dlogk) * I11 * I11 * Pk+I12
        dpk12[ia, :] = norm12*((2.2380952381-dpk/3)*i11_1*i11_2*pk+i12_12)
        dpk34[ia, :] = norm34*((2.2380952381-dpk/3)*i11_3*i11_4*pk+i12_34)

        # Counter terms for clustering (i.e. - (bA + bB) * PAB
        if prof1.is_number_counts or (prof2 is None or prof2.is_number_counts):
            b1 = b2 = np.zeros_like(k_use)
            i02_12 = hmc.I_0_2(cosmo, k_use, aa, prof1, prof12_2pt, prof2)
            P_12 = norm12 * (pk * i11_1 * i11_2 + i02_12)

            if prof1.is_number_counts:
                b1 = i11_1 * norm1

            if prof2 is None:
                b2 = b1
            elif prof2.is_number_counts:
                b2 = i11_2 * norm2

            dpk12[ia, :] -= (b1 + b2) * P_12

        if prof3_bak.is_number_counts or \
                ((prof3_bak.is_number_counts and prof4 is None) or
                 (prof4 is not None) and prof4.is_number_counts):
            b3 = b4 = np.zeros_like(k_use)
            if (prof3 is None) and (prof4 is None) and (prof34_2pt is None):
                i02_34 = i02_12
            else:
                i02_34 = hmc.I_0_2(cosmo, k_use, aa, prof3_bak, prof34_2pt_bak,
                                   prof4)
            P_34 = norm34 * (pk * i11_3 * i11_4 + i02_34)

            if prof3 is None:
                b3 = b1
            elif prof3.is_number_counts:
                b3 = i11_3 * norm3

            if prof4 is None:
                b4 = b3
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
