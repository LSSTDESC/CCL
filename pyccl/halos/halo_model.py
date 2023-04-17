from .massdef import MassDef
from .hmfunc import MassFunc
from .hbias import HaloBias
from ..pyutils import _spline_integrate
from .. import background
from ..base import CCLAutoreprObject, unlock_instance
from ..parameters import physical_constants as const
import numpy as np


__all__ = ("HMCalculator",)


class HMCalculator(CCLAutoreprObject):
    """This class implements a set of methods that can be used to
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
    __repr_attrs__ = __eq_attrs__ = ("_massfunc", "_hbias", "_mdef", "_prec",)

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

        # Cache last results for mass function and halo bias.
        self._cosmo_mf = self._cosmo_bf = None
        self._a_mf = self._a_bf = -1

    def _integ_spline(self, fM, lM):
        # Spline integrator
        return _spline_integrate(lM, fM, lM[0], lM[-1])

    @unlock_instance(mutate=False)
    def _get_mass_function(self, cosmo, a, rho0):
        # Compute the mass function at this cosmo and a.
        if a != self._a_mf or cosmo != self._cosmo_mf:
            massfunc = self._massfunc.get_mass_function
            self._mf = massfunc(cosmo, self._mass, a)
            integ = self._integrator(self._mf*self._mass, self._lmass)
            self._mf0 = (rho0 - integ) / self._m0
            self._cosmo_mf, self._a_mf = cosmo, a  # cache

    @unlock_instance(mutate=False)
    def _get_halo_bias(self, cosmo, a, rho0):
        # Compute the halo bias at this cosmo and a.
        if cosmo != self._cosmo_bf or a != self._a_bf:
            hbias = self._hbias.get_halo_bias
            self._bf = hbias(cosmo, self._mass, a)
            integ = self._integrator(self._mf*self._bf*self._mass, self._lmass)
            self._mbf0 = (rho0 - integ) / self._m0
            self._cosmo_bf, self._a_bf = cosmo, a  # cache

    def _get_ingredients(self, cosmo, a, get_bf):
        """Compute mass function and halo bias at some scale factor."""
        rho0 = const.RHO_CRITICAL * cosmo["Omega_m"] * cosmo["h"]**2
        self._get_mass_function(cosmo, a, rho0)
        if get_bf:
            self._get_halo_bias(cosmo, a, rho0)

    def _integrate_over_mf(self, array_2):
        i1 = self._integrator(self._mf[..., :] * array_2,
                              self._lmass)
        return i1 + self._mf0 * array_2[..., 0]

    def _integrate_over_mbf(self, array_2):
        i1 = self._integrator((self._mf * self._bf)[..., :] * array_2,
                              self._lmass)
        return i1 + self._mbf0 * array_2[..., 0]

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
        self._get_ingredients(cosmo, a, False)
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
        dh = const.CLIGHT_HMPC / cosmo['h']
        dvdz = dh * dc**2 / ez
        dvda = dvdz * abs_dzda

        # now do m intergrals in a loop
        mint = np.zeros_like(a)
        for i, _a in enumerate(a):
            self._get_ingredients(cosmo, _a, False)
            _selm = np.atleast_2d(sel(self._mass, _a)).T
            mint[i] = self._integrator(
                dvda[i] * self._mf[..., :] * _selm[..., :],
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
        self._get_ingredients(cosmo, a, False)
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
        self._get_ingredients(cosmo, a, True)
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
            prof1 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
                profile.
            prof_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
                a profile covariance object
                returning the the two-point moment of the two profiles
                being correlated.
            prof2 (:class:`~pyccl.halos.profiles.HaloProfile` or None):
                a second halo profile. If `None`, `prof1` will be used as
                `prof2`.

        Returns:
             float or array_like: integral values evaluated at each
             value of `k`.
        """
        if prof2 is None:
            prof2 = prof1

        # Compute mass function
        self._get_ingredients(cosmo, a, False)
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
            prof1 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
                profile.
            prof_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
                a profile covariance object
                returning the the two-point moment of the two profiles
                being correlated.
            prof2 (:class:`~pyccl.halos.profiles.HaloProfile`): a
                second halo profile. If `None`, `prof1` will be used as
                `prof2`.

        Returns:
             float or array_like: integral values evaluated at each
             value of `k`.
        """
        if prof2 is None:
            prof2 = prof1

        # Compute mass function
        self._get_ingredients(cosmo, a, True)
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
                third halo profile. If `None`, `prof1` will be used as
                `prof3`.
            prof34_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
                a profile covariance object returning the the
                two-point moment of `prof3` and `prof4`.
            prof4 (:class:`~pyccl.halos.profiles.HaloProfile`): a
                fourth halo profile. If `None`, `prof2` will be used as
                `prof4`.

        Returns:
             float or array_like: integral values evaluated at each
             value of `k`.
        """
        if prof3 is None:
            prof3 = prof1
        if prof4 is None:
            prof4 = prof2

        if prof34_2pt is None:
            prof34_2pt = prof12_2pt

        self._get_ingredients(cosmo, a, False)
        uk12 = prof12_2pt.fourier_2pt(prof1, cosmo, k, self._mass, a,
                                      prof2=prof2, mass_def=self._mdef).T
        if (prof1, prof2) == (prof3, prof4):
            # 4pt approximation of the same profile
            uk34 = uk12
        else:
            uk34 = prof34_2pt.fourier_2pt(prof3, cosmo, k, self._mass, a,
                                          prof2=prof4, mass_def=self._mdef).T
        i04 = self._integrate_over_mf(uk12[None, :, :] * uk34[:, None, :])
        return i04
