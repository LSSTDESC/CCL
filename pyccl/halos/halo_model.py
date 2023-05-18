__all__ = ("HMCalculator",)

import numpy as np

from .. import CCLAutoRepr, unlock_instance
from .. import physical_constants as const
from . import MassDef
from ..pyutils import _spline_integrate


class HMCalculator(CCLAutoRepr):
    """This class implements a set of methods that can be used to
    compute various halo model quantities. A lot of these quantities
    will involve integrals of the sort:

    .. math::
       \\int dM\\,n(M,a)\\,f(M,k,a),

    where :math:`n(M,a)` is the halo mass function, and :math:`f` is
    an arbitrary function of mass, scale factor and Fourier scales.

    Args:
        mass_function (str or :class:`~pyccl.halos.halo_model_base.MassFunc`):
            the mass function to use
        halo_bias (str or :class:`~pyccl.halos.halo_model_base.HaloBias`):
            the halo bias function to use
        mass_def (str or :class:`~pyccl.halos.massdef.MassDef`):
            the halo mass definition to use
        log10M_min (:obj:`float`): lower bound of the mass integration range
            (base-10 logarithmic).
        log10M_max (:obj:`float`): lower bound of the mass integration range
            (base-10 logarithmic).
        nM (:obj:`int`): number of uniformly-spaced samples in :math:`\\log_{10}(M)`
            to be used in the mass integrals.
        integration_method_M (:obj:`str`): integration method to use
            in the mass integrals. Options: "simpson" and "spline".
    """ # noqa
    __repr_attrs__ = __eq_attrs__ = (
        "mass_function", "halo_bias", "mass_def", "precision",)

    def __init__(self, *, mass_function, halo_bias, mass_def=None,
                 log10M_min=8., log10M_max=16., nM=128,
                 integration_method_M='simpson'):
        # Initialize halo model ingredients.
        out = MassDef.from_specs(mass_def, mass_function=mass_function,
                                 halo_bias=halo_bias)
        if len(out) != 3:
            raise ValueError("A valid mass function and halo bias is "
                             "needed")
        self.mass_def, self.mass_function, self.halo_bias = out

        self.precision = {
            'log10M_min': log10M_min, 'log10M_max': log10M_max, 'nM': nM,
            'integration_method_M': integration_method_M}
        self._lmass = np.linspace(log10M_min, log10M_max, nM)
        self._mass = 10.**self._lmass
        self._m0 = self._mass[0]

        if integration_method_M == "simpson":
            from scipy.integrate import simpson
            self._integrator = simpson
        elif integration_method_M == "spline":
            self._integrator = self._integ_spline
        else:
            raise ValueError("Invalid integration method.")

        # Cache last results for mass function and halo bias.
        self._cosmo_mf = self._cosmo_bf = None
        self._a_mf = self._a_bf = -1

    def _integ_spline(self, fM, log10M):
        # Spline integrator
        return _spline_integrate(log10M, fM, log10M[0], log10M[-1])

    def _check_mass_def(self, *others):
        # Verify that internal & external mass definitions are consistent.
        if set([x.mass_def for x in others]) != set([self.mass_def]):
            raise ValueError("Inconsistent mass definitions.")

    @unlock_instance(mutate=False)
    def _get_mass_function(self, cosmo, a, rho0):
        # Compute the mass function at this cosmo and a.
        if a != self._a_mf or cosmo != self._cosmo_mf:
            self._mf = self.mass_function(cosmo, self._mass, a)
            integ = self._integrator(self._mf*self._mass, self._lmass)
            self._mf0 = (rho0 - integ) / self._m0
            self._cosmo_mf, self._a_mf = cosmo, a  # cache

    @unlock_instance(mutate=False)
    def _get_halo_bias(self, cosmo, a, rho0):
        # Compute the halo bias at this cosmo and a.
        if a != self._a_bf or cosmo != self._cosmo_bf:
            self._bf = self.halo_bias(cosmo, self._mass, a)
            integ = self._integrator(self._mf*self._bf*self._mass, self._lmass)
            self._mbf0 = (rho0 - integ) / self._m0
            self._cosmo_bf, self._a_bf = cosmo, a  # cache

    def _get_ingredients(self, cosmo, a, *, get_bf):
        """Compute mass function and halo bias at some scale factor."""
        rho0 = const.RHO_CRITICAL * cosmo["Omega_m"] * cosmo["h"]**2
        self._get_mass_function(cosmo, a, rho0)
        if get_bf:
            self._get_halo_bias(cosmo, a, rho0)

    def _integrate_over_mf(self, array_2):
        #  ∫ dM n(M) f(M)
        i1 = self._integrator(self._mf * array_2, self._lmass)
        return i1 + self._mf0 * array_2[..., 0]

    def _integrate_over_mbf(self, array_2):
        #  ∫ dM n(M) b(M) f(M)
        i1 = self._integrator(self._mf * self._bf * array_2, self._lmass)
        return i1 + self._mbf0 * array_2[..., 0]

    def integrate_over_massfunc(self, func, cosmo, a):
        """ Returns the integral over mass of a given funcion times
        the mass function:

        .. math::
            \\int dM\\,n(M,a)\\,f(M)

        Args:
            func (:obj:`callable`): a function accepting an array of halo masses
                as a single argument, and returning an array of the
                same size.
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
            a (:obj:`float`): scale factor.

        Returns:
            :obj:`float`: integral value.
        """ # noqa
        fM = func(self._mass)
        self._get_ingredients(cosmo, a, get_bf=False)
        return self._integrate_over_mf(fM)

    def number_counts(self, cosmo, *, selection,
                      a_min=None, a_max=1.0, na=128):
        """ Solves the integral:

        .. math::
            nc(sel) = \\int dM\\int da\\,\\frac{dV}{dad\\Omega}\\,
            n(M,a)\\,sel(M,a)

        where :math:`n(M,a)` is the halo mass function, and
        :math:`sel(M,a)` is the selection function as a function of halo mass
        and scale factor.

        Note that the selection function is normalized to integrate to unity
        and assumed to represent the selection probaility per unit scale factor
        and per unit mass.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
            selection (:obj:`callable`): function of mass and scale factor
                that returns the selection function. This function
                should take in floats or arrays with a signature ``sel(m, a)``
                and return an array with shape ``(len(m), len(a))`` according
                to the numpy broadcasting rules.
            a_min (:obj:`float`): the minimum scale factor at which to start integrals
                over the selection function.
                Default: value of ``cosmo.cosmo.spline_params.A_SPLINE_MIN``
            a_max (:obj:`float`): the maximum scale factor at which to end integrals
                over the selection function.
            na (:obj:`int`): number of samples in scale factor to be used in
                the integrals.

        Returns:
            :obj:`float`: the total number of clusters/halos.
        """ # noqa
        # get a values for integral
        if a_min is None:
            a_min = cosmo.cosmo.spline_params.A_SPLINE_MIN
        a = np.linspace(a_min, a_max, na)

        # compute the volume element
        abs_dzda = 1 / a / a
        dc = cosmo.comoving_angular_distance(a)
        ez = cosmo.h_over_h0(a)
        dh = const.CLIGHT_HMPC / cosmo['h']
        dvdz = dh * dc**2 / ez
        dvda = dvdz * abs_dzda

        # now do m intergrals in a loop
        mint = np.zeros_like(a)
        for i, _a in enumerate(a):
            self._get_ingredients(cosmo, _a, get_bf=False)
            _selm = np.atleast_2d(selection(self._mass, _a)).T
            mint[i] = self._integrator(
                dvda[i] * self._mf[..., :] * _selm[..., :],
                self._lmass
            )

        # now do scale factor integral
        return self._integrator(mint, a)

    def I_0_1(self, cosmo, k, a, prof):
        """ Solves the integral:

        .. math::
            I^0_1(k,a|u) = \\int dM\\,n(M,a)\\,\\langle u(k,a|M)\\rangle,

        where :math:`n(M,a)` is the halo mass function, and
        :math:`\\langle u(k,a|M)\\rangle` is the halo profile as a
        function of scale, scale factor and halo mass.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
            k (:obj:`float` or `array`): comoving wavenumber.
            a (:obj:`float`): scale factor.
            prof (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`):
                halo profile.

        Returns:
            (:obj:`float` or `array`): integral values evaluated at each
            value of ``k``.
        """
        self._fix_profile_mass_def(prof)
        self._check_mass_def(prof)
        self._get_ingredients(cosmo, a, get_bf=False)
        uk = prof.fourier(cosmo, k, self._mass, a).T
        return self._integrate_over_mf(uk)

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
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
            k (:obj:`float` or `array`): comoving wavenumber.
            a (:obj:`float`): scale factor.
            prof (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`):
                halo profile.

        Returns:
            (:obj:`float` or `array`): integral values evaluated at each
            value of ``k``.
        """
        self._fix_profile_mass_def(prof)
        self._check_mass_def(prof)
        self._get_ingredients(cosmo, a, get_bf=True)
        uk = prof.fourier(cosmo, k, self._mass, a).T
        return self._integrate_over_mbf(uk)

    def I_0_2(self, cosmo, k, a, prof, *, prof2=None, prof_2pt):
        """ Solves the integral:

        .. math::
            I^0_2(k,a|u,v) = \\int dM\\,n(M,a)\\,
            \\langle u(k,a|M) v(k,a|M)\\rangle,

        where :math:`n(M,a)` is the halo mass function, and
        :math:`\\langle u(k,a|M) v(k,a|M)\\rangle` is the two-point
        moment of the two halo profiles.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
            k (:obj:`float` or `array`): comoving wavenumber.
            a (:obj:`float`): scale factor.
            prof (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`):
                halo profile.
            prof2 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): a
                second halo profile. If ``None``, ``prof`` will be used as
                ``prof2``.
            prof_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
                a profile covariance object
                returning the the two-point moment of the two profiles
                being correlated.

        Returns:
             (:obj:`float` or `array`): integral values evaluated at each
             value of ``k``.
        """
        if prof2 is None:
            prof2 = prof
        self._fix_profile_mass_def(prof)
        self._fix_profile_mass_def(prof2)

        self._check_mass_def(prof, prof2)
        self._get_ingredients(cosmo, a, get_bf=False)
        uk = prof_2pt.fourier_2pt(cosmo, k, self._mass, a, prof, prof2=prof2).T
        return self._integrate_over_mf(uk)

    def I_1_2(self, cosmo, k, a, prof, *, prof2=None, prof_2pt):
        """ Solves the integral:

        .. math::
            I^1_2(k,a|u,v) = \\int dM\\,n(M,a)\\,b(M,a)\\,
            \\langle u(k,a|M) v(k,a|M)\\rangle,

        where :math:`n(M,a)` is the halo mass function,
        :math:`b(M,a)` is the halo bias, and
        :math:`\\langle u(k,a|M) v(k,a|M)\\rangle` is the two-point
        moment of the two halo profiles.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
            k (:obj:`float` or `array`): comoving wavenumber.
            a (:obj:`float`): scale factor.
            prof (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`):
                halo profile.
            prof2 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): a
                second halo profile. If ``None``, ``prof`` will be used as
                ``prof2``.
            prof_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
                a profile covariance object
                returning the the two-point moment of the two profiles
                being correlated.

        Returns:
             (:obj:`float` or `array`): integral values evaluated at each
             value of ``k``.
        """
        if prof2 is None:
            prof2 = prof
        self._fix_profile_mass_def(prof)
        self._fix_profile_mass_def(prof2)

        self._check_mass_def(prof, prof2)
        self._get_ingredients(cosmo, a, get_bf=True)
        uk = prof_2pt.fourier_2pt(cosmo, k, self._mass, a, prof, prof2=prof2).T
        return self._integrate_over_mbf(uk)

    def I_0_22(self, cosmo, k, a, prof, *,
               prof2=None, prof3=None, prof4=None,
               prof12_2pt, prof34_2pt=None):
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
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
            k (:obj:`float` or `array`): comoving wavenumber.
            a (:obj:`float`): scale factor.
            prof (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`):
                halo profile.
            prof2 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): a
                second halo profile. If ``None``, ``prof`` will be used as
                ``prof2``.
            prof3 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): a
                third halo profile. If ``None``, ``prof`` will be used as
                ``prof3``.
            prof4 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): a
                fourth halo profile. If ``None``, ``prof2`` will be used as
                ``prof4``.
            prof12_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
                a profile covariance object returning the the
                two-point moment of ``prof`` and ``prof2``.
            prof34_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
                a profile covariance object returning the the
                two-point moment of ``prof3`` and ``prof4``. If ``None``,
                ``prof12_2pt`` will be used.

        Returns:
             (:obj:`float` or `array`): integral values evaluated at each
             value of ``k``.
        """
        if prof3 is None:
            prof3 = prof
        if prof4 is None:
            prof4 = prof2

        if prof34_2pt is None:
            prof34_2pt = prof12_2pt

        self._fix_profile_mass_def(prof)
        self._fix_profile_mass_def(prof2)
        self._fix_profile_mass_def(prof3)
        self._fix_profile_mass_def(prof4)
        self._check_mass_def(prof, prof2, prof3, prof4)
        self._get_ingredients(cosmo, a, get_bf=False)
        uk12 = prof12_2pt.fourier_2pt(
            cosmo, k, self._mass, a, prof, prof2=prof2).T

        if (prof, prof2, prof12_2pt) == (prof3, prof4, prof34_2pt):
            # 4pt approximation of the same profile
            uk34 = uk12
        else:
            uk34 = prof34_2pt.fourier_2pt(
                cosmo, k, self._mass, a, prof3, prof2=prof4).T

        return self._integrate_over_mf(uk12[None, :, :] * uk34[:, None, :])
