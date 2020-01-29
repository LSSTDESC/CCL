from .. import ccllib as lib
from .hmfunc import MassFunc
from .hbias import HaloBias
from .profiles import HaloProfile
from ..core import check
from ..pk2d import Pk2D
from ..power import linear_matter_power, nonlin_matter_power
from ..background import rho_x
from ..pyutils import _spline_integrate
import numpy as np


class ProfileCovar(object):
    """ This class implements the 1-halo covariance between two
    halo profiles. In the simplest case, this covariance is just
    the product of both profiles in Fourier space.
    More complicated cases should be implemented by subclassing
    this class and overloading the :meth:`~ProfileCovar.fourier_covar`
    method.
    """
    def __init__(self):
        pass

    def fourier_covar(self, prof, cosmo, k, M, a,
                      prof_2=None, mass_def=None):
        """ Returns the Fourier-space two-point moment between
        two profiles:

        .. math::
           \\langle\\rho_1(k)\\rho_2(k)\\rangle.

        Args:
            prof (:class:`~pyccl.halos.profiles.HaloProfile`):
                halo profile for which the second-order moment
                is desired.
            cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
            k (float or array_like): comoving wavenumber in Mpc^-1.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            prof_2 (:class:`~pyccl.halos.profiles.HaloProfile`):
                second halo profile for which the second-order moment
                is desired. If `None`, the assumption is that you want
                an auto-correlation, and `prof` will be used as `prof_2`.
            mass_def (:obj:`~pyccl.halos.massdef.MassDef`): a mass
                definition object.

        Returns:
            float or array_like: second-order Fourier-space
            moment. The shape of the output will be `(N_M, N_k)`
            where `N_k` and `N_m` are the sizes of `k` and `M`
            respectively. If `k` or `M` are scalars, the
            corresponding dimension will be squeezed out on output.
        """
        if not isinstance(prof, HaloProfile):
            raise TypeError("prof must be of type `HaloProfile`")
        uk1 = prof.fourier(cosmo, k, M, a, mass_def=mass_def)

        if prof_2 is None:
            uk2 = uk1
        else:
            if not isinstance(prof_2, HaloProfile):
                raise TypeError("prof_2 must be of type "
                                "`HaloProfile` or `None`")

            uk2 = prof_2.fourier(cosmo, k, M, a, mass_def=mass_def)

        return uk1 * uk2


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
        l10M_min (float): logarithmic mass (in units of solar mass)
            corresponding to the lower bound of the integrals in
            mass. Default: 8.
        l10M_max (float): logarithmic mass (in units of solar mass)
            corresponding to the upper bound of the integrals in
            mass. Default: 16.
        nl10M (int): number of samples in log(Mass) to be used in
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
    def __init__(self, cosmo, **kwargs):
        self.rho0 = rho_x(cosmo, 1., 'matter', is_comoving=True)
        self.precision = {'l10M_min': 8.,
                          'l10M_max': 16.,
                          'nl10M': 128,
                          'integration_method_M': 'simpson',
                          'k_min': 1E-5}
        self.precision.update(kwargs)
        self.lmass = np.linspace(self.precision['l10M_min'],
                                 self.precision['l10M_max'],
                                 self.precision['nl10M'])
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

    def _u_k_from_arrays(self, hmf_a, hmf0, uk_a):
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

    def _b_k_from_arrays(self, hmf_a, hbf_a, hmf0, uk_a):
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

    def _check_massfunc(self, massfunc):
        if not isinstance(massfunc, MassFunc):
            raise TypeError("massfunc must be of type `MassFunc`")

    def _check_hbias(self, hbias):
        if not isinstance(hbias, HaloBias):
            raise TypeError("hbias must be of type `HaloBias`")

    def _check_prof(self, prof):
        if not isinstance(prof, HaloProfile):
            raise TypeError("prof must be of type `HaloProfile`")

    def mean_profile(self, cosmo, k, a, massfunc, prof,
                     normprof=False, mass_def=None):
        """ Solves the integral:

        .. math::
            I^1_0(k,a|u) = \\int dM\\,n(M,a)\\,\\langle u(k,a|M)\\rangle,

        where :math:`n(M,a)` is the halo mass function, and
        :math:`\\langle u(k,a|M)\\rangle` is the halo profile as a
        function of scale, scale factor and halo mass.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
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
        self._check_massfunc(massfunc)
        self._check_prof(prof)

        na = len(a_use)
        nk = len(k_use)
        out = np.zeros([na, nk])
        for ia, aa in enumerate(a_use):
            mf = self._hmf(massfunc, cosmo, aa, mass_def=mass_def)
            mf0 = (self.rho0 -
                   self.integrator(mf * self.mass,
                                   self.lmass)) / self.m0
            uk = prof.fourier(cosmo, k_use, self.mass, aa,
                              mass_def=mass_def).T
            if normprof:
                uk0 = prof.fourier(cosmo,
                                   self.precision['k_min'],
                                   self.mass, aa,
                                   mass_def=mass_def).T
                norm = 1. / self._u_k_from_arrays(mf, mf0, uk0)
            else:
                norm = 1.
            out[ia, :] = self._u_k_from_arrays(mf, mf0, uk) * norm

        if np.ndim(a) == 0:
            out = np.squeeze(out, axis=0)
        if np.ndim(k) == 0:
            out = np.squeeze(out, axis=-1)
        return out

    def bias(self, cosmo, k, a, massfunc, hbias, prof,
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
        self._check_massfunc(massfunc)
        self._check_hbias(hbias)
        self._check_prof(prof)

        na = len(a_use)
        nk = len(k_use)
        out = np.zeros([na, nk])
        for ia, aa in enumerate(a_use):
            mf = self._hmf(massfunc, cosmo, aa, mass_def=mass_def)
            bf = self._hbf(hbias, cosmo, aa, mass_def=mass_def)
            mbf0 = (self.rho0 -
                    self.integrator(mf * bf * self.mass,
                                    self.lmass)) / self.m0
            uk = prof.fourier(cosmo, k_use, self.mass, aa,
                              mass_def=mass_def).T
            if normprof:
                mf0 = (self.rho0 -
                       self.integrator(mf * self.mass,
                                       self.lmass)) / self.m0
                uk0 = prof.fourier(cosmo,
                                   self.precision['k_min'],
                                   self.mass, aa,
                                   mass_def=mass_def).T
                norm = 1. / self._u_k_from_arrays(mf, mf0, uk0)
            else:
                norm = 1.
            out[ia, :] = self._b_k_from_arrays(mf, bf,
                                               mbf0, uk) * norm

        if np.ndim(a) == 0:
            out = np.squeeze(out, axis=0)
        if np.ndim(k) == 0:
            out = np.squeeze(out, axis=-1)
        return out

    def pk(self, cosmo, k, a, massfunc, hbias, prof,
           covprof=None, prof_2=None, p_of_k_a=None,
           normprof_1=False, normprof_2=False,
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
            k (float or array_like): comoving wavenumber in Mpc^-1.
            a (float or array_like): scale factor.
            massfunc (:class:`~pyccl.halos.hmfunc.MassFunc`): a mass
                function object.
            hbias (:class:`~pyccl.halos.hbias.HaloBias`): a halo bias
                object.
            prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
                profile.
            covprof (:class:`ProfileCovar`): a profile covariance object
                returning the the two-point moment of the two profiles
                being correlated. If `None`, the default second moment
                will be used, corresponding to the products of the means
                of both profiles.
            prof_2 (:class:`~pyccl.halos.profiles.HaloProfile`): a
                second halo profile. If `None`, `prof` will be used as
                `prof_2`.
            p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
                be used as the linear matter power spectrum. If `None`,
                the power spectrum stored within `cosmo` will be used.
            normprof_1 (bool): if `True`, this integral will be
                normalized by :math:`I^1_0(k\\rightarrow 0,a|u)`
                (see :meth:`~HMCalculator.mean_profile`), where
                :math:`u` is the profile represented by `prof`.
            normprof_2 (bool): if `True`, this integral will be
                normalized by :math:`I^1_0(k\\rightarrow 0,a|v)`
                (see :meth:`~HMCalculator.mean_profile`), where
                :math:`v` is the profile represented by `prof_2`.
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
        self._check_massfunc(massfunc)
        self._check_hbias(hbias)
        self._check_prof(prof)
        if (prof_2 is not None) and (not isinstance(prof_2, HaloProfile)):
            raise TypeError("prof_2 must be of type `HaloProfile` or `None`")
        if covprof is None:
            covprof = ProfileCovar()
        elif not isinstance(covprof, ProfileCovar):
            raise TypeError("covprof must be of type "
                            "`ProfileCovar` or `None`")
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
            mf = self._hmf(massfunc, cosmo, aa, mass_def=mass_def)
            mf0 = (self.rho0 -
                   self.integrator(mf * self.mass,
                                   self.lmass)) / self.m0

            # Compute normalization
            if normprof_1:
                uk01 = prof.fourier(cosmo,
                                    self.precision['k_min'],
                                    self.mass, aa,
                                    mass_def=mass_def).T
                norm1 = 1. / self._u_k_from_arrays(mf, mf0, uk01)
            else:
                norm1 = 1.
            if prof_2 is None:
                norm2 = norm1
            else:
                if normprof_2:
                    uk02 = prof_2.fourier(cosmo,
                                          self.precision['k_min'],
                                          self.mass, aa,
                                          mass_def=mass_def).T
                    norm2 = 1. / self._u_k_from_arrays(mf, mf0, uk02)
                else:
                    norm2 = 1.
            norm = norm1 * norm2

            if get_2h:
                bf = self._hbf(hbias, cosmo, aa, mass_def=mass_def)
                # Compute first bias factor
                mbf0 = (self.rho0 -
                        self.integrator(mf * bf * self.mass,
                                        self.lmass)) / self.m0
                uk_1 = prof.fourier(cosmo, k_use, self.mass, aa,
                                    mass_def=mass_def).T
                bk_1 = self._b_k_from_arrays(mf, bf, mbf0, uk_1)

                # Compute second bias factor
                if prof_2 is None:
                    bk_2 = bk_1
                else:
                    uk_2 = prof_2.fourier(cosmo, k_use, self.mass, aa,
                                          mass_def=mass_def).T
                    bk_2 = self._b_k_from_arrays(mf, bf, mbf0, uk_2)

                # Compute power spectrum
                pk_2h = pkf(aa) * bk_1 * bk_2
            else:
                pk_2h = 0.

            if get_1h:
                # 1-halo term
                uk2 = covprof.fourier_covar(prof, cosmo, k_use,
                                            self.mass, aa,
                                            prof_2=prof_2,
                                            mass_def=mass_def).T
                pk_1h = self._u_k_from_arrays(mf, mf0, uk2)
            else:
                pk_1h = 0.

            # 2-halo term
            out[ia, :] = (pk_1h + pk_2h) * norm

        if np.ndim(a) == 0:
            out = np.squeeze(out, axis=0)
        if np.ndim(k) == 0:
            out = np.squeeze(out, axis=-1)
        return out

    def get_Pk2D(self, cosmo, massfunc, hbias, prof,
                 covprof=None, prof_2=None, p_of_k_a=None,
                 normprof_1=False, normprof_2=False,
                 mass_def=None, get_1h=True, get_2h=True,
                 lk_arr=None, a_arr=None,
                 extrap_order_lok=1, extrap_order_hik=2):
        """ Returns a :class:`~pyccl.pk2d.Pk2D` object containing
        the halo-model power spectrum for two quantities defined by
        their respective halo profiles. See :meth:`~HMCalculator.pk`
        for more details about the actual calculation.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
            massfunc (:class:`~pyccl.halos.hmfunc.MassFunc`): a mass
                function object.
            hbias (:class:`~pyccl.halos.hbias.HaloBias`): a halo bias
                object.
            prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
                profile.
            covprof (:class:`ProfileCovar`): a profile covariance object
                returning the the two-point moment of the two profiles
                being correlated. If `None`, the default second moment
                will be used, corresponding to the products of the means
                of both profiles.
            prof_2 (:class:`~pyccl.halos.profiles.HaloProfile`): a
                second halo profile. If `None`, `prof` will be used as
                `prof_2`.
            p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
                be used as the linear matter power spectrum. If `None`,
                the power spectrum stored within `cosmo` will be used.
            normprof_1 (bool): if `True`, this integral will be
                normalized by :math:`I^1_0(k\\rightarrow 0,a|u)`
                (see :meth:`~HMCalculator.mean_profile`), where
                :math:`u` is the profile represented by `prof`.
            normprof_2 (bool): if `True`, this integral will be
                normalized by :math:`I^1_0(k\\rightarrow 0,a|v)`
                (see :meth:`~HMCalculator.mean_profile`), where
                :math:`v` is the profile represented by `prof_2`.
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

        pk_arr = self.pk(cosmo, np.exp(lk_arr), a_arr,
                         massfunc, hbias, prof, covprof=covprof,
                         prof_2=prof_2, p_of_k_a=p_of_k_a,
                         normprof_1=normprof_1, normprof_2=normprof_2,
                         mass_def=mass_def, get_1h=get_1h, get_2h=get_2h)

        pk2d = Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr,
                    extrap_order_lok=extrap_order_lok,
                    extrap_order_hik=extrap_order_hik,
                    cosmo=cosmo, is_logp=False)
        return pk2d
