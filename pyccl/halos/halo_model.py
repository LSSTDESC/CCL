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
from ..base import CCLHalosObject, unlock_instance, warn_api
from ..parameters import physical_constants as const
import numpy as np


class HMCalculator(CCLHalosObject):
    """This class implements a set of methods that can be used to
    compute various halo model quantities. A lot of these quantities
    will involve integrals of the sort:

    .. math::
       \\int dM\\,n(M,a)\\,f(M,k,a),

    where :math:`n(M,a)` is the halo mass function, and :math:`f` is
    an arbitrary function of mass, scale factor and Fourier scales.

    Args:
        mass_function (str or :class:`~pyccl.halos.hmfunc.MassFunc`):
            the mass function to use
        halo_bias (str or :class:`~pyccl.halos.hbias.HaloBias`):
            the halo bias function to use
        mass_def (str or :class:`~pyccl.halos.massdef.MassDef`):
            the halo mass definition to use
        lM_min, lM_max (float): lower and upper integration bounds
            of logarithmic (base-10) mass (in units of solar mass).
            Default range: 8, 16.
        nlM (int): number of uniformly-spaced samples in log(Mass)
            to be used in the mass integrals. Default: 128.
        integration_method_M (string): integration method to use
            in the mass integrals. Options: "simpson" and "spline".
            Default: "simpson".
        k_norm (float): some of the integrals solved by this class
            will often be normalized by their value on very large
            scales. This parameter (in units of inverse Mpc)
            determines what is considered a "very large" scale.
            Default: 1E-5.
    """
    __repr_attrs__ = ("mass_function", "halo_bias", "mass_def", "precision",)

    @warn_api(pairs=[("massfunc", "mass_function"), ("hbias", "halo_bias"),
                     ("log10M_min", "lM_min"), ("log10M_max", "lM_max"),
                     ("nlog10M", "nlM"), ("k_min", "k_norm")])
    def __init__(self, *, mass_function, halo_bias, mass_def,
                 lM_min=8., lM_max=16., nlM=128,
                 integration_method_M='simpson', k_norm=1E-5):
        # halo mass definition
        if isinstance(mass_def, MassDef):
            self.mass_def = mass_def
        elif isinstance(mass_def, str):
            self.mass_def = MassDef.from_name(mass_def)()
        else:
            raise TypeError("mass_def must be of type `MassDef` "
                            "or a mass definition name string")
        # halo mass function
        if isinstance(mass_function, MassFunc):
            self.mass_function = mass_function
        elif isinstance(mass_function, str):
            nMclass = MassFunc.from_name(mass_function)
            self.mass_function = nMclass(mass_def=self.mass_def)
        else:
            raise TypeError("mass_function must be of type `MassFunc` "
                            "or a mass function name string")
        # halo bias function
        if isinstance(halo_bias, HaloBias):
            self.halo_bias = halo_bias
        elif isinstance(halo_bias, str):
            bMclass = HaloBias.from_name(halo_bias)
            self.halo_bias = bMclass(mass_def=self.mass_def)
        else:
            raise TypeError("halo_bias must be of type `HaloBias` "
                            "or a halo bias name string")

        self.precision = {
            'log10M_min': lM_min, 'log10M_max': lM_max, 'nlM': nlM,
            'integration_method_M': integration_method_M, 'k_norm': k_norm}
        self._lmass = np.linspace(self.precision['log10M_min'],
                                  self.precision['log10M_max'],
                                  self.precision['nlM'])
        self._mass = 10.**self._lmass
        self._m0 = self._mass[0]

        if self.precision['integration_method_M'] not in ['spline', 'simpson']:
            raise NotImplementedError("Only \'simpson\' and 'spline' "
                                      "supported as integration methods")
        elif self.precision['integration_method_M'] == 'simpson':
            from scipy.integrate import simpson
            self._integrator = simpson
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
            massfunc = self.mass_function.get_mass_function
            self._mf = massfunc(cosmo, self._mass, a)
            integ = self._integrator(self._mf*self._mass, self._lmass)
            self._mf0 = (rho0 - integ) / self._m0
            self._cosmo_mf, self._a_mf = cosmo, a  # cache

    @unlock_instance(mutate=False)
    def _get_halo_bias(self, cosmo, a, rho0):
        # Compute the halo bias at this cosmo and a.
        if cosmo != self._cosmo_bf or a != self._a_bf:
            hbias = self.halo_bias.get_halo_bias
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
        uk0 = prof.fourier(cosmo, self.precision['k_norm'],
                           self._mass, a, mass_def=self.mass_def).T
        norm = 1. / self._integrate_over_mf(uk0)
        return norm

    @warn_api(pairs=[("sel", "selection"),
                     ("amin", "a_min"),
                     ("amax", "a_max")],
              reorder=["na", "a_min", "a_max"])
    def number_counts(self, cosmo, *, selection,
                      a_min=None, a_max=1.0, na=128):
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
            selection (callable): function of mass and scale factor
                that returns the selection function. This function
                should take in floats or arrays with a signature ``sel(m, a)``
                and return an array with shape ``(len(m), len(a))`` according
                to the numpy broadcasting rules.
            a_min (float): the minimum scale factor at which to start integrals
                over the selection function.
                Default: value of ``cosmo.cosmo.spline_params.A_SPLINE_MIN``
            a_max (float): the maximum scale factor at which to end integrals
                over the selection function.
                Default: 1.0
            na (int): number of samples in scale factor to be used in
                the integrals. Default: 128.

        Returns:
            float: the total number of clusters
        """  # noqa

        # get a values for integral
        if a_min is None:
            a_min = cosmo.cosmo.spline_params.A_SPLINE_MIN
        a = np.linspace(a_min, a_max, na)

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
            _selm = np.atleast_2d(selection(self._mass, _a)).T
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
                          mass_def=self.mass_def).T
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
                          mass_def=self.mass_def).T
        i11 = self._integrate_over_mbf(uk)
        return i11

    @warn_api(pairs=[("prof1", "prof")], reorder=["prof_2pt", "prof2"])
    def I_0_2(self, cosmo, k, a, prof, *, prof2=None, prof_2pt):
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
            prof2 (:class:`~pyccl.halos.profiles.HaloProfile`): a
                second halo profile. If `None`, `prof` will be used as
            prof_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
                a profile covariance object
                returning the the two-point moment of the two profiles
                being correlated.
            prof_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
                a profile covariance object returning the the two-point
                moment of the two profiles being correlated.

        Returns:
             float or array_like: integral values evaluated at each
             value of `k`.
        """
        if prof2 is None:
            prof2 = prof

        # Compute mass function
        self._get_ingredients(cosmo, a, False)
        uk = prof_2pt.fourier_2pt(cosmo, k, self._mass, a, prof,
                                  prof2=prof2,
                                  mass_def=self.mass_def).T
        i02 = self._integrate_over_mf(uk)
        return i02

    @warn_api(pairs=[("prof1", "prof")], reorder=["prof_2pt", "prof2"])
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
            cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
            k (float or array_like): comoving wavenumber in Mpc^-1.
            a (float): scale factor.
            prof1 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
                profile.
            prof2 (:class:`~pyccl.halos.profiles.HaloProfile`): a
                second halo profile. If `None`, `prof` will be used as
                `prof2`.
            prof_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
                a profile covariance object
                returning the the two-point moment of the two profiles
                being correlated.

        Returns:
             float or array_like: integral values evaluated at each
             value of `k`.
        """
        if prof2 is None:
            prof2 = prof

        # Compute mass function
        self._get_ingredients(cosmo, a, True)
        uk = prof_2pt.fourier_2pt(cosmo, k, self._mass, a, prof,
                                  prof2=prof2,
                                  mass_def=self.mass_def).T
        i02 = self._integrate_over_mbf(uk)
        return i02

    @warn_api(pairs=[("prof1", "prof")],
              reorder=["prof12_2pt", "prof2", "prof3", "prof34_2pt", "prof4"])
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
            cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
            k (float or array_like): comoving wavenumber in Mpc^-1.
            a (float): scale factor.
            prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
                profile.
            prof2 (:class:`~pyccl.halos.profiles.HaloProfile`): a
                second halo profile. If `None`, `prof` will be used as
                `prof2`.
            prof3 (:class:`~pyccl.halos.profiles.HaloProfile`): a
                third halo profile. If `None`, `prof` will be used as
                `prof3`.
            prof4 (:class:`~pyccl.halos.profiles.HaloProfile`): a
                fourth halo profile. If `None`, `prof2` will be used as
                `prof4`.
            prof12_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
                a profile covariance object returning the the
                two-point moment of `prof` and `prof2`.
            prof34_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
                a profile covariance object returning the the
                two-point moment of `prof3` and `prof4`.

        Returns:
             float or array_like: integral values evaluated at each
             value of `k`.
        """
        if prof3 is None:
            prof3 = prof
        if prof4 is None:
            prof4 = prof2

        if prof34_2pt is None:
            prof34_2pt = prof12_2pt

        self._get_ingredients(cosmo, a, False)
        uk12 = prof12_2pt.fourier_2pt(
            cosmo, k, self._mass, a, prof,
            prof2=prof2, mass_def=self.mass_def).T

        if (prof, prof2) == (prof3, prof4):
            # 4pt approximation of the same profile
            uk34 = uk12
        else:
            uk34 = prof34_2pt.fourier_2pt(
                cosmo, k, self._mass, a, prof3,
                prof2=prof4, mass_def=self.mass_def).T

        i04 = self._integrate_over_mf(uk12[None, :, :] * uk34[:, None, :])
        return i04


@warn_api
def halomod_mean_profile_1pt(cosmo, hmc, k, a, prof, *, normprof=False):
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
    a_use = np.atleast_1d(a).astype(float)
    k_use = np.atleast_1d(k).astype(float)

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


@warn_api
def halomod_bias_1pt(cosmo, hmc, k, a, prof, *, normprof=False):
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
    a_use = np.atleast_1d(a).astype(float)
    k_use = np.atleast_1d(k).astype(float)

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


@warn_api(pairs=[("normprof1", "normprof"), ("supress_1h", "suppress_1h")],
          reorder=["prof_2pt", "prof2", "p_of_k_a", "normprof", "normprof2"])
def halomod_power_spectrum(cosmo, hmc, k, a, prof, *,
                           prof2=None, prof_2pt=None,
                           normprof=False, normprof2=False,
                           p_of_k_a=None,
                           get_1h=True, get_2h=True,
                           smooth_transition=None, suppress_1h=None):
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
        prof2 (:class:`~pyccl.halos.profiles.HaloProfile`): a
            second halo profile. If `None`, `prof` will be used as
            `prof2`.
        normprof (bool): if `True`, this integral will be
            normalized by :math:`I^0_1(k\\rightarrow 0,a|u)`
            (see :meth:`~HMCalculator.I_0_1`), where
            :math:`u` is the profile represented by `prof`.
        normprof2 (bool): if `True`, this integral will be
            normalized by :math:`I^0_1(k\\rightarrow 0,a|v)`
            (see :meth:`~HMCalculator.I_0_1`), where
            :math:`v` is the profile represented by `prof2`.
        prof_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object
            returning the the two-point moment of the two profiles
            being correlated. If `None`, the default second moment
            will be used, corresponding to the products of the means
            of both profiles.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If `None`,
            the power spectrum stored within `cosmo` will be used.
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
        suppress_1h (function or None):
            Suppress the 1-halo large scale contribution by a
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
    a_use = np.atleast_1d(a).astype(float)
    k_use = np.atleast_1d(k).astype(float)

    # Check inputs
    if not isinstance(prof, HaloProfile):
        raise TypeError("prof must be of type `HaloProfile`")
    if prof2 is None:
        prof2 = prof
    elif not isinstance(prof2, HaloProfile):
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
        if not callable(smooth_transition):
            raise TypeError("smooth_transition must be "
                            "a function of `a` or None")
    if suppress_1h is not None:
        if not get_1h:
            raise ValueError("can't suppress the 1-halo term "
                             "when get_1h is False")
        if not callable(suppress_1h):
            raise TypeError("suppress_1h must be "
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
        if normprof:
            norm1 = hmc.profile_norm(cosmo, aa, prof)
        else:
            norm1 = 1
        # Compute second profile normalization
        if prof2 == prof:
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
            if prof2 == prof:
                i11_2 = i11_1
            else:
                i11_2 = hmc.I_1_1(cosmo, k_use, aa, prof2)

            # Compute 2-halo power spectrum
            pk_2h = pkf(aa) * i11_1 * i11_2
        else:
            pk_2h = 0.

        if get_1h:
            pk_1h = hmc.I_0_2(cosmo, k_use, aa, prof,
                              prof2=prof2, prof_2pt=prof_2pt)
            if suppress_1h is not None:
                ks = suppress_1h(aa)
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


@warn_api(pairs=[("normprof1", "normprof"), ("supress_1h", "suppress_1h")],
          reorder=["prof_2pt", "prof2", "p_of_k_a", "normprof", "normprof2"])
def halomod_Pk2D(cosmo, hmc, prof, *,
                 prof2=None, prof_2pt=None,
                 normprof=False, normprof2=False,
                 p_of_k_a=None,
                 get_1h=True, get_2h=True,
                 lk_arr=None, a_arr=None,
                 extrap_order_lok=1, extrap_order_hik=2,
                 smooth_transition=None, suppress_1h=None):
    """ Returns a :class:`~pyccl.pk2d.Pk2D` object containing
    the halo-model power spectrum for two quantities defined by
    their respective halo profiles. See :meth:`halomod_power_spectrum`
    for more details about the actual calculation.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        prof (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile.
        prof2 (:class:`~pyccl.halos.profiles.HaloProfile`): a
            second halo profile. If `None`, `prof` will be used as
            `prof2`.
        prof_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object
            returning the the two-point moment of the two profiles
            being correlated. If `None`, the default second moment
            will be used, corresponding to the products of the means
            of both profiles.
        normprof (bool): if `True`, this integral will be
            normalized by :math:`I^0_1(k\\rightarrow 0,a|u)`
            (see :meth:`~HMCalculator.I_0_1`), where
            :math:`u` is the profile represented by `prof`.
        normprof2 (bool): if `True`, this integral will be
            normalized by :math:`I^0_1(k\\rightarrow 0,a|v)`
            (see :meth:`~HMCalculator.I_0_1`), where
            :math:`v` is the profile represented by `prof2`.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If `None`,
            the power spectrum stored within `cosmo` will be used.
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
        suppress_1h (function or None):
            Suppress the 1-halo large scale contribution by a
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
                                    normprof=normprof, normprof2=normprof2,
                                    get_1h=get_1h, get_2h=get_2h,
                                    smooth_transition=smooth_transition,
                                    suppress_1h=suppress_1h)

    pk2d = Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik,
                is_logp=False)
    return pk2d


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
