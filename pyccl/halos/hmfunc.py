from .. import ccllib as lib
from ..core import check
from ..background import omega_x
from ..pyutils import warn_api, deprecate_attr, deprecated
from .massdef import MassDef, MassDef200m, MassDef200c
from ..emulator import Emulator, EmulatorObject
from ..parameters import physical_constants
from ..base import CCLHalosObject
import numpy as np
from scipy.interpolate import interp1d


class MassFunc(CCLHalosObject):
    """ This class enables the calculation of halo mass functions.
    We currently assume that all mass functions can be written as

    .. math::
        \\frac{dn}{d\\log_{10}M} = f(\\sigma_M)\\,\\frac{\\rho_M}{M}\\,
        \\frac{d\\log \\sigma_M}{d\\log_{10} M}

    where :math:`\\sigma_M^2` is the overdensity variance on spheres with a
    radius given by the Lagrangian radius for mass M.
    All sub-classes implementing specific mass function parametrizations
    can therefore be simply created by replacing this class'
    `_get_fsigma` method.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object that fixes
            the mass definition used by this mass function
            parametrization.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    name = 'default'

    @warn_api
    def __init__(self, *, mass_def=None, mass_def_strict=True):
        self.mass_def_strict = mass_def_strict
        # Check if mass definition was provided and check that it's sensible.
        if mass_def is not None:
            if self._check_mass_def(mass_def):
                raise ValueError("Mass function " + self.name +
                                 " is not compatible with mass definition" +
                                 " Delta = %s, " % (mass_def.Delta) +
                                 " rho = " + mass_def.rho_type)
            self.mass_def = mass_def
        else:
            self._default_mass_def()
        self._setup()

    @deprecate_attr(pairs=[("mass_def", "mdef")])
    def __getattr__(self, name):
        return

    def _default_mass_def(self):
        """ Assigns a default mass definition for this object if
        none is passed at initialization.
        """
        self.mass_def = MassDef('fof', 'matter')

    def _setup(self):
        """ Use this function to initialize any internal attributes
        of this object. This function is called at the very end of the
        constructor call.
        """
        pass

    def _check_mass_def_strict(self, mass_def):
        return False

    def _check_mass_def(self, mass_def):
        """ Return False if the input mass definition agrees with
        the definitions for which this mass function parametrization
        works. True otherwise. This function gets called at the
        start of the constructor call.

        Args:
            mass_def (:class:`~pyccl.halos.massdef.MassDef`):
                a mass definition object.

        Returns:
            bool: True if the mass definition is not compatible with \
                this mass function parametrization. False otherwise.
        """
        if self.mass_def_strict:
            return self._check_mass_def_strict(mass_def)
        return False

    def _get_consistent_mass(self, cosmo, M, a, mass_def_other):
        """ Transform a halo mass with a given mass definition into
        the corresponding mass definition that was used to initialize
        this object.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            mass_def_other (:class:`~pyccl.halos.massdef.MassDef`):
                a mass definition object.

        Returns:
            float or array_like: mass according to this object's \
                mass definition.
        """
        if mass_def_other is not None:
            M_use = mass_def_other.translate_mass(
                cosmo, M, a,
                mass_def_other=self.mass_def)
        else:
            M_use = M
        return np.log10(M_use)

    def _get_Delta_m(self, cosmo, a):
        """ For SO-based mass definitions, this returns the corresponding
        value of Delta for a rho_matter-based definition. This is useful
        mostly for the Tinker mass functions, which are defined for any
        SO mass in general, but explicitly only for Delta_matter.
        """
        delta = self.mass_def.get_Delta(cosmo, a)
        if self.mass_def.rho_type == 'matter':
            return delta
        else:
            om_this = omega_x(cosmo, a, self.mass_def.rho_type)
            om_matt = omega_x(cosmo, a, 'matter')
            return delta * om_this / om_matt

    @warn_api(pairs=[("mass_def_other", "mdef_other")])
    def get_mass_function(self, cosmo, M, a, *, mass_def_other=None):
        """ Returns the mass function for input parameters.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            mass_def_other (:class:`~pyccl.halos.massdef.MassDef`):
                the mass definition object that defines M.

        Returns:
            float or array_like: mass function \
                :math:`dn/d\\log_{10}M` in units of Mpc^-3 (comoving).
        """
        # Initialize sigma(M) splines if needed
        cosmo.compute_sigma()

        M_use = np.atleast_1d(M)
        logM = self._get_consistent_mass(cosmo, M_use,
                                         a, mass_def_other)

        # sigma(M)
        status = 0
        sigM, status = lib.sigM_vec(cosmo.cosmo, a, logM,
                                    len(logM), status)
        check(status)
        # dlogsigma(M)/dlog10(M)
        dlns_dlogM, status = lib.dlnsigM_dlogM_vec(cosmo.cosmo, a, logM,
                                                   len(logM), status)
        check(status)

        rho = (physical_constants.RHO_CRITICAL *
               cosmo['Omega_m'] * cosmo['h']**2)
        f = self._get_fsigma(cosmo, sigM, a, 2.302585092994046 * logM)
        mf = f * rho * dlns_dlogM / M_use

        if np.ndim(M) == 0:
            mf = mf[0]
        return mf

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        """ Get the :math:`f(\\sigma_M)` function for this mass function
        object (see description of this class for details).

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            sigM (float or array_like): standard deviation in the
                overdensity field on the scale of this halo.
            a (float): scale factor.
            lnM (float or array_like): natural logarithm of the
                halo mass in units of M_sun (provided in addition
                to sigM for convenience in some mass function
                parametrizations).

        Returns:
            float or array_like: :math:`f(\\sigma_M)` function.
        """
        raise NotImplementedError("Use one of the non-default "
                                  "MassFunction classes")

    @classmethod
    def from_name(cls, name):
        """ Returns mass function subclass from name string

        Args:
            name (string): a mass function name

        Returns:
            MassFunc subclass corresponding to the input name.
        """
        mass_functions = {c.name: c for c in cls.__subclasses__()}
        if name in mass_functions:
            return mass_functions[name]
        else:
            raise ValueError(f"Mass function {name} not implemented.")


class MassFuncPress74(MassFunc):
    """ Implements mass function described in 1974ApJ...187..425P.
    This parametrization is only valid for 'fof' masses.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts FoF masses only.
            If `None`, FoF masses will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    name = 'Press74'

    @warn_api
    def __init__(self, *, mass_def=None, mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _default_mass_def(self):
        self.mass_def = MassDef('fof', 'matter')

    def _setup(self):
        self.norm = np.sqrt(2/np.pi)

    def _check_mass_def_strict(self, mass_def):
        if mass_def.Delta != 'fof':
            return True
        return False

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        delta_c = 1.68647

        nu = delta_c/sigM
        return self.norm * nu * np.exp(-0.5 * nu**2)


class MassFuncSheth99(MassFunc):
    """ Implements mass function described in arXiv:astro-ph/9901122
    This parametrization is only valid for 'fof' masses.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts FoF masses only.
            If `None`, FoF masses will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
        use_delta_c_fit (bool): if True, use delta_crit given by
            the fit of Nakamura & Suto 1997. Otherwise use
            delta_crit = 1.68647.
    """
    name = 'Sheth99'

    @warn_api
    def __init__(self, *, mass_def=None, mass_def_strict=True,
                 use_delta_c_fit=False):
        self.use_delta_c_fit = use_delta_c_fit
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _default_mass_def(self):
        self.mass_def = MassDef('fof', 'matter')

    def _setup(self):
        self.A = 0.21615998645
        self.p = 0.3
        self.a = 0.707

    def _check_mass_def_strict(self, mass_def):
        if mass_def.Delta != 'fof':
            return True

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        if self.use_delta_c_fit:
            status = 0
            delta_c, status = lib.dc_NakamuraSuto(cosmo.cosmo, a, status)
            check(status)
        else:
            delta_c = 1.68647

        nu = delta_c / sigM
        return nu * self.A * (1. + (self.a * nu**2)**(-self.p)) * \
            np.exp(-self.a * nu**2/2.)


class MassFuncJenkins01(MassFunc):
    """ Implements mass function described in astro-ph/0005260.
    This parametrization is only valid for 'fof' masses.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts FoF masses only.
            If `None`, FoF masses will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    name = 'Jenkins01'

    @warn_api
    def __init__(self, *, mass_def=None, mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _default_mass_def(self):
        self.mass_def = MassDef('fof', 'matter')

    def _setup(self):
        self.A = 0.315
        self.b = 0.61
        self.q = 3.8

    def _check_mass_def_strict(self, mass_def):
        if mass_def.Delta != 'fof':
            return True
        return False

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        return self.A * np.exp(-np.fabs(-np.log(sigM) + self.b)**self.q)


class MassFuncTinker08(MassFunc):
    """ Implements mass function described in arXiv:0803.2706.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts SO masses with
            200 < Delta < 3200 with respect to the matter density.
            If `None`, Delta = 200 (matter) will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    name = 'Tinker08'

    @warn_api
    def __init__(self, *, mass_def=None, mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _default_mass_def(self):
        self.mass_def = MassDef200m()

    def _pd(self, ld):
        return 10.**(-(0.75/(ld - 1.8750612633))**1.2)

    def _setup(self):
        from scipy.interpolate import interp1d

        delta = np.array([200.0, 300.0, 400.0, 600.0, 800.0,
                          1200.0, 1600.0, 2400.0, 3200.0])
        alpha = np.array([0.186, 0.200, 0.212, 0.218, 0.248,
                          0.255, 0.260, 0.260, 0.260])
        beta = np.array([1.47, 1.52, 1.56, 1.61, 1.87,
                         2.13, 2.30, 2.53, 2.66])
        gamma = np.array([2.57, 2.25, 2.05, 1.87, 1.59,
                          1.51, 1.46, 1.44, 1.41])
        phi = np.array([1.19, 1.27, 1.34, 1.45, 1.58,
                        1.80, 1.97, 2.24, 2.44])
        ldelta = np.log10(delta)
        self.pA0 = interp1d(ldelta, alpha)
        self.pa0 = interp1d(ldelta, beta)
        self.pb0 = interp1d(ldelta, gamma)
        self.pc = interp1d(ldelta, phi)

    def _check_mass_def_strict(self, mass_def):
        if mass_def.Delta == 'fof':
            return True
        return False

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        ld = np.log10(self._get_Delta_m(cosmo, a))
        pA = self.pA0(ld) * a**0.14
        pa = self.pa0(ld) * a**0.06
        pb = self.pb0(ld) * a**self._pd(ld)
        return pA * ((pb / sigM)**pa + 1) * np.exp(-self.pc(ld)/sigM**2)


class MassFuncDespali16(MassFunc):
    """ Implements mass function described in arXiv:1507.05627.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts any SO masses.
            If `None`, Delta = 200 (matter) will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
        ellipsoidal (bool): use the ellipsoidal parametrization.
    """
    name = 'Despali16'

    @warn_api
    def __init__(self, *, mass_def=None, mass_def_strict=True,
                 ellipsoidal=False):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)
        self.ellipsoidal = ellipsoidal

    def _default_mass_def(self):
        self.mass_def = MassDef200m()

    def _setup(self):
        pass

    def _check_mass_def_strict(self, mass_def):
        if mass_def.Delta == 'fof':
            return True
        return False

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        status = 0
        delta_c, status = lib.dc_NakamuraSuto(cosmo.cosmo, a, status)
        check(status)

        Dv, status = lib.Dv_BryanNorman(cosmo.cosmo, a, status)
        check(status)

        x = np.log10(self.mass_def.get_Delta(cosmo, a) *
                     omega_x(cosmo, a, self.mass_def.rho_type) / Dv)

        if self.ellipsoidal:
            A = -0.1768 * x + 0.3953
            a = 0.3268 * x**2 + 0.2125 * x + 0.7057
            p = -0.04570 * x**2 + 0.1937 * x + 0.2206
        else:
            A = -0.1362 * x + 0.3292
            a = 0.4332 * x**2 + 0.2263 * x + 0.7665
            p = -0.1151 * x**2 + 0.2554 * x + 0.2488

        nu = delta_c/sigM
        nu_p = a * nu**2

        return 2.0 * A * np.sqrt(nu_p / 2.0 / np.pi) * \
            np.exp(-0.5 * nu_p) * (1.0 + nu_p**-p)


class MassFuncTinker10(MassFunc):
    """ Implements mass function described in arXiv:1001.3162.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts SO masses with
            200 < Delta < 3200 with respect to the matter density.
            If `None`, Delta = 200 (matter) will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
        norm_all_z (bool): should we normalize the mass function
            at z=0 or at all z?
    """
    name = 'Tinker10'

    @warn_api
    def __init__(self, *, mass_def=None, mass_def_strict=True,
                 norm_all_z=False):
        self.norm_all_z = norm_all_z
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _default_mass_def(self):
        self.mass_def = MassDef200m()

    def _setup(self):
        from scipy.interpolate import interp1d

        delta = np.array([200.0, 300.0, 400.0, 600.0, 800.0,
                          1200.0, 1600.0, 2400.0, 3200.0])
        alpha = np.array([0.368, 0.363, 0.385, 0.389, 0.393,
                          0.365, 0.379, 0.355, 0.327])
        beta = np.array([0.589, 0.585, 0.544, 0.543, 0.564,
                         0.623, 0.637, 0.673, 0.702])
        gamma = np.array([0.864, 0.922, 0.987, 1.09, 1.20,
                          1.34, 1.50, 1.68, 1.81])
        phi = np.array([-0.729, -0.789, -0.910, -1.05, -1.20,
                        -1.26, -1.45, -1.50, -1.49])
        eta = np.array([-0.243, -0.261, -0.261, -0.273, -0.278,
                        -0.301, -0.301, -0.319, -0.336])

        ldelta = np.log10(delta)
        self.pA0 = interp1d(ldelta, alpha)
        self.pa0 = interp1d(ldelta, eta)
        self.pb0 = interp1d(ldelta, beta)
        self.pc0 = interp1d(ldelta, gamma)
        self.pd0 = interp1d(ldelta, phi)
        if self.norm_all_z:
            p = np.array([-0.158, -0.195, -0.213, -0.254, -0.281,
                          -0.349, -0.367, -0.435, -0.504])
            q = np.array([0.0128, 0.0128, 0.0143, 0.0154, 0.0172,
                          0.0174, 0.0199, 0.0203, 0.0205])
            self.pp0 = interp1d(ldelta, p)
            self.pq0 = interp1d(ldelta, q)

    def _check_mass_def_strict(self, mass_def):
        if mass_def.Delta == 'fof':
            return True
        return False

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        ld = np.log10(self._get_Delta_m(cosmo, a))
        nu = 1.686 / sigM
        # redshift evolution only up to z=3
        a = np.clip(a, 0.25, 1)
        pa = self.pa0(ld) * a**(-0.27)
        pb = self.pb0(ld) * a**(-0.20)
        pc = self.pc0(ld) * a**0.01
        pd = self.pd0(ld) * a**0.08
        pA0 = self.pA0(ld)
        if self.norm_all_z:
            z = 1./a - 1
            pp = self.pp0(ld)
            pq = self.pq0(ld)
            pA0 *= np.exp(z*(pp+pq*z))
        return nu * pA0 * (1 + (pb * nu)**(-2 * pd)) * \
            nu**(2 * pa) * np.exp(-0.5 * pc * nu**2)


class MassFuncBocquet16(MassFunc):
    """ Implements mass function described in arXiv:1502.07357.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts SO masses with
            Delta = 200 (matter, critical) and 500 (critical).
            If `None`, Delta = 200 (matter) will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
        hydro (bool): if `False`, use the parametrization found
            using dark-matter-only simulations. Otherwise, include
            baryonic effects (default).
    """
    name = 'Bocquet16'

    @warn_api
    def __init__(self, *, mass_def=None, mass_def_strict=True,
                 hydro=True):
        self.hydro = hydro
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _default_mass_def(self):
        self.mass_def = MassDef200m()

    def _setup(self):
        if int(self.mass_def.Delta) == 200:
            if self.mass_def.rho_type == 'matter':
                self.mass_def_type = '200m'
            elif self.mass_def.rho_type == 'critical':
                self.mass_def_type = '200c'
        elif int(self.mass_def.Delta) == 500:
            if self.mass_def.rho_type == 'critical':
                self.mass_def_type = '500c'
        if self.mass_def_type == '200m':
            if self.hydro:
                self.A0 = 0.228
                self.a0 = 2.15
                self.b0 = 1.69
                self.c0 = 1.30
                self.Az = 0.285
                self.az = -0.058
                self.bz = -0.366
                self.cz = -0.045
            else:
                self.A0 = 0.175
                self.a0 = 1.53
                self.b0 = 2.55
                self.c0 = 1.19
                self.Az = -0.012
                self.az = -0.040
                self.bz = -0.194
                self.cz = -0.021
        elif self.mass_def_type == '200c':
            if self.hydro:
                self.A0 = 0.202
                self.a0 = 2.21
                self.b0 = 2.00
                self.c0 = 1.57
                self.Az = 1.147
                self.az = 0.375
                self.bz = -1.074
                self.cz = -0.196
            else:
                self.A0 = 0.222
                self.a0 = 1.71
                self.b0 = 2.24
                self.c0 = 1.46
                self.Az = 0.269
                self.az = 0.321
                self.bz = -0.621
                self.cz = -0.153
        elif self.mass_def_type == '500c':
            if self.hydro:
                self.A0 = 0.180
                self.a0 = 2.29
                self.b0 = 2.44
                self.c0 = 1.97
                self.Az = 1.088
                self.az = 0.150
                self.bz = -1.008
                self.cz = -0.322
            else:
                self.A0 = 0.241
                self.a0 = 2.18
                self.b0 = 2.35
                self.c0 = 2.02
                self.Az = 0.370
                self.az = 0.251
                self.bz = -0.698
                self.cz = -0.310

    def _check_mass_def_strict(self, mass_def):
        if isinstance(mass_def.Delta, str):
            return True
        elif int(mass_def.Delta) == 200:
            if mass_def.rho_type not in ['matter', 'critical']:
                return True
        elif int(mass_def.Delta) == 500:
            if mass_def.rho_type != 'critical':
                return True
        else:
            return True
        return False

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        zp1 = 1./a
        AA = self.A0 * zp1**self.Az
        aa = self.a0 * zp1**self.az
        bb = self.b0 * zp1**self.bz
        cc = self.c0 * zp1**self.cz

        f = AA * ((sigM / bb)**-aa + 1.0) * np.exp(-cc / sigM**2)

        if self.mass_def_type == '200c':
            z = 1./a-1
            Omega_m = omega_x(cosmo, a, "matter")
            gamma0 = 3.54E-2 + Omega_m**0.09
            gamma1 = 4.56E-2 + 2.68E-2 / Omega_m
            gamma2 = 0.721 + 3.50E-2 / Omega_m
            gamma3 = 0.628 + 0.164 / Omega_m
            delta0 = -1.67E-2 + 2.18E-2 * Omega_m
            delta1 = 6.52E-3 - 6.86E-3 * Omega_m
            gamma = gamma0 + gamma1 * np.exp(-((gamma2 - z) / gamma3)**2)
            delta = delta0 + delta1 * z
            M200c_M200m = gamma + delta * lnM
            f *= M200c_M200m
        elif self.mass_def_type == '500c':
            z = 1./a-1
            Omega_m = omega_x(cosmo, a, "matter")
            alpha0 = 0.880 + 0.329 * Omega_m
            alpha1 = 1.00 + 4.31E-2 / Omega_m
            alpha2 = -0.365 + 0.254 / Omega_m
            alpha = alpha0 * (alpha1 * z + alpha2) / (z + alpha2)
            beta = -1.7E-2 + 3.74E-3 * Omega_m
            M500c_M200m = alpha + beta * lnM
            f *= M500c_M200m
        return f


class MassFuncWatson13(MassFunc):
    """ Implements mass function described in arXiv:1212.0095.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts fof and any SO masses.
            If `None`, Delta = 200 (matter) will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    name = 'Watson13'

    @warn_api
    def __init__(self, *, mass_def=None, mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _default_mass_def(self):
        self.mass_def = MassDef200m()

    def _setup(self):
        self.is_fof = self.mass_def.Delta == 'fof'

    def _check_mass_def_strict(self, mass_def):
        if mass_def.Delta == 'vir':
            return True
        return False

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        if self.is_fof:
            pA = 0.282
            pa = 2.163
            pb = 1.406
            pc = 1.210
            return pA * ((pb / sigM)**pa + 1.) * np.exp(-pc / sigM**2)
        else:
            om = omega_x(cosmo, a, "matter")
            Delta_178 = self.mass_def.Delta / 178.0

            if a == 1.0:
                pA = 0.194
                pa = 1.805
                pb = 2.267
                pc = 1.287
            elif a < 0.14285714285714285:  # z>6
                pA = 0.563
                pa = 3.810
                pb = 0.874
                pc = 1.453
            else:
                pA = om * (1.097 * a**3.216 + 0.074)
                pa = om * (5.907 * a**3.058 + 2.349)
                pb = om * (3.136 * a**3.599 + 2.344)
                pc = 1.318

            f_178 = pA * ((pb / sigM)**pa + 1.) * np.exp(-pc / sigM**2)
            C = np.exp(0.023 * (Delta_178 - 1.0))
            d = -0.456 * om - 0.139
            Gamma = (C * Delta_178**d *
                     np.exp(0.072 * (1.0 - Delta_178) / sigM**2.130))
            return f_178 * Gamma


class MassFuncAngulo12(MassFunc):
    """ Implements mass function described in arXiv:1203.3216.
    This parametrization is only valid for 'fof' masses.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts FoF masses only.
            If `None`, FoF masses will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    name = 'Angulo12'

    @warn_api
    def __init__(self, *, mass_def=None, mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _default_mass_def(self):
        self.mass_def = MassDef('fof', 'matter')

    def _setup(self):
        self.A = 0.201
        self.a = 2.08
        self.b = 1.7
        self.c = 1.172

    def _check_mass_def_strict(self, mass_def):
        if mass_def.Delta != 'fof':
            return True
        return False

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        return self.A * ((self.a / sigM)**self.b + 1.) * \
            np.exp(-self.c / sigM**2)


class MassFuncBocquet20(MassFunc, Emulator):
    """ Emulated mass function described in arXiv:2003.12116.

    This emulator is based on a Mira-Titan Universe suite of
    cosmological N-body simulations.

    Parameters:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            A mass definition object.
            This parametrization accepts SO masses with
            Delta = 200 critical.
        mass_def_strict (bool):
            If False, consistency of the mass definition
            will be ignored. The default is True.
        extrapolate (bool):
            If True, the queried mass range outside of the emulator's
            training mass range will be extrapolated in log-space,
            linearly for the low masses and quadratically for the
            high masses. Otherwise, it will return zero for those
            masses. The default is True.
    """
    name = 'Bocquet20'

    def __init__(self, *, mass_def=None, mass_def_strict=True,
                 extrapolate=True):
        self.extrapolate = extrapolate
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)

    def _default_mass_def(self):
        self.mass_def = MassDef200c()

    def _load_emu(self):
        from MiraTitanHMFemulator import Emulator as HMFemu
        model = HMFemu()
        # build the emulator bounds
        bounds = model.param_limits.copy()
        bounds["z"] = [0., 2.02]
        bounds["M_min"] = [1e13, np.inf]
        return EmulatorObject(model, bounds)

    def _build_parameters(self, cosmo=None, M=None, a=None):
        from ..neutrinos import Omega_nu_h2
        # check input
        if (cosmo is not None) and (a is None):
            raise ValueError("Need value for scale factor")

        self._parameters = {}
        if cosmo is not None:
            h = cosmo["h"]
            m_nu = np.sum(cosmo["m_nu"])
            T_CMB = cosmo["T_CMB"]
            Omega_c = cosmo["Omega_c"]
            Omega_b = cosmo["Omega_b"]
            Omega_nu_h2 = Omega_nu_h2(a, m_nu=m_nu, T_CMB=T_CMB)

            self._parameters["Ommh2"] = (Omega_c + Omega_b)*h**2 + Omega_nu_h2
            self._parameters["Ombh2"] = Omega_b * h**2
            self._parameters["Omnuh2"] = Omega_nu_h2
            self._parameters["n_s"] = cosmo["n_s"]
            self._parameters["h"] = cosmo["h"]
            self._parameters["sigma_8"] = cosmo["sigma8"]
            self._parameters["w_0"] = cosmo["w0"]
            self._parameters["w_b"] = (-cosmo["wa"] - cosmo["w0"])**0.25

            self._parameters["z"] = 1/a - 1
            if not self.extrapolate:
                self._parameters["M_min"] = np.min(M*h)

    def _finalize_parameters(self, wa):
        # Translate parameters to final emulator input
        self._parameters["w_a"] = wa
        self._parameters.pop("w_b")
        self._parameters.pop("z")
        if not self.extrapolate:
            self._parameters.pop("M_min")

    def _extrapolate_hmf(self, hmf, M, eps=1e-12):
        M_use = np.atleast_1d(M)
        # indices where the emulator outputs reasonable values
        idx = np.where(hmf >= eps)[0]

        # extrapolate low masses linearly...
        M_lo, hmf_lo = M_use[idx][:2], hmf[idx][:2]
        F_lo = interp1d(np.log(M_lo), np.log(hmf_lo), kind="linear",
                        bounds_error=False, fill_value="extrapolate")
        # ...and high masses quadratically
        M_hi, hmf_hi = M_use[idx][-3:], hmf[idx][-3:]
        F_hi = interp1d(np.log(M_hi), np.log(hmf_hi), kind="quadratic",
                        bounds_error=False, fill_value="extrapolate")

        hmf[:idx[0]] = np.exp(F_lo(np.log(M_use[:idx[0]])))
        hmf[idx[-1]:] = np.exp(F_hi(np.log(M_use[idx[-1]:])))
        return hmf

    def get_mass_function(self, cosmo, M, a):
        # load and build parameters
        emu = self._load_emu()
        self._build_parameters(cosmo, M, a)
        emu.check_bounds(self._parameters)
        self._finalize_parameters(cosmo["wa"])

        def hmf_dummy(cosmo, M, a):
            # Populate the queried masses with some emulator-friendly
            # values and re-calculate the mass function.
            M = np.atleast_1d(M)
            M_dummy = np.logspace(13, 16, 64)
            M_dummy = np.sort(np.append(M_dummy, M))
            idx_ask = np.searchsorted(M_dummy, M).tolist()
            hmf = self.get_mass_function(cosmo, M_dummy, a)
            return hmf, idx_ask

        M_use = np.atleast_1d(M) * cosmo["h"]
        hmf = np.zeros_like(M_use)
        # keep only the masses inside the emulator's range
        idx = np.where(M_use > 1e13)[0]
        if len(idx) > 0:
            # Under normal use, this block runs.
            M_emu = M_use[idx]
            hmf[idx] = emu.model.predict(
                self._parameters, 1/a-1, M_emu,
                get_errors=False)[0]
            hmf *= cosmo["h"]**3
        else:
            # No masses inside the emulator range.
            # Create a dummy mass array, extrapolate,
            # and throw away all but the queried masses.
            hmf, idx_ask = hmf_dummy(cosmo, M_use/cosmo["h"], a)
            hmf = hmf[idx_ask]

        if np.any(hmf < 1e-12):
            # M_lo == 0 ; M_hi == O(1e-300)
            # If this is the case, extrapolate to replace the small values.
            if np.size(M) >= 3:
                # Quadratic interpolation/extrapolation requires
                # at least 3 points.
                if self.extrapolate:
                    hmf = self._extrapolate_hmf(hmf, M, 1e-12)
            else:
                # Masses partially inside the emulator range,
                # but too few points, so can't safely extrapolate.
                # Create a dummy mass array and extrapolate,
                # and throw away all but the queried masses.
                hmf, idx_ask = hmf_dummy(cosmo, M_use/cosmo["h"], a)
                hmf = hmf[idx_ask]

        if np.ndim(M) == 0:
            hmf = hmf[0]
        return hmf


@deprecated(new_function=MassFunc.from_name)
def mass_function_from_name(name):
    """ Returns mass function subclass from name string

    Args:
        name (string): a mass function name

    Returns:
        MassFunc subclass corresponding to the input name.
    """
    return MassFunc.from_name(name)
