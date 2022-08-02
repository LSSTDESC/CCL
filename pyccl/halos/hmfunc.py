from .. import ccllib as lib
from ..core import check
from ..background import omega_x
from ..parameters import physical_constants
from .massdef import MassDef, MassDef200m
import numpy as np
import functools


class MassFunc(object):
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
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object that fixes
            the mass definition used by this mass function
            parametrization.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    name = 'default'

    def __init__(self, cosmo, mass_def=None, mass_def_strict=True):
        # Initialize sigma(M) splines if needed
        cosmo.compute_sigma()
        self.mass_def_strict = mass_def_strict
        # Check if mass function was provided and check that it's
        # sensible.
        if mass_def is not None:
            if self._check_mdef(mass_def):
                raise ValueError("Mass function " + self.name +
                                 " is not compatible with mass definition" +
                                 " Delta = %s, " % (mass_def.Delta) +
                                 " rho = " + mass_def.rho_type)
            self.mdef = mass_def
        else:
            self._default_mdef()
        self._setup(cosmo)

    def _default_mdef(self):
        """ Assigns a default mass definition for this object if
        none is passed at initialization.
        """
        self.mdef = MassDef('fof', 'matter')

    def _setup(self, cosmo):
        """ Use this function to initialize any internal attributes
        of this object. This function is called at the very end of the
        constructor call.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        """
        pass

    def _check_mdef_strict(self, mdef):
        return False

    def _check_mdef(self, mdef):
        """ Return False if the input mass definition agrees with
        the definitions for which this mass function parametrization
        works. True otherwise. This function gets called at the
        start of the constructor call.

        Args:
            mdef (:class:`~pyccl.halos.massdef.MassDef`):
                a mass definition object.

        Returns:
            bool: True if the mass definition is not compatible with \
                this mass function parametrization. False otherwise.
        """
        if self.mass_def_strict:
            return self._check_mdef_strict(mdef)
        return False

    def _get_consistent_mass(self, cosmo, M, a, mdef_other):
        """ Transform a halo mass with a given mass definition into
        the corresponding mass definition that was used to initialize
        this object.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            mdef_other (:class:`~pyccl.halos.massdef.MassDef`):
                a mass definition object.

        Returns:
            float or array_like: mass according to this object's \
                mass definition.
        """
        if mdef_other is not None:
            M_use = mdef_other.translate_mass(cosmo, M, a, self.mdef)
        else:
            M_use = M
        return np.log10(M_use)

    def _get_Delta_m(self, cosmo, a):
        """ For SO-based mass definitions, this returns the corresponding
        value of Delta for a rho_matter-based definition. This is useful
        mostly for the Tinker mass functions, which are defined for any
        SO mass in general, but explicitly only for Delta_matter.
        """
        delta = self.mdef.get_Delta(cosmo, a)
        if self.mdef.rho_type == 'matter':
            return delta
        else:
            om_this = omega_x(cosmo, a, self.mdef.rho_type)
            om_matt = omega_x(cosmo, a, 'matter')
            return delta * om_this / om_matt

    def get_mass_function(self, cosmo, M, a, mdef_other=None):
        """ Returns the mass function for input parameters.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            mdef_other (:class:`~pyccl.halos.massdef.MassDef`):
                the mass definition object that defines M.

        Returns:
            float or array_like: mass function \
                :math:`dn/d\\log_{10}M` in units of Mpc^-3 (comoving).
        """
        M_use = np.atleast_1d(M)
        logM = self._get_consistent_mass(cosmo, M_use,
                                         a, mdef_other)

        # sigma(M)
        status = 0
        sigM, status = lib.sigM_vec(cosmo.cosmo, a, logM,
                                    len(logM), status)
        check(status, cosmo=cosmo)
        # dlogsigma(M)/dlog10(M)
        dlns_dlogM, status = lib.dlnsigM_dlogM_vec(cosmo.cosmo, a, logM,
                                                   len(logM), status)
        check(status, cosmo=cosmo)

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
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts FoF masses only.
            If `None`, FoF masses will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    name = 'Press74'

    def __init__(self, cosmo, mass_def=None, mass_def_strict=True):
        super(MassFuncPress74, self).__init__(cosmo,
                                              mass_def,
                                              mass_def_strict)

    def _default_mdef(self):
        self.mdef = MassDef('fof', 'matter')

    def _setup(self, cosmo):
        self.norm = np.sqrt(2/np.pi)

    def _check_mdef_strict(self, mdef):
        if mdef.Delta != 'fof':
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
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
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

    def __init__(self, cosmo, mass_def=None, mass_def_strict=True,
                 use_delta_c_fit=False):
        self.use_delta_c_fit = use_delta_c_fit
        super(MassFuncSheth99, self).__init__(cosmo,
                                              mass_def,
                                              mass_def_strict)

    def _default_mdef(self):
        self.mdef = MassDef('fof', 'matter')

    def _setup(self, cosmo):
        self.A = 0.21615998645
        self.p = 0.3
        self.a = 0.707

    def _check_mdef_strict(self, mdef):
        if mdef.Delta != 'fof':
            return True

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        if self.use_delta_c_fit:
            status = 0
            delta_c, status = lib.dc_NakamuraSuto(cosmo.cosmo, a, status)
            check(status, cosmo=cosmo)
        else:
            delta_c = 1.68647

        nu = delta_c / sigM
        return nu * self.A * (1. + (self.a * nu**2)**(-self.p)) * \
            np.exp(-self.a * nu**2/2.)


class MassFuncJenkins01(MassFunc):
    """ Implements mass function described in astro-ph/0005260.
    This parametrization is only valid for 'fof' masses.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts FoF masses only.
            If `None`, FoF masses will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    name = 'Jenkins01'

    def __init__(self, cosmo, mass_def=None, mass_def_strict=True):
        super(MassFuncJenkins01, self).__init__(cosmo,
                                                mass_def=mass_def,
                                                mass_def_strict=True)

    def _default_mdef(self):
        self.mdef = MassDef('fof', 'matter')

    def _setup(self, cosmo):
        self.A = 0.315
        self.b = 0.61
        self.q = 3.8

    def _check_mdef_strict(self, mdef):
        if mdef.Delta != 'fof':
            return True
        return False

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        return self.A * np.exp(-np.fabs(-np.log(sigM) + self.b)**self.q)


class MassFuncTinker08(MassFunc):
    """ Implements mass function described in arXiv:0803.2706.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts SO masses with
            200 < Delta < 3200 with respect to the matter density.
            If `None`, Delta = 200 (matter) will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    name = 'Tinker08'

    def __init__(self, cosmo, mass_def=None, mass_def_strict=True):
        super(MassFuncTinker08, self).__init__(cosmo,
                                               mass_def,
                                               mass_def_strict)

    def _default_mdef(self):
        self.mdef = MassDef200m()

    def _pd(self, ld):
        return 10.**(-(0.75/(ld - 1.8750612633))**1.2)

    def _setup(self, cosmo):
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

    def _check_mdef_strict(self, mdef):
        if mdef.Delta == 'fof':
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
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts any SO masses.
            If `None`, Delta = 200 (matter) will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    name = 'Despali16'

    def __init__(self, cosmo, mass_def=None, mass_def_strict=True,
                 ellipsoidal=False):
        super(MassFuncDespali16, self).__init__(cosmo,
                                                mass_def,
                                                mass_def_strict)
        self.ellipsoidal = ellipsoidal

    def _default_mdef(self):
        self.mdef = MassDef200m()

    def _setup(self, cosmo):
        pass

    def _check_mdef_strict(self, mdef):
        if mdef.Delta == 'fof':
            return True
        return False

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        status = 0
        delta_c, status = lib.dc_NakamuraSuto(cosmo.cosmo, a, status)
        check(status, cosmo=cosmo)

        Dv, status = lib.Dv_BryanNorman(cosmo.cosmo, a, status)
        check(status, cosmo=cosmo)

        x = np.log10(self.mdef.get_Delta(cosmo, a) *
                     omega_x(cosmo, a, self.mdef.rho_type) / Dv)

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
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
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

    def __init__(self, cosmo, mass_def=None, mass_def_strict=True,
                 norm_all_z=False):
        self.norm_all_z = norm_all_z
        super(MassFuncTinker10, self).__init__(cosmo,
                                               mass_def,
                                               mass_def_strict)

    def _default_mdef(self):
        self.mdef = MassDef200m()

    def _setup(self, cosmo):
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

    def _check_mdef_strict(self, mdef):
        if mdef.Delta == 'fof':
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
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
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

    def __init__(self, cosmo, mass_def=None, mass_def_strict=True,
                 hydro=True):
        self.hydro = hydro
        super(MassFuncBocquet16, self).__init__(cosmo,
                                                mass_def,
                                                mass_def_strict)

    def _default_mdef(self):
        self.mdef = MassDef200m()

    def _setup(self, cosmo):
        if int(self.mdef.Delta) == 200:
            if self.mdef.rho_type == 'matter':
                self.mdef_type = '200m'
            elif self.mdef.rho_type == 'critical':
                self.mdef_type = '200c'
        elif int(self.mdef.Delta) == 500:
            if self.mdef.rho_type == 'critical':
                self.mdef_type = '500c'
        if self.mdef_type == '200m':
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
        elif self.mdef_type == '200c':
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
        elif self.mdef_type == '500c':
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

    def _check_mdef_strict(self, mdef):
        if isinstance(mdef.Delta, str):
            return True
        elif int(mdef.Delta) == 200:
            if (mdef.rho_type != 'matter') and \
               (mdef.rho_type != 'critical'):
                return True
        elif int(mdef.Delta) == 500:
            if mdef.rho_type != 'critical':
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

        if self.mdef_type == '200c':
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
        elif self.mdef_type == '500c':
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
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts fof and any SO masses.
            If `None`, Delta = 200 (matter) will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    name = 'Watson13'

    def __init__(self, cosmo, mass_def=None, mass_def_strict=True):
        super(MassFuncWatson13, self).__init__(cosmo,
                                               mass_def,
                                               mass_def_strict)

    def _default_mdef(self):
        self.mdef = MassDef200m()

    def _setup(self, cosmo):
        self.is_fof = self.mdef.Delta == 'fof'

    def _check_mdef_strict(self, mdef):
        if mdef.Delta == 'vir':
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
            Delta_178 = self.mdef.Delta / 178.0

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
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        mass_def (:class:`~pyccl.halos.massdef.MassDef`):
            a mass definition object.
            this parametrization accepts FoF masses only.
            If `None`, FoF masses will be used.
        mass_def_strict (bool): if False, consistency of the mass
            definition will be ignored.
    """
    name = 'Angulo12'

    def __init__(self, cosmo, mass_def=None, mass_def_strict=True):
        super(MassFuncAngulo12, self).__init__(cosmo,
                                               mass_def,
                                               mass_def_strict)

    def _default_mdef(self):
        self.mdef = MassDef('fof', 'matter')

    def _setup(self, cosmo):
        self.A = 0.201
        self.a = 2.08
        self.b = 1.7
        self.c = 1.172

    def _check_mdef_strict(self, mdef):
        if mdef.Delta != 'fof':
            return True
        return False

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        return self.A * ((self.a / sigM)**self.b + 1.) * \
            np.exp(-self.c / sigM**2)


@functools.wraps(MassFunc.from_name)
def mass_function_from_name(name):
    return MassFunc.from_name(name)
