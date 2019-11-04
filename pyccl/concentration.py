from . import ccllib as lib
from .background import growth_factor
from .hmfunc import sigmaM
from .massdef import MassDef, mass2radius_lagrangian
import numpy as np
from .power import linear_matter_power
from scipy.interpolate import InterpolatedUnivariateSpline


class Concentration(object):
    """ This class enables the calculation of halo concentrations.

    Args:
        mass_def (:obj:`MassDef`): a mass definition object that fixes
            the mass definition used by this c(M)
            parametrization.
    """
    name = 'default'

    def __init__(self, mass_def=None):
        if mass_def is not None:
            if self._check_mdef(mass_def):
                raise ValueError("c(M) relation " + self.name +
                                 " is not compatible with mass definition" +
                                 " Delta = %s, " % (mass_def.Delta) +
                                 " rho = " + mass_def.rho_type)
            self.mdef = mass_def
        else:
            self._default_mdef()
        self._setup()

    def _default_mdef(self):
        """ Assigns a default mass definition for this object if
        none is passed at initialization.
        """
        self.mdef = MassDef('fof', 'matter')

    def _setup(self):
        """ Use this function to initialize any internal attributes
        of this object. This function is called at the very end of the
        constructor call.
        """
        pass

    def _check_mdef(self, mdef):
        """ Return False if the input mass definition agrees with
        the definitions for which this concentration-mass relation
        works. True otherwise. This function gets called at the
        start of the constructor call.

        Args:
            mdef (:obj:`MassDef`): a mass definition object.

        Returns:
            bool: True if the mass definition is not compatible with
                this parametrization. False otherwise.
        """
        return False

    def _get_consistent_mass(self, cosmo, M, a, mdef_other):
        """ Transform a halo mass with a given mass definition into
        the corresponding mass definition that was used to initialize
        this object.

        Args:
            cosmo (:obj:`Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            mdef_other (:obj:`MassDef`): a mass definition object.

        Returns:
            float or array_like: mass according to this object's
            mass definition.
        """
        if mdef_other is not None:
            M_use = mdef_other.translate_mass(cosmo, M, a, self.mdef)
        else:
            M_use = M
        return M_use

    def get_concentration(self, cosmo, M, a, mdef_other=None):
        """ Returns the concentration for input parameters.

        Args:
            cosmo (:obj:`Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            mdef_other (:obj:`MassDef`): the mass definition object
                that defines M.

        Returns:
            float or array_like: concentration.
        """
        M_use = self._get_consistent_mass(cosmo,
                                          np.atleast_1d(M),
                                          a, mdef_other)

        c = self.concentration(cosmo, M_use, a)
        if np.isscalar(M):
            c = c[0]
        return c


class ConcentrationDiemer15(Concentration):
    name = 'Diemer15'

    def __init__(self, mdef=None):
        super(ConcentrationDiemer15, self).__init__(mdef)

    def _default_mdef(self):
        self.mdef = MassDef(200, 'critical')

    def _setup(self):
        self.kappa = 1.0
        self.phi_0 = 6.58
        self.phi_1 = 1.27
        self.eta_0 = 7.28
        self.eta_1 = 1.56
        self.alpha = 1.08
        self.beta = 1.77

    def _check_mdef(self, mdef):
        if (np.fabs(mdef.Delta - 200.) > 1E-4) and \
           (mdef.rho_type != 'critical'):
            return True
        return False

    def concentration(self, cosmo, M, a):
        M_use = np.atleast_1d(M)

        # Compute power spectrum slope
        R = mass2radius_lagrangian(cosmo, M_use)
        lk_R = np.log10(2.0 * np.pi / R * self.kappa)
        # Using spline interpolation
        # TODO: it'd be worth getting C to return the
        #       spline derivative.
        lkmin = np.amin(lk_R-0.05)
        lkmax = np.amax(lk_R+0.05)
        logk = np.arange(lkmin, lkmax, 0.01)
        lpk = np.log10(linear_matter_power(cosmo, 10**logk, a))
        interp = InterpolatedUnivariateSpline(logk, lpk)
        n = interp(lk_R, nu=1)

        status = 0
        delta_c, status = lib.dc_NakamuraSuto(cosmo.cosmo, a, status)
        sig = sigmaM(cosmo, M_use, a)
        nu = delta_c / sig

        floor = self.phi_0 + n * self.phi_1
        nu0 = self.eta_0 + n * self.eta_1
        c = 0.5 * floor * ((nu0 / nu)**self.alpha +
                           (nu / nu0)**self.beta)
        if np.isscalar(M):
            c = c[0]

        return c


class ConcentrationBhattacharya13(Concentration):
    name = 'Bhattacharya13'

    def __init__(self, mdef=None):
        super(ConcentrationBhattacharya13, self).__init__(mdef)

    def _default_mdef(self):
        self.mdef = MassDef(200, 'critical')

    def _check_mdef(self, mdef):
        if mdef.Delta != 'vir':
            if np.fabs(mdef.Delta - 200.) > 1E-4:
                return True
        return False

    def _setup(self):
        if self.mdef.Delta == 'vir':
            self.A = 7.7
            self.B = 0.9
            self.C = -0.29
        else:  # Now Delta has to be 200
            if self.mdef.rho_type == 'matter':
                self.A = 9.0
                self.B = 1.15
                self.C = -0.29
            else:  # Now rho_type has to be critical
                self.A = 5.9
                self.B = 0.54
                self.C = -0.35

    def concentration(self, cosmo, M, a):
        gz = growth_factor(cosmo, a)
        status = 0
        delta_c, status = lib.dc_NakamuraSuto(cosmo.cosmo, a, status)
        sig = sigmaM(cosmo, M, a)
        nu = delta_c / sig
        return self.A * gz**self.B * nu**self.C


class ConcentrationPrada12(Concentration):
    name = 'Prada12'

    def __init__(self, mdef=None):
        super(ConcentrationPrada12, self).__init__(mdef)

    def _default_mdef(self):
        self.mdef = MassDef(200, 'critical')

    def _check_mdef(self, mdef):
        if (np.fabs(mdef.Delta - 200.) > 1E-4) and \
           (mdef.rho_type != 'critical'):
            return True
        return False

    def _setup(self):
        self.c0 = 3.681
        self.c1 = 5.033
        self.al = 6.948
        self.x0 = 0.424
        self.i0 = 1.047
        self.i1 = 1.646
        self.be = 7.386
        self.x1 = 0.526
        self.cnorm = 1. / self.cmin(1.393)
        self.inorm = 1. / self.imin(1.393)

    def cmin(self, x):
        return self.c0 + (self.c1 - self.c0) * \
            (np.arctan(self.al * (x - self.x0)) / np.pi + 0.5)

    def imin(self, x):
        return self.i0 + (self.i1 - self.i0) * \
            (np.arctan(self.be * (x - self.x1)) / np.pi + 0.5)

    def concentration(self, cosmo, M, a):
        sig = sigmaM(cosmo, M, a)
        om = cosmo.cosmo.params.Omega_m
        ol = cosmo.cosmo.params.Omega_l
        x = a * (ol / om)**(1. / 3.)
        B0 = self.cmin(x) * self.cnorm
        B1 = self.imin(x) * self.inorm
        sig_p = B1 * sig
        Cc = 2.881 * ((sig_p / 1.257)**1.022 + 1) * np.exp(0.060 / sig_p**2)
        return B0 * Cc


class ConcentrationKlypin11(Concentration):
    name = 'Klypin11'

    def __init__(self, mdef=None):
        super(ConcentrationKlypin11, self).__init__(mdef)

    def _default_mdef(self):
        self.mdef = MassDef('vir', 'critical')

    def _check_mdef(self, mdef):
        if mdef.Delta != 'vir':
            return True
        return False

    def concentration(self, cosmo, M, a):
        M_pivot_inv = cosmo.cosmo.params.h * 1E-12
        return 9.6 * (M * M_pivot_inv)**-0.075


class ConcentrationDuffy08(Concentration):
    name = 'Duffy08'

    def __init__(self, mdef=None):
        super(ConcentrationDuffy08, self).__init__(mdef)

    def _default_mdef(self):
        self.mdef = MassDef(200, 'critical')

    def _check_mdef(self, mdef):
        if mdef.Delta != 'vir':
            if np.fabs(mdef.Delta - 200.) > 1E-4:
                return True
        return False

    def _setup(self):
        if self.mdef.Delta == 'vir':
            self.A = 7.85
            self.B = -0.081
            self.C = -0.47
        else:  # Now Delta has to be 200
            if self.mdef.rho_type == 'matter':
                self.A = 10.14
                self.B = -0.081
                self.C = -1.01
            else:  # Now rho_type has to be critical
                self.A = 5.71
                self.B = -0.084
                self.C = -0.47

    def concentration(self, cosmo, M, a):
        M_pivot_inv = cosmo.cosmo.params.h * 5E-13
        return self.A * (M * M_pivot_inv)**self.B * a**(-self.C)


def concentration_from_name(name):
    """ Returns halo concentration subclass from name string

    Args:
        name (string): a concentration name

    Returns:
        Concentration subclass corresponding to the input name.
    """
    concentrations = {c.name: c
                      for c in Concentration.__subclasses__()}
    if name in concentrations:
        return concentrations[name]
    else:
        raise ValueError("Concentration %s not implemented")
