from . import ccllib as lib
from .core import check
import numpy as np


class HBiasFunc(object):
    """ This class enables the calculation of halo bias functions.
    We currently assume that all mass functions can be written as
    functions that depend on M only through sigma_M (where
    sigma_M^2 is the overdensity variance on spheres with a
    radius given by the Lagrangian radius for mass M).
    All sub-classes implementing specific parametrizations
    can therefore be simply created by replacing this class'
    get_bsigma method.

    Args:
        name (str): a name for this mass function object.
        cosmo (:obj:`Cosmology`): A Cosmology object.
        mass_def (:obj:`HMDef`): a mass definition object that fixes
            the mass definition used by this mass function
            parametrization.
    """
    def __init__(self, name, cosmo, mass_def):
        cosmo.compute_sigma()
        self.name = name
        if self._check_mdef(mass_def):
            raise ValueError("Mass function " + name +
                             " is not compatible with mass definition" +
                             " Delta = %.1lf, " % (mass_def.Delta) +
                             " rho = " + mass_def.rho_type)
        self.mdef = mass_def
        self._setup(cosmo)

    def _setup(self, cosmo):
        """ Use this function to initialize any internal attributes
        of this object. This function is called at the very end of the
        constructor call.

        Args:
            cosmo (:obj:`Cosmology`): A Cosmology object.
        """
        pass

    def _check_mdef(self, mdef):
        """ Return False if the input mass definition agrees with
        the definitions for which this mass function parametrization
        works. True otherwise. This function gets called at the
        start of the constructor call.

        Args:
            mdef (:obj:`HMDef`): a mass definition object.

        Returns:
            bool: True if the mass definition is not compatible with
                this mass function parametrization. False otherwise.
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
            mdef_other (:obj:`HMDef`): a mass definition object.

        Returns:
            float or array_like: mass according to this object's
            mass definition.
        """
        if mdef_other is not None:
            M_use = mdef_other.translate_mass(cosmo, M, a, self.mdef)
        else:
            M_use = M
        return np.log10(M_use)

    def get_halo_bias(self, cosmo, M, a, mdef_other=None):
        """ Returns the hmass function for input parameters.

        Args:
            cosmo (:obj:`Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            mdef_other (:obj:`HMDef`): the mass definition object
                that defines M.

        Returns:
            float or array_like: halo bias.
        """
        M_use = np.atleast_1d(M)
        logM = self._get_consistent_mass(cosmo, M_use,
                                         a, mdef_other)

        # sigma(M)
        status = 0
        sigM, status = lib.sigM_vec(cosmo.cosmo, a, logM,
                                    len(logM), status)
        check(status)

        b = self.get_bsigma(cosmo, sigM, a)
        if np.isscalar(M):
            b = b[0]
        return b

    def get_bsigma(self, cosmo, sigM, a):
        """ Get the halo bias as a function of sigmaM.

        Args:
            cosmo (:obj:`Cosmology`): A Cosmology object.
            sigM (float or array_like): standard deviation in the
                overdensity field on the scale of this halo.
            a (float): scale factor.

        Returns:
            float or array_like: f(sigma_M) function.
        """
        raise NotImplementedError("Use one of the non-default "
                                  "HBiasFunc classes")


class HBiasFuncSheth99(HBiasFunc):
    """ Implements halo bias described in 1999MNRAS.308..119S

    Args:
        cosmo (:obj:`Cosmology`): A Cosmology object.
        mass_def (:obj:`HMDef`): a mass definition object.
            this parametrization only accepts 'fof' masses.
    """
    def __init__(self, cosmo, mass_def):
        super(HBiasFuncSheth99, self).__init__("Sheth99",
                                               cosmo,
                                               mass_def)

    def _setup(self, cosmo):
        self.p = 0.3
        self.a = 0.707
        self.dc = 1.68647

    def _check_mdef(self, mdef):
        if mdef.Delta != 'fof':
            return True
        return False

    def get_bsigma(self, cosmo, sigM, a):
        nu = self.dc/sigM
        anu2 = self.a * nu**2
        return 1. + (anu2 - 1. + 2. * self.p / (1. + anu2**self.p))/self.dc


class HBiasFuncSheth01(HBiasFunc):
    """ Implements halo bias described in 2001MNRAS.323....1S

    Args:
        cosmo (:obj:`Cosmology`): A Cosmology object.
        mass_def (:obj:`HMDef`): a mass definition object.
            this parametrization only accepts 'fof' masses.
    """
    def __init__(self, cosmo, mass_def):
        super(HBiasFuncSheth01, self).__init__("Sheth01",
                                               cosmo,
                                               mass_def)

    def _setup(self, cosmo):
        self.a = 0.707
        self.sqrta = 0.84083292038
        self.b = 0.5
        self.c = 0.6
        self.dc = 1.68647

    def _check_mdef(self, mdef):
        if mdef.Delta != 'fof':
            return True
        return False

    def get_bsigma(self, cosmo, sigM, a):
        nu = self.dc/sigM
        anu2 = self.a * nu**2
        anu2c = anu2**self.c
        t1 = self.b * (1.0 - self.c) * (1.0 - 0.5 * self.c)
        return 1. + (self.sqrta * anu2 * (1 + self.b / anu2c) -
                     anu2c / (anu2c + t1)) / (self.sqrta * self.dc)


class HBiasFuncBhattacharya11(HBiasFunc):
    """ Implements halo bias described in 2011ApJ...732..122B

    Args:
        cosmo (:obj:`Cosmology`): A Cosmology object.
        mass_def (:obj:`HMDef`): a mass definition object.
            this parametrization only accepts 'fof' masses.
    """
    def __init__(self, cosmo, mass_def):
        super(HBiasFuncBhattacharya11, self).__init__("Bhattacharya11",
                                                      cosmo,
                                                      mass_def)

    def _setup(self, cosmo):
        self.a = 0.788
        self.az = 0.01
        self.p = 0.807
        self.q = 1.795
        self.dc = 1.68647

    def _check_mdef(self, mdef):
        if mdef.Delta != 'fof':
            return True
        return False

    def get_bsigma(self, cosmo, sigM, a):
        nu = self.dc / sigM
        a = self.a * a**self.az
        anu2 = a * nu**2
        return 1. + (anu2 - self.q + 2*self.p / (1 + anu2**self.p)) / self.dc


class HBiasFuncTinker10(HBiasFunc):
    """ Implements mass function described in 2010ApJ...724..878T

    Args:
        cosmo (:obj:`Cosmology`): A Cosmology object.
        mass_def (:obj:`HMDef`): a mass definition object.
            this parametrization accepts SO masses with
            200 < Delta < 3200 with respect to the matter density.
    """
    def __init__(self, cosmo, mass_def):
        super(HBiasFuncTinker10, self).__init__("Tinker10",
                                                cosmo,
                                                mass_def)

    def _setup(self, cosmo):
        ld = np.log10(self.mdef.Delta)
        xp = np.exp(-(4./ld)**4.)
        self.A = 1.0 + 0.24 * ld * xp
        self.a = 0.44 * ld - 0.88
        self.B = 0.183
        self.b = 1.5
        self.C = 0.019 + 0.107 * ld + 0.19*xp
        self.c = 2.4
        self.dc = 1.68647

    def _check_mdef(self, mdef):
        if (mdef.Delta < 200.) or (mdef.Delta > 3200.) or \
           (mdef.rho_type != 'matter'):
            return True
        return False

    def get_bsigma(self, cosmo, sigM, a):
        nu = self.dc / sigM
        nupa = nu**self.a

        return 1. - self.A * nupa / (nupa + self.dc**self.a) + \
            self.B * nu**self.b + self.C * nu**self.c
