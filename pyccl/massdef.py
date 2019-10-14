from . import ccllib as lib
from .core import check
from .concentration import *
from .background import species_types, rho_x, omega_x
import numpy as np


def massfunc_m2r(cosmo, M):
    return (M / (4.18879020479 * rho_x(cosmo, a, self.rho_type)))**(1./3.)


def get_new_concentration_py(cosmo, c_old, Delta_old, Delta_new):
    """ Computes the concentration parameter for a different mass definition.
    This is done assuming an NFW profile. The output concentration `c_new` is
    found by solving the equation:
        f(c_old) * D_old = f(c_new) * D_new
    where f(x) = x^3/[ln(1+x) - x/(1+x)].

    Args:
        cosmo (:obj:`Cosmology`): A Cosmology object.
        c_old (float or array_like): concentration to translate from.
        Delta_old (float): Delta parameter associated to the input concentration.
            See description of the HMDef class.
        Delta_new (float): Delta parameter associated to the output concentration.

    Returns:
        float or array_like: concentration parameter for the new mass definition.
    """
    status = 0
    c_old_use = np.atleast_1d(c_old)
    c_new, status = lib.get_new_concentration_vec(cosmo.cosmo,
                                                  Delta_old, c_old_use,
                                                  Delta_new, c_old_use.size,
                                                  status)
    if np.isscalar(c_old):
        c_new = c_new[0]

    check(status)
    return c_new

class HMDef(object):
    """Halo mass definition. Halo masses are defined in terms of an overdensity
    parameter Delta and an associated density X (either the matter density or
    the critical density):
        M = 4 * pi * Delta * rho_X * R^3 / 3
    where R is the halo radius. This object also holds methods to translate
    between R and M, and to translate masses between different definitions
    if a concentration-mass relation is provided.

    Args:
        Delta (float): overdensity parameter.
        rho_type (string): either 'critical' or 'matter'.
        c_m_relation (function, optional): concentration-mass relation.
            Provided as a function with signature c(cosmo, M, a), where
            'cosmo' is a Cosmology object, 'M' is halo mass and 'a' is
            scale factor. If `None`, no c(M) relation will be attached to
            this mass definition (and hence one can't translate into other
            definitions).
    """
    def __init__(self, Delta, rho_type, c_m_relation=None):
        # Check it makes sense
        if Delta <= 0:
            raise ValueError("Delta must be a positive number")
        self.Delta = Delta
        # Can only be matter or critical
        if rho_type not in ['matter','critical']:
            raise ValueError("rho_type must be either \'matter\' or \'critical\'")
        self.rho_type = rho_type
        self.species = species_types[rho_type]
        # c(M) relation
        self.concentration = c_m_relation

    def __eq__(self, other):
        return (self.Delta == other.Delta) and (self.rho_type == other.rho_type)

    def get_mass(self, cosmo, R, a):
        """ Translates a halo radius into a mass
            M =  (4 * pi / 3) * rho_X * Delta * R^3

        Args:
            cosmo (:obj:`Cosmology`): A Cosmology object.
            R (float or array_like): halo radius in units of Mpc (physical, not
                comoving).
            a (float): scale factor.

        Returns:
            float or array_like: halo mass in units of M_sun.
        """
        return 4.18879020479 * rho_x(cosmo, a, self.rho_type) * self.Delta * R**3

    def get_radius(self, cosmo, M, a): 
        """ Translates a halo mass into a radius
            M =  (4 * pi / 3) * rho_X * Delta * R^3

        Args:
            cosmo (:obj:`Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.

        Returns:
            float or array_like: halo radius in units of Mpc (physical, not
                comoving).
        """
        return (M / (4.18879020479 * self.Delta * rho_x(cosmo, a, self.rho_type)))**(1./3.)

    def get_concentration(self, cosmo, M, a):
        """ Returns concentration for this mass definition.

        Args:
            cosmo (:obj:`Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.

        Returns:
            float or array_like: halo concentration.
        """
        if self.concentration is None:
            raise RuntimeError("This mass definition doesn't have an associated"
                               "c(M) relation")
        else:
            return self.concentration(cosmo, M, a)

    def translate_mass(self, cosmo, M, a, m_def_other):
        """ Translate halo mass in this definition into another definition

        Args:
            cosmo (:obj:`Cosmology`): A Cosmology object.
            M (float or array_like): halo mass in units of M_sun.
            a (float): scale factor.
            m_def_other (:obj:`HMDef`): another mass definition.

        Returns:
            float or array_like: halo masses in new definition.
        """
        if self == m_def_other:
            return M
        else:
            if self.concentration is None:
                raise RuntimeError("This mass definition doesn't have an associated"
                                   "c(M) relation")
            else:
                D_this = self.Delta * omega_x(cosmo, a, self.rho_type)
                c_this = self.get_concentration(cosmo, M, a)
                R_this = self.get_radius(cosmo, M, a)
                D_new = m_def_other.Delta * omega_x(cosmo, a, m_def_other.rho_type)
                c_new = get_new_concentration_py(cosmo, c_this, D_this, D_new)
                R_new = c_new * R_this / c_this
                return m_def_other.get_mass(cosmo, R_new, a)


class HMDef200mat(HMDef):
    """`HMDef` class for the mass definition with Delta=200 times the matter
    density. Available concentration-mass relations (values for `c_m`):
      * 'Duffy08': concentration-mass relation in arXiv:0804.2486.
      * 'Bhattacharya11': c(M) relation in arXiv:1112.5479.

    Args:
        c_m (string): concentration-mass relation.
    """
    def __init__(self, c_m='Duffy08'):
        if c_m == 'Duffy08':
            c_m_f = concentration_duffy08_200mat
        elif c_m == 'Bhattacharya11':
            c_m_f = concentration_bhattacharya11_200mat
        else:
            raise NotImplementedError("Unknwon c(M) relation " + c_m)

        super(HMDef200mat, self).__init__(200,
                                          'matter',
                                          c_m_relation = c_m_f)


class HMDef200crit(HMDef):
    """`HMDef` class for the mass definition with Delta=200 times the critical
    density. Available concentration-mass relations (values for `c_m`):
      * 'Duffy08': concentration-mass relation in arXiv:0804.2486.
      * 'Bhattacharya11': c(M) relation in arXiv:1112.5479.

    Args:
        c_m (string): concentration-mass relation.
    """
    def __init__(self, c_m='Duffy08'):
        if c_m == 'Duffy08':
            c_m_f = concentration_duffy08_200crit
        elif c_m == 'Bhattacharya11':
            c_m_f = concentration_bhattacharya11_200crit
        else:
            raise NotImplementedError("Unknwon c(M) relation " + c_m)

        super(HMDef200crit, self).__init__(200,
                                           'critical',
                                           c_m_relation = c_m_f)
