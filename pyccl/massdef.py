from . import ccllib as lib
from .core import check
from .concentration import *
from .background import species_types, rho_x, omega_x
import numpy as np


def get_new_concentration_py(c_old, Delta_old, Delta_new):
    status = 0
    c_old_use = np.atleast_1d(c_old)
    c_new, status = lib.get_new_concentration_vec(Delta_old, c_old_use,
                                                  Delta_new, c_old_use.size,
                                                  status)
    if np.isscalar(c_old):
        c_new = c_new[0]

    check(status)
    return c_new

class HMDef(object):
    def __init__(self, Delta, rho_type, c_m_relation=None):
        if Delta <= 0:
            raise ValueError("Delta must be a positive number")
        self.Delta = Delta
        if rho_type not in ['matter','critical']:
            raise ValueError("rho_type must be either \'matter\' or \'critical\'")
        self.rho_type = rho_type
        self.species = species_types[rho_type]
        self.concentration = c_m_relation

    def __eq__(self, other):
        return (self.Delta == other.Delta) and (self.rho_type == other.rho_type)

    def get_mass(self, cosmo, R, a):
        # (4 * pi / 3) * rho * Delta * R^3
        return 4.18879020479 * rho_x(cosmo, a, self.rho_type) * self.Delta * R**3

    def get_radius(self, cosmo, M, a): 
        # [ M / (4 * pi * rho * Delta * R^3) ]^(1/3)
        return (M / (4.18879020479 * self.Delta * rho_x(cosmo, a, self.rho_type)))**(1./3.)

    def get_concentration(self, cosmo, M, a):
        if self.concentration is None:
            raise RuntimeError("This mass definition doesn't have an associated"
                               "c(M) relation")
        else:
            return self.concentration(cosmo, M, a)

    def translate_mass(self, cosmo, M, a, m_def_other):
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
                c_new = get_new_concentration_py(c_this, D_this, D_new)
                R_new = c_new * R_this / c_this
                return m_def_other.get_mass(cosmo, R_new, a)


class HMDef200mat(HMDef):
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
