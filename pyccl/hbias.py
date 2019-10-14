from . import ccllib as lib
from .core import check
from .background import species_types, rho_x, omega_x
import numpy as np

class HBiasFunc(object):
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
        pass

    def _check_mdef(self, mdef):
        return False

    def _get_consistent_mass(self, cosmo, M, a, mdef_other):
        if mdef_other is not None:
            M_use = mdef_other.translate_mass(cosmo, M, a, self.mdef)
        else:
            M_use = M
        return np.log10(M_use)

    def get_halo_bias(self, cosmo, M, a, mdef_other=None):
        M_use = np.atleast_1d(M)
        logM = self._get_consistent_mass(cosmo, M_use,
                                         a, mdef_other)

        # sigma(M)
        status = 0
        sigM, status = lib.sigM_vec(cosmo.cosmo, a, logM,
                                    len(logM), status)
        check(status);

        b = self.get_bsigma(cosmo, sigM, a)
        if np.isscalar(M):
            b = b[0]
        return b

    def get_bsigma(self, cosmo, sigM, a):
        raise NotImplementedError("Use one of the non-default HBiasFunc classes")


class HBiasFuncShethTormen(HBiasFunc):
    def __init__(self, cosmo, mass_def):
        super(HBiasFuncShethTormen, self).__init__("S&T",
                                                   cosmo,
                                                   mass_def)

    def _setup(self, cosmo):
        self.p = 0.3;
        self.a = 0.707;

    def get_bsigma(self, cosmo, sigM, a):
        status = 0
        delta_c, status = lib.deltac_NakamuraSuto(cosmo.cosmo, a, status)
        check(status);

        nu = delta_c/sigM
        return 1. + (self.a * nu**2 - 1. + 2. * self.p / (1. + (self.a * nu**2)**self.p))/delta_c;


class HBiasFuncTinker10(HBiasFunc):
    def __init__(self, cosmo, mass_def):
        super(HBiasFuncTinker10, self).__init__("Tinker10",
                                                cosmo,
                                                mass_def)
 
    def _setup(self, cosmo):
        ld = np.log10(self.mdef.Delta)
        xp = np.exp(-(4./ld)**4.);
        self.A = 1.0 + 0.24 * ld * xp
        self.a = 0.44 * ld - 0.88
        self.B = 0.183
        self.b = 1.5
        self.C = 0.019 + 0.107 * ld + 0.19*xp
        self.c = 2.4

    def _check_mdef(self, mdef):
        if (mdef.Delta<200.) or (mdef.Delta>3200.) or (mdef.rho_type!='matter'):
            return True
        return False

    def get_bsigma(self, cosmo, sigM, a):
        delta_c = 1.686
        nu = delta_c / sigM
        nupa=nu**self.a
        
        return 1. - self.A * nupa / (nupa + delta_c**self.a) + \
            self.B * nu**self.b + self.C * nu**self.c
