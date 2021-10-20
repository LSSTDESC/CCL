from .profiles import HaloProfile, HaloProfileNFW
from .profiles_2pt import Profile2pt
from .concentration import Concentration
import numpy as np
from scipy.integrate import simps
from scipy.optimize import brentq


class HaloProfileCIBShang12(HaloProfile):
    one_over_4pi = 0.07957747154

    def __init__(self, c_M_relation, nu_GHz, beta=1.5, Td=34., gamma=2.,
                 s_z=3.0, log10meff=12.5, sigLM=0.5, Mmin=3E11, L0=4E-20):
        if not isinstance(c_M_relation, Concentration):
            raise TypeError("c_M_relation must be of type `Concentration`)")

        self.nu = nu_GHz
        self.beta = beta
        self.Td = Td
        self.gamma = gamma
        self.s_z = s_z
        self.l10meff = log10meff
        self.sigLM = sigLM
        self.Mmin = Mmin
        self.L0 = L0
        self._set_nu0()
        self.pNFW = HaloProfileNFW(c_M_relation)
        super(HaloProfileCIBShang12, self).__init__()

    def dNsub_dlnM_TinkerWetzel10(self, Msub, Mparent):
        return 0.13*(Msub/Mparent)**(-0.7)*np.exp(-9.9*(Msub/Mparent)**2.5)

    def update_parameters(self, nu_GHz=None,
                          beta=None, Td=None, gamma=None,
                          s_z=None, log10meff=None, sigLM=None,
                          Mmin=None, L0=None):
        newspec = False
        if nu_GHz is not None:
            self.nu = nu_GHz
        if beta is not None:
            self.beta = beta
            newspec = True
        if Td is not None:
            self.Td = Td
            newspec = True
        if gamma is not None:
            self.gamma = gamma
            newspec = True
        if s_z is not None:
            self.s_z = s_z
        if log10meff is not None:
            self.l10meff = log10meff
        if sigLM is not None:
            self.sigLM = sigLM
        if Mmin is not None:
            self.Mmin = Mmin
        if L0 is not None:
            self.L0 = L0

        if newspec:
            self._set_nu0()

    def _mBB(self, nu):
        # h*nu_GHZ / k_B / Td_K
        x = 0.0479924466*nu/self.Td
        ex = np.exp(x)
        return x**(3+self.beta)/(ex-1)

    def _plaw(self, nu):
        return self.mBB0*(nu/self.nu0)**(-self.gamma)

    def spectrum(self, nu):
        return np.piecewise(nu, [nu <= self.nu0],
                            [self._mBB, self._plaw])/self.mBB0

    def _set_nu0(self):
        x0 = self.beta+3+self.gamma

        def f_0(x):
            ex = np.exp(x)
            return x0 - x*ex/(ex-1)
        x_0 = brentq(f_0, x0/2, 2*x0)
        self.nu0 = x_0*self.Td/0.0479924466
        self.mBB0 = self._mBB(self.nu0)

    def _Lum(self, l10M, a):
        # Redshift evolution
        phi_z = a**(-self.s_z)
        # Mass dependence
        # M/sqrt(2*pi*sigLM^2)
        sig_pref = 10**l10M/(2.50662827463*self.sigLM)
        sigma_m = sig_pref * np.exp(-0.5*((l10M - self.l10meff)/self.sigLM)**2)
        return self.L0*phi_z*sigma_m

    def _fcen(self, M, a):
        Lum = self._Lum(np.log10(M), a)
        fcen = np.heaviside(M-self.Mmin, 1)*Lum*self.one_over_4pi
        return fcen

    def _fsat(self, M, a):
        fsat = np.zeros_like(M)
        # Loop over Mparent
        # TODO: if this is too slow we could move it to C
        # and parallelize
        for iM, Mparent in enumerate(M):
            if Mparent > self.Mmin:
                # Array of Msubs (log-spaced with 10 samples per dex)
                nm = max(2, int(np.log10(Mparent/self.Mmin)*10))
                msub = np.geomspace(self.Mmin, Mparent, nm+1)
                # Sample integrand
                dnsubdlnm = self.dNsub_dlnM_TinkerWetzel10(msub, Mparent)
                Lum = self._Lum(np.log10(msub), a)
                integ = dnsubdlnm*Lum
                fsat[iM] = simps(integ, x=np.log(msub))*self.one_over_4pi
        return fsat

    def _real(self, cosmo, r, M, a, mass_def):
        M_use = np.atleast_1d(M)
        r_use = np.atleast_1d(r)

        # (redshifted) Frequency dependence
        spec_nu = self.spectrum(self.nu/a)

        fs = self._fsat(M_use, a)
        ur = self.pNFW._real(cosmo, r_use, M_use, a, mass_def)

        prof = fs[:, None]*ur*spec_nu

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        # (redshifted) Frequency dependence
        spec_nu = self.spectrum(self.nu/a)

        fc = self._fcen(M_use, a)
        fs = self._fsat(M_use, a)
        uk = self.pNFW._fourier(cosmo, k_use, M_use, a, mass_def)

        prof = fc[:, None]+fs[:, None]*uk*spec_nu

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier_variance(self, cosmo, k, M, a, mass_def, nu_other=None):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        spec_nu1 = self.spectrum(self.nu/a)
        if nu_other is None:
            spec_nu2 = self.spectrum(nu_other/a)
        else:
            spec_nu2 = spec_nu1

        fc = self._fcen(M_use, a)
        fs = self._fsat(M_use, a)
        uk = self.pNFW._fourier(cosmo, k_use, M_use, a, mass_def)

        prof = fs[:, None]*uk
        prof = 2*fc[:, None]*prof + prof**2
        prof *= spec_nu1*spec_nu2

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class Profile2ptCIB(Profile2pt):
    def fourier_2pt(self, prof, cosmo, k, M, a,
                    prof2=None, mass_def=None):
        if not isinstance(prof, HaloProfileCIBShang12):
            raise TypeError("prof must be of type `HaloProfileCIB`")

        nu2 = None
        if prof2 is not None:
            if not isinstance(prof, HaloProfileCIBShang12):
                raise TypeError("prof must be of type `HaloProfileCIB`")
            nu2 = prof2.nu
        return prof._fourier_variance(cosmo, k, M, a, mass_def,
                                      nu_other=nu2)
