from .profiles import HaloProfile, HaloProfileNFW
from .profiles_2pt import Profile2pt
from .concentration import Concentration
import numpy as np
from scipy.integrate import simps
from scipy.special import lambertw


class HaloProfileCIBShang12(HaloProfile):
    one_over_4pi = 0.07957747154

    def __init__(self, c_M_relation, nu_GHz, alpha=0.36, T0=24.4, beta=1.75,
                 gamma=1.7, s_z=3.6, log10meff=12.6, sigLM=0.707, Mmin=1E10,
                 L0=6.4E-8):
        if not isinstance(c_M_relation, Concentration):
            raise TypeError("c_M_relation must be of type `Concentration`)")

        self.nu = nu_GHz
        self.alpha = alpha
        self.T0 = T0
        self.beta = beta
        self.gamma = gamma
        self.s_z = s_z
        self.l10meff = log10meff
        self.sigLM = sigLM
        self.Mmin = Mmin
        self.L0 = L0
        self.pNFW = HaloProfileNFW(c_M_relation)
        super(HaloProfileCIBShang12, self).__init__()

    def dNsub_dlnM_TinkerWetzel10(self, Msub, Mparent):
        return 0.30*(Msub/Mparent)**(-0.7)*np.exp(-9.9*(Msub/Mparent)**2.5)

    def update_parameters(self, nu_GHz=None,
                          alpha=None, beta=None, T0=None, gamma=None,
                          s_z=None, log10meff=None, sigLM=None,
                          Mmin=None, L0=None):
        if nu_GHz is not None:
            self.nu = nu_GHz
        if alpha is not None:
            self.alpha = alpha
        if T0 is not None:
            self.T0 = T0
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
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

    def spectrum(self, nu, a):
        # h*nu_GHZ / k_B / Td_K
        h_GHz_o_kB_K = 0.0479924466
        Td = self.T0/a**self.alpha
        x = h_GHz_o_kB_K * nu / Td

        # Find nu_0
        q = self.beta+3+self.gamma
        x0 = q+np.real(lambertw(-q*np.exp(-q), k=0))

        def mBB(x):
            ex = np.exp(x)
            return x**(3+self.beta)/(ex-1)

        mBB0 = mBB(x0)

        def plaw(x):
            return mBB0*(x0/x)**self.gamma

        return np.piecewise(x, [x <= x0],
                            [mBB, plaw])/mBB0

    def _Lum(self, l10M, a):
        # Redshift evolution
        phi_z = a**(-self.s_z)
        # Mass dependence
        # M/sqrt(2*pi*sigLM^2)
        sig_pref = 10**l10M/(2.50662827463*self.sigLM)
        sigma_m = sig_pref * np.exp(-0.5*((l10M - self.l10meff)/self.sigLM)**2)
        return self.L0*phi_z*sigma_m

    def _Lumcen(self, M, a):
        Lum = self._Lum(np.log10(M), a)
        Lumcen = np.heaviside(M-self.Mmin, 1)*Lum
        return Lumcen

    def _Lumsat(self, M, a):
        Lumsat = np.zeros_like(M)
        # Loop over Mparent
        # TODO: if this is too slow we could move it to C
        # and parallelize
        for iM, Mparent in enumerate(M):
            if Mparent > self.Mmin:
                # Array of Msubs (log-spaced with 10 samples per dex)
                nm = max(2, int(np.log10(Mparent/1E10)*10))
                msub = np.geomspace(1E10, Mparent, nm+1)
                # Sample integrand
                dnsubdlnm = self.dNsub_dlnM_TinkerWetzel10(msub, Mparent)
                Lum = self._Lum(np.log10(msub), a)
                integ = dnsubdlnm*Lum
                Lumsat[iM] = simps(integ, x=np.log(msub))
        return Lumsat

    def _real(self, cosmo, r, M, a, mass_def):
        M_use = np.atleast_1d(M)
        r_use = np.atleast_1d(r)

        # (redshifted) Frequency dependence
        spec_nu = self.spectrum(self.nu/a, a)

        Ls = self._Lumsat(M_use, a)
        ur = self.pNFW._real(cosmo, r_use, M_use, a, mass_def)

        prof = Ls[:, None]*ur*spec_nu*self.one_over_4pi

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier(self, cosmo, k, M, a, mass_def):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        # (redshifted) Frequency dependence
        spec_nu = self.spectrum(self.nu/a, a)

        Lc = self._Lumcen(M_use, a)
        Ls = self._Lumsat(M_use, a)
        uk = self.pNFW._fourier(cosmo, k_use, M_use,
                                a, mass_def)/M_use[:, None]

        prof = (Lc[:, None]+Ls[:, None]*uk)*spec_nu*self.one_over_4pi

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _fourier_variance(self, cosmo, k, M, a, mass_def, nu_other=None):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        spec_nu1 = self.spectrum(self.nu/a, a)
        if nu_other is None:
            spec_nu2 = spec_nu1
        else:
            spec_nu2 = self.spectrum(nu_other/a, a)

        Lc = self._Lumcen(M_use, a)
        Ls = self._Lumsat(M_use, a)
        uk = self.pNFW._fourier(cosmo, k_use, M_use,
                                a, mass_def)/M_use[:, None]

        prof = Ls[:, None]*uk
        prof = 2*Lc[:, None]*prof + prof**2
        prof *= spec_nu1*spec_nu2*self.one_over_4pi**2

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
