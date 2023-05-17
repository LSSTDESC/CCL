from .pspec_base import PowerSpectrum
from ..pk2d import Pk2D
from ..errors import CCLError
import numpy as np
try:
    import isitgr  # noqa: F401
except ModuleNotFoundError:
    pass  # prevent nans from isitgr


__all__ = ("PowerSpectrumOnCAMB", "PowerSpectrumCAMB", "PowerSpectrumISITGR",)


class PowerSpectrumOnCAMB(PowerSpectrum):
    """Base for Boltzmann solvers based on CAMB (e.g. CAMB, ISiTGR)."""
    rescale_s8 = True

    def __init__(self, *, package, kmax, lmax, dark_energy_model, nonlin,
                 **precision):
        self.package = package
        self.kmax = kmax
        self.lmax = lmax
        self.dark_energy_model = dark_energy_model
        self.nonlin = nonlin
        self.precision = precision

    def _setup(self, cosmo):
        """Set up the parameters and initialize the power spectrum."""
        a_arr, lk_arr = self._get_spline_arrays(cosmo)
        cp = self.package.model.CAMBparams()
        for param, value in self.precision.items():
            setattr(cp.Accuracy, param, value)

        # Configuration.
        cp.WantCls = False
        cp.DoLensing = False
        cp.Want_CMB = False
        cp.Want_CMB_lensing = False
        cp.Want_cl_2D_array = False
        cp.WantTransfer = True

        # Background parameters.
        h2 = cosmo['h']**2
        cp.H0 = cosmo['h'] * 100
        cp.ombh2 = cosmo['Omega_b'] * h2
        cp.omch2 = cosmo['Omega_c'] * h2
        cp.omk = cosmo['Omega_k']
        cp.TCMB = cosmo['T_CMB']

        # Neutrinos.
        # We maually setup the CAMB neutrinos to match
        # the adjustments CLASS makes to their temperatures.
        cp.share_delta_neff = False
        cp.omnuh2 = cosmo['Omega_nu_mass'] * h2
        cp.num_nu_massless = cosmo['N_nu_rel']
        cp.num_nu_massive = int(cosmo['N_nu_mass'])
        cp.nu_mass_eigenstates = int(cosmo['N_nu_mass'])
        # hard-code the Neff value CAMB is calibrated with
        # TODO: This has now changed
        delta_neff = cosmo['Neff'] - 3.046  # used for BBN YHe

        # * CAMB defines a neutrino degeneracy factor as `T_i = g^(1/4)*T_nu`
        #   where `T_nu` is the standard neutrino temperature from first order
        #   computations.
        # * CLASS defines the temperature of each neutrino species to be
        #   `T_i_eff = T_ncdm * T_cmb` where `T_ncdm` is a fudge factor to get
        #   the total mass in terms of eV to match second-order computations
        #   of the relationship between m_nu and Omega_nu.
        # * Trying to get the codes to use the same neutrino temperature,
        #   we set `T_i_eff = T_i = g^(1/4) * T_nu` and solve for the right
        #   value of g for CAMB. We get `g = (T_ncdm / (11/4)^(-1/3))^4`.
        g = (cosmo["T_ncdm"] / (11/4)**(-1/3))**4

        if cosmo['N_nu_mass'] > 0:
            nu_mass_fracs = [m / cosmo["sum_nu_masses"] for m in cosmo["m_nu"]]

            cp.nu_mass_numbers = np.ones(cosmo['N_nu_mass'], dtype=np.int)
            cp.nu_mass_fractions = nu_mass_fracs
            cp.nu_mass_degeneracies = np.ones(cosmo['N_nu_mass']) * g

        # YHe from BBN.
        cp.bbn_predictor = self.package.bbn.get_predictor()
        Tr = self.package.constants.COBE_CMBTemp / cp.TCMB
        cp.YHe = cp.bbn_predictor.Y_He(cp.ombh2 * (Tr)**3, delta_neff)

        # Dark energy.
        camb_de_models = ['DarkEnergyPPF', 'ppf', 'DarkEnergyFluid', 'fluid']
        if self.dark_energy_model not in camb_de_models:
            raise ValueError("CCL only supports fluid and ppf dark energy with"
                             f"{self.package.__name__}.")
        cp.set_classes(dark_energy_model=self.dark_energy_model)

        is_ppf = "ppf" in self.dark_energy_model.lower()
        w0, wa, eps = cosmo["w0"], cosmo["wa"], 1e-6
        if not is_ppf and wa != 0 and (w0 < -(1+eps) or 1+w0+wa < -eps):
            raise ValueError("For w < -1, use the 'ppf' dark energy model.")
        cp.DarkEnergy.set_params(w=cosmo['w0'], wa=cosmo['wa'])

        # Initialize power spectrum.
        zs = 1.0 / a_arr - 1
        cp.set_matter_power(redshifts=zs.tolist(), kmax=self.kmax,
                            nonlinear=self.nonlin, silent=True)

        # Power spectrum normalization.
        # If A_s is not given, we just get close and CCL will normalize it.
        A_s, sigma8 = cosmo["A_s"], cosmo["sigma8"]
        A_s_fid = A_s if np.isfinite(A_s) else self._sigma8_to_As(sigma8)

        cp.set_for_lmax(int(self.lmax))
        cp.InitPower.set_params(As=A_s_fid, ns=cosmo['n_s'])
        return cp

    def _get_camb_power(self, res, cosmo, nonlin):
        k, z, pk = res.get_linear_matter_power_spectrum(
            hubble_units=False, k_hunit=False, nonlinear=nonlin)
        np.log(k, out=k)
        np.log(pk, out=pk)
        # reverse the time axis because CAMB uses z
        return Pk2D(a_arr=(1/(1+z))[::-1], lk_arr=k, pk_arr=pk[::-1],
                    is_logp=True, extrap_order_lok=1, extrap_order_hik=2)


class PowerSpectrumCAMB(PowerSpectrumOnCAMB):
    """
    """
    name = "boltzmann_camb"
    rescale_mg = True

    def __init__(self, kmax=10, lmax=5000, dark_energy_model="fluid",
                 halofit_version="mead2020_feedback", HMCode_A_baryon=3.13,
                 HMCode_eta_baryon=0.603, HMCode_logT_AGN=7.8, nonlin=False,
                 **precision):
        import camb
        self.halofit_version = halofit_version
        self.HMCode_A_baryon = HMCode_A_baryon
        self.HMCode_eta_baryon = HMCode_eta_baryon
        self.HMCode_logT_AGN = HMCode_logT_AGN
        super().__init__(
            package=camb, kmax=kmax, lmax=lmax,
            dark_energy_model=dark_energy_model,
            nonlin=nonlin, **precision)

    def _get_power_spectrum(self, cosmo):
        import camb
        cp = self._setup(cosmo)

        if self.nonlin:
            if not np.isfinite(cosmo["A_s"]):
                raise CCLError("CAMB does not rescale the non-linear "
                               "spectrum consistently with sigma8.")

            # Set non-linear model parameters.
            cp.NonLinearModel = camb.nonlinear.Halofit()
            cp.NonLinearModel.set_params(
                halofit_version=self.halofit_version,
                HMCode_A_baryon=self.HMCode_A_baryon,
                HMCode_eta_baryon=self.HMCode_eta_baryon,
                HMCode_logT_AGN=self.HMCode_logT_AGN)

        res = camb.get_results(cp)
        pkl = self._get_camb_power(res, cosmo=cosmo, nonlin=False)
        if self.nonlin:
            if not np.isfinite(cosmo["A_s"]):
                raise CCLError("CAMB doesn't rescale non-linear power spectra "
                               "consistently without A_s.")
            pknl = self._get_camb_power(res, cosmo=cosmo, nonlin=True)
            return pkl, pknl
        return pkl


class PowerSpectrumISITGR(PowerSpectrumOnCAMB):
    """
    """
    name = "boltzmann_isitgr"
    rescale_mg = False

    def __init__(self, kmax=10, lmax=5000, dark_energy_model="fluid",
                 **precision):
        import isitgr  # noqa: F811
        super().__init__(
            package=isitgr, kmax=kmax, lmax=lmax,
            dark_energy_model=dark_energy_model,
            nonlin=False, **precision)

    def _get_power_spectrum(self, cosmo):
        cp = self._setup(cosmo=cosmo)

        cp.GR = 1  # modified GR
        cp.ISiTGR_muSigma = True
        cp.mu0 = cosmo['mu_0']
        cp.Sigma0 = cosmo['sigma_0']
        cp.c1 = cosmo['c1_mg']
        cp.c2 = cosmo['c2_mg']
        cp.Lambda = cosmo['lambda_mg']

        res = isitgr.get_results(cp)
        return self._get_camb_power(res, cosmo=cosmo, nonlin=False)
