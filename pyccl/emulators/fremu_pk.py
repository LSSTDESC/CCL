__all__ = ("FREmu",)

import numpy as np
from .. import Pk2D
from . import EmulatorPk

class FREmu(EmulatorPk):
    """ Nonlinear power spectrum emulator from fremu with MGCAMB support for k < 0.1.
    """
    def __init__(self, n_sampling_a=100):
        # avoid tensorflow warnings
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            from fremu import fremu
            self.mpk = fremu.emulator()
        self.a_min = 0.25
        self.a_max = 1
        self.k_min = 1e-4
        self.k_max = 10
        self.n_sampling_a = n_sampling_a

    def __str__(self) -> str:
        return """fremu Pk module,
k_min,k_max = ({}, {}),
a_min,a_max = ({}, {})""".format(
            self.k_min, self.k_max, self.a_min, self.a_max)

    def _get_pk_at_a(self, cosmo, a):
        # set cosmo for fremu
        h = cosmo['h']
        mnu = np.sum(cosmo['m_nu'])
        Onu = mnu / 93.14 / h**2
        Om = cosmo['Omega_c'] + cosmo['Omega_b'] + Onu
        Ob = cosmo['Omega_b']
        ns = cosmo['n_s']
        sigma8 = cosmo['sigma8']
        fR0 = cosmo['extra_parameters']['fR0']

        self.mpk.set_cosmo(Om=Om, Ob=Ob, h=h, ns=ns, sigma8=sigma8, mnu=mnu, fR0=fR0)

        k_hubble = np.logspace(-4, 1, 500)
        pk_hubble = []
        
        # Calculate pk for each a
        for a_ in a:
            pk_hubble_ = self.mpk.get_power_spectrum(k=k_hubble, z=1/a_-1)
            pk_hubble.append(pk_hubble_)

        pk_hubble = np.array(pk_hubble)

        # For k < 0.1, replace fremu values with MGCAMB values
        k_mg = np.logspace(-4, -1, 100)  # k-range for MGCAMB
        pk_mgcamb = self._get_mgcamb_pk(cosmo, a, k_mg)

        # Combine the results
        for i, a_ in enumerate(a):
            # Interpolating MGCAMB results onto the same k grid
            pk_interp_mg = np.interp(k_hubble[k_hubble < 0.1], k_mg, pk_mgcamb[i])
            pk_hubble[i][k_hubble < 0.1] = pk_interp_mg

        return k_hubble * h, pk_hubble / h**3

    def _get_pk2d(self, cosmo):
        a = np.linspace(self.a_min, 1, self.n_sampling_a)
        k, pk = self._get_pk_at_a(cosmo, a)
        return Pk2D(a_arr=a, lk_arr=np.log(k), pk_arr=np.log(pk), is_logp=True,
                    extrap_order_lok=1, extrap_order_hik=2)

    def _get_mgcamb_pk(self, cosmo, a, k):
        import camb

        # Set cosmological parameters for MGCAMB
        h = cosmo['h']
        mnu = np.sum(cosmo['m_nu'])
        Onu = mnu / 93.14 / h**2
        Om = cosmo['Omega_c'] + cosmo['Omega_b'] + Onu
        Ob = cosmo['Omega_b']
        ns = cosmo['n_s']
        sigma8 = cosmo['sigma8']
        fR0 = cosmo['extra_parameters']['fR0']

        pk_mgcamb = []

        for a_ in a:
            z = 1/a_ - 1  # Redshift for each scale factor
            pars = camb.CAMBparams()
            pars.set_mgparams(MG_flag=3, QSA_flag=4, F_R0=-fR0)
            pars.set_cosmology(H0=h * 100, ombh2=Ob * h**2, omch2=(Om - Ob - Onu) * h**2, mnu=mnu, omk=0, num_massive_neutrinos=3)
            pars.InitPower.set_params(ns=ns)

            # Initialize scalar amplitude
            initial_scalar_amp = 1e-9
            pars.InitPower.set_params(As=initial_scalar_amp)
            pars.set_matter_power(redshifts=[z], kmax=0.1, nonlinear=False)

            # Get results to compute sigma8 adjustment
            results = camb.get_results(pars)
            sigma8_computed = results.get_sigma8_0()
            ratio = sigma8 / sigma8_computed
            new_scalar_amp = initial_scalar_amp * ratio ** 2

            # Update scalar amplitude and get final power spectrum
            pars.InitPower.set_params(As=new_scalar_amp)
            pars.set_matter_power(redshifts=[z], kmax=0.1, nonlinear=False)
            results = camb.get_results(pars)

            # Get matter power interpolator and evaluate at the given k values
            pk_interpolator = results.get_matter_power_interpolator(nonlinear=False)
            pk_mgcamb.append([pk_interpolator.P(z, k_val) for k_val in k])

        return np.array(pk_mgcamb)
