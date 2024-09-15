__all__ = ("FREmu",)

import numpy as np

from .. import Pk2D
from . import EmulatorPk


class FREmu(EmulatorPk):
    """ Nonlinear power spectrum emulator from fremu
    
    """
    def __init__(self,n_sampling_a=100):
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
        Onu = mnu/ 93.14 / h**2
        self.mpk.set_cosmo(Om=cosmo['Omega_c'] + cosmo['Omega_b'] + Onu,
                           Ob=cosmo['Omega_b'],
                           h=cosmo['h'],
                           ns=cosmo['n_s'],
                           sigma8=cosmo['sigma8'],
                           mnu=mnu,
                           fR0=cosmo['extra_parameters']['fR0'])
        pk_hubble = []
        k_hubble = np.logspace(-4,1,500)
        for a_ in a:
            pk_hubble_ = self.mpk.get_power_spectrum(k=k_hubble, z=1/a_-1)
            pk_hubble.append(pk_hubble_)
        pk_hubble = np.array(pk_hubble)           
        return k_hubble * h, pk_hubble / h**3

    def _get_pk2d(self, cosmo):
        a = np.linspace(self.a_min, 1, self.n_sampling_a)
        k, pk = self.get_pk_at_a(cosmo, a)
        return Pk2D(a_arr=a, lk_arr=np.log(k), pk_arr=np.log(pk), is_logp=True,
                    extrap_order_lok=1, extrap_order_hik=2)
