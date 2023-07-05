__all__ = ("EuclidEmulator2Nonlinear",)

import numpy as np

from .. import Pk2D
from .. import CCLError
from . import EmulatorPk


class EuclidEmulator2Nonlinear(EmulatorPk):
    """See https://arxiv.org/pdf/2010.11288.pdf
    and https://github.com/miknab/EuclidEmulator2

    Args:
        n_sampling_a (:obj:`int`): number of expansion factor values used for
                                    building the 2d pk interpolator
    """
    def __init__(self, n_sampling_a=100):
        import euclidemu2 as ee2
        self.ee2 = ee2
        self.a_min = 1/(1+3)
        self.a_max = 1
        self.k_min = 0.01
        self.k_max = 10.0
        self.n_sampling_a = n_sampling_a

    def __str__(self) -> str:
        return """EuclidEmulator2 nonlinear Pk module,
k_min,k_max = ({}, {}),
a_min,a_max = ({}, {})""".format(
            self.k_min, self.k_max, self.a_min, self.a_max)

    def _get_pk_at_a(self, a, cosmo):
        if np.isnan(cosmo['A_s']):
            raise CCLError('euclid emulator 2 needs A_s as input')
        redshifts = 1 / np.atleast_1d(a) - 1
        emupars = {'As': cosmo['A_s'],
                   'ns': cosmo['n_s'],
                   'Omb': cosmo['Omega_b'],
                   'Omm': (cosmo['Omega_c'] + cosmo['Omega_b'] +
                           np.sum(cosmo['m_nu']) / 93.14 / cosmo['h']**2),
                   'h': cosmo['h'],
                   'mnu': np.sum(cosmo['m_nu']),
                   'w': cosmo['w0'],
                   'wa': cosmo['wa']}

        h = cosmo['h']
        k_hubble, _pk_hubble, _, _ = self.ee2.get_pnonlin(emupars, redshifts)
        pk_hubble = np.array([_pk_hubble[index] for index in _pk_hubble])
        pk_hubble = np.squeeze(pk_hubble)
        return k_hubble * h, pk_hubble / h**3

    def _get_pk2d(self, cosmo):
        a = np.linspace(self.a_min, 1, self.n_sampling_a)
        k, pk = self._get_pk_at_a(a, cosmo)
        return Pk2D(a_arr=a, lk_arr=np.log(k), pk_arr=np.log(pk), is_logp=True,
                    extrap_order_lok=1, extrap_order_hik=2)
