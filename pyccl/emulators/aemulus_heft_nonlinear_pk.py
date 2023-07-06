__all__ = ("AemulusHEFTNonlinear",)

import numpy as np

from .. import Pk2D
from .. import CCLError
from . import EmulatorPk


class AemulusHEFTNonlinear(EmulatorPk):
    """ Nonlinear power spectrum emulator from aemulus heft

    This is an emulator of N-body matter Pk as a function of 7 cosmological
    parameters (Omega_b h^2, Omega_c h^2, w, n_s, 1e9As, H0, Mnu) and
    redshift.

    The emulator is the same for the HEFT model for galaxies and, taking
    only the first argument of its output, for the matter power spectrum.

    In this class we discard anything that is not the matter power spectrum.

    See https://arxiv.org/abs/2303.09762
    and https://github.com/AemulusProject/aemulus_heft

    Args:
        n_sampling_a (:obj:`int`): number of scale factor values used for
                                    building the 2d pk interpolator
    """
    def __init__(self, n_sampling_a=100):
        from aemulus_heft.heft_emu import HEFTEmulator
        from aemulus_heft.utils import lpt_spectra
        self.emu = HEFTEmulator()
        self.lpt_spectra = lpt_spectra
        self.a_min = 1 / (1 + self.emu.zs[0])
        self.a_max = 1 / (1 + self.emu.zs[-1])
        self.k_min = 0.01
        self.k_max = self.emu.k.max()
        self.n_sampling_a = n_sampling_a

    def __str__(self) -> str:
        return """baccoemu nonlinear Pk module,
k_min,k_max = ({}, {}),
a_min,a_max = ({}, {})""".format(
            self.k_min, self.k_max, self.a_min, self.a_max)

    def _get_pk_at_a(self, a, cosmo):
        if np.isnan(cosmo['A_s']):
            raise CCLError('aemulus_heft needs A_s as input')
        # create the array of cosmo params in the format required
        # by aemulus_heft
        emupars = [cosmo['Omega_b'] * cosmo['h']**2,
                   cosmo['Omega_c'] * cosmo['h']**2,
                   cosmo['w0'],
                   cosmo['n_s'],
                   1e9 * cosmo['A_s'],
                   100 * cosmo['h'],
                   np.sum(cosmo['m_nu'])]
        h = cosmo['h']
        a = np.atleast_1d(a)
        extend_low_k = np.logspace(np.log10(self.k_min),
                                   np.log10(self.emu.k[0]),
                                   10)[:-1]
        k_hubble = np.concatenate([extend_low_k, self.emu.k])
        pk = []
        for _a in a:
            z = 1 / _a - 1
            # get the lpt spectra and corresponding sigma8
            # from velocileptor at the exact ks defined in the emulator
            spec_lpt, sigma8z = self.lpt_spectra(k_hubble, z, emupars)
            newemupars = np.concatenate([emupars, [sigma8z]])
            # get the spectra from heft
            spec_heft = self.emu.predict(k_hubble,
                                         np.array(newemupars),
                                         spec_lpt)
            # only consider the matter pk
            pk_hubble = spec_heft[0, :]
            pk.append(pk_hubble / h**3)
        return k_hubble * h, np.squeeze(pk)

    def _get_pk2d(self, cosmo):
        a = np.linspace(self.a_min, 1, self.n_sampling_a)
        k, pk = self.get_pk_at_a(a, cosmo)
        return Pk2D(a_arr=a, lk_arr=np.log(k), pk_arr=np.log(pk), is_logp=True,
                    extrap_order_lok=1, extrap_order_hik=2)
