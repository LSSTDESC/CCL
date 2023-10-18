__all__ = ("BaccoemuLinear",)

import numpy as np

from .. import Pk2D
from . import EmulatorPk


class BaccoemuLinear(EmulatorPk):
    """ Linear power spectrum emulator from baccoemu

    This is an emulator of the linear matter power spectrum as a function
    of 8 cosmological parameters (:math:`\\omega_c`, :math:`\\omega_b`,
    :math:`A_s` or :math:`\\sigma_{8,c}`, :math:`n_s`, :math:`h`,
    :math:`M_\\nu`, :math:`w_0`, :math:`w_a`) and the scale factor.

    See `Arico et al. 2021 <https://arxiv.org/abs/2104.14568>`_
    and https://bacco.dipc.org/emulator.html
    """
    def __init__(self):
        # avoid tensorflow warnings
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import baccoemu
            self.mpk = baccoemu.Matter_powerspectrum()
        self.a_min = self.mpk.emulator['linear']['bounds'][-1][0]
        self.a_max = self.mpk.emulator['linear']['bounds'][-1][1]
        self.k_min = self.mpk.emulator['linear']['k'][0]
        self.k_max = self.mpk.emulator['linear']['k'][-1]

    def __str__(self) -> str:
        return """baccoemu linear Pk module,
k_min,k_max = ({}, {}),
a_min,a_max = ({}, {})""".format(
            self.k_min, self.k_max, self.a_min, self.a_max)

    def _sigma8tot_2_sigma8cold(self, emupars, sigma8tot):
        """Use baccoemu to convert sigma8 total matter to sigma8 cdm+baryons
        """
        if hasattr(emupars['omega_cold'], '__len__'):
            _emupars = {}
            for pname in emupars:
                _emupars[pname] = emupars[pname][0]
        else:
            _emupars = emupars
        A_s_fid = 2.1e-9
        sigma8tot_fid = self.mpk.get_sigma8(cold=False, A_s=A_s_fid,
                                            **_emupars)
        A_s = (sigma8tot / sigma8tot_fid)**2 * A_s_fid
        return self.mpk.get_sigma8(cold=True, A_s=A_s, **_emupars)

    def _get_pk_at_a(self, cosmo, a):
        # First create the dictionary passed to baccoemu
        # if a is an array, make sure all the other parameters passed to the
        # emulator have the same len
        if hasattr(a, '__len__'):
            emupars = dict(
                omega_cold=np.full((len(a)),
                                   cosmo['Omega_c'] + cosmo['Omega_b']),
                omega_baryon=np.full((len(a)), cosmo['Omega_b']),
                ns=np.full((len(a)), cosmo['n_s']),
                hubble=np.full((len(a)), cosmo['h']),
                neutrino_mass=np.full((len(a)), np.sum(cosmo['m_nu'])),
                w0=np.full((len(a)), cosmo['w0']),
                wa=np.full((len(a)), cosmo['wa']),
                expfactor=a
            )
        else:
            emupars = dict(
                omega_cold=cosmo['Omega_c'] + cosmo['Omega_b'],
                omega_baryon=cosmo['Omega_b'],
                ns=cosmo['n_s'],
                hubble=cosmo['h'],
                neutrino_mass=np.sum(cosmo['m_nu']),
                w0=cosmo['w0'],
                wa=cosmo['wa'],
                expfactor=a
            )

        # if cosmo contains sigma8, we use it for baccoemu, otherwise we pass
        # A_s to the emulator
        if np.isnan(cosmo['A_s']):
            sigma8tot = cosmo['sigma8']
            sigma8cold = self._sigma8tot_2_sigma8cold(emupars, sigma8tot)
            if hasattr(a, '__len__'):
                emupars['sigma8_cold'] = np.full((len(a)), sigma8cold)
            else:
                emupars['sigma8_cold'] = sigma8cold
        else:
            if hasattr(a, '__len__'):
                emupars['A_s'] = np.full((len(a)), cosmo['A_s'])
            else:
                emupars['A_s'] = cosmo['A_s']

        h = cosmo['h']
        k_hubble, pk_hubble = self.mpk.get_linear_pk(cold=False, **emupars)
        return k_hubble * h, pk_hubble / h**3

    def _get_pk2d(self, cosmo):
        a = cosmo.get_pk_spline_a()
        a_for_baccoemu = a[a >= self.a_min]
        a_extrapolated = a[a < self.a_min]
        # we directly use the emulator for the expansion factors within its
        # range
        k, pk = self.get_pk_at_a(cosmo, a_for_baccoemu)
        # for the expansion factors requested by ccl but outside the emulator
        # range, we extrapolate from the earliest pk available with linear
        # growth factors.
        # NOTE: ccl computes scale independent growth factors, this is not
        # correct with massive neutrinos
        growth_factors = cosmo.growth_factor(a_extrapolated)
        ref_growth_factor = cosmo.growth_factor(a_for_baccoemu[0])
        pk_extrapolated = (np.repeat([pk[0]], [len(a_extrapolated)], axis=0).T
                           * (growth_factors / ref_growth_factor)**2).T
        # now we combine the extrapolated and direct spectra
        pk_final = np.concatenate([pk_extrapolated, pk])
        return Pk2D(a_arr=a, lk_arr=np.log(k), pk_arr=np.log(pk_final),
                    is_logp=True, extrap_order_lok=1, extrap_order_hik=2)
