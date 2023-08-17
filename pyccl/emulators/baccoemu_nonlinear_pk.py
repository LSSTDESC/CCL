__all__ = ("BaccoemuNonlinear",)

import numpy as np

from .. import Pk2D
from . import EmulatorPk


class BaccoemuNonlinear(EmulatorPk):
    """ Nonlinear power spectrum emulator from baccoemu

    This is an emulator of the non-linear matter power spectrum as a function
    of 8 cosmological parameters (:math:`\\omega_c`, :math:`\\omega_b`,
    :math:`A_s` or :math:`\\sigma_{8,c}`, :math:`n_s`, :math:`h`,
    :math:`M_\\nu`, :math:`w_0`, :math:`w_a`) and the scale factor.

    If `nonlinear_emu_path` and `nonlinear_emu_details` are not specified, the
    installed public version of the emulator is used. If the user has access to
    a locally saved baccoemu version, the path can be specified with these two
    keyword arguments.

    See `Angulo et al. 2020 <https://arxiv.org/pdf/2004.06245.pdf>`_
    and https://bacco.dipc.org/emulator.html.

    Args:
        nonlinear_emu_path (:obj:`str`): path to a folder containing a
                                          specific baccoemu version
        nonlinear_emu_details (:obj:`str`): name of the details file in the
                                             given baccoemu version
        n_sampling_a (:obj:`int`): number of scale factor values used for
                                    building the 2d pk interpolator
    """
    def __init__(self, nonlinear_emu_path=None, nonlinear_emu_details=None,
                 n_sampling_a=100):
        # avoid tensorflow warnings
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import baccoemu
            self.mpk = baccoemu.Matter_powerspectrum(
                nonlinear_emu_path=nonlinear_emu_path,
                nonlinear_emu_details=nonlinear_emu_details)
        self.a_min = self.mpk.emulator['nonlinear']['bounds'][-1][0]
        self.a_max = self.mpk.emulator['nonlinear']['bounds'][-1][1]
        self.k_min = self.mpk.emulator['nonlinear']['k'][0]
        self.k_max = self.mpk.emulator['nonlinear']['k'][-1]
        self.n_sampling_a = n_sampling_a

    def __str__(self) -> str:
        return """baccoemu nonlinear Pk module,
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
        k_hubble, pk_hubble = self.mpk.get_nonlinear_pk(cold=False, **emupars)
        return k_hubble * h, pk_hubble / h**3

    def _get_pk2d(self, cosmo):
        a = np.linspace(self.a_min, 1, self.n_sampling_a)
        k, pk = self.get_pk_at_a(cosmo, a)
        return Pk2D(a_arr=a, lk_arr=np.log(k), pk_arr=np.log(pk), is_logp=True,
                    extrap_order_lok=1, extrap_order_hik=2)
