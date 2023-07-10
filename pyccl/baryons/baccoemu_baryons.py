__all__ = ("BaccoemuBaryons",)

import numpy as np
from copy import deepcopy

from .. import Pk2D
from . import Baryons


class BaccoemuBaryons(Baryons):
    """ The baryonic boost factor computed with the baccoemu baryons emulators.

    See https://arxiv.org/abs/2011.15018 and
    https://bacco.dipc.org/emulator.html

    Note that masses are in units of :math:`M_\\odot`, differently from the
    original paper and baccoemu public code (where they are
    in :math:`M_\\odot/h`)

    Args:
        log10_M_c (:obj:`float`): characteristic halo mass to model baryon
                                   mass fraction (in :math:`M_\\odot`)
        log10_eta (:obj:`float`): extent of ejected gas
        log10_beta (:obj:`float`): slope of power law describing baryon mass
                                    fraction
        log10_M1_z0_cen (:obj:`float`): characteristic halo mass scale for
                                         central galaxies (in :math:`M_\\odot`)
        log10_theta_out (:obj:`float`):  outer slope of density profiles of
                                          hot gas in haloes
        log10_theta_inn (:obj:`float`): inner slope of density profiles of
                                         hot gas in haloes
        log10_M_inn (:obj:`float`): transition mass of density profiles of
                                     hot gas in haloes (in :math:`M_\\odot`)
    """
    name = 'BaccoemuBaryons'
    __repr_attrs__ = __eq_attrs__ = ("bcm_params",)

    def __init__(self, log10_M_c=14.174, log10_eta=-0.3, log10_beta=-0.22,
                 log10_M1_z0_cen=10.674, log10_theta_out=0.25,
                 log10_theta_inn=-0.86, log10_M_inn=13.574):
        # avoid tensorflow warnings
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            import baccoemu
            self.mpk = baccoemu.Matter_powerspectrum()
        self.a_min = self.mpk.emulator['baryon']['bounds'][-1][0]
        self.a_max = self.mpk.emulator['baryon']['bounds'][-1][1]
        self.k_min = self.mpk.emulator['baryon']['k'][0]
        self.k_max = self.mpk.emulator['baryon']['k'][-1]

        self.bcm_params = {
            'M_c': log10_M_c,
            'eta': log10_eta,
            'beta': log10_beta,
            'M1_z0_cen': log10_M1_z0_cen,
            'theta_out': log10_theta_out,
            'theta_inn': log10_theta_inn,
            'M_inn': log10_M_inn
        }

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
        sigma8tot_fid = self.mpk.get_sigma8(cold=False,
                                            A_s=A_s_fid, **_emupars)
        A_s = (sigma8tot / sigma8tot_fid)**2 * A_s_fid
        return self.mpk.get_sigma8(cold=True, A_s=A_s, **_emupars)

    def boost_factor(self, cosmo, k, a):
        """The baccoemu BCM model boost factor for baryons.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
            k (:obj:`float` or `array`): Wavenumber (in :math:`{\\rm Mpc}^{-1}`).
            a (:obj:`float` or `array`): Scale factor.

        Returns:
            :obj:`float` or `array`: Correction factor to apply to
                the power spectrum.
        """ # noqa

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
            # note that ccl parametrises sigma8 of the total matter power
            # spectrum while baccoemu defines it in terms of the cdm+baryons
            # power spectrum; so we have to convert from total to cold sigma8
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

        # change masses from Msun to Msun/h
        _bcm_params = deepcopy(self.bcm_params)
        for key in ['M_c', 'M1_z0_cen', 'M_inn']:
            _bcm_params[key] = np.log10((10**_bcm_params[key])
                                        * emupars['hubble'])

        # baccoemu internally interpolates k with a cubic spline
        # it returns k, boost, so, since we are already requesting a specific
        # k-vector we can ignore the first returned object
        _, fka = self.mpk.get_baryonic_boost(k=k / cosmo['h'],
                                             **{**emupars, **_bcm_params})

        return fka

    def update_parameters(self, log10_M_c=None, log10_eta=None,
                          log10_beta=None, log10_M1_z0_cen=None,
                          log10_theta_out=None, log10_theta_inn=None,
                          log10_M_inn=None):
        """Update BCM parameters. All parameters set to ``None`` will
        be left untouched.
        """
        _kwargs = locals()
        _new_bcm_params = {key: _kwargs[key] for key in
                           set(list(_kwargs.keys())) - set(['self'])}
        new_bcm_params = {}
        for key in _new_bcm_params:
            if _new_bcm_params[key] is not None:
                new_bcm_params[key[6:]] = _new_bcm_params[key]
        self.bcm_params.update(new_bcm_params)

    def _include_baryonic_effects(self, cosmo, pk):
        # Applies boost factor
        a_arr, lk_arr, pk_arr = pk.get_spline_arrays()
        k_arr = np.exp(lk_arr)
        fka = self.boost_factor(cosmo, k_arr, a_arr)
        pk_arr *= fka

        if pk.psp.is_log:
            np.log(pk_arr, out=pk_arr)  # in-place log

        return Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr,
                    is_logp=pk.psp.is_log,
                    extrap_order_lok=pk.extrap_order_lok,
                    extrap_order_hik=pk.extrap_order_hik)
