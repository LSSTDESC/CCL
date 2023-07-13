__all__ = ("CosmicemuMTIVPk",)

import numpy as np
import os
import inspect
from scipy.interpolate import interp1d

from .. import Pk2D
from . import EmulatorPk


class CosmicemuMTIVPk(EmulatorPk):
    """ Nonlinear power spectrum emulator for CosmicEmu (Mira-Titan IV,
    2022 version).

    This is an emulator of the non-linear matter power spectrum from
    N-body simulations, as a function of 8 cosmological parameters:
    (:math:`\\omega_m`, :math:`\\omega_b`, :math:`\\sigma_8`,
    :math:`n_s`, :math:`h`, :math:`w_0`, :math:`w_a`, and
    :math:`\\omega_\\nu`). The power spectrum is interpolated in this
    parameter space at a fixed grid of redshifts and wavenumbers.

    See `Moran et al. 2022 <https://arxiv.org/abs/2207.12345>`_.
    and https://github.com/lanl/CosmicEmu.

    Args:
        kind (:obj:`str`): type of matter power spectrum to use.
            Options are `'tot'` (for the total matter power spectrum)
            or `'cb'` (for the CDM+baryons power spectrum).
    """
    def __init__(self, kind='tot'):
        data_path = os.path.join(os.path.dirname(
            os.path.abspath(inspect.stack()[0][1])), 'data')
        fname = os.path.join(data_path,
                             f"CosmicEmu_MTIV_2022_P{kind}.npz")
        d = np.load(fname)
        self.x = d['x']
        self.xmin = d['xmin']
        self.xrange = d['xrange']
        self.xmax = d['xmax']
        self.z = d['z']
        self.z_asc = d['z_asc']
        self.ks = d['mode']
        self.K = d['K']
        self.w = d['w']
        self.mean = d['mean']
        self.sd = d['sd']
        self.beta = d['beta']
        self.lamws = d['lamws']
        self.lamz = d['lamz']
        self.nsim, self.npar = self.x.shape
        self.nz = len(self.z)
        self.nk = len(self.ks)
        self.peta = len(self.lamz)

        # Compute KrigBasis
        x2diff = (self.x[:, None, :]-self.x[None, :, :])**2
        logc = np.sum(x2diff[None, :, :, :]*self.beta[:, None, None, :],
                      axis=-1)
        sigma_sim = np.exp(-logc)/self.lamz[:, None, None]
        # Add to diagonal
        for i, lw in enumerate(self.lamws):
            sigma_sim[i, :, :] += np.diag(np.full(self.nsim, 1./lw))
        self.KrigBasis = np.linalg.solve(sigma_sim, self.w)

        # Parameter order
        self.pnames = ['omega_m', 'omega_b', 'sigma8', 'h', 'n_s',
                       'w_0', 'wtild', 'omega_nu']

    def _cosmo_to_x(self, cosmo):
        # Translates cosmology to an array of parameters used
        # by CosmicEmu.
        h = cosmo['h']
        omega_m = cosmo['Omega_m']*h**2
        omega_b = cosmo['Omega_b']*h**2
        sigma8 = cosmo['sigma8']
        if np.isnan(sigma8):
            raise ValueError("sigma8 must be provided to use CosmicEmu")
        wtild = (-cosmo['w0']-cosmo['wa'])**0.25
        omega_nu = cosmo.omega_x(1.0, 'neutrinos_massive')*h**2
        return np.array([omega_m, omega_b, sigma8, h,
                         cosmo['n_s'], cosmo['w0'], wtild, omega_nu])

    def _get_pk_full(self, cosmo):
        # Computes power spectrum at full grid of redshifts and k
        # for this cosmology.
        xstar = self._cosmo_to_x(cosmo)
        # Check for out of bounds
        out_of_bounds = (xstar < self.xmin) | (xstar > self.xmax)
        if np.any(out_of_bounds):
            for ip in np.where(out_of_bounds)[0]:
                pname = self.pnames[ip]
                pmin = self.xmin[ip]
                pmax = self.xmax[ip]
                raise ValueError(f'{pname} must be between {pmin} and {pmax}')
        # Standardize
        xstarstd = (xstar-self.xmin)/self.xrange
        # Compute covariance with new point
        Sigmastar = np.exp(-np.sum(self.beta[:, None, :] *
                                   ((self.x-xstarstd[None, :])**2)[None, :, :],
                                   axis=-1))/self.lamz[:, None]
        # Project and reshape
        wstar = np.sum(Sigmastar*self.KrigBasis, axis=-1)
        ystaremu = (np.dot(self.K, wstar)*self.sd+self.mean).reshape([self.nz,
                                                                      self.nk])
        pk = 10**ystaremu*2*np.pi**2/self.ks**1.5
        return self.z, self.ks, pk

    def _get_pk_at_a(self, cosmo, a):
        z, ks, pk = self._get_pk_full(cosmo)
        pki = interp1d(z, pk.T, kind='cubic')
        pks = pki(1./a-1).T
        return ks, pks

    def _get_pk2d(self, cosmo):
        z, ks, pk = self._get_pk_full(cosmo)
        return Pk2D(a_arr=1./(1+z), lk_arr=np.log(ks), pk_arr=np.log(pk),
                    is_logp=True, extrap_order_lok=1, extrap_order_hik=2)
