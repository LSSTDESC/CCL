__all__ = ("CosmicemuMTIVPk",)

import numpy as np
import os
import inspect

from .. import Pk2D
from . import EmulatorPk


class CosmicemuMTIVPk(EmulatorPk):
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
        h = cosmo['h']
        omega_m = cosmo['Omega_m']*h**2
        omega_b = cosmo['Omega_b']*h**2
        sigma8 = cosmo.sigma8()
        wtild = (-cosmo['w0']-cosmo['wa'])**0.25
        omega_nu = cosmo.omega_x(1.0, 'neutrinos_massive')*h**2
        return np.array([omega_m, omega_b, sigma8, h,
                         cosmo['n_s'], cosmo['w0'], wtild, omega_nu])

    def _get_pk_full(self, cosmo):
        xstar = self._cosmo_to_x(cosmo)
        out_of_bounds = (xstar < self.xmin) | (xstar > self.xmax)
        if np.any(out_of_bounds):
            for ip in np.where(out_of_bounds)[0]:
                pname = self.pnames[ip]
                pmin = self.xmin[ip]
                pmax = self.xmax[ip]
                raise ValueError(f'{pname} must be between {pmin} and {pmax}')
        xstarstd = (xstar - self.xmin)/self.xrange
        Sigmastar = np.exp(-np.sum(self.beta[:, None, :]*
                                   ((self.x-xstarstd[None, :])**2)[None, :, :],
                                   axis=-1))/self.lamz[:, None]
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
