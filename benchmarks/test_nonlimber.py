import numpy as np
import pyccl as ccl
import time
import os
import pytest


def get_cosmological_parameters(self):
    return {'Omega_m': 0.3156,
            'Omega_b': 0.0492,
            'w0': -1.0,
            'h': 0.6727,
            'A_s': 2.12107E-9,
            'n_s': 0.9645}

def get_tracer_parameters(self):
    # Per-bin galaxy bias
    b_g = np.array([1.376695, 1.451179, 1.528404,
                    1.607983, 1.689579, 1.772899,
                    1.857700, 1.943754, 2.030887,
                    2.118943])
    return {'b_g': b_g}

def get_tracer_kernels(self):
    filename = 'data/nonlimber/kernels_fullwidth.npz'
    d = np.load(filename)
    kernels_cl = d['kernels_cl']
    kernels_sh = d['kernels_sh']
    return {'z_cl': d['z_cl'],
            'chi_cl': d['chi_cl'],
            'kernels_cl': kernels_cl,
            'z_sh': d['z_sh'],
            'chi_sh': d['chi_sh'],
            'kernels_sh': kernels_sh}

@pytest.fixture(scope='module')
def set_up():
    par = self.get_cosmological_parameters()
    self.cosmo = ccl.Cosmology(Omega_c=par['Omega_m']-par['Omega_b'],
                                   Omega_b=par['Omega_b'],
                                   h=par['h'], n_s=par['n_s'],
                                   A_s=par['A_s'], w0=par['w0'])
    tpar = self.get_tracer_parameters()
    ker = self.get_tracer_kernels()

    a_g = 1./(1+ker['z_cl'][::-1])    
    t_g = []
    for k in ker['kernels_cl']:
        t = ccl.Tracer()
        barr = np.ones_like(a_g)
        t.add_tracer(self.cosmo,
                        (ker['chi_cl'], k),
                        transfer_a=(a_g, barr))
        t_g.append(t)
    t_s = []
    for k in ker['kernels_sh']:
        t = ccl.Tracer()
        t.add_tracer(self.cosmo,
                        kernel=(ker['chi_sh'], k),
                        der_bessel=-1, der_angles=2)
        t_s.append(t)
    return cosmo, t_g, t_s
    
@pytest.mark.parametrize("method",['FKEM'])
def test_cells(set_up, method):
    cosmo, t_g, t_s = set_up
    ### stuff below is github pilot prefilled and clearly wrong
    t0 = time.time()
    cls = ccl.angular_cl(cosmo, t_g, t_g, ells, method=method)
    t1 = time.time()
    print(f"Time taken for {method} = {t1-t0}")
    return cls




class NonLimberTest:

    def __init__(self):
        self.nb_g = 10
        self.nb_s = 5
        self.pk = get_pk()
        self.background = get_background()
        self.cosmo = get_cosmological_parameters()

    def get_pk(self):
        return np.load('data/nonlimber/pk.npz')

    def get_background(self):
        return np.load('data/nonlimber/background.npz')



    def get_tracer_dndzs(self):
        filename = 'data/nonlimber/dNdzs_fullwidth.npz'
        dNdz_file = np.load(filename)
        z_sh = dNdz_file['z_sh']
        dNdz_sh = dNdz_file['dNdz_sh']
        z_cl = dNdz_file['z_cl']
        dNdz_cl = dNdz_file['dNdz_cl']
        return {'z_sh': z_sh, 'dNdz_sh': dNdz_sh,
                'z_cl': z_cl, 'dNdz_cl': dNdz_cl}

    def get_noise_biases(self):
        from scipy.integrate import simps

        # Lens sample: 40 gals/arcmin^2
        ndens_c = 40.
        # Source sample: 27 gals/arcmin^2
        ndens_s = 27.
        # Ellipticity scatter per component
        e_rms = 0.28
        
        ndic = self.get_tracer_dndzs()
        nc_ints = np.array([simps(n, x=ndic['z_cl'])
                            for n in ndic['dNdz_cl'].T])
        ns_ints = np.array([simps(n, x=ndic['z_sh'])
                            for n in ndic['dNdz_sh'].T])
        nc_ints *= ndens_c / np.sum(nc_ints)
        ns_ints *= ndens_s / np.sum(ns_ints)
        tosrad = (180*60/np.pi)**2
        nl_cl = 1./(nc_ints*tosrad)
        nl_sh = e_rms**2/(ns_ints*tosrad)
        return nl_cl, nl_sh


    def get_ells(self):
        return np.unique(np.geomspace(2, 2000, 128).astype(int)).astype(float)

    def get_nmodes_fullsky(self):
        """ Returns the number of modes in each ell bin"""
        ls = self.get_ells()
        nmodes = list(ls[1:]**2-ls[:-1]**2)
        lp = ls[-1]**2/ls[-2]
        nmodes.append(lp**2-ls[-1]**2)
        return np.array(nmodes)*0.5

    def get_num_cls(self):
        ngg = (self.nb_g * (self.nb_g + 1)) // 2
        nss = (self.nb_s * (self.nb_s + 1)) // 2
        ngs = self.nb_g * self.nb_s
        return ngg, ngs, nss


