import numpy as np
import pyccl as ccl
# Just adding n5k
# Stole from https://gist.github.com/JungeAlexander/6ce0a5213f3af56d7369
import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import n5k


cal = n5k.N5KCalculatorBase('tests/config.yml')
par = cal.get_cosmological_parameters()
cosmo = ccl.Cosmology(Omega_c=par['Omega_m'] - par['Omega_b'],
                      Omega_b=par['Omega_b'], h=par['h'],
                      w0=par['w0'], n_s=par['n_s'], A_s=par['A_s'])

# Generate power spectrum
z = np.linspace(0, 3.5, 50)
k = np.logspace(-4, 2, 200)
pkln = np.array([ccl.linear_matter_power(cosmo, k, a)
                 for a in 1/(1+z)])
pknl = np.array([ccl.nonlin_matter_power(cosmo, k, a)
                 for a in 1/(1+z)])
np.savez('input/pk.npz', k=k, z=z, pk_nl=pknl, pk_lin=pkln)

# Generate tracer kernels
dndzs = cal.get_tracer_dndzs(filename='input/additional_dNdzs/dNdzs_3src_6lens.npz')
z_cl = dndzs['z_cl']
z_sh = dndzs['z_sh']
tpar = cal.get_tracer_parameters()

Cl_tracers = [ccl.NumberCountsTracer(cosmo, False,
                                     (z_cl, dndzs['dNdz_cl'][:, ni]),
                                     bias=(z_cl, np.full(len(z_cl), b)))
              for (ni, b) in zip(range(0, 6), tpar['b_g'])]
Sh_tracers = [ccl.WeakLensingTracer(cosmo, (z_sh, dndzs['dNdz_sh'][:, ni]),
                                    True)
              for ni in range(0, 3)]


def bias(i):
    return np.full(len(z_cl), tpar['b_g'][i])


chi_sh = ccl.comoving_radial_distance(cosmo, 1./(1.+z_sh))
chi_cl = ccl.comoving_radial_distance(cosmo, 1./(1.+z_cl))
kernels_cl = np.array([t.get_kernel(chi_cl)[0]*bias(i)
                       for i, t in enumerate(Cl_tracers)])
kernels_sh = np.array([t.get_kernel(chi_sh)[0] for t in Sh_tracers])
np.savez('input/kernels_3src_6lens.npz',
         z_cl=z_cl, chi_cl=chi_cl, kernels_cl=kernels_cl,
         z_sh=z_sh, chi_sh=chi_sh, kernels_sh=kernels_sh)

# Generate background arrays
#zs = np.linspace(0, 100, 1024)
#chis = ccl.comoving_radial_distance(cosmo, 1./(1.+zs))
#ez = ccl.h_over_h0(cosmo, 1./(1.+zs))
#np.savez('input/background.npz', z=zs, chi=chis, Ez=ez)
