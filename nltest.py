import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl

# Redshift distribution
z = np.linspace(0, 4.72, 60)
nz = z**2*np.exp(-0.5*((z-1.5)/0.7)**2)

# Bias
bz = 0.278*((1 + z)**2 - 6.565) + 2.393
bz=2.0*np.ones(len(z))
#plt.figure()
#plt.plot(z, nz*bz)
#plt.xlabel(r'$z$', fontsize=16)
#plt.ylabel(r'$b(z)\,\phi(z)$', fontsize=16)
#plt.show()
# Power spectra
ls = np.unique(np.linspace(2, 2000, 1998).astype(int)).astype(float)
cosmo = ccl.CosmologyVanillaLCDM()
tracer_gal = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z, nz), bias=(z, bz))
cl_gg = ccl.angular_cl(cosmo, tracer_gal, tracer_gal, ls,
                       l_limber=-1)
cl_ggn = ccl.angular_cl(cosmo, tracer_gal, tracer_gal, ls,
                        l_limber=1000)

plt.figure()
plt.plot(ls, cl_gg, 'k-', label='Limber')
plt.plot(ls, cl_ggn, 'r--', label='Non-Limber')
plt.loglog()
plt.legend()
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$C_\ell$', fontsize=16)
plt.show()