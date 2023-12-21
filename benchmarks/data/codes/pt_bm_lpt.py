import numpy as np
import pyccl as ccl
from velocileptors.EPT.cleft_kexpanded_resummed_fftw import RKECLEFT

cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.96, sigma8=0.8,
                      transfer_function='boltzmann_camb', T_CMB=2.725)
lkmin = -4
lkmax = 2
n_per_decade = 20
z_s = np.array([0., 1.])
a_s = 1./(1 + z_s)
ks = np.logspace(lkmin, lkmax, (lkmax - lkmin) * n_per_decade)

pklz0 = cosmo.linear_matter_power(ks, 1.0)
g = cosmo.growth_factor(a_s)

h = cosmo['h']
cleft = RKECLEFT(ks / h, pklz0 * h ** 3)
lpt_table = []
for gz in g:
       cleft.make_ptable(D=gz, kmin=ks[0] / h,
                         kmax=ks[-1] / h, nk=ks.size)
       lpt_table.append(cleft.pktable)
lpt_table = np.array(lpt_table)
lpt_table[:, :, 1:] /= h ** 3

b11 = b12 = 1.3
b21 = b22 = 1.5
bs1 = bs2 = 1.7
b3nl1 = b3nl2 = 1.9
bk21 = bk22 = 0.1

# Transform from Eulerian to Lagrangian biases
bL11 = b11 - 1
bL12 = b12 - 1

# Pgg
Pdmdm = lpt_table[:, :, 1]
Pdmd1 = 0.5*lpt_table[:, :, 2]
Pd1d1 = lpt_table[:, :, 3]
pgg = (Pdmdm + (bL11+bL12) * Pdmd1 +
       (bL11*bL12) * Pd1d1)

Pdmd2 = 0.5*lpt_table[:, :, 4]
Pd1d2 = 0.5*lpt_table[:, :, 5]
Pd2d2 = lpt_table[:, :, 6]
Pdms2 = 0.25*lpt_table[:, :, 7]
Pd1s2 = 0.25*lpt_table[:, :, 8]
Pd2s2 = 0.25*lpt_table[:, :, 9]
Ps2s2 = 0.25*lpt_table[:, :, 10]
Pdmo3 = 0.25*lpt_table[:, :, 11]
Pd1o3 = 0.25*lpt_table[:, :, 12]

Pdmk2 = 0.5*Pdmdm * (ks**2)[None, :]
Pd1k2 = 0.5*Pdmd1 * (ks**2)[None, :]
Pd2k2 = Pdmd2 * (ks**2)[None, :]
Ps2k2 = Pdms2 * (ks**2)[None, :]
Pk2k2 = 0.25*Pdmdm * (ks**4)[None, :]

pgg += ((b21 + b22) * Pdmd2 +
       (bs1 + bs2) * Pdms2 +
       (bL11*b22 + bL12*b21) * Pd1d2 +
       (bL11*bs2 + bL12*bs1) * Pd1s2 +
       (b21*b22) * Pd2d2 +
       (b21*bs2 + b22*bs1) * Pd2s2 +
       (bs1*bs2) * Ps2s2 +
       (b3nl1 + b3nl2) * Pdmo3 +
       (bL11*b3nl2 + bL12*b3nl1) * Pd1o3)

pgg += ((bk21 + bk22) * Pdmk2 +
       (bL12 * bk21 + bL11 * bk22) * Pd1k2 +
       (b22 * bk21 + b21 * bk22) * Pd2k2 +
       (bs2 * bk21 + bs1 * bk22) * Ps2k2 +
       (bk21 * bk22) * Pk2k2)

# Pgm
pgm = Pdmdm + bL11 * Pdmd1
pgm += (b21 * Pdmd2 +
       bs1 * Pdms2 +
       b3nl1 * Pdmo3 +
       bk21 * Pdmk2)

np.savetxt("../pt_bm_lpt_z0.txt",
           np.transpose([ks, pgg[0], pgm[0]]),
           header='[0]-k  [1]-GG [2]-GM')
np.savetxt("../pt_bm_lpt_z1.txt",
           np.transpose([ks, pgg[1], pgm[1]]),
           header='[0]-k  [1]-GG [2]-GM')
