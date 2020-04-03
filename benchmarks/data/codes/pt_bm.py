import numpy as np
import pyccl as ccl
import fastpt as fpt

cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.96, sigma8=0.8,
                      transfer_function='bbks')
lkmin = -4
lkmax = 2
n_per_decade = 20
z_s = np.array([0., 1.])
a_s = 1./(1 + z_s)
ks = np.logspace(lkmin, lkmax, (lkmax - lkmin) * n_per_decade)
pk = ccl.linear_matter_power(cosmo, ks, 1.)
gf = ccl.growth_factor(cosmo, a_s)
g4 = gf**4
pklin = np.array([ccl.linear_matter_power(cosmo, ks, a)
                  for a in a_s])

C_window=.75
P_window=None
n_pad=int(0.5*len(ks))
to_do = ['one_loop_dd', 'dd_bias', 'IA']
pt_ob=fpt.FASTPT(ks,to_do=to_do,low_extrap=-5,high_extrap=3,n_pad=n_pad)
oloop_dd = pt_ob.one_loop_dd(pk,
                             P_window=P_window,
                             C_window=C_window)
Pd1d1 = pklin + g4[:, None] * oloop_dd[0][None, :]
dd_bias = pt_ob.one_loop_dd_bias(pk,
                                 P_window=P_window,
                                 C_window=C_window)

ia_ta = pt_ob.IA_ta(pk,
                    P_window=P_window,
                    C_window=C_window)
ia_tt = pt_ob.IA_tt(pk,
                    P_window=P_window,
                    C_window=C_window)
ia_mix = pt_ob.IA_mix(pk,
                      P_window=P_window,
                      C_window=C_window)

g4 = g4[:, None]
Pd1d2 = g4 * dd_bias[2][None, :]
Pd2d2 = g4 * dd_bias[3][None, :]
Pd1s2 = g4 * dd_bias[4][None, :]
Pd2s2 = g4 * dd_bias[5][None, :]
Ps2s2 = g4 * dd_bias[6][None, :]
a00e = g4 * ia_ta[0][None, :]
c00e = g4 * ia_ta[1][None, :]
a0e0e = g4 * ia_ta[2][None, :]
a0b0b = g4 * ia_ta[3][None, :]
ae2e2 = g4 * ia_tt[0][None, :]
ab2b2 = g4 * ia_tt[1][None, :]
a0e2 = g4 * ia_mix[0][None, :]
b0e2 = g4 * ia_mix[1][None, :]
d0ee2 = g4 * ia_mix[2][None, :]
d0bb2 = g4 * ia_mix[3][None, :]
                                        
b1 = 1.3
b2 = 1.5
bs = 1.7
c1 = 1.9
c2 = 2.1
cd = 2.3

pgg = (b1**2 * Pd1d1 +
       b1*b2 * Pd1d2 +
       0.25 * b2**2 * Pd2d2 +
       b1 * bs * Pd1s2 +
       0.5 * b2 * bs * Pd2s2 +
       0.25 * bs**2 * Ps2s2)

pgm = (b1 * Pd1d1 +
       0.5* b2 * Pd1d2 +
       0.5 * bs * Pd1s2)
pgi = b1 * (c1 * Pd1d1 +
            cd * (a00e + c00e) +
            c2 * (a0e2 + b0e2))
pii = (c1**2 * Pd1d1 +
       2 * c1 * cd * (a00e + c00e) +
       cd**2 * a0e0e +
       c2**2 * ae2e2 +
       2 * c1 * c2 * (a0e2 + b0e2) +
       2 * cd * c2 * d0ee2)
pii_bb = (cd**2 * a0b0b +
          c2**2 * ab2b2 +
          2 * cd * c2 * d0bb2)
pim = (c1 * Pd1d1 +
       cd * (a00e + c00e) +
       c2 * (a0e2 + b0e2))

np.savetxt("../pt_bm_z0.txt",
           np.transpose([ks, pgg[0], pgm[0], pgi[0],
                         pii[0], pii_bb[0], pim[0]]),
           header='[0]-k  [1]-GG [2]-GM [3]-GI [4]-II [5]-II_BB [6]-IM')
np.savetxt("../pt_bm_z1.txt",
           np.transpose([ks, pgg[1], pgm[1], pgi[1],
                         pii[1], pii_bb[1], pim[1]]),
           header='[0]-k  [1]-GG [2]-GM [3]-GI [4]-II [5]-II_BB [6]-IM')
