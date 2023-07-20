import pysz_gal.pysz_gal as szgal
import numpy as np
from scipy.integrate import simps


# Set N(z)
def _nz_2mrs(z):
    # From 1706.05422
    m = 1.31
    beta = 1.64
    x = z / 0.0266
    return x**m * np.exp(-x**beta)
z1 = 1e-5
z2 = 0.1
z_arr = np.linspace(z1, z2, 10024)
dndz = _nz_2mrs(z_arr)
dndz /= simps(dndz, z_arr)

# Set things up
pyszgal_cl = szgal.tsz_gal_cl()
ell = np.unique(np.geomspace(2, 1000, 20).astype(int)).astype(float)

# Set params
h0 = 0.67
Omb = 0.05
Omc = 0.25
As = 2E-9
ns = 0.9645
mnu = 0.00001
ombh2 = Omb*h0**2
omch2 = Omc*h0**2
bias = 1.55
lMcut = 11.8
Mcut = 10**lMcut
lM1 = 11.73
M1 = 10**lM1
kappa = 1.0
sigma_Ncen = 0.15
alp_Nsat = 0.77
rmax = 4.39
rgs = 1.17
pars = {'h0':h0, 'obh2':ombh2,'och2':omch2,\
        'As':As,'ns':ns,'mnu':mnu,\
        'mass_bias':bias,\
        'Mcut':Mcut,'M1':M1,'kappa':kappa,'sigma_Ncen':sigma_Ncen,'alp_Nsat':alp_Nsat,\
        'rmax':rmax,'rgs':rgs,\
        'flag_nu':True,'flag_tll':False}

# Run pysz_gal
gg, gy, tll, ng, tmp, ks, zs, pks = pyszgal_cl.get_tsz_cl(ell, pars, dndz,\
                                                          z1, z2, z1, z2, 101, 1001)
# Save benchmark data
np.savetxt("../k_hod.txt", ks)
np.savetxt("../z_hod.txt", zs)
np.savetxt("../pk_hod.txt", pks)
np.savetxt("../cl_hod.txt", np.transpose([ell, gg[0]+gg[1]]))
