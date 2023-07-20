import pyccl as ccl
import numpy as np
import scipy.integrate
import math
from functools import partial

cosmoin=ccl.Cosmology(
    Omega_c=0.25,
    Omega_b=0.05,
    h=0.7,
    sigma8=0.8,
    n_s=0.96,
    Neff=0,
    m_nu=0.0,
    w0=-1.,
    wa=0.,
    T_CMB=2.7,
    m_nu_type='normal',
    Omega_g=0,
    Omega_k=0,
    transfer_function='bbks',
    matter_power_spectrum='linear')

k_lin=np.logspace(-4., 3., 10000)
a_arr = np.logspace(-1., 0., 10)
knl_arr = np.zeros((len(a_arr),))
for i in range(len(a_arr)):
    a = a_arr[i]
    #I will need linear power at z
    pk_lin_z=ccl.linear_matter_power(cosmoin, k_lin, a)
    interp_pk_lin_z=scipy.interpolate.interp1d(k_lin,pk_lin_z)


    #---kNL prediction----
    [intknl,errintknl]=scipy.integrate.quad(interp_pk_lin_z,min(k_lin),max(k_lin),epsabs=0,epsrel=1e-6)
    print("Check error is small:",errintknl/intknl)
    knlemin2=1./(6*math.pi**2)*intknl
    knl=1./math.sqrt(knlemin2)
    print('knl[1/Mpc]=',knl)
    knl_arr[i] = knl

header_knl="[0] a, [1] k_NL Mpc^-1"

np.savetxt("kNL.txt", np.transpose(np.vstack((a_arr ,knl_arr))), header=header_knl)
