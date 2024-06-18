import numpy as np
import pyccl as ccl
from scipy.integrate import simpson


def test_iswcl():
    # Cosmology
    Ob = 0.05
    Oc = 0.25
    h = 0.7
    COSMO = ccl.Cosmology(
        Omega_b=Ob,
        Omega_c=Oc,
        h=h,
        n_s=0.96,
        sigma8=0.8,
        transfer_function='bbks')

    # CCL calculation
    ls = np.arange(2, 100)
    zs = np.linspace(0, 0.6, 256)
    nz = np.exp(-0.5*((zs-0.3)/0.05)**2)
    bz = np.ones_like(zs)
    tr_n = ccl.NumberCountsTracer(COSMO, has_rsd=False,
                                  dndz=(zs, nz), bias=(zs, bz))
    tr_i = ccl.ISWTracer(COSMO)
    cl = ccl.angular_cl(COSMO, tr_n, tr_i, ls)

    # Benchmark from Eq. 6 in 1710.03238
    pz = nz / simpson(nz, x=zs)
    H0 = h / ccl.physical_constants.CLIGHT_HMPC
    # Prefactor
    prefac = 3*COSMO['T_CMB']*(Oc+Ob)*H0**3/(ls+0.5)**2
    # H(z)/H0
    ez = ccl.h_over_h0(COSMO, 1./(1+zs))
    # Linear growth and derivative
    dz = ccl.growth_factor(COSMO, 1./(1+zs))
    gz = np.gradient(dz*(1+zs), zs[1]-zs[0])/dz
    # Comoving distance
    chi = ccl.comoving_radial_distance(COSMO, 1/(1+zs))
    # P(k)
    pks = np.array([ccl.nonlin_matter_power(COSMO, (ls+0.5)/(c+1E-6), 1./(1+z))
                    for c, z in zip(chi, zs)]).T
    # Limber integral
    cl_int = pks[:, :]*(pz*ez*gz)[None, :]
    clbb = simpson(cl_int, x=zs)
    clbb *= prefac

    assert np.all(np.fabs(cl/clbb-1) < 1E-3)
