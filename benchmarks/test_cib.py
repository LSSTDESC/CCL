import numpy as np
import pyccl as ccl


def test_cibcl():
    # Read benchmarks
    bm = np.loadtxt("benchmarks/data/cib_class_sz_szpowerspectrum.txt",
                    unpack=True)

    # Initialize cosmology and halo model ingredients
    cosmo = ccl.Cosmology(Omega_b=0.05,
                          Omega_c=0.25,
                          h=0.7,
                          n_s=0.9645,
                          A_s=2.02E-9,
                          Neff=3.046, T_CMB=2.725)
    mdef = ccl.halos.MassDef200m
    cM = ccl.halos.ConcentrationDuffy08(mass_def=mdef)
    nM = ccl.halos.MassFuncTinker10(mass_def=mdef, norm_all_z=True)
    bM = ccl.halos.HaloBiasTinker10(mass_def=mdef)
    hmc = ccl.halos.HMCalculator(mass_function=nM, halo_bias=bM,
                                 mass_def=mdef)
    pr = ccl.halos.HaloProfileCIBShang12(concentration=cM, nu_GHz=217,
                                         Mmin=1E10, mass_def=mdef)
    pr.update_parameters(nu_GHz=217,
                         alpha=0.36,
                         T0=24.4,
                         beta=1.75,
                         gamma=1.7,
                         s_z=3.6,
                         log10Meff=12.6,
                         siglog10M=np.sqrt(0.5),
                         Mmin=1E10,
                         L0=6.4E-8)

    pr2pt = ccl.halos.Profile2ptCIB()

    # CIB tracer
    tr = ccl.CIBTracer(cosmo, z_min=0.07)

    # 3D power spectrum
    pk = ccl.halos.halomod_Pk2D(cosmo, hmc, pr, prof_2pt=pr2pt)

    # Angular power spectrum
    ls = bm[0]
    cl = ccl.angular_cl(cosmo, tr, tr, ell=ls, p_of_k_a=pk)
    dl = cl*ls*(ls+1)/(2*np.pi)

    # Compare
    assert np.all(np.fabs(dl/(bm[31]+bm[32])-1) < 1E-2)
