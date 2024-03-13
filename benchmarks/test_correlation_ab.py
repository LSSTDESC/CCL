import numpy as np
import pyccl as ccl
import os


def test_correlation_ab():
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)
    z_eval = 0.

    dirdat = os.path.dirname(__file__) + '/data/'
    benchmark_file = os.path.join(dirdat, 'wgplus.out')
    rp_benchmark, wgplus_benchmark = np.loadtxt(benchmark_file, unpack=True, delimiter=' ')

    C1rhocrit=0.0134 # standard IA normalisation
    Pk_GI = ccl.Pk2D.from_function(pkfunc=lambda k, a: C1rhocrit * cosmo['Omega_m']
                                                       / cosmo.growth_factor(a) *
                                                       cosmo.nonlin_matter_power(k, a),
                                   is_logp=False)
    wgplus = cosmo.correlation_ab(r_p=rp_benchmark, z=z_eval, p_of_k_a=Pk_GI, type='g+')

    assert np.all(np.abs((wgplus-wgplus_benchmark)/wgplus_benchmark)<5e-2)
