import numpy as np
import pyccl as ccl
import time


def test_timing():
    ls = np.unique(np.geomspace(2, 2000, 128).astype(int)).astype(float)
    nl = len(ls)
    b_g = np.array([1.376695, 1.451179, 1.528404,
                    1.607983, 1.689579, 1.772899,
                    1.857700, 1.943754, 2.030887,
                    2.118943])
    dNdz_file = np.load('benchmarks/data/dNdzs.npz')
    z_s = dNdz_file['z_sh']
    dNdz_s = dNdz_file['dNdz_sh'].T
    z_g = dNdz_file['z_cl']
    dNdz_g = dNdz_file['dNdz_cl'].T
    nt = len(dNdz_s) + len(dNdz_g)
    nx = (nt*(nt+1))//2

    cosmo = ccl.CosmologyVanillaLCDM(transfer_function='boltzmann_class')
    cosmo.compute_nonlin_power()

    start = time.time()
    t_g = [ccl.NumberCountsTracer(cosmo, True,
                                  (z_g, ng),
                                  bias=(z_g, np.full(len(z_g), b)))
           for ng, b in zip(dNdz_g, b_g)]
    t_s = [ccl.WeakLensingTracer(cosmo, (z_s, ns)) for ns in dNdz_s]
    t_all = t_g + t_s

    cls = np.zeros([nx, nl])
    ind1, ind2 = np.triu_indices(nt)
    for ix, (i1, i2) in enumerate(zip(ind1, ind2)):
        cls[ix, :] = ccl.angular_cl(cosmo, t_all[i1], t_all[i2], ls)
    end = time.time()
    t_seconds = end - start
    print(end-start)
    assert t_seconds < 3.
