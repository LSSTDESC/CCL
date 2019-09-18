import numpy as np
import pyccl as ccl

import pytest


# FIXME: these are not real standards
# tolerence on abs difference in r^2 xi(r) for the range
# r = 0.1 - 100 Mpc (40 points in r) for z=0,1,2,3,4,5
CORR_TOLERANCE1 = [3.0e-2, 3.0e-2, 3.0e-2, 3.0e-2, 3.0e-2, 3.0e-2]
# tolerence on abs difference in r^2 xi(r) for the range
# r = 50 - 250 Mpc (100 points in r) for z=0,1,2,3,4,5
CORR_TOLERANCE2 = [3.0e-2, 3.0e-2, 3.0e-2, 3.0e-2, 3.0e-2, 3.0e-2]


@pytest.mark.parametrize('model', list(range(3)))
def test_correlation_3d(model):
    Omega_v = [0.7, 0.7, 0.7, 0.65, 0.75]
    w_0 = [-1.0, -0.9, -0.9, -0.9, -0.9]
    w_a = [0.0, 0.0, 0.1, 0.1, 0.1]

    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        Neff=3.046,
        m_nu_type='normal',
        Omega_g=0,
        Omega_k=1.0 - 0.25 - 0.05 - Omega_v[model],
        w0=w_0[model],
        wa=w_a[model],
        transfer_function='bbks',
        matter_power_spectrum='halofit')

    data = np.loadtxt("./benchmarks/data/model%d_xi.txt" % (model+1))
    N1 = 40
    data1 = data[:N1, :]
    r1 = data1[:, 0]
    data2 = data[N1:, :]
    r2 = data2[:, 0]

    for z in np.arange(6):
        zind = int(z)
        a = 1.0 / (1 + z)

        xi1 = ccl.correlation_3d(cosmo, a, r1)
        err = np.abs(r1*r1*(xi1-data1[:, zind+1]))
        assert np.allclose(err, 0, rtol=0, atol=CORR_TOLERANCE1[zind])

        xi2 = ccl.correlation_3d(cosmo, a, data2[:, 0])
        err = np.abs(r2*r2*(xi2-data2[:, zind+1]))
        assert np.allclose(err, 0, rtol=0, atol=CORR_TOLERANCE1[zind])
