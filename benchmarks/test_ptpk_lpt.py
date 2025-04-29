import os
import numpy as np
import pyccl as ccl
import pyccl.nl_pt as pt
import pytest

# Updated to a higher tolerance because of recent
# changes to velocileptors with respect to the
# version used to generate benchmarks.
LPTPK_TOLERANCE = 5e-4

# Set cosmology
COSMO = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05,
                      h=0.7, sigma8=0.8, n_s=0.96,
                      T_CMB=2.725,
                      transfer_function='boltzmann_camb')

# Redshifts
zs = np.array([0., 1.])

# Tracers
ptt = {}
ptt['g'] = pt.PTNumberCountsTracer(b1=1.3, b2=1.5, bs=1.7, b3nl=1.9, bk2=0.1)
ptt['m'] = pt.PTMatterTracer()

# Calculator
a_arr = 1./(1+np.array([0., 0.25, 0.5, 0.75, 1.]))[::-1]
ptc = pt.LagrangianPTCalculator(log10k_min=-4, log10k_max=2,
                                nk_per_decade=20, a_arr=a_arr,
                                b1_pk_kind='pt', bk2_pk_kind='pt')
ptc.update_ingredients(COSMO)

# Read data
dirdat = os.path.join(os.path.dirname(__file__), 'data')
data = []
data.append(np.loadtxt(os.path.join(dirdat, 'pt_bm_lpt_z0.txt'), unpack=True))
data.append(np.loadtxt(os.path.join(dirdat, 'pt_bm_lpt_z1.txt'), unpack=True))
order = ['gg', 'gm']


@pytest.mark.parametrize('comb', enumerate(order))
def test_pt_pk(comb):
    i_d, cc = comb
    t1, t2 = cc

    ptt1 = ptt[t1]
    ptt2 = ptt[t2]

    pk = ptc.get_biased_pk2d(ptt1, tracer2=ptt2)

    for iz, z in enumerate(zs):
        a = 1./(1+z)
        kin = data[iz][0]
        ind = np.where((kin < 1.0) & (kin > 1E-3))
        k = kin[ind]
        dpk = data[iz][i_d+1][ind]
        tpk = pk(k, a, COSMO)
        assert np.all(
            np.fabs(tpk / dpk - 1) < LPTPK_TOLERANCE
        )
