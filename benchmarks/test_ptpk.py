import os
import numpy as np
import pyccl as ccl
import pyccl.nl_pt as pt
import pytest

# Set cosmology
COSMO = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05,
                      h=0.7, sigma8=0.8, n_s=0.96,
                      T_CMB=2.725,
                      transfer_function='bbks')

# Redshifts
zs = np.array([0., 1.])

# Tracers
ptt = {}
ptt['g'] = pt.PTNumberCountsTracer(b1=1.3, b2=1.5, bs=1.7, b3nl=1.9, bk2=0.1)
ptt['i'] = pt.PTIntrinsicAlignmentTracer(c1=1.9, c2=2.1, cdelta=2.3)
ptt['m'] = pt.PTMatterTracer()

# Calculator
a_arr = 1./(1+np.array([0., 0.25, 0.5, 0.75, 1.]))[::-1]
ptc = pt.EulerianPTCalculator(with_NC=True, with_IA=True,
                              log10k_min=-4, log10k_max=2,
                              nk_per_decade=20, a_arr=a_arr,
                              b1_pk_kind='pt', bk2_pk_kind='pt',
                              pad_factor=0.5)
ptc.update_ingredients(COSMO)

# Read data
dirdat = os.path.join(os.path.dirname(__file__), 'data')
data = []
data.append(np.loadtxt(os.path.join(dirdat, 'pt_bm_z0.txt'), unpack=True))
data.append(np.loadtxt(os.path.join(dirdat, 'pt_bm_z1.txt'), unpack=True))
order = ['gg', 'gm', 'gi', 'ii', 'ib', 'im']


@pytest.mark.parametrize('comb', enumerate(order))
def test_pt_pk(comb):
    i_d, cc = comb
    t1, t2 = cc

    return_bb = False
    if t2 == 'b':
        t2 = 'i'
        return_bb = True

    ptt1 = ptt[t1]
    ptt2 = ptt[t2]

    pk = ptc.get_biased_pk2d(ptt1, tracer2=ptt2, return_ia_bb=return_bb)

    for iz, z in enumerate(zs):
        a = 1./(1+z)
        kin = data[iz][0]
        ind = np.where((kin < 1.0) & (kin > 1E-3))
        k = kin[ind]
        dpk = data[iz][i_d+1][ind]
        tpk = pk(k, a, COSMO)
        print('If this fails, try updating or re-installing fast-pt:')
        print('pip install -U fast-pt')
        assert np.all(np.fabs(tpk / dpk - 1) < 1E-5)
