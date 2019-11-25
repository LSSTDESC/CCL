import numpy as np
import pytest

import pyccl as ccl

HALOPROFILE_TOLERANCE = 1E-3
M200 = ccl.halos.MassDef(200, 'matter')

# We need a custom-made concentration object
class ConcentrationConstant(ccl.halos.Concentration):
    def __init__(self, c, mdef=None):
        self.c = c
        super(ConcentrationConstant, self).__init__(mdef)

    def _default_mdef(self):
        self.mdef = M200

    def _check_mdef(self, mdef):
        return False

    def _concentration(self, cosmo, M, a):
        if np.ndim(M) == 0:
            return self.c
        else:
            return self.c * np.ones(M.size)


@pytest.mark.parametrize(
    'model', ['nfw', 'projected_nfw', 'einasto', 'hernquist'])
def test_haloprofile(model):
    cosmo = ccl.Cosmology(
        Omega_b=0.0486,
        Omega_c=0.2603,
        h=0.6774,
        n_s=0.9667,
        sigma8=0.8159,
        w0=-1,
        wa=0,
        m_nu=0,
        m_nu_type='normal',
        Neff=3.046,
        Omega_k=0,
        transfer_function='eisenstein_hu',
        mass_function='shethtormen')

    data = np.loadtxt("./benchmarks/data/haloprofile_%s_colossus.txt" % model)
    a = 1.0
    concentration = 5
    halomass = 6e13
    halomassdef = 200
    rmin = 0.01
    rmax = 100
    r = np.exp(
        np.log(rmin) +
        np.log(rmax/rmin) * np.arange(data.shape[0]) / (data.shape[0]-1))

    if model == 'nfw':
        c = ConcentrationConstant(concentration)
        p = ccl.halos.HaloProfileNFW(c, truncated=False)
        def prof_func(csm, cnc, m, hmd, a, r):
            return p.profile_real(csm, r, m, a, M200)
        #prof_func = ccl.nfw_profile_3d
    elif model == 'projected_nfw':
        prof_func = ccl.nfw_profile_2d
    elif model == 'einasto':
        prof_func = ccl.einasto_profile_3d
    elif model == 'hernquist':
        prof_func = ccl.hernquist_profile_3d
    prof = prof_func(cosmo, concentration, halomass, halomassdef, a, r)

    tol = np.clip(np.abs(HALOPROFILE_TOLERANCE * data[:, 1]), 1e-12, np.inf)
    err = np.abs(prof - data[:, 1])
    assert np.all(err <= tol)
