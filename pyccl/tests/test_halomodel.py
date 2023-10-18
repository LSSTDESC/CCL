import numpy as np
import pytest
import pyccl as ccl


def get_pk_new(mf, c, cosmo, a, k, get_1h, get_2h):
    mdef = ccl.halos.MassDef('vir', 'matter')
    if mf == 'shethtormen':
        hmf = ccl.halos.MassFuncSheth99(mass_def=mdef,
                                        mass_def_strict=False,
                                        use_delta_c_fit=True)
        hbf = ccl.halos.HaloBiasSheth99(mass_def=mdef,
                                        mass_def_strict=False)
    elif mf == 'tinker10':
        hmf = ccl.halos.MassFuncTinker10(mass_def=mdef,
                                         mass_def_strict=False)
        hbf = ccl.halos.HaloBiasTinker10(mass_def=mdef,
                                         mass_def_strict=False)

    if c == 'constant_concentration':
        cc = ccl.halos.ConcentrationConstant(4., mass_def=mdef)
    elif c == 'duffy2008':
        cc = ccl.halos.ConcentrationDuffy08(mass_def=mdef)
    elif c == 'bhattacharya2011':
        cc = ccl.halos.ConcentrationBhattacharya13(mass_def=mdef)
    prf = ccl.halos.HaloProfileNFW(mass_def=mdef, concentration=cc)
    hmc = ccl.halos.HMCalculator(mass_function=hmf, halo_bias=hbf,
                                 mass_def=mdef)
    return ccl.halos.halomod_power_spectrum(cosmo, hmc, k, a, prf,
                                            get_1h=get_1h,
                                            get_2h=get_2h)


@pytest.mark.parametrize('mf_c', [['shethtormen', 'bhattacharya2011'],
                                  ['shethtormen', 'duffy2008'],
                                  ['shethtormen', 'constant_concentration'],
                                  ['tinker10', 'constant_concentration']])
def test_halomodel_choices_smoke(mf_c):
    mf, c = mf_c
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        transfer_function='bbks', matter_power_spectrum='linear')
    a = 0.8
    k = np.geomspace(1E-2, 1, 10)
    p1h = get_pk_new(mf, c, cosmo, a, k, True, False)
    p2h = get_pk_new(mf, c, cosmo, a, k, False, True)
    pt = get_pk_new(mf, c, cosmo, a, k, True, True)

    assert np.all(np.isfinite(p1h))
    assert np.all(np.isfinite(p2h))
    assert np.all(np.isfinite(pt))
    assert np.allclose(pt, p1h+p2h, atol=0)
