from pathlib import Path

import numpy as np
import pytest
import pyccl as ccl

DATA = Path(__file__).parent / "data"

COSMO = ccl.CosmologyVanillaLCDM(transfer_function='boltzmann_camb',
                                 matter_power_spectrum='linear')
M200 = ccl.halos.MassDef200m
HMF = ccl.halos.MassFuncTinker08(mass_def=M200)
HBF = ccl.halos.HaloBiasTinker10(mass_def=M200)
CON = ccl.halos.ConcentrationDuffy08(mass_def=M200)
NFW = ccl.halos.HaloProfileNFW(mass_def=M200, concentration=CON,
                               fourier_analytic=True)
HMC = ccl.halos.HMCalculator(mass_function=HMF, halo_bias=HBF,
                             mass_def=M200, nM=512)
K = np.geomspace(1e-3, 10, 60)
A = 0.7

SELECTIONS = {
    'lognormal': lambda M, a: np.exp(-0.5 * ((np.log10(M) - 14.3) / 0.3)**2),
    'tophat': lambda M, a: ((M > 1e14) & (M < 1e15)) * 1.,
}


def _selection(name):
    return ccl.halos.HaloProfileSelection(mass_def=M200,
                                          selection=SELECTIONS[name])


@pytest.mark.parametrize('name', list(SELECTIONS))
def test_bispectrum_squeezed_vs_onecovariance(name):
    B_oc = np.load(DATA / f"onecov_bispectrum_squeezed_{name}.npz")['bispec']
    B = ccl.halos.halomod_bispectrum_squeezed(COSMO, HMC, K, A,
                                              _selection(name), prof2=NFW)
    assert np.all(np.abs((B_oc - B) / B_oc) < 1e-4)


@pytest.mark.parametrize('name', list(SELECTIONS))
def test_bispectrum_squeezed_ssc_vs_onecovariance(name):
    # The reference follows OneCovariance's powspec_responses, which uses a
    # smoothing spline for dln(k^3 P)/dlnk that washes out the BAO wiggles.
    # This differs from the CCL derivative by up to ~5% around k~0.1-0.3/Mpc.
    B_oc = np.load(DATA / f"onecov_bispectrum_ssc_{name}.npz")['bispec']
    B = ccl.halos.halomod_bispectrum_squeezed_SSC(COSMO, HMC, K, A,
                                                  _selection(name),
                                                  prof2=NFW)
    assert np.all(np.abs((B_oc - B) / B_oc) < 6e-2)


@pytest.mark.parametrize('name', list(SELECTIONS))
def test_bispectrum_squeezed_ssc_same_derivative(name):
    # Isolates the formula and the normalisation from the treatment of the
    # derivative.
    ref = np.load(DATA / f"onecov_bispectrum_ssc_{name}.npz")
    dlnpk = COSMO.get_linear_power()(K, A, derivative=True)
    B_oc = ref['phi'] * ((47/21 - dlnpk / 3) * ref['i_m']**2 * ref['pk']
                         + ref['i_mm'])
    B = ccl.halos.halomod_bispectrum_squeezed_SSC(COSMO, HMC, K, A,
                                                  _selection(name),
                                                  prof2=NFW)
    assert np.all(np.abs((B_oc - B) / B_oc) < 1e-3)
