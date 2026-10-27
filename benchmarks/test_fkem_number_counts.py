"""FKEM spectra compared with direct spherical-Bessel quadrature."""
import json
from pathlib import Path

import numpy as np
import pytest

import pyccl as ccl


DATA = json.loads((Path(__file__).parent / 'data' /
                   'fkem_number_counts_reference.json').read_text())


def setup_case(case_index):
    case = DATA['cases'][case_index]
    cosmo = ccl.Cosmology(**case['cosmology'],
                          transfer_function='eisenstein_hu',
                          matter_power_spectrum='linear')
    tracers = []
    for _, lo, hi in case['windows']:
        z = np.linspace(lo, hi, 4097)
        nz = np.sin(np.pi*(z-lo)/(hi-lo))**4 / ((hi-lo)*3/8)
        bias = z, (1+z)**case['bias_power']
        for b, rsd in [(bias, False), (None, True), (bias, True)]:
            tracers.append(ccl.NumberCountsTracer(
                cosmo, dndz=(z, nz), bias=b, has_rsd=rsd))
    pairs = {tuple(pair): i for i, pair in enumerate(case['pairs'])}
    return cosmo, tracers, case['spectra'], pairs


@pytest.fixture(scope='module')
def original():
    return setup_case(0)


@pytest.mark.parametrize('nchi', [128, 256, 512])
@pytest.mark.parametrize('ell_index', range(3))
@pytest.mark.parametrize('pair', [(0, 0), (0, 2), (2, 2)])
def test_compact_number_counts(original, nchi, ell_index, pair):
    cosmo, tracers, spectra, pairs = original
    row = spectra[ell_index]
    left, right = pair
    actual = ccl.angular_cl(
        cosmo, tracers[left], tracers[right], row['ell'], l_limber=100,
        fkem_Nchi=nchi, fkem_chi_min=1.)
    expected = row['values'][pairs[pair]]
    # Coarse radial sampling has a separate, explicit accuracy requirement.
    tolerance = 1.e-3 if nchi < 512 else 1.e-4
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=0.)


@pytest.fixture(scope='module', params=[1, 2])
def expanded(request):
    return setup_case(request.param)


@pytest.mark.parametrize('ell_index', range(5))
@pytest.mark.parametrize('pair_index', range(45))
def test_number_counts_cross_spectra(expanded, ell_index, pair_index):
    cosmo, tracers, spectra, pairs = expanded
    left, right = list(pairs)[pair_index]
    row = spectra[ell_index]
    actual = ccl.angular_cl(
        cosmo, tracers[left], tracers[right], row['ell'], l_limber=1000)
    expected = row['values'][pair_index]
    norm = np.sqrt(row['values'][pairs[left, left]]
                   * row['values'][pairs[right, right]])
    # Auto spectra supply a stable scale when a cross spectrum is near zero.
    np.testing.assert_allclose(actual/norm, expected/norm,
                               rtol=0., atol=1.e-4)
