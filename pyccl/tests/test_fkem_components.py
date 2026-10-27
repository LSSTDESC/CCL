import itertools

import numpy as np
import pytest

import pyccl as ccl


@pytest.mark.parametrize("nchi", [128, 256])
def test_number_counts_component_additivity(nchi):
    # With identical linear/nonlinear power, the two Limber terms cancel.
    # Density and RSD use identical radial sampling, so adding their
    # components must commute with projection.
    cosmo = ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96,
        transfer_function="eisenstein_hu", matter_power_spectrum="linear")
    z = np.linspace(0.01, 2., 200)
    nz = np.exp(-((z - 1.)**2) / 0.1)
    bias = z, np.full_like(z, 1.5)
    tracers = [ccl.NumberCountsTracer(
        cosmo, dndz=(z, nz), bias=b, has_rsd=rsd)
        for b, rsd in [(bias, False), (None, True), (bias, True)]]
    ell = np.array([2., 5., 8., 10., 25.])
    kwargs = dict(l_limber=100, fkem_Nchi=nchi, fkem_chi_min=1.e-6)
    parts = []
    for i, j in itertools.combinations_with_replacement(range(2), 2):
        value = ccl.angular_cl(cosmo, tracers[i], tracers[j], ell, **kwargs)
        parts.append(value if i == j else 2 * value)
    combined = ccl.angular_cl(cosmo, tracers[2], tracers[2], ell, **kwargs)
    np.testing.assert_allclose(
        combined, np.sum(parts, axis=0), rtol=1.e-10, atol=0.)
    # The cached transforms must produce the same result on a repeated call.
    repeated = ccl.angular_cl(cosmo, tracers[2], tracers[2], ell, **kwargs)
    np.testing.assert_array_equal(repeated, combined)
