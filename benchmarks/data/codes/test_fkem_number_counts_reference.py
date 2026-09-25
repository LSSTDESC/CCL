"""Numerical checks of the direct reference integrals, not FKEM tolerances."""
import numpy as np
import pytest
from scipy.special import roots_legendre, spherical_jn

from fkem_number_counts_reference import (
    ZLO, ZHI, bessel_second_derivative, redshift_distribution)


def test_redshift_distribution_normalization():
    nodes, weights = roots_legendre(64)
    z = ZLO + (nodes + 1) * (ZHI - ZLO) / 2
    norm = weights @ redshift_distribution(z) * (ZHI - ZLO) / 2
    np.testing.assert_allclose(norm, 1, rtol=0, atol=2e-14)
    np.testing.assert_array_equal(
        redshift_distribution(np.array([ZLO - 1, ZHI + 1])), [0, 0])


@pytest.mark.parametrize("ell", [2, 5, 10])
def test_second_derivative_using_order_recurrence(ell):
    x = np.geomspace(1e-6, 1e3, 257)
    lower = ell * (ell - 1) / ((2 * ell + 1) * (2 * ell - 1))
    central = -(ell**2 / (2 * ell - 1)
                + (ell + 1)**2 / (2 * ell + 3)) / (2 * ell + 1)
    upper = (ell + 1) * (ell + 2) / ((2 * ell + 1) * (2 * ell + 3))
    recurrence = (lower * spherical_jn(ell - 2, x)
                  + central * spherical_jn(ell, x)
                  + upper * spherical_jn(ell + 2, x))
    np.testing.assert_allclose(
        bessel_second_derivative(ell, x), recurrence,
        rtol=1e-11, atol=2e-15)


@pytest.mark.parametrize("ell", [2, 5, 10])
def test_derivative_projection_by_parts(ell):
    # W and W' vanish at both endpoints. Transferring both derivatives to
    # W independently checks differentiation of the whole argument k*chi.
    nodes, weights = roots_legendre(128)
    chi = 2 + nodes
    phase = np.pi * (chi - 1) / 2
    window = np.sin(phase)**4
    window_second = (np.pi / 2)**2 * (
        12 * np.sin(phase)**2 * np.cos(phase)**2 - 4 * np.sin(phase)**4)
    k = np.array([.01, .1, 1., 10.])
    x = k[:, None] * chi[None, :]
    direct = bessel_second_derivative(ell, x) @ (weights * window)
    by_parts = spherical_jn(ell, x) @ (weights * window_second) / k**2
    np.testing.assert_allclose(direct, by_parts, rtol=2e-10, atol=2e-14)
