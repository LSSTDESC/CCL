import numpy as np
import pyccl as ccl
import pytest  # pyright: ignore[reportMissingImports]
from typing import Callable, cast

SPK_TOLERANCE = 1e-5
SPK_A = 0.8


def test_spk_power_law_boost_against_reference():
    pytest.importorskip("pyspk")
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        Neff=3.046,
        mass_split="normal",
        m_nu=0.0,
        Omega_g=0,
        Omega_k=0,
        w0=-1,
        wa=0,
        transfer_function="bbks",
    )

    data = np.loadtxt("./benchmarks/data/spk_power_law_fk.txt")
    h = cast(float, cosmo["h"])
    k = data[:, 0] * h
    fk_ref = data[:, 1]

    baryons = ccl.BaryonsSPK(
        SO=200,
        relation_kind="power_law",
        fb_a=0.4,
        fb_pow=0.3,
        fb_pivot=10**13.5,
        k_min_hmpc=1e-2,
        k_max_hmpc=2.0,
        n_k=k.size,
        out_of_bounds_policy="error",
    )

    fk = baryons.boost_factor(cosmo, k, SPK_A)
    err = np.abs(fk_ref / fk - 1)
    assert np.allclose(err, 0, atol=SPK_TOLERANCE, rtol=0)


def test_spk_power_law_include_matches_boost():
    pytest.importorskip("pyspk")
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        Neff=3.046,
        mass_split="normal",
        m_nu=0.0,
        Omega_g=0,
        Omega_k=0,
        w0=-1,
        wa=0,
        transfer_function="bbks",
    )

    h = cast(float, cosmo["h"])
    k = np.geomspace(1e-2, 1.0, 128) * h

    baryons = ccl.BaryonsSPK(
        SO=200,
        relation_kind="power_law",
        fb_a=0.4,
        fb_pow=0.3,
        fb_pivot=10**13.5,
        k_min_hmpc=1e-2,
        k_max_hmpc=2.0,
        n_k=k.size,
        out_of_bounds_policy="error",
    )

    pk_nobar = ccl.nonlin_matter_power(cosmo, k, SPK_A)
    fk = baryons.boost_factor(cosmo, k, SPK_A)
    pkb = baryons.include_baryonic_effects(cosmo, cosmo.get_nonlin_power())
    pkb_eval = cast(Callable[[np.ndarray, float], np.ndarray], pkb)
    pk_wbar = pkb_eval(k, SPK_A)

    err = np.abs(pk_wbar / (pk_nobar * fk) - 1)
    assert np.allclose(err, 0, atol=SPK_TOLERANCE, rtol=0)
