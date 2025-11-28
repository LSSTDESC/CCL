"""Benchmark tests for the Baccoemu linear transfer function."""

from pathlib import Path

import numpy as np

import pyccl as ccl

DATA_PATH = Path(__file__).parent / "data" / "baccoemu_linear.txt"
DATA = np.loadtxt(DATA_PATH)

BEMULIN_TOLERANCE = 1e-3


def _check_baccoemu_linear(cosmo: ccl.Cosmology, bemu: ccl.BaccoemuLinear) -> None:
    """Checks that BACCOemu baryons match the reference fk(k)."""
    k = DATA[:, 0] * cosmo["h"]
    pk = DATA[:, 1] / cosmo["h"] ** 3
    a = 1.0

    # 1) Cosmology linear power vs reference
    linpk = cosmo.get_linear_power()
    err = np.abs(pk / linpk(k, a) - 1.0)
    np.testing.assert_allclose(err, 0.0, atol=BEMULIN_TOLERANCE, rtol=0.0)

    # 2) Emulator helper vs reference
    ktest, pktest = bemu.get_pk_at_a(cosmo, a)
    pktest = np.exp(np.interp(np.log(k), np.log(ktest), np.log(pktest)))
    err = np.abs(pktest / pk - 1.0)
    np.testing.assert_allclose(err, 0.0, atol=BEMULIN_TOLERANCE, rtol=0.0)


def test_baccoemu_linear_sigma8():
    """Checks that BACCOemu linear T(k) matches the sigma8 reference."""
    bemu = ccl.BaccoemuLinear()
    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.05,
        h=0.67,
        sigma8=0.83,
        n_s=0.96,
        Neff=3.046,
        mass_split="normal",
        m_nu=0.1,
        Omega_g=0.0,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
        transfer_function=bemu,
    )

    _check_baccoemu_linear(cosmo, bemu)


def test_baccoemu_linear_A_s():
    """Checks that BACCOemu linear T(k) matches the A_s reference."""
    bemu = ccl.BaccoemuLinear()
    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.05,
        h=0.67,
        A_s=2.2194e-9,
        n_s=0.96,
        Neff=3.046,
        mass_split="normal",
        m_nu=0.1,
        Omega_g=0.0,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
        transfer_function=bemu,
    )

    _check_baccoemu_linear(cosmo, bemu)
