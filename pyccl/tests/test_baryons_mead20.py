"""Tests for HMCode/Mead2020 nonlinear P(k) with baryonic feedback."""

import pytest

import numpy as np

import pyccl as ccl


camb = pytest.importorskip("camb")

TOL_MEAD20 = 1e-5


def _compute_camb_mead2020_pk(
    logT_AGN=7.93,
    Omega_c=0.25,
    Omega_b=0.05,
    A_s=2.1e-9,
    n_s=0.97,
    h=0.7,
    T_CMB=2.7255,
):
    """Returns k, z, P_nl(k,z) from CAMB with Mead2020 baryons."""
    p = camb.CAMBparams(
        WantTransfer=True,
        NonLinearModel=camb.nonlinear.Halofit(
            halofit_version="mead2020_feedback",
            HMCode_logT_AGN=logT_AGN,
        ),
    )
    # This affects k_min
    p.WantCls = False
    p.DoLensing = False
    p.Want_CMB = False
    p.Want_CMB_lensing = False
    p.Want_cl_2D_array = False

    p.set_cosmology(
        H0=h * 100,
        omch2=Omega_c * h**2,
        ombh2=Omega_b * h**2,
        mnu=0.0,
        TCMB=T_CMB,
    )
    p.share_delta_neff = False
    p.InitPower.set_params(As=A_s, ns=n_s)

    zs = [0.0, 0.5, 1.0, 1.5]
    p.set_matter_power(redshifts=zs, kmax=10.0, nonlinear=True)
    p.set_for_lmax(5000)

    results = camb.get_results(p)
    k, z_out, pk_nl = results.get_nonlinear_matter_power_spectrum(
        hubble_units=False, k_hunit=False
    )
    return k, z_out, pk_nl


def _make_ccl_mead2020_cosmo(
    logT_AGN=7.93,
    Omega_c=0.25,
    Omega_b=0.05,
    A_s=2.1e-9,
    n_s=0.97,
    h=0.7,
):
    """Returns a CCL Cosmology using CAMB+Mead2020_feedback+HMCode."""
    extras = {
        "camb": {
            "halofit_version": "mead2020_feedback",
            "HMCode_logT_AGN": logT_AGN,
        }
    }
    return ccl.Cosmology(
        Omega_c=Omega_c,
        Omega_b=Omega_b,
        h=h,
        m_nu=0.0,
        A_s=A_s,
        n_s=n_s,
        transfer_function="boltzmann_camb",
        matter_power_spectrum="camb",
        extra_parameters=extras,
    )


def test_baryons_mead20_hmcode_matches_camb():
    """Tests that CCL's CAMB HMCode Mead2020 matches CAMB's output."""
    k, zs, pk_camb_all = _compute_camb_mead2020_pk()
    cosmo_ccl = _make_ccl_mead2020_cosmo()

    for z_val, pk_camb in zip(zs, pk_camb_all):
        a = 1.0 / (1.0 + z_val)
        pk_ccl = ccl.nonlin_matter_power(cosmo_ccl, k, a)
        # restrict to a safe range in case CAMB changes behavior at very low or
        # high k
        sel = (k >= 1e-3) & (k <= 5.0)
        np.testing.assert_allclose(pk_camb[sel], pk_ccl[sel], rtol=TOL_MEAD20)


def test_baryons_mead20_pk_responds_to_logt_agn():
    """Tests that changing logT_AGN changes the matter power spectrum."""
    cosmo_lo = _make_ccl_mead2020_cosmo(logT_AGN=7.0)
    cosmo_hi = _make_ccl_mead2020_cosmo(logT_AGN=8.3)

    a = 1.0
    k = np.logspace(-2, 1, 50)  # 0.01–10 1/Mpc

    pk_lo = ccl.nonlin_matter_power(cosmo_lo, k, a)
    pk_hi = ccl.nonlin_matter_power(cosmo_hi, k, a)

    ratio = pk_hi / pk_lo

    assert np.all(np.isfinite(ratio))
    assert np.any(np.abs(ratio - 1.0) > 1e-3)


def test_baryons_mead20_cls_finite():
    """Tests that angular C_ells can be computed with Mead2020 baryons."""
    cosmo = _make_ccl_mead2020_cosmo()
    z = np.linspace(0.0, 2.0, 50)
    nz = z**2 * np.exp(-0.5 * ((z - 1.0) / 0.3) ** 2)
    tr = ccl.WeakLensingTracer(cosmo, dndz=(z, nz))
    ells = np.geomspace(10, 1000, 20)
    cl = ccl.angular_cl(cosmo, tr, tr, ells)
    assert np.all(np.isfinite(cl))


def test_baryons_mead20_invalid_combo_raises():
    """Tests that invalid (trf, mps) combo raises CCLError."""
    with pytest.raises(ccl.CCLError):
        ccl.CosmologyVanillaLCDM(
            transfer_function="boltzmann_class",
            matter_power_spectrum="camb",
        )
