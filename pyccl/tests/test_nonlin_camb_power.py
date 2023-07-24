import numpy as np
import pytest
import pyccl as ccl


def test_nonlin_camb_power():
    import camb

    logT_AGN = 7.93
    Omega_c = 0.25
    Omega_b = 0.05
    A_s = 2.1e-9
    n_s = 0.97
    h = 0.7
    # Needs to be set for good agreements between CCL and CAMB
    T_CMB = 2.7255

    p = camb.CAMBparams(WantTransfer=True,
                        NonLinearModel=camb.nonlinear.Halofit(
                            halofit_version="mead2020_feedback",
                            HMCode_logT_AGN=logT_AGN))
    # This affects k_min
    p.WantCls = False
    p.DoLensing = False
    p.Want_CMB = False
    p.Want_CMB_lensing = False
    p.Want_cl_2D_array = False
    p.set_cosmology(H0=h*100, omch2=Omega_c*h**2, ombh2=Omega_b*h**2,
                    mnu=0.0, TCMB=T_CMB)
    p.share_delta_neff = False
    p.InitPower.set_params(As=A_s, ns=n_s)

    z = [0.0, 0.5, 1.0, 1.5]
    p.set_matter_power(redshifts=z, kmax=10.0, nonlinear=True)
    p.set_for_lmax(5000)

    r = camb.get_results(p)

    k, z, pk_nonlin_camb = r.get_nonlinear_matter_power_spectrum(
        hubble_units=False, k_hunit=False)

    ccl_cosmo = ccl.Cosmology(
        Omega_c=Omega_c, Omega_b=Omega_b, h=h, m_nu=0.0,
        A_s=A_s, n_s=n_s,
        transfer_function="boltzmann_camb",
        matter_power_spectrum="camb",
        extra_parameters={"camb": {"halofit_version": "mead2020_feedback",
                                   "HMCode_logT_AGN": logT_AGN}})

    for z_, pk_camb in zip(z, pk_nonlin_camb):
        pk_nonlin_ccl = ccl.nonlin_matter_power(ccl_cosmo, k, 1/(1+z_))

        assert np.allclose(pk_camb, pk_nonlin_ccl, rtol=3e-5)


def test_nonlin_camb_power_raises():
    # Test that it raises when (trf, mps) == (no camb, camb).
    with pytest.raises(ccl.CCLError):
        ccl.CosmologyVanillaLCDM(transfer_function="boltzmann_class",
                                 matter_power_spectrum="camb")
