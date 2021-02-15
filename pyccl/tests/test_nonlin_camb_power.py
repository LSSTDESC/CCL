import numpy as np

import pyccl as ccl


def test_nonlin_camb_power():
    import camb

    logT_AGN = 7.93
    Omega_c = 0.25
    Omega_b = 0.05
    A_s = 2.1e-9
    n_s = 0.97
    h = 0.7

    p = camb.CAMBparams(WantTransfer=True,
                        NonLinearModel=camb.nonlinear.Halofit(
                            halofit_version="mead2020",
                            HMCode_logT_AGN=logT_AGN))
    p.set_cosmology(H0=h*100, omch2=Omega_c*h**2, ombh2=Omega_b*h**2, mnu=0.0)
    p.share_delta_neff = False
    p.InitPower.set_params(
        As=A_s,
        ns=n_s)
    z = [0.0]
    p.set_matter_power(redshifts=z, kmax=10.0, nonlinear=True)
    p.set_for_lmax(5000)

    r = camb.get_results(p)

    k, z, pk_nonlin_camb = r.get_nonlinear_matter_power_spectrum(
        hubble_units=False, k_hunit=False)

    ccl_cosmo = ccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h, m_nu=0.0,
                              A_s=A_s, n_s=n_s,
                              transfer_function="boltzmann_camb",
                              matter_power_spectrum="camb",
                              extra_parameters={"camb":
                                                {"halofit_version": "mead2020",
                                                 "HMCode_logT_AGN": logT_AGN}})
    pk_nonlin_ccl = ccl.nonlin_matter_power(ccl_cosmo, k, 1.0)

    # import matplotlib.pyplot as plt

    # plt.loglog(k, pk_nonlin_camb[0])
    # plt.loglog(k, pk_nonlin_ccl)
    # plt.semilogx(k, pk_nonlin_ccl/pk_nonlin_camb[0]-1)
    # plt.show()
    assert np.allclose(pk_nonlin_camb, pk_nonlin_ccl, rtol=3e-3)


# if __name__ == "__main__":
#     test_nonlin_camb_power()
