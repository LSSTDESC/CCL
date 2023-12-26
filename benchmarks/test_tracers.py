import numpy as np
import pyccl as ccl
import pytest


def get_prediction(ells, chi_i, chi_f, alpha, beta, gamma,
                   der_bessel, der_angles):
    exponent = 2 * gamma - alpha - 2 * beta - 1
    fl = np.ones_like(ells)
    if der_bessel == -1:
        fl *= 1. / (ells + 0.5)**4
    if der_angles == 2:
        fl *= (ells + 2.) * (ells + 1.) * ells * (ells - 1)
    elif der_angles == 1:
        fl *= ((ells + 1.) * ells)**2
    lfac = fl * (ells + 0.5)**(alpha + 2 * beta)
    norm = (chi_f**exponent-chi_i**exponent) / exponent
    return lfac * norm


@pytest.fixture(scope='module')
def set_up():
    ccl.gsl_params.INTEGRATION_LIMBER_EPSREL = 1E-4
    ccl.gsl_params.INTEGRATION_EPSREL = 1E-4
    cosmo = ccl.Cosmology(Omega_c=0.30, Omega_b=0.00, Omega_g=0, Omega_k=0,
                          h=0.7, sigma8=0.8, n_s=0.96, Neff=0, m_nu=0.0,
                          w0=-1, wa=0, T_CMB=2.7, transfer_function='bbks',
                          matter_power_spectrum='linear')

    ccl.gsl_params.reload()
    return cosmo


@pytest.mark.parametrize("alpha,beta,gamma,is_factorizable,"
                         "w_transfer,mismatch,der_bessel,der_angles",
                         # non-factorizable
                         [(-2., -1., -1., False, True, False, 0, 0),
                          (-2., 0., -1., False, True, False, 0, 0),
                          (-2., -1., 0., False, True, False, 0, 0),
                          (-2., 0., 0., False, True, False, 0, 0),
                          (0., 0., 0., False, True, False, 0, 0),
                          # factorizable transfer functions
                          (-2., -1., -1., True, True, False, 0, 0),
                          (-2., 0., -1., True, True, False, 0, 0),
                          (-2., -1., 0., True, True, False, 0, 0),
                          (-2., 0., 0., True, True, False, 0, 0),
                          (0., 0., 0., True, True, False, 0, 0),
                          # Unit transfer functions
                          (-2., 0., 0., False, False, False, 0, 0),
                          (0., 0., 0., False, False, False, 0, 0),
                          # Mismatch in kernel-transfer coverage
                          (-2., -1., 0., True, True, True, 0, 0),
                          (-2., -1., 0., False, True, True, 0, 0),
                          # non-zero der_bessel
                          (-2., -1., -1., False, True, False, -1, 0),
                          # non-zero der_angles
                          (-2., -1., -1., False, True, False, 0, 1),
                          (-2., -1., -1., False, True, False, 0, 2),
                          (-2., -1., -1., False, True, False, -1, 2)])
def test_tracers_analytic(set_up, alpha, beta, gamma,
                          is_factorizable, w_transfer,
                          mismatch, der_bessel, der_angles):
    cosmo = set_up
    zmax = 0.8
    zi = 0.4
    zf = 0.6
    nchi = 1024
    nk = 512

    # x arrays
    chii = ccl.comoving_radial_distance(cosmo, 1. / (1 + zi))
    chif = ccl.comoving_radial_distance(cosmo, 1. / (1 + zf))
    chimax = ccl.comoving_radial_distance(cosmo, 1. / (1 + zmax))
    chiarr = np.linspace(0.1, chimax, nchi)
    if mismatch:
        # Remove elements around the edges
        mask = (chiarr < 0.9 * chif) & (chiarr > 1.1 * chii)
        chiarr_transfer = chiarr[mask]
    else:
        chiarr_transfer = chiarr
    aarr_transfer = ccl.scale_factor_of_chi(cosmo, chiarr_transfer)[::-1]
    aarr = ccl.scale_factor_of_chi(cosmo, chiarr)[::-1]
    lkarr = np.log(10.**np.linspace(-6, 3, nk))

    # Kernel
    wchi = np.ones_like(chiarr)
    wchi[chiarr < chii] = 0
    wchi[chiarr > chif] = 0

    # Transfer
    t = ccl.Tracer()
    if w_transfer:
        ta = (chiarr_transfer**gamma)[::-1]
        tk = np.exp(beta*lkarr)
        if is_factorizable:
            # 1D
            t.add_tracer(cosmo,
                         kernel=(chiarr, wchi),
                         transfer_k=(lkarr, tk),
                         transfer_a=(aarr_transfer, ta),
                         der_bessel=der_bessel,
                         der_angles=der_angles)
        else:
            # 2D
            tka = ta[:, None] * tk[None, :]
            t.add_tracer(cosmo,
                         kernel=(chiarr, wchi),
                         transfer_ka=(aarr_transfer, lkarr, tka),
                         der_bessel=der_bessel,
                         der_angles=der_angles)
    else:
        t.add_tracer(cosmo,
                     kernel=(chiarr, wchi),
                     der_bessel=der_bessel,
                     der_angles=der_angles)

    # Power spectrum
    pkarr = np.ones_like(aarr)[:, None] * (np.exp(alpha * lkarr))[None, :]
    pk2d = ccl.Pk2D(a_arr=aarr, lk_arr=lkarr, pk_arr=pkarr, is_logp=False)

    # C_ells
    larr = np.linspace(2, 3000, 100)
    # 1D
    cl = ccl.angular_cl(cosmo, t, t, larr, p_of_k_a=pk2d)
    # Prediction
    clpred = get_prediction(larr, chii, chif,
                            alpha, beta, gamma,
                            der_bessel, der_angles)

    assert np.all(np.fabs(cl / clpred - 1) < 5E-3)
