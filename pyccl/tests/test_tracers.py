import numpy as np
import pytest
from . import pyccl as ccl
from . import CCLWarning

COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')


def dndz(z):
    return np.exp(-((z-0.5)/0.1)**2)


def get_tracer(tracer_type, cosmo=None, **tracer_kwargs):
    if cosmo is None:
        cosmo = COSMO
    z = np.linspace(0., 1., 2000)
    n = dndz(z)
    b = np.sqrt(1. + z)

    if tracer_type == 'nc':
        ntr = 3
        tr = ccl.NumberCountsTracer(cosmo, has_rsd=True,
                                    dndz=(z, n),
                                    bias=(z, b),
                                    mag_bias=(z, b),
                                    **tracer_kwargs)
    elif tracer_type == 'wl':
        ntr = 2
        tr = ccl.WeakLensingTracer(cosmo,
                                   dndz=(z, n),
                                   ia_bias=(z, b),
                                   **tracer_kwargs)
    elif tracer_type == 'cl':
        ntr = 1
        tr = ccl.CMBLensingTracer(cosmo, z_source=1100., **tracer_kwargs)
    else:
        ntr = 0
        tr = ccl.Tracer(**tracer_kwargs)
    return tr, ntr


@pytest.mark.parametrize('tracer_type', ['nc', 'wl'])
def test_tracer_dndz_smoke(tracer_type):
    tr, _ = get_tracer(tracer_type)
    for z in [np.linspace(0.5, 0.6, 10),
              0.5]:
        n1 = dndz(z)
        n2 = tr.get_dndz(z)
        assert np.all(np.fabs(n1 / n2 - 1) < 1E-5)


@pytest.mark.parametrize('tracer_type', ['cl', 'not'])
def test_tracer_dndz_errors(tracer_type):
    tr, _ = get_tracer(tracer_type)
    with pytest.raises(NotImplementedError):
        tr.get_dndz(0.5)


@pytest.mark.parametrize('tracer_type', ['nc', 'wl', 'cl', 'not'])
def test_tracer_kernel_smoke(tracer_type):
    tr, ntr = get_tracer(tracer_type)
    for chi in [np.linspace(0., 3000., 128),
                [100., 1000.],
                100.]:
        w = tr.get_kernel(chi=chi)

        assert w.shape[0] == ntr
        if ntr > 0:
            for ww in w:
                assert np.shape(ww) == np.shape(chi)


@pytest.mark.parametrize('tracer_type', ['nc', 'wl', 'cl', 'not'])
def test_tracer_der_bessel_smoke(tracer_type):
    tr, ntr = get_tracer(tracer_type)

    if tracer_type == 'nc':
        dd = np.array([0, 2, -1])
    elif tracer_type == 'wl':
        dd = np.array([-1, -1])
    elif tracer_type == 'cl':
        dd = np.array([-1])
    else:
        dd = np.array([])

    d = tr.get_bessel_derivative()
    assert np.all(d == dd)


@pytest.mark.parametrize('tracer_type', ['nc', 'wl', 'cl', 'not'])
def test_tracer_f_ell_smoke(tracer_type):
    tr, ntr = get_tracer(tracer_type)
    for ell in [np.linspace(0., 3000., 128),
                [100., 1000.],
                100.]:
        fl = tr.get_f_ell(ell)

        assert fl.shape[0] == ntr
        if ntr > 0:
            for f in fl:
                assert np.shape(f) == np.shape(ell)


@pytest.mark.parametrize('tracer_type', ['nc', 'wl', 'cl', 'not'])
def test_tracer_transfer_smoke(tracer_type):
    tr, ntr = get_tracer(tracer_type)
    for lk in [np.linspace(-3., 1., 10),
               [-2., 0.],
               -1.]:
        for a in [np.linspace(0.5, 1., 8),
                  [0.4, 1.],
                  0.9]:
            tf = tr.get_transfer(lk, a)

            assert tf.shape[0] == ntr
            if ntr > 0:
                if np.ndim(a) == 0:
                    if np.ndim(lk) == 0:
                        shap = (ntr, )
                    else:
                        shap = (ntr, len(lk))
                else:
                    if np.ndim(lk) == 0:
                        shap = (ntr, len(a))
                    else:
                        shap = (ntr, len(lk), len(a))
            else:
                shap = (0, )
            assert tf.shape == shap


def test_tracer_nz_support():
    z_max = 1.0
    a = np.linspace(1/(1+z_max), 1.0, 100)

    background_def = {"a": a,
                      "chi": ccl.comoving_radial_distance(COSMO, a),
                      "h_over_h0": ccl.h_over_h0(COSMO, a)}

    calculator_cosmo = ccl.CosmologyCalculator(
        Omega_c=0.27, Omega_b=0.045, h=0.67,
        sigma8=0.8, n_s=0.96,
        background=background_def)

    z = np.linspace(0., 2., 2000)
    n = dndz(z)

    with pytest.raises(ValueError):
        _ = ccl.WeakLensingTracer(calculator_cosmo, dndz=(z, n))

    with pytest.raises(ValueError):
        _ = ccl.NumberCountsTracer(calculator_cosmo, has_rsd=False,
                                   dndz=(z, n), bias=(z, np.ones_like(z)))

    with pytest.raises(ValueError):
        _ = ccl.CMBLensingTracer(calculator_cosmo, z_source=2.0)


def new_simple_cosmo():
    return ccl.CosmologyVanillaLCDM(transfer_function='bbks',
                                    matter_power_spectrum="linear")


def test_tracer_nz_norm_spline_vs_gsl_intergation():
    # Create a new Cosmology object so that we're not messing with the other
    # tests
    ccl.gsl_params.NZ_NORM_SPLINE_INTEGRATION = True
    cosmo = new_simple_cosmo()
    tr_wl, _ = get_tracer("wl", cosmo)
    tr_nc, _ = get_tracer("nc", cosmo)

    w_wl_spline, _ = tr_wl.get_kernel(chi=None)
    w_nc_spline, _ = tr_nc.get_kernel(chi=None)

    ccl.gsl_params.NZ_NORM_SPLINE_INTEGRATION = False
    cosmo = new_simple_cosmo()
    tr_wl, _ = get_tracer("wl", cosmo)
    tr_nc, _ = get_tracer("nc", cosmo)

    w_wl_gsl, _ = tr_wl.get_kernel(chi=None)
    w_nc_gsl, _ = tr_nc.get_kernel(chi=None)

    for w_spline, w_gsl in zip(w_wl_spline, w_wl_gsl):
        assert np.allclose(w_spline, w_gsl, atol=0, rtol=1e-8)
    for w_spline, w_gsl in zip(w_nc_spline, w_nc_gsl):
        assert np.allclose(w_spline, w_gsl, atol=0, rtol=1e-8)

    ccl.gsl_params.reload()  # reset to the default parameters


@pytest.mark.parametrize('z_min, z_max, n_z_samples', [(0.0, 1.0, 2000),
                                                       (0.0, 1.0, 1000),
                                                       (0.0, 1.0, 500),
                                                       (0.0, 1.0, 100),
                                                       (0.3, 1.0, 1000)])
def test_tracer_lensing_kernel_spline_vs_gsl_intergation(z_min, z_max,
                                                         n_z_samples):
    # Create a new Cosmology object so that we're not messing with the other
    # tests
    z = np.linspace(z_min, z_max, n_z_samples)
    n = dndz(z)

    # Make sure case where z[0] > 0 and n[0] > 0 is tested for
    if z_min > 0:
        assert n[0] > 0

    ccl.gsl_params.LENSING_KERNEL_SPLINE_INTEGRATION = True
    cosmo = new_simple_cosmo()
    if n_z_samples >= 256:
        tr_wl = ccl.WeakLensingTracer(cosmo, dndz=(z, n))
    else:
        with pytest.warns(CCLWarning):
            tr_wl = ccl.WeakLensingTracer(cosmo, dndz=(z, n))
    w_wl_spline, _ = tr_wl.get_kernel(chi=None)

    ccl.gsl_params.LENSING_KERNEL_SPLINE_INTEGRATION = False
    cosmo = new_simple_cosmo()
    tr_wl = ccl.WeakLensingTracer(cosmo, dndz=(z, n))
    w_wl_gsl, chi = tr_wl.get_kernel(chi=None)

    # Peak of kernel is ~1e-5
    if n_z_samples >= 1000:
        assert np.allclose(w_wl_spline[0], w_wl_gsl[0], atol=1e-10, rtol=1e-9)
    else:
        assert np.allclose(w_wl_spline[0], w_wl_gsl[0], atol=5e-9, rtol=1e-5)

    ccl.gsl_params.reload()  # reset to the default parameters


def test_tracer_delta_function_nz():
    z = np.linspace(0., 1., 2000)
    z_s_idx = int(z.size*0.8)
    z_s = z[z_s_idx]
    n = np.zeros_like(z)
    n[z_s_idx] = 2.0

    tr_wl = ccl.WeakLensingTracer(COSMO, dndz=(z, n))

    # Single source plane tracer to compare against
    chi_kappa, w_kappa = ccl.tracers.get_kappa_kernel(COSMO, z_source=z_s,
                                                      n_samples=100)

    # Use the same comoving distances
    w = tr_wl.get_kernel(chi=chi_kappa)

    assert np.allclose(w[0], w_kappa, atol=1e-8, rtol=1e-6)
    # at z=z_source, interpolation becomes apparent, so for this test we
    # ignore these data points.
    assert np.allclose(w[0][:-2], w_kappa[:-2], atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize('tracer_type', ['nc', 'wl', 'cl'])
def test_tracer_n_sample_smoke(tracer_type):
    tr, ntr = get_tracer(tracer_type, n_samples=50)
    if tracer_type != "cl":
        # n_samples=None should fall back to using the samples from the n(z).
        tr, ntr = get_tracer(tracer_type, n_samples=None)


def test_tracer_n_sample_wl():
    z = np.linspace(0., 1., 2000)
    n = dndz(z)

    n_samples = 50
    tr_wl = ccl.WeakLensingTracer(COSMO, dndz=(z, n), n_samples=n_samples)
    w, chi = tr_wl.get_kernel(chi=None)

    assert w[0].shape[-1] == n_samples
    assert chi[0].shape[-1] == n_samples


def test_tracer_n_sample_warn():
    z = np.linspace(0., 1., 50)
    n = dndz(z)

    with pytest.warns(CCLWarning):
        _ = ccl.WeakLensingTracer(COSMO, dndz=(z, n))


def test_tracer_bool():
    assert bool(ccl.Tracer()) is False
    assert bool(ccl.CMBLensingTracer(COSMO, z_source=1100)) is True


def test_tracer_chi_min_max():
    # Test that it can access the C-level chi_min and chi_max.
    tr = ccl.CMBLensingTracer(COSMO, z_source=1100)
    assert tr.chi_min == tr._trc[0].chi_min
    assert tr.chi_max == tr._trc[0].chi_max

    # Raises an error if chi_min or chi_max is not the same.
    chi = np.linspace(tr.chi_min+0.05, tr.chi_max+0.05, 128)
    wchi = np.ones_like(chi)
    tr.add_tracer(COSMO, kernel=(chi, wchi))
    with pytest.raises(AttributeError):
        tr.chi_min
    with pytest.raises(AttributeError):
        tr.chi_max


def test_tracer_repr():
    """Check that the repr works as intended."""
    # Equal Tracers with a kernel.
    tr1 = ccl.CMBLensingTracer(COSMO, z_source=1100)
    tr2 = ccl.CMBLensingTracer(COSMO, z_source=1100)
    assert tr1 == tr2
    # Unequal.
    tr3 = ccl.CMBLensingTracer(COSMO, z_source=1101)
    assert tr3 != tr1
    # We add an extra tracer and check unequal.
    chi = np.linspace(tr1.chi_min, tr1.chi_max, 128)
    wchi = np.ones_like(chi)
    tr2.add_tracer(COSMO, kernel=(chi, wchi))
    assert tr2 != tr1
    # Check empty tracer.
    z = np.linspace(0, 0.5, 128)
    nz = np.ones_like(z)
    tr4 = ccl.Tracer()
    tr5 = ccl.NumberCountsTracer(COSMO, dndz=(z, nz), has_rsd=False)  # all off
    assert tr4 == tr5
    # Check tracers with transfer functions.
    # transfer_a
    tr6 = ccl.NumberCountsTracer(COSMO, dndz=(z, nz), has_rsd=True)
    tr7 = ccl.NumberCountsTracer(COSMO, dndz=(z, nz), has_rsd=True)
    assert tr6 == tr7
    # transfer_k
    lk = np.linspace(-5, 2, 8)
    t_k = np.ones_like(lk)
    tr6.add_tracer(COSMO, transfer_k=(lk, t_k))
    tr7.add_tracer(COSMO, transfer_k=(lk, t_k))
    # transfer_ka
    a = np.linspace(0.5, 1.0, 8)
    t_ka = np.ones((a.size, lk.size))
    tr6.add_tracer(COSMO, transfer_ka=(a, lk, t_ka))
    tr7.add_tracer(COSMO, transfer_ka=(a, lk, t_ka))
    assert tr6 == tr7
    # different extrap orders
    tr6.add_tracer(COSMO, transfer_ka=(a, lk, t_ka), extrap_order_lok=0)
    tr7.add_tracer(COSMO, transfer_ka=(a, lk, t_ka), extrap_order_lok=1)
    assert tr6 != tr7
