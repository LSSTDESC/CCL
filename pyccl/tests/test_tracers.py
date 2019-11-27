import numpy as np
import pytest
import pyccl as ccl

COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')


def get_tracer(tracer_type):
    z = np.linspace(0., 1., 200)
    n = np.exp(-((z-0.5)/0.1)**2)
    b = np.sqrt(1. + z)

    if tracer_type == 'nc':
        ntr = 3
        tr = ccl.NumberCountsTracer(COSMO, True,
                                    dndz=(z, n),
                                    bias=(z, b),
                                    mag_bias=(z, b))
    elif tracer_type == 'wl':
        ntr = 2
        tr = ccl.WeakLensingTracer(COSMO,
                                   dndz=(z, n),
                                   ia_bias=(z, b))
    elif tracer_type == 'cl':
        ntr = 1
        tr = ccl.CMBLensingTracer(COSMO, 1100.)
    else:
        ntr = 0
        tr = ccl.Tracer()
    return tr, ntr


@pytest.mark.parametrize('tracer_type', ['nc', 'wl', 'cl', 'not'])
def test_tracer_kernel_smoke(tracer_type):
    tr, ntr = get_tracer(tracer_type)
    for chi in [np.linspace(0., 3000., 128),
                [100., 1000.],
                100.]:
        w = tr.get_kernel(chi)

        assert w.shape[0] == ntr
        if ntr > 0:
            for ww in w:
                assert np.shape(ww) == np.shape(chi)


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
