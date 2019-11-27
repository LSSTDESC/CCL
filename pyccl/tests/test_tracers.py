import numpy as np
import pytest
import pyccl as ccl

COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')


@pytest.mark.parametrize('tracer_type', ['nc', 'wl', 'cl', 'not'])
def test_tracer_kernel_smoke(tracer_type):
    z = np.linspace(0., 1., 200)
    n = np.exp(-((z-0.5)/0.1)**2)
    b = np.sqrt(1. + z)
    chi = np.linspace(0., 3000., 128)

    if tracer_type == 'nc':
        shap = (3, 128)
        tr = ccl.NumberCountsTracer(COSMO, True,
                                    dndz=(z, n),
                                    bias=(z, b),
                                    mag_bias=(z, b))
    elif tracer_type == 'wl':
        shap = (2, 128)
        tr = ccl.WeakLensingTracer(COSMO,
                                   dndz=(z, n),
                                   ia_bias=(z, b))
    elif tracer_type == 'cl':
        shap = (1, 128)
        tr = ccl.CMBLensingTracer(COSMO, 1100.)
    else:
        shap = (0, )
        tr = ccl.Tracer()

    w = tr.get_kernel(chi)
    assert w.shape == shap
