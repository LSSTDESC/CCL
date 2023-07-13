import numpy as np
import pyccl as ccl
import pytest

CEMU_TOL = 1E-3


@pytest.mark.parametrize("kind", ["tot", "cb"])
def test_cemu(kind):
    cemu = ccl.CosmicemuMTIVPk(kind)
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.67,
        sigma8=0.8,
        n_s=0.96,
        m_nu=0.0,
        w0=-1.0,
        wa=0.0,
        matter_power_spectrum=cemu)

    zs = np.array([0.0, 1.0])
    for iz, a in enumerate(1/(1+zs)):
        k, pk = np.loadtxt(f"./benchmarks/data/cosmo1_{kind}_{iz}.txt",
                           unpack=True)
        pkh = cosmo.nonlin_matter_power(k, a)
        assert np.allclose(pk, pkh, atol=0, rtol=CEMU_TOL)

        ktest, pktest = cemu.get_pk_at_a(cosmo, a)
        pkh = np.exp(np.interp(np.log(k), np.log(ktest), np.log(pktest)))
        assert np.allclose(pk, pkh, atol=0, rtol=CEMU_TOL)
