import numpy as np
import pytest
import pyccl as ccl


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
PKA = ccl.Pk2D(lambda k, a: np.log(a/k), cosmo=COSMO)
HMFS_FOF = [ccl.halos.MassFuncPress74,
            ccl.halos.MassFuncSheth99,
            ccl.halos.MassFuncJenkins01,
            ccl.halos.MassFuncAngulo12]
HMFS_SO = [ccl.halos.MassFuncTinker08,
           ccl.halos.MassFuncTinker10,
           ccl.halos.MassFuncWatson13,
           ccl.halos.MassFuncDespali16,
           ccl.halos.MassFuncBocquet16]
HMFS = HMFS_FOF + HMFS_SO
MS = [1E13, [1E12, 1E15], np.array([1E12, 1E15])]
MFOF = ccl.halos.MassDef('fof', 'matter')
MVIR = ccl.halos.MassDef('vir', 'critical')
MDFS = [MFOF, MFOF, MVIR, MFOF, MFOF]


@pytest.mark.parametrize('nM_class', HMFS)
def test_nM_subclasses_smoke(nM_class):
    nM = nM_class(COSMO)
    for m in MS:
        n = nM.get_mass_function(COSMO, m, 0.9)
        assert np.all(np.isfinite(n))


@pytest.mark.parametrize('nM_pair', zip(HMFS_SO, MDFS))
def test_nM_mdef_raises(nM_pair):
    nM_class, mdef = nM_pair
    with pytest.raises(ValueError):
        nM_class(COSMO, mdef)
