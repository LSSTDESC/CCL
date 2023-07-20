import numpy as np
import pytest
import pyccl as ccl
from .test_cclobject import check_eq_repr_hash


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
CONCS = [ccl.halos.ConcentrationDiemer15,
         ccl.halos.ConcentrationBhattacharya13,
         ccl.halos.ConcentrationPrada12,
         ccl.halos.ConcentrationKlypin11,
         ccl.halos.ConcentrationDuffy08,
         ccl.halos.ConcentrationIshiyama21,
         ccl.halos.ConcentrationConstant]
MS = [1E13, [1E12, 1E15], np.array([1E12, 1E15])]
# None of our concentration-mass relations
# are defined for FoF halos, or 400 critical.
MDEF = ccl.halos.MassDefFof
M400 = ccl.halos.MassDef(400, 'critical')


def test_Concentration_eq_repr_hash():
    # Test eq, repr, hash for Concentration.
    CM1 = ccl.halos.Concentration.from_name("Duffy08")()
    CM2 = ccl.halos.ConcentrationDuffy08()
    assert check_eq_repr_hash(CM1.mass_def, CM2.mass_def)
    assert check_eq_repr_hash(CM1, CM2)

    M200m = ccl.halos.MassDef200m
    CM3 = ccl.halos.ConcentrationDuffy08(mass_def=M200m)
    assert check_eq_repr_hash(CM1.mass_def, CM3.mass_def, equal=False)
    assert check_eq_repr_hash(CM1, CM3, equal=False)


@pytest.mark.parametrize('cM_class', CONCS)
def test_cM_subclasses_smoke(cM_class):
    cM = cM_class()
    for m in MS:
        c = cM(COSMO, m, 0.9)
        assert np.all(np.isfinite(c))
        assert np.shape(c) == np.shape(m)


def test_cM_duffy_smoke():
    md = ccl.halos.MassDef('vir', 'critical')
    cM = ccl.halos.ConcentrationDuffy08(mass_def=md)
    for m in MS:
        c = cM(COSMO, m, 0.9)
        assert np.all(np.isfinite(c))
        assert np.shape(c) == np.shape(m)


@pytest.mark.parametrize('cM_class', CONCS[:-1])
def test_cM_mdef_raises(cM_class):
    # testing strings
    with pytest.raises(ValueError):
        cM_class(mass_def=MDEF)
    # testing numbers
    with pytest.raises(ValueError):
        cM_class(mass_def=M400)

    if cM_class.name == "Ishiyama21":
        with pytest.raises(ValueError):
            cM_class(mass_def="500c", Vmax=True)

        with pytest.raises(ValueError):
            cM_class(mass_def="200m")


@pytest.mark.parametrize('name', ['Duffy08', 'Diemer15'])
def test_cM_from_string(name):
    cM_class = ccl.halos.Concentration.from_name(name)
    assert cM_class == ccl.halos.Concentration.from_name(name)
    cM = cM_class()
    for m in MS:
        c = cM(COSMO, m, 0.9)
        assert np.all(np.isfinite(c))
        assert np.shape(c) == np.shape(m)


def test_cM_from_string_raises():
    with pytest.raises(KeyError):
        ccl.halos.Concentration.from_name('Duffy09')
