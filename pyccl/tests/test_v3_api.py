import pyccl as ccl
import pytest

# TODO: Remove for CCLv3.


def test_unexpected_argument_raises():
    # Test that if an argument has been renamed it will still raise when
    # a wrong argument is passed.
    with pytest.raises(TypeError):
        # here, `c_m` was renamed to `concentration`
        ccl.halos.MassDef200c(hello="Duffy08")


def test_mass_def_api():
    # Check API preserved in functions where `mass_def` is deprecated.
    cosmo = ccl.CosmologyVanillaLCDM(transfer_function="bbks")
    mdef = ccl.halos.MassDef200c()
    prof0 = ccl.halos.HaloProfilePressureGNFW(mass_def=mdef)
    res0 = prof0.real(cosmo, 1, 1e14, 1)

    with pytest.warns(ccl.CCLDeprecationWarning):
        prof1 = ccl.halos.HaloProfilePressureGNFW()
    with pytest.warns(ccl.CCLDeprecationWarning):
        res1 = prof1.real(cosmo, 1, 1e14, 1, mdef)
    with pytest.warns(ccl.CCLDeprecationWarning):
        res2 = prof1.real(cosmo, 1, 1e14, 1, mass_def=mdef)
    assert res0 == res1 == res2
