import numpy as np
import pyccl as ccl
import pytest

# Set tolerances
BOOST_TOLERANCE = 1e-5

# Set up the cosmological parameters to be used
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67,
                      sigma8=0.8, n_s=0.96,
                      transfer_function='eisenstein_hu')
k = 0.5*cosmo['h']
a = 1.

# Set up power without baryons
pk2D_no_baryons = cosmo.get_nonlin_power()

fbarcvec = np.linspace(0.25, 1, 20)
mdef_vec = ['500c', '200c']


def compare_boost():
    vdboost = []
    vdboost_expect = []
    pk_nl = pk2D_no_baryons(k, a)
    for mdef in mdef_vec:
        for f in fbarcvec:
            # Takes ftilde as argument
            vD19 = ccl.BaryonsvanDaalen19(fbar=f, mass_def=mdef)
            pk2D_with_baryons = vD19.include_baryonic_effects(
                cosmo, pk2D_no_baryons)
            # Takes k in units of 1/Mpc as argument
            pk_nl_bar = pk2D_with_baryons(k, a)
            vdboost.append(pk_nl_bar/pk_nl-1)
            if mdef == '500c':
                vdboost_expect.append(-np.exp(-5.99*f-0.5107))
            else:
                vdboost_expect.append(-np.exp(-5.816*f-0.4005))
    assert np.allclose(
        vdboost, vdboost_expect, atol=1e-5, rtol=BOOST_TOLERANCE)


def test_boost_model():
    compare_boost()


def test_baryons_from_name():
    baryons = ccl.BaryonsvanDaalen19(fbar=0.7, mass_def='500c')
    bar2 = ccl.Baryons.from_name('vanDaalen19')
    assert baryons.name == bar2.name
    assert baryons.name == 'vanDaalen19'


def test_baryons_vd19_raises():
    with pytest.raises(ValueError):
        ccl.BaryonsvanDaalen19(fbar=0.7, mass_def='blah')
    b = ccl.BaryonsvanDaalen19(fbar=0.7, mass_def='500c')
    with pytest.raises(ValueError):
        b.update_parameters(mass_def='blah')


def test_update_params():
    b = ccl.BaryonsvanDaalen19(fbar=0.7, mass_def='500c')
    b.update_parameters(fbar=0.6, mass_def='200c')
    assert b.mass_def == '200c'
    assert b.fbar == 0.6
