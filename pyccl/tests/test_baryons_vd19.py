import numpy as np
import pyccl as ccl
import pytest

# Set tolerances
BOOST_TOLERANCE = 1e-5

# Set up the cosmological parameters to be used 
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96, 
                      transfer_function='boltzmann_class')
k=0.5*cosmo['h']
a=1.

# Set up power without baryons
pk2D_no_baryons = cosmo.get_nonlin_power()

fbarcvec=np.linspace(0.25,1,20)
baryons = ccl.BaryonsvanDaalen19(fbar500c=0.7)
pk2D_with_baryons = baryons.include_baryonic_effects(cosmo, pk2D_no_baryons)

def compare_boost():
    
    vdboost=[]
    vdboost_expect=[]
    pk_nl = pk2D_no_baryons(k, a)
    for f in fbarcvec:
        baryons = ccl.BaryonsvanDaalen19(fbar500c=f) #takes ftilde as argument
        pk2D_with_baryons = baryons.include_baryonic_effects(cosmo, pk2D_no_baryons)
        pk_nl_bar = pk2D_with_baryons(k,a) #takes k in units of 1/Mpc as argument
        vdboost.append(pk_nl_bar/pk_nl-1)
        vdboost_expect.append(-np.exp(-5.99*f-0.5107))
    assert np.allclose(
        vdboost,vdboost_expect, atol=1e-5, rtol=BOOST_TOLERANCE)


def test_boost_model():
    compare_boost()


def test_baryons_from_name():
    bar2 = ccl.Baryons.from_name('vanDaalen19')
    assert baryons.name == bar2.name
    assert baryons.name == 'vanDaalen19'
