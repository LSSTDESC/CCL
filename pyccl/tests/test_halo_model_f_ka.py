import numpy as np
import pyccl as ccl
from pyccl.halos import halomod_power_spectrum

# cosmology
cosmo = ccl.CosmologyVanillaLCDM()
# halo model calculator
hmd = ccl.halos.MassDef(200, "matter")
nM = ccl.halos.mass_function_from_name("Tinker08")(cosmo, mass_def=hmd)
bM = ccl.halos.halo_bias_from_name("Tinker10")(cosmo, mass_def=hmd)
hmc = ccl.halos.HMCalculator(cosmo, nM, bM, hmd)
# k and a
k = np.logspace(-3, 2, 256)
a = 1.
# profile
cM = ccl.halos.ConcentrationDuffy08(mdef=hmd)
prof = ccl.halos.HaloProfileNFW(cM)
# f(k,a)
noboost = lambda k, a, cosmo: np.ones_like(k)
mu, sig = -0.4, 0.35
gauss = lambda x, mu, sig: np.exp(-((x - mu) / sig)**2)
def boost(k, a, cosmo):
    """ Let's give the power specturm a gaussian boost at k=mu. """
    boost = (1 + gauss(np.log10(k), mu, sig))
    return boost
# power spectra
Pk0 = halomod_power_spectrum(cosmo, hmc, k, a, prof)
Pk1 = halomod_power_spectrum(cosmo, hmc, k, a, prof, f_ka=boost)
Pk2 = halomod_power_spectrum(cosmo, hmc, k, a, prof, f_ka=noboost)

B = gauss(np.log10(k), mu, sig)  # this is the boost we gave to P(k,a)
assert np.allclose(Pk1/Pk0-1, B, rtol=0)
assert np.allclose(Pk0, Pk2, rtol=0)
