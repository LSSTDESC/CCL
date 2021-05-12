import numpy as np
import pyccl as ccl
from pyccl.halos import HMCalculator
from pyccl.halos import Profile2pt, Profile2ptR
from pyccl.halos.profiles import HaloProfileNFW
from pyccl.halos.concentration import ConcentrationDuffy08

cM = ConcentrationDuffy08()
prof = HaloProfileNFW(cM)
cosmo = ccl.CosmologyVanillaLCDM()
a = 1
k = np.logspace(-3, 2, 128)
M = np.logspace(7, 16, 128)


## Test directly with `fourier_2pt` ##
# null case: when r_corr=0 it should produce the same result as `Profile2pt`
F0 = Profile2pt().fourier_2pt(prof, cosmo, k, M, a, mass_def=cM.mdef)
F1 = Profile2ptR(r_corr=0).fourier_2pt(prof, cosmo, k, M, a, mass_def=cM.mdef)

assert np.allclose(F0, F1, atol=0)

# case 1: when r_corr=-1 it the 1h terms are completely uncorrelated
#         and hence it should return an array of zeros
F2 = Profile2ptR(r_corr=-1).fourier_2pt(prof, cosmo, k, M, a, mass_def=cM.mdef)

assert np.allclose(F2, 0)


## Test with the Halo Model Calculator ##
from pyccl.halos.hmfunc import MassFuncTinker10
from pyccl.halos.hbias import HaloBiasTinker10
nM = MassFuncTinker10(cosmo, cM.mdef)
bM = HaloBiasTinker10(cosmo, cM.mdef)
hmc = HMCalculator(cosmo, nM, bM, cM.mdef)

I0 = hmc.I_0_2(cosmo, k, 1, prof, Profile2pt())
I1 = hmc.I_0_2(cosmo, k, 1, prof, Profile2ptR(r_corr=0))

assert np.allclose(I0, I1, atol=0)

I2 = hmc.I_0_2(cosmo, k, 1, prof, Profile2ptR(r_corr=-1))

assert np.allclose(I2, 0)