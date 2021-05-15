import pyccl as ccl
from pyccl.halos.profiles import HaloProfileHOD
from pyccl.halos.concentration import ConcentrationDuffy08
import numpy as np

# global
k = np.logspace(-3, 2, 256)
a = 1

# cosmo
cosmo = ccl.CosmologyVanillaLCDM()
cM = ConcentrationDuffy08()
hmd = cM.mdef

# profiles
p1 = HaloProfileHOD(cM, ns_independent=False)
p2 = HaloProfileHOD(cM, ns_independent=True)
assert p1.lMmin_0 == p2.lMmin_0


## FOURIER SPACE ##
# test 1: non-zero fourier profile when M < Mmin
p1f = p1._fourier(cosmo, k, 10**(p1.lMmin_0-2), a, hmd)
p2f = p2._fourier(cosmo, k, 10**(p2.lMmin_0-2), a, hmd)
assert np.all(p1f == 0) and np.all(p2f > 0)

# test 2: equal fourier profiles when M > Mmin
p1f = p1._fourier(cosmo, k, 10**(p1.lMmin_0+2), a, hmd)
p2f = p2._fourier(cosmo, k, 10**(p2.lMmin_0+2), a, hmd)
assert np.allclose(p1f, p2f, rtol=0)

# test 3: half of each other when M == Mmin
p1f = p1._fourier(cosmo, k, 10**(p1.lMmin_0), a, hmd)
p2f = p2._real(cosmo, k, 10**(p2.lMmin_0), a, hmd)
assert np.allclose(2*p1f, p2f+0.5, rtol=0)


## REAL SPACE ##
r = 1/k[::-1]

# test 1: non-zero fourier profile when M < Mmin
p1r = p1._real(cosmo, r, 10**(p1.lMmin_0-2), a, hmd)
p2r = p2._real(cosmo, r, 10**(p2.lMmin_0-2), a, hmd)
assert np.all(p1r == 0) and np.all(p2r > 0)

# test 2: equal fourier profiles when M > Mmin
p1r = p1._real(cosmo, r, 10**(p1.lMmin_0+2), a, hmd)
p2r = p2._real(cosmo, r, 10**(p2.lMmin_0+2), a, hmd)
assert np.allclose(p1r, p2r, rtol=0)

# test 3: half of each other when M == Mmin
p1r = p1._real(cosmo, r, 10**(p1.lMmin_0), a, hmd)
p2r = p2._real(cosmo, r, 10**(p2.lMmin_0), a, hmd)
assert np.allclose(2*p1r, p2r+0.5, rtol=0)