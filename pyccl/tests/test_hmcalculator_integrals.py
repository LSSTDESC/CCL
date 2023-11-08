import numpy as np
import pyccl as ccl

cosmo = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
mdef = ccl.halos.MassDef200m
hmf = ccl.halos.MassFuncTinker10(mass_def=mdef, mass_def_strict=False)
hbf = ccl.halos.HaloBiasTinker10(mass_def=mdef, mass_def_strict=False)

hmc = ccl.halos.HMCalculator(mass_function=hmf, halo_bias=hbf, mass_def=mdef)

# Profiles
con = ccl.halos.ConcentrationDuffy08(mass_def=mdef)
P1 = ccl.halos.HaloProfileNFW(mass_def=mdef, concentration=con,
                              fourier_analytic=True)
P3 = ccl.halos.HaloProfilePressureGNFW(mass_def=mdef)

# Profiles2pt
PKC = ccl.halos.Profile2pt()
PKCH = ccl.halos.Profile2ptHOD()

#
nk = 32
k_use = np.geomspace(1E-3, 10, nk)
aa = 1


def test_hmcalculator_I_0_1():
    prof1 = P1

    I = hmc.I_0_1(cosmo, k_use, aa, prof1)

    # Test correct shape
    assert I.shape == (nk, )


def test_hmcalculator_I_1_1():
    prof1 = P1

    I = hmc.I_1_1(cosmo, k_use, aa, prof1)
    I01 = hmc.I_0_1(cosmo, k_use, aa, prof1)

    # Test correct shape
    assert I.shape == (nk, )

    # Check that you're accounting for the bias
    assert np.max(np.abs(I / I01 - 1)) > 0.1


def test_hmcalculator_I_1_3():
    prof1 = prof2 = P1
    prof3 = P3
    prof12_2pt = PKC

    # 1, 23
    I = hmc.I_1_3(cosmo, k_use, aa, prof1, prof_2pt=prof12_2pt, prof2=prof2,
                  prof3=prof3)

    # Test correct shape
    assert I.shape == (nk, nk)


def test_hmcalculator_I_0_2():
    prof1 = P1
    prof2 = P3
    prof12_2pt = PKC

    I = hmc.I_0_2(cosmo, k_use, aa, prof1, prof_2pt=prof12_2pt, prof2=prof2)

    # Test correct shape
    assert I.shape == (nk,)


def test_hmcalculator_I_1_2():
    prof1 = P1
    prof2 = P3
    prof12_2pt = PKC

    I = hmc.I_1_2(cosmo, k_use, aa, prof1, prof_2pt=prof12_2pt, prof2=prof2)
    I2 = hmc.I_1_2(cosmo, k_use, aa, prof1, prof_2pt=prof12_2pt, prof2=prof2,
                   diag=False)

    # Test correct shape
    assert I.shape == (nk,)
    assert I2.shape == (nk, nk)
    assert np.all(np.diag(I2) == I)


def test_hmcalculator_I_0_22():
    prof1 = prof2 = P1
    prof3 = prof4 = P3
    prof12_2pt = prof34_2pt = PKC

    I = hmc.I_0_22(cosmo, k_use, aa, prof1, prof12_2pt=prof12_2pt, prof2=prof2,
                   prof3=prof3, prof34_2pt=prof34_2pt, prof4=prof4)

    # Test correct shape
    assert I.shape == (nk, nk)
