import numpy as np
import pyccl as ccl


def test_hmcalculator_number_counts_smoke():
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        transfer_function='bbks', matter_power_spectrum='linear')
    mdef = ccl.halos.MassDef(200, 'matter')
    hmf = ccl.halos.MassFuncTinker10(cosmo, mdef,
                                     mass_def_strict=False)
    hbf = ccl.halos.HaloBiasTinker10(cosmo, mass_def=mdef,
                                     mass_def_strict=False)

    hmc = ccl.halos.HMCalculator(cosmo, hmf, hbf, mdef)

    def sel(m, a):
        m = np.atleast_1d(m)
        a = np.atleast_1d(a)
        val = np.zeros_like(m.reshape(-1, 1) * a.reshape(1, -1))
        msk_a = (a > 0.5) & (a < 1.0)
        msk_m = (m > 1e14) & (m < 1e16)
        val[msk_m, :] += 1
        val[:, msk_a] += 1
        msk = val == 2
        val[~msk] = 0
        return val

    nc = hmc.number_counts(cosmo, sel) * 4.0 * np.pi
    assert np.isfinite(nc)
    assert not np.allclose(nc, 0)


def test_hmcalculator_number_counts_zero():
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        transfer_function='bbks', matter_power_spectrum='linear')
    mdef = ccl.halos.MassDef(200, 'matter')
    hmf = ccl.halos.MassFuncTinker10(cosmo, mdef,
                                     mass_def_strict=False)
    hbf = ccl.halos.HaloBiasTinker10(cosmo, mass_def=mdef,
                                     mass_def_strict=False)

    hmc = ccl.halos.HMCalculator(cosmo, hmf, hbf, mdef)

    def sel(m, a):
        m = np.atleast_1d(m)
        a = np.atleast_1d(a)
        val = np.zeros_like(m.reshape(-1, 1) * a.reshape(1, -1))
        return val

    nc = hmc.number_counts(cosmo, sel) * 4.0 * np.pi
    assert np.isfinite(nc)
    assert np.allclose(nc, 0)


def test_hmcalculator_number_counts_norm():
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        transfer_function='bbks', matter_power_spectrum='linear')
    mdef = ccl.halos.MassDef(200, 'matter')
    hmf = ccl.halos.MassFuncTinker10(cosmo, mdef,
                                     mass_def_strict=False)
    hbf = ccl.halos.HaloBiasTinker10(cosmo, mass_def=mdef,
                                     mass_def_strict=False)

    hmc = ccl.halos.HMCalculator(cosmo, hmf, hbf, mdef)

    def sel2(m, a):
        m = np.atleast_1d(m)
        a = np.atleast_1d(a)
        val = np.zeros_like(m.reshape(-1, 1) * a.reshape(1, -1))
        msk_a = (a > 0.5) & (a < 1.0)
        msk_m = (m > 1e14) & (m < 1e16)
        val[msk_m, :] += 1
        val[:, msk_a] += 1
        msk = val == 2
        val[~msk] = 0
        return val

    def sel4(m, a):
        m = np.atleast_1d(m)
        a = np.atleast_1d(a)
        val = np.zeros_like(m.reshape(-1, 1) * a.reshape(1, -1))
        msk_a = (a > 0.5) & (a < 1.0)
        msk_m = (m > 1e14) & (m < 1e16)
        val[msk_m, :] += 2
        val[:, msk_a] += 2
        msk = val == 4
        val[~msk] = 0
        return val

    nc2 = hmc.number_counts(cosmo, sel2) * 4.0 * np.pi
    assert np.isfinite(nc2)
    assert not np.allclose(nc2, 0)
    nc4 = hmc.number_counts(cosmo, sel4) * 4.0 * np.pi
    assert np.isfinite(nc4)
    assert not np.allclose(nc4, 0)

    assert np.allclose(nc2, nc4)
