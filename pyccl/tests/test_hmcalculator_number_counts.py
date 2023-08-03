import numpy as np
import pyccl as ccl
import scipy.integrate


def test_hmcalculator_number_counts_smoke():
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        transfer_function='bbks', matter_power_spectrum='linear')
    mdef = ccl.halos.MassDef(200, 'matter')
    hmf = ccl.halos.MassFuncTinker10(mass_def=mdef, mass_def_strict=False)
    hbf = ccl.halos.HaloBiasTinker10(mass_def=mdef, mass_def_strict=False)

    hmc = ccl.halos.HMCalculator(mass_function=hmf, halo_bias=hbf,
                                 mass_def=mdef)

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

    nc = hmc.number_counts(cosmo, selection=sel) * 4.0 * np.pi
    assert np.isfinite(nc)
    assert not np.allclose(nc, 0)


def test_hmcalculator_number_counts_zero():
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        transfer_function='bbks', matter_power_spectrum='linear')
    mdef = ccl.halos.MassDef(200, 'matter')
    hmf = ccl.halos.MassFuncTinker10(mass_def=mdef, mass_def_strict=False)
    hbf = ccl.halos.HaloBiasTinker10(mass_def=mdef, mass_def_strict=False)

    hmc = ccl.halos.HMCalculator(mass_function=hmf, halo_bias=hbf,
                                 mass_def=mdef)

    def sel(m, a):
        m = np.atleast_1d(m)
        a = np.atleast_1d(a)
        val = np.zeros_like(m.reshape(-1, 1) * a.reshape(1, -1))
        return val

    nc = hmc.number_counts(cosmo, selection=sel) * 4.0 * np.pi
    assert np.isfinite(nc)
    assert np.allclose(nc, 0)


def test_hmcalculator_number_counts_norm():
    cosmo = ccl.Cosmology(
        Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
        transfer_function='bbks', matter_power_spectrum='linear')
    mdef = ccl.halos.MassDef(200, 'matter')
    hmf = ccl.halos.MassFuncTinker10(mass_def=mdef, mass_def_strict=False)
    hbf = ccl.halos.HaloBiasTinker10(mass_def=mdef, mass_def_strict=False)

    hmc = ccl.halos.HMCalculator(mass_function=hmf, halo_bias=hbf,
                                 mass_def=mdef)

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

    nc2 = hmc.number_counts(cosmo, selection=sel2) * 4.0 * np.pi
    assert np.isfinite(nc2)
    assert not np.allclose(nc2, 0)
    nc4 = hmc.number_counts(cosmo, selection=sel4) * 4.0 * np.pi
    assert np.isfinite(nc4)
    assert not np.allclose(nc4, 0)

    assert np.allclose(nc2 * 2, nc4)


def test_hmcalculator_number_counts_scipy_dblquad():
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        n_s=0.96,
        sigma8=0.8,
        Omega_k=0.0,
        Omega_g=0,
        Neff=0.0,
        w0=-1.0,
        wa=0.0,
        T_CMB=2.7245,
        transfer_function='eisenstein_hu',
        matter_power_spectrum='linear'
    )
    mdef = ccl.halos.MassDef(200, 'matter')
    hmf = ccl.halos.MassFuncTinker08(mass_def=mdef, mass_def_strict=False)
    hbf = ccl.halos.HaloBiasTinker10(mass_def=mdef, mass_def_strict=False)

    amin = 0.75
    amax = 1.0
    mmin = 1e14
    mmax = 1e15

    hmc = ccl.halos.HMCalculator(
        mass_function=hmf, halo_bias=hbf, mass_def=mdef,
        log10M_min=np.log10(mmin),
        log10M_max=np.log10(mmax),
        integration_method_M='spline')

    def sel(m, a):
        m = np.atleast_1d(m)
        a = np.atleast_1d(a)
        val = np.zeros_like(m.reshape(-1, 1) * a.reshape(1, -1))
        msk_a = (a > amin) & (a < amax)
        msk_m = (m > mmin) & (m < mmax)
        val[msk_m, :] += 2
        val[:, msk_a] += 2
        msk = val == 4
        val[~msk] = 0
        val[msk] = 1.0
        return val

    def _func(m, a):
        abs_dzda = 1 / a / a
        dc = ccl.comoving_angular_distance(cosmo, a)
        ez = ccl.h_over_h0(cosmo, a)
        dh = ccl.physical_constants.CLIGHT_HMPC / cosmo['h']
        dvdz = dh * dc**2 / ez
        dvda = dvdz * abs_dzda

        val = hmf(cosmo, 10**m, a) * sel(10**m, a)
        return val[0, 0] * dvda

    mtot, _ = scipy.integrate.dblquad(
        _func,
        amin,
        amax,
        lambda x: hmc.precision['log10M_min'],
        lambda x: hmc.precision['log10M_max'],
    )

    mtot_hmc = hmc.number_counts(cosmo, selection=sel, a_min=amin, a_max=amax)
    assert np.allclose(mtot_hmc, mtot, atol=0, rtol=0.02)
