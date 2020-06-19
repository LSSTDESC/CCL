import numpy as np
import pyccl as ccl


def test_hmcalculator_number_counts_numcosmo():
    cosmo = ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7,
        sigma8=0.894959309349923,
        n_s=0.9742,
        transfer_function='eisenstein_hu',
        matter_power_spectrum='linear',
        Omega_g=0,
    )
    mdef = ccl.halos.MassDef(200, 'matter')
    hmf = ccl.halos.MassFuncTinker10(cosmo, mdef,
                                     mass_def_strict=False)
    hbf = ccl.halos.HaloBiasTinker10(cosmo, mass_def=mdef,
                                     mass_def_strict=False)

    hmc = ccl.halos.HMCalculator(
        cosmo, hmf, hbf, mdef, log10M_min=14, nlog10M=1024)

    print("")
    print(ccl.sigma8(cosmo))
    print(ccl.background.h_over_h0(cosmo, 0.9))
    print(ccl.linear_matter_power(cosmo, 1.0, 0.9))
    print(ccl.growth_factor(cosmo, 0.9))

    a_2 = 1.0 / (1.0 + 2.0)

    def sel(m, a):
        m = np.atleast_1d(m)
        a = np.atleast_1d(a)
        val = np.zeros_like(m.reshape(-1, 1) * a.reshape(1, -1))
        msk_a = (a > a_2) & (a < 1.0)
        msk_m = (m > 1e14) & (m < 1e16)
        val[msk_m, :] += 1
        val[:, msk_a] += 1
        msk = val == 2
        val[~msk] = 0
        # the selection function needs to be constant per
        # dlog10(m)
        # our code expects it per dm, so we need a factor of
        #  dlog10(m)/dm
        val = val / (np.log(10) * m.reshape(-1, 1))
        return val

    area = 200 * (np.pi / 180)**2

    nc = hmc.number_counts(cosmo, sel) * area
    assert np.isfinite(nc)
    assert not np.allclose(nc, 0)
    print(nc)
