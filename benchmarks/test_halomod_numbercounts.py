import numpy as np
import pyccl as ccl


def test_hmcalculator_number_counts_numcosmo():
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
        mu_0=0.0,
        transfer_function='eisenstein_hu',
        matter_power_spectrum='linear'
    )
    mdef = ccl.halos.MassDef(200, 'matter')
    hmf = ccl.halos.MassFuncTinker08(cosmo, mdef,
                                     mass_def_strict=False)
    hbf = ccl.halos.HaloBiasTinker10(cosmo, mass_def=mdef,
                                     mass_def_strict=False)

    benches = np.loadtxt("./benchmarks/data/numcosmo_cluster_counts.txt")

    for i in range(benches.shape[0]):
        bench = benches[i, :]
        hmc = ccl.halos.HMCalculator(
            cosmo, hmf, hbf, mdef,
            log10M_min=np.log10(bench[1]),
            log10M_max=np.log10(bench[2]),
            integration_method_M='spline')

        a_2 = 1.0 / (1.0 + bench[4])
        a_1 = 1.0 / (1.0 + bench[3])

        def sel(m, a):
            m = np.atleast_1d(m)
            a = np.atleast_1d(a)
            val = np.zeros_like(m.reshape(-1, 1) * a.reshape(1, -1))
            msk_a = (a > a_2) & (a < a_1)
            msk_m = (m > bench[1]) & (m < bench[2])
            val[msk_m, :] += 1
            val[:, msk_a] += 1
            msk = val == 2
            val[~msk] = 0
            val[msk] = 1.0
            return val

        area = 200 * (np.pi / 180)**2

        nc = hmc.number_counts(cosmo, sel, amin=a_2, amax=a_1) * area
        assert np.isfinite(nc)
        assert not np.allclose(nc, 0)

        tol = max(0.013, np.sqrt(bench[0]) / bench[0] / 10)
        print(nc, bench[0], nc/bench[0]-1, tol)
        assert np.allclose(nc, bench[0], atol=0, rtol=tol)


def test_hmcalculator_number_counts_numcosmo_highacc():
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
        mu_0=0.0,
        transfer_function='eisenstein_hu',
        matter_power_spectrum='linear'
    )
    mdef = ccl.halos.MassDef(200, 'matter')
    hmf = ccl.halos.MassFuncTinker08(cosmo, mdef,
                                     mass_def_strict=False)
    hbf = ccl.halos.HaloBiasTinker10(cosmo, mass_def=mdef,
                                     mass_def_strict=False)

    benches = np.loadtxt("./benchmarks/data/numcosmo_cluster_counts.txt")

    for i in range(benches.shape[0]):
        bench = benches[i, :]
        hmc = ccl.halos.HMCalculator(
            cosmo, hmf, hbf, mdef,
            log10M_min=np.log10(bench[1]),
            log10M_max=np.log10(bench[2]),
            integration_method_M='spline',
            nlog10M=4096,
        )

        a_2 = 1.0 / (1.0 + bench[4])
        a_1 = 1.0 / (1.0 + bench[3])

        def sel(m, a):
            m = np.atleast_1d(m)
            a = np.atleast_1d(a)
            val = np.zeros_like(m.reshape(-1, 1) * a.reshape(1, -1))
            msk_a = (a > a_2) & (a < a_1)
            msk_m = (m > bench[1]) & (m < bench[2])
            val[msk_m, :] += 1
            val[:, msk_a] += 1
            msk = val == 2
            val[~msk] = 0
            val[msk] = 1.0
            return val

        area = 200 * (np.pi / 180)**2

        nc = hmc.number_counts(
            cosmo, sel,
            amin=a_2,
            amax=a_1,
            na=4096,
        ) * area
        assert np.isfinite(nc)
        assert not np.allclose(nc, 0)

        tol = 1e-3
        print(nc, bench[0], nc/bench[0]-1, tol)
        assert np.allclose(nc, bench[0], atol=0, rtol=tol)
