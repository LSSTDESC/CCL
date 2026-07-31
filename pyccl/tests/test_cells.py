import numpy as np
import pytest
import pyccl as ccl
from pyccl.modified_gravity import MuSigmaMG
from pyccl import CCLWarning


ccl.gsl_params.LENSING_KERNEL_SPLINE_INTEGRATION = False
COSMO = ccl.Cosmology(
    Omega_c=0.27,
    Omega_b=0.045,
    h=0.67,
    sigma8=0.8,
    n_s=0.96,
    transfer_function="bbks",
    matter_power_spectrum="linear",
)
PKA = ccl.Pk2D.from_function(pkfunc=lambda k, a: np.log(a / k))
ZZ = np.linspace(0.0, 1.0, 200)
NN = np.exp(-(((ZZ - 0.5) / 0.1) ** 2))
LENS = ccl.WeakLensingTracer(COSMO, dndz=(ZZ, NN))


@pytest.mark.parametrize("p_of_k_a", [ccl.DEFAULT_POWER_SPECTRUM, PKA, None])
def test_cells_smoke(p_of_k_a):
    # make a set of tracers to test with
    z = np.linspace(0.0, 1.0, 200)
    n = np.exp(-(((z - 0.5) / 0.1) ** 2))
    b = np.sqrt(1.0 + z)
    lens1 = ccl.WeakLensingTracer(COSMO, dndz=(z, n))
    lens2 = ccl.WeakLensingTracer(COSMO, dndz=(z, n), ia_bias=(z, n))
    nc1 = ccl.NumberCountsTracer(
        COSMO, has_rsd=False, dndz=(z, n), bias=(z, b)
    )
    nc2 = ccl.NumberCountsTracer(COSMO, has_rsd=True, dndz=(z, n), bias=(z, b))
    nc3 = ccl.NumberCountsTracer(
        COSMO, has_rsd=True, dndz=(z, n), bias=(z, b), mag_bias=(z, b)
    )
    cmbl = ccl.CMBLensingTracer(COSMO, z_source=1100.0)
    tracers = [lens1, lens2, nc1, nc2, nc3, cmbl]

    ell_scl = 4.0
    ell_int = 4
    ell_lst = [2, 3, 4, 5]
    ell_arr = np.arange(2, 5)
    ells = [ell_int, ell_scl, ell_lst, ell_arr]

    for i in range(len(tracers)):
        for j in range(i, len(tracers)):
            for ell in ells:
                corr = ccl.angular_cl(
                    COSMO, tracers[i], tracers[j], ell, p_of_k_a=p_of_k_a
                )
                assert np.all(np.isfinite(corr))
                assert np.shape(corr) == np.shape(ell)

                # reversing should be fine
                corr_rev = ccl.angular_cl(
                    COSMO, tracers[j], tracers[i], ell, p_of_k_a=p_of_k_a
                )
                assert np.allclose(corr, corr_rev)

    # Check invalid dndz
    with pytest.raises(ValueError):
        ccl.NumberCountsTracer(COSMO, has_rsd=False, dndz=z, bias=(z, b))
    with pytest.raises(ValueError):
        ccl.NumberCountsTracer(
            COSMO, has_rsd=False, dndz=(z, n, n), bias=(z, b)
        )
    with pytest.raises(ValueError):
        ccl.NumberCountsTracer(COSMO, has_rsd=False, dndz=(z,), bias=(z, b))
    with pytest.raises(ValueError):
        ccl.NumberCountsTracer(COSMO, has_rsd=False, dndz=(1, 2), bias=(z, b))
    with pytest.raises(ValueError):
        ccl.WeakLensingTracer(COSMO, dndz=z)
    with pytest.raises(ValueError):
        ccl.WeakLensingTracer(COSMO, dndz=(z, n, n))
    with pytest.raises(ValueError):
        ccl.WeakLensingTracer(COSMO, dndz=(z,))
    with pytest.raises(ValueError):
        ccl.WeakLensingTracer(COSMO, dndz=(1, 2))


@pytest.mark.parametrize("ells", [[3, 2, 1], [1, 3, 2], [2, 3, 1]])
def test_cells_raise_ell_reversed(ells):
    with pytest.raises(ValueError):
        ccl.angular_cl(COSMO, LENS, LENS, ells)


def test_cells_raise_integ_method():
    ells = [10, 11]
    with pytest.raises(ValueError):
        ccl.angular_cl(
            COSMO, LENS, LENS, ells, limber_integration_method="guad"
        )

    with pytest.raises(ValueError):
        LENS_2 = ccl.WeakLensingTracer(COSMO, dndz=(ZZ, np.zeros(len(ZZ))))
        ccl.angular_cl(
            COSMO, LENS, LENS_2, ells, limber_integration_method="quad"
        )


def test_cells_raise_nonlimber_methods():
    ells = [10, 11]
    with pytest.raises(ValueError):
        ccl.angular_cl(
            COSMO, LENS, LENS, ells, non_limber_integration_method="FEKM"
        )
    with pytest.raises(ValueError):
        ccl.angular_cl(
            COSMO, LENS, LENS, ells, l_limber='auoto',
            non_limber_integration_method="FKEM"
        )
    cl, meta = ccl.angular_cl(COSMO, LENS, LENS,
                              ells, l_limber=100,
                              non_limber_integration_method="FKEM",
                              return_meta=True
                              )
    assert (meta['l_limber'] == 100)


def test_cells_raise_weird_pk():
    ells = [10, 11]
    with pytest.raises(ValueError):
        ccl.angular_cl(COSMO, LENS, LENS, ells, p_of_k_a=lambda k, a: 10)


def test_fkem_chi_params():
    # Redshift distribution
    z = np.linspace(0, 4.72, 60)
    nz = z**2*np.exp(-0.5*((z-1.5)/0.7)**2)

    # Bias
    bz = np.ones_like(z)

    # Power spectra
    ls = np.unique(np.geomspace(2, 2000, 128).astype(int)).astype(float)
    cosmo = ccl.CosmologyVanillaLCDM()
    tracer_gal = ccl.NumberCountsTracer(cosmo, has_rsd=False,
                                        dndz=(z, nz), bias=(z, bz))
    cl_gg = ccl.angular_cl(cosmo, tracer_gal, tracer_gal, ls,
                           l_limber=-1)
    cl_ggn = ccl.angular_cl(cosmo, tracer_gal, tracer_gal, ls,
                            l_limber=1000)
    cl_ggn_b = ccl.angular_cl(cosmo, tracer_gal, tracer_gal, ls,
                              l_limber=1000, fkem_chi_min=1.0,
                              fkem_Nchi=100)

    ell_good = ls > 100

    # Check that, above ell, the non-Limber calculation does not
    # agree with Limber when using the default FKEM sampling params
    assert not np.all(np.fabs(cl_ggn/cl_gg-1)[ell_good] < 0.01)
    # Check that it works with custom ones.
    assert np.all(np.fabs(cl_ggn_b/cl_gg-1)[ell_good] < 0.01)


def test_fkem_warn_bad_params():
    # Simple NC tracer setup
    z = np.linspace(0, 2.0, 50)
    nz = z**2 * np.exp(-z)
    bz = np.ones_like(z)
    ells = np.array([10.0, 30.0, 100.0])
    cosmo = ccl.CosmologyVanillaLCDM()
    nc = ccl.NumberCountsTracer(
        cosmo, has_rsd=False, dndz=(z, nz), bias=(z, bz)
    )

    # Nchi invalid (<=0) -> warning and defaulting
    with pytest.warns(CCLWarning, match="Nchi must be a positive integer"):
        cl = ccl.angular_cl(
            cosmo,
            nc,
            nc,
            ells,
            non_limber_integration_method="FKEM",
            l_limber=1000,
            fkem_Nchi=0,
        )
        assert np.all(np.isfinite(cl))

    # chi_min invalid (<=0) -> warning and defaulting
    with pytest.warns(CCLWarning, match="chi_min must be greater than zero"):
        cl = ccl.angular_cl(
            cosmo,
            nc,
            nc,
            ells,
            non_limber_integration_method="FKEM",
            l_limber=1000,
            fkem_chi_min=0.0,
        )
        assert np.all(np.isfinite(cl))

    # limber_max_error invalid (<=0) -> warning and defaulting
    with pytest.warns(CCLWarning,
                      match="limber_max_error must be greater than zero"):
        cl = ccl.angular_cl(
            cosmo,
            nc,
            nc,
            ells,
            non_limber_integration_method="FKEM",
            l_limber=1000,
            limber_max_error=0.0,
        )
        assert np.all(np.isfinite(cl))


def test_fkem_warn_mismatched_pk_types_fallback_to_limber():
    # If p_of_k_a and p_of_k_a_lin are of different types,
    # FKEM should warn and fall back
    z = np.linspace(0, 2.0, 50)
    nz = z**2 * np.exp(-z)
    bz = np.ones_like(z)
    ells = np.array([10.0, 30.0, 100.0])
    cosmo = ccl.CosmologyVanillaLCDM()
    nc = ccl.NumberCountsTracer(
        cosmo, has_rsd=False, dndz=(z, nz), bias=(z, bz)
    )

    pka_obj = ccl.Pk2D.from_function(pkfunc=lambda k, a: a / (1.0 + k))
    with pytest.warns(CCLWarning, match="p_of_k_a and "
                      "p_of_k_a_lin must be of the same type"):
        cl, meta = ccl.angular_cl(
            cosmo,
            nc,
            nc,
            ells,
            non_limber_integration_method="FKEM",
            l_limber=1000,
            p_of_k_a=pka_obj,
            p_of_k_a_lin=ccl.DEFAULT_POWER_SPECTRUM,  # string default
            return_meta=True,
        )
        # Should have fallen back to Limber for all ells
        assert np.all(np.isfinite(cl))
        assert meta["l_limber"] == -1


def test_fkem_multicomponent_tracer():
    # Smoke test to make sure multi-component tracers do not fail when
    # FKEM non-Limber is calculated.
    z = np.linspace(0.0, 2.0, 60)
    nz = np.exp(-0.5 * ((z - 0.5) / 0.3) ** 2)
    bz = np.ones_like(z)
    mbz = 0.5 * np.ones_like(z)
    ells = np.array([2.0, 5.0, 10.0, 20.0])
    cosmo = ccl.CosmologyVanillaLCDM()

    nc = ccl.NumberCountsTracer(
        cosmo,
        has_rsd=True,
        dndz=(z, nz),
        bias=(z, bz),
        mag_bias=(z, mbz),
    )

    # gg auto test
    cl_gg = ccl.angular_cl(
        cosmo, nc, nc, ells,
        non_limber_integration_method="FKEM",
        l_limber=1000,
        fkem_Nchi=100,
        fkem_chi_min=1.0,
    )
    assert np.all(np.isfinite(cl_gg))

    # cross test with weak lensing
    lens = ccl.WeakLensingTracer(cosmo, dndz=(z, nz))
    cl_gk = ccl.angular_cl(
        cosmo, nc, lens, ells,
        non_limber_integration_method="FKEM",
        l_limber=1000,
        fkem_Nchi=100,
        fkem_chi_min=1.0,
    )
    assert np.all(np.isfinite(cl_gk))


def test_fkem_decomposed_matches_direct_number_counts_mag():
    z = np.linspace(0.01, 2.0, 200)
    nz = np.exp(-((z - 1.0) ** 2) / 0.1)
    ells = np.arange(2, 200)
    cosmo = ccl.CosmologyVanillaLCDM()

    ccl_kwargs = dict(
        l_limber=210,
        non_limber_integration_method="FKEM",
        fkem_Nchi=200,
        fkem_chi_min=1e-6,
    )

    tracer_den = ccl.NumberCountsTracer(
        cosmo,
        dndz=(z, nz),
        bias=(z, 1.5 * np.ones_like(z)),
        has_rsd=False,
        mag_bias=None,
    )
    tracer_mag = ccl.NumberCountsTracer(
        cosmo,
        dndz=(z, nz),
        bias=None,
        has_rsd=False,
        mag_bias=(z, 1.0 * np.ones_like(z)),
    )
    tracer_full = ccl.NumberCountsTracer(
        cosmo,
        dndz=(z, nz),
        bias=(z, 1.5 * np.ones_like(z)),
        has_rsd=False,
        mag_bias=(z, 1.0 * np.ones_like(z)),
    )

    cl_dd = ccl.angular_cl(cosmo, tracer_den, tracer_den, ells, **ccl_kwargs)
    cl_dM = ccl.angular_cl(cosmo, tracer_den, tracer_mag, ells, **ccl_kwargs)
    cl_MM = ccl.angular_cl(cosmo, tracer_mag, tracer_mag, ells, **ccl_kwargs)
    cl_decomposed = cl_dd + 2.0 * cl_dM + cl_MM

    cl_direct = ccl.angular_cl(cosmo, tracer_full, tracer_full, ells,
                               **ccl_kwargs)

    rel = np.abs(cl_direct / cl_decomposed - 1.0)
    assert np.all(np.isfinite(rel))
    assert np.max(rel) < 5e-3


def test_fkem_decomposed_matches_direct_weak_lensing_ia():
    z = np.linspace(0.01, 2.0, 200)
    nz = np.exp(-((z - 1.0) ** 2) / 0.1)
    ia_amp = 0.5 * np.ones_like(z)
    ells = np.arange(2, 200)
    cosmo = ccl.CosmologyVanillaLCDM()

    ccl_kwargs = dict(
        l_limber=210,
        non_limber_integration_method="FKEM",
        fkem_Nchi=200,
        fkem_chi_min=1e-6,
    )

    tracer_lens = ccl.WeakLensingTracer(
        cosmo,
        dndz=(z, nz),
        has_shear=True,
        ia_bias=None,
    )
    tracer_ia = ccl.WeakLensingTracer(
        cosmo,
        dndz=(z, nz),
        has_shear=False,
        ia_bias=(z, ia_amp),
        use_A_ia=False,
    )
    tracer_full = ccl.WeakLensingTracer(
        cosmo,
        dndz=(z, nz),
        has_shear=True,
        ia_bias=(z, ia_amp),
        use_A_ia=False,
    )

    cl_gg = ccl.angular_cl(cosmo, tracer_lens, tracer_lens, ells,
                           **ccl_kwargs)
    cl_gi = ccl.angular_cl(cosmo, tracer_lens, tracer_ia, ells,
                           **ccl_kwargs)
    cl_ii = ccl.angular_cl(cosmo, tracer_ia, tracer_ia, ells, **ccl_kwargs)
    cl_decomposed = cl_gg + 2.0 * cl_gi + cl_ii

    cl_direct = ccl.angular_cl(cosmo, tracer_full, tracer_full, ells,
                               **ccl_kwargs)

    rel = np.abs(cl_direct / cl_decomposed - 1.0)
    assert np.all(np.isfinite(rel))
    assert np.max(rel) < 5e-3


def test_cells_mg():
    # Check that if we feed the non-linear matter power spectrum from a MG
    # cosmology into a Calculator and get Cells using MG tracers, we get the
    # same results.

    # set up a MG cosmology
    cosmo_MG = ccl.CosmologyVanillaLCDM(mg_parametrization=MuSigmaMG(
                                        mu_0=0.5, sigma_0=0.5),
                                        transfer_function="bbks",
                                        matter_power_spectrum="linear")
    cosmo_MG.compute_nonlin_power()
    pk2d = cosmo_MG.get_nonlin_power()

    # copy it into a calculator
    a, lk, pk = pk2d.get_spline_arrays()
    pk_nonlin = {"a": a, "k": np.exp(lk), "delta_matter:delta_matter": pk}
    cosmo_calc = ccl.CosmologyCalculator(
        Omega_c=0.25, Omega_b=0.05, h=0.67, n_s=0.96, sigma8=0.81,
        mg_parametrization=MuSigmaMG(mu_0=0.5, sigma_0=0.5),
        pk_nonlin=pk_nonlin)

    # get the Cells
    ell = np.geomspace(2, 2000, 128)
    tr_MG = ccl.CMBLensingTracer(cosmo_MG, z_source=1100.0)
    tr_calc = ccl.CMBLensingTracer(cosmo_calc, z_source=1100.0)

    cl0 = ccl.angular_cl(cosmo_MG, tr_MG, tr_MG, ell)
    cosmo_calc.compute_growth()
    cl1 = ccl.angular_cl(cosmo_calc, tr_calc, tr_calc, ell)
    assert np.all(np.fabs(1 - cl1 / cl0) < 1e-10)


ccl.gsl_params.reload()  # reset to the default parameters
