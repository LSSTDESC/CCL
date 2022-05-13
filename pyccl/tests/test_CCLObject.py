import pytest
import numpy as np
from . import pyccl as ccl


def test_CCLObject():
    # Test eq --> repr <-- hash for all kinds of CCL objects.

    # 1.1. Using a complicated Cosmology object.
    extras = {"camb": {"halofit_version": "mead2020", "HMCode_logT_AGN": 7.8}}
    kwargs = {"transfer_function": "bbks",
              "matter_power_spectrum": "emu",
              "z_mg": np.ones(10),
              "df_mg": np.ones(10),
              "extra_parameters": extras}
    COSMO1 = ccl.CosmologyVanillaLCDM(**kwargs)
    COSMO2 = ccl.CosmologyVanillaLCDM(**kwargs)
    assert COSMO1 == COSMO2
    kwargs["df_mg"] *= 2
    COSMO2 = ccl.CosmologyVanillaLCDM(**kwargs)
    assert COSMO1 != COSMO2

    # 2. Using a Pk2D object.
    cosmo = ccl.CosmologyVanillaLCDM(transfer_function="bbks")
    cosmo.compute_linear_power()
    PK1 = cosmo.get_linear_power()
    PK2 = ccl.Pk2D.pk_from_model(cosmo, "bbks")
    assert PK1 == PK2
    assert ccl.Pk2D(empty=True) == ccl.Pk2D(empty=True)

    # 3.1. Using a factorizable Tk3D object.
    a_arr, lk_arr, pk_arr = PK1.get_spline_arrays()
    TK1 = ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr,
                   pk1_arr=pk_arr, pk2_arr=pk_arr, is_logt=False)
    TK2 = ccl.Tk3D(a_arr=a_arr, lk_arr=lk_arr,
                   pk1_arr=pk_arr, pk2_arr=pk_arr, is_logt=False)
    assert TK1 == TK2

    # 3.2. Using a non-factorizable Tk3D object.
    a_arr_2 = np.arange(0.5, 0.9, 0.1)
    lk_arr_2 = np.linspace(-2, 1, 8)
    TK1 = ccl.Tk3D(
        a_arr=a_arr_2, lk_arr=lk_arr_2,
        tkk_arr=np.ones((a_arr_2.size, lk_arr_2.size, lk_arr_2.size)))
    TK2 = ccl.Tk3D(
        a_arr=a_arr_2, lk_arr=lk_arr_2,
        tkk_arr=np.ones((a_arr_2.size, lk_arr_2.size, lk_arr_2.size)))
    assert TK1 == TK2

    # 4. Using a CosmologyCalculator.
    pk_linear = {"a": a_arr,
                 "k": np.exp(lk_arr),
                 "delta_matter:delta_matter": pk_arr}
    COSMO1 = ccl.CosmologyCalculator(
        Omega_c=0.25, Omega_b=0.05, h=0.67, n_s=0.96, sigma8=0.81,
        pk_linear=pk_linear, pk_nonlin=pk_linear)
    COSMO2 = ccl.CosmologyCalculator(
        Omega_c=0.25, Omega_b=0.05, h=0.67, n_s=0.96, sigma8=0.81,
        pk_linear=pk_linear, pk_nonlin=pk_linear)
    assert COSMO1 == COSMO2

    # 5. Using a Tracer object.
    TR1 = ccl.CMBLensingTracer(cosmo, z_source=1101)
    TR2 = ccl.CMBLensingTracer(cosmo, z_source=1101)
    assert TR1 == TR2


def test_CCLHalosObject():
    # Test eq --> repr <-- hash for all kinds of CCL halo objects.

    # 1. Build a halo model calculator using the default parametrizations.
    cosmo = ccl.CosmologyVanillaLCDM(transfer_function="bbks")
    HMC = ccl.halos.HMCalculator(
        cosmo, massfunc="Tinker08", hbias="Tinker10", mass_def="200m")

    # 2. Define separate default halo model ingredients.
    MDEF = ccl.halos.MassDef200m()
    HMF = ccl.halos.MassFuncTinker08(cosmo, mass_def=MDEF)
    HBF = ccl.halos.HaloBiasTinker10(cosmo, mass_def=MDEF)

    # 3. Test equivalence.
    assert MDEF == HMC._mdef
    assert HMF == HMC._massfunc
    assert HBF == HMC._hbias
    # TODO: Uncomment once CCLv3 is released (uniform keyword naming).
    # HMC2 = ccl.halos.HMCalculator(
    #     cosmo, massfunc=HMF, hbias=HBF, mass_def=MDEF)
    # assert HMC == HMC2

    # 4. Test halo profiles.
    CM1 = ccl.halos.Concentration.from_name("Duffy08")()
    CM2 = ccl.halos.ConcentrationDuffy08()
    assert CM1 == CM2

    P1 = ccl.halos.HaloProfileHOD(c_M_relation=CM1)
    P2 = ccl.halos.HaloProfileHOD(c_M_relation=CM2)
    assert P1 == P2

    PCOV1 = ccl.halos.Profile2pt(r_corr=1.5)
    PCOV2 = ccl.halos.Profile2pt(r_corr=1.0)
    assert PCOV1 != PCOV2
    PCOV2.update_parameters(r_corr=1.5)
    assert PCOV1 == PCOV2


def test_CCLObject_immutable():
    # test `CCLObject` lock
    obj = ccl.CCLObject()
    obj._object_lock.unlock()
    assert "locked=False" in repr(obj._object_lock)
    obj._object_lock.lock()
    assert "locked=True" in repr(obj._object_lock)

    # `update_parameters` not implemented.
    cosmo = ccl.CosmologyVanillaLCDM()
    with pytest.raises(AttributeError):
        cosmo.my_attr = "hello_world"
    with pytest.raises(NotImplementedError):
        cosmo.update_parameters(A_SPLINE_NA=120)

    # `update_parameters` implemented.
    prof = ccl.halos.HaloProfilePressureGNFW(mass_bias=0.5)
    with pytest.raises(AttributeError):
        prof.mass_bias = 0.7
    assert prof.mass_bias == 0.5
    prof.update_parameters(mass_bias=0.7)
    assert prof.mass_bias == 0.7

    # Check that the hash repr is deleted as required.
    prof1 = ccl.halos.HaloProfilePressureGNFW(mass_bias=0.5)
    prof2 = ccl.halos.HaloProfilePressureGNFW(mass_bias=0.5)
    assert prof1 == prof2                   # repr is cached
    prof2.update_parameters(mass_bias=0.7)  # cached repr is deleted
    assert prof1 != prof2                   # repr is cached again


def test_CCLObject_default_behavior():
    # Test that if `__repr__` is not defined the fall back is safe.
    MyType = type("MyType", (ccl.CCLObject,), {"test": 0})
    instances = [MyType() for _ in range(2)]
    assert instances[0] != instances[1]
    assert hash(instances[0]) != hash(instances[1])
