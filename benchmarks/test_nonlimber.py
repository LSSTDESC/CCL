import numpy as np
import pyccl as ccl

import time
import pytest

root = "benchmarks/data/nonlimber/"


def get_cosmological_parameters():
    return {
        "Omega_m": 0.3156,
        "Omega_b": 0.0492,
        "w0": -1.0,
        "h": 0.6727,
        "A_s": 2.12107e-9,
        "n_s": 0.9645,
        "Neff": 3.046,
        "T_CMB": 2.725,
    }


def get_tracer_parameters():
    # Per-bin galaxy bias
    b_g = np.array(
        [
            1.376695,
            1.451179,
            1.528404,
            1.607983,
            1.689579,
            1.772899,
            1.857700,
            1.943754,
            2.030887,
            2.118943,
        ]
    )
    return {"b_g": b_g}


def get_ells():
    return np.unique(np.geomspace(2, 2000, 128).astype(int)).astype(float)


def get_nmodes(fsky=0.4):
    """Returns the number of modes in each ell bin"""
    ls = get_ells()
    nmodes = list(ls[1:] ** 2 - ls[:-1] ** 2)
    lp = ls[-1] ** 2 / ls[-2]
    nmodes.append(lp**2 - ls[-1] ** 2)
    return np.array(nmodes) * 0.5 * fsky


def get_tracer_dNdzs():
    filename = root + "/dNdzs_fullwidth.npz"
    d = np.load(filename)
    return {
        "z_sh": d["z_sh"],
        "dNdz_sh": d["dNdz_sh"],
        "z_cl": d["z_cl"],
        "dNdz_cl": d["dNdz_cl"],
    }


def read_cls():
    d = np.load(root + "/benchmarks_nl_full_clgg.npz")
    ls = d["ls"]
    cls_gg = d["cls"]
    d = np.load(root + "/benchmarks_nl_full_clgs.npz")
    cls_gs = d["cls"]
    d = np.load(root + "/benchmarks_nl_full_clss.npz")
    cls_ss = d["cls"]
    return ls, cls_gg, cls_gs, cls_ss


@pytest.fixture(scope="module")
def set_up():
    par = get_cosmological_parameters()
    cosmo = ccl.Cosmology(
        Omega_c=par["Omega_m"] - par["Omega_b"],
        Omega_b=par["Omega_b"],
        h=par["h"],
        n_s=par["n_s"],
        A_s=par["A_s"],
        w0=par["w0"],
    )
    tpar = get_tracer_parameters()
    Nzs = get_tracer_dNdzs()

    t_g = []
    # we pass unity bias here and will multiply by the actual bias later
    # this allows us to use the same setup
    # for both direct and PTtracer approach
    for Nz, bg in zip(Nzs["dNdz_cl"].T, tpar['b_g']):
        t = ccl.NumberCountsTracer(
            cosmo,
            has_rsd=False,
            dndz=(Nzs["z_cl"], Nz),
            bias=(Nzs["z_cl"], bg*np.ones(len(Nzs["z_cl"]))))
        t_g.append(t)
    t_s = []
    for Nz in Nzs["dNdz_sh"].T:
        t = ccl.WeakLensingTracer(cosmo, dndz=(Nzs["z_sh"], Nz))
        t_s.append(t)
    ells = get_ells()
    raw_truth = read_cls()
    indices_gg = []
    indices_gs = []
    indices_ss = []
    rind_gg = {}
    rind_gs = {}
    rind_ss = {}
    Ng, Ns = len(t_g), len(t_s)
    for i1 in range(Ng):
        for i2 in range(i1, Ng):
            rind_gg[(i1, i2)] = len(indices_gg)
            rind_gg[(i2, i1)] = len(indices_gg)
            indices_gg.append((i1, i2))

        for i2 in range(Ns):
            rind_gs[(i1, i2)] = len(indices_gs)
            rind_gs[(i2, i1)] = len(indices_gs)
            indices_gs.append((i1, i2))

    for i1 in range(Ns):
        for i2 in range(i1, Ns):
            rind_ss[(i1, i2)] = len(indices_ss)
            rind_ss[(i2, i1)] = len(indices_ss)
            indices_ss.append((i1, i2))

    # Sanity checks
    assert np.allclose(raw_truth[0], ells)
    Nell = len(ells)
    tgg, tgs, tss = raw_truth[1:]
    assert tgg.shape == (len(indices_gg), Nell)
    assert tgs.shape == (len(indices_gs), Nell)
    assert tss.shape == (len(indices_ss), Nell)

    # now generate errors
    err_gg = []
    err_gs = []
    err_ss = []
    nmodes = get_nmodes()
    for i1, i2 in indices_gg:
        err_gg.append(
            np.sqrt(
                (
                    tgg[rind_gg[(i1, i1)]] * tgg[rind_gg[(i2, i2)]]
                    + tgg[rind_gg[(i1, i2)]] ** 2
                )
                / nmodes
            )
        )
    for i1, i2 in indices_gs:
        err_gs.append(
            np.sqrt(
                (
                    tgg[rind_gg[(i1, i1)]] * tss[rind_ss[(i2, i2)]]
                    + tgs[rind_gs[(i1, i2)]] ** 2
                )
                / nmodes
            )
        )
    for i1, i2 in indices_ss:
        err_ss.append(
            np.sqrt(
                (
                    tss[rind_ss[(i1, i1)]] * tss[rind_ss[(i2, i2)]]
                    + tss[rind_ss[(i1, i2)]] ** 2
                )
                / nmodes
            )
        )

    tracers1 = {"gg": t_g, "gs": t_g, "ss": t_s}
    tracers2 = {"gg": t_g, "gs": t_s, "ss": t_s}
    truth = {"gg": tgg, "gs": tgs, "ss": tss}
    errors = {"gg": err_gg, "gs": err_gs, "ss": err_ss}
    indices = {"gg": indices_gg, "gs": indices_gs, "ss": indices_ss}
    return cosmo, ells, tracers1, tracers2, truth, errors, indices


@pytest.mark.parametrize("method", ["FKEM"])
@pytest.mark.parametrize("cross_type", ["gg", "gs", "ss"])
def test_cells(set_up, method, cross_type):
    cosmo, ells, tracers1, tracers2, truth, errors, indices = set_up
    t0 = time.time()
    chi2max = 0
    for pair_index, (i1, i2) in enumerate(indices[cross_type]):
        p_of_k_a = p_of_k_a_lin = ccl.DEFAULT_POWER_SPECTRUM
        cls, meta = ccl.angular_cl(
            cosmo,
            tracers1[cross_type][i1],
            tracers2[cross_type][i2],
            ells,
            p_of_k_a=p_of_k_a,
            p_of_k_a_lin=p_of_k_a_lin,
            l_limber='auto',
            non_limber_integration_method=method,
            return_meta=True
        )
        l_limber = meta['l_limber']
        chi2 = (cls - truth[cross_type][pair_index, :]) ** 2 / \
            errors[cross_type][pair_index]**2
        chi2max = max(chi2.max(), chi2max)
        assert np.all(chi2 < 0.3)
    t1 = time.time()
    print(
        f'Time taken for {method} on {cross_type} = {(t1-t0):3.2f};\
        worst chi2 = {chi2max:5.3f}      l_limber = {l_limber}'
    )
