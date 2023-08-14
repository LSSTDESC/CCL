import matplotlib.pyplot as plt

import numpy as np

if __name__ == "__main__":
    names = ["FKEM"]  # Add more names to include in the SN vs ell plot

    for name in names:
        comp_vs_bm = np.load("tests/tester_comp_%s.npz"%(name))

        sn_per_l = comp_vs_bm["sn_per_l"]
        cumulative_sn_per_l = np.sqrt(np.cumsum(sn_per_l**2))

        ell = comp_vs_bm["ls"]
        plt.semilogx(ell, comp_vs_bm["sn_per_l"], label="SN %s"%(name))
        plt.semilogx(ell, cumulative_sn_per_l, label="Cumulative SN %s"%(name))

    print(r"l at l<200:", ell[ell<200])
    print(r"cumulative $\Delta\chi^2$ at l<200:", cumulative_sn_per_l[ell<200]**2)
    plt.axhline(0, c="k", lw=1)


    plt.legend(frameon=False)
    plt.xlabel("$\\ell$")
    plt.ylabel(r"$\sqrt{\Delta\chi^2}$")
    plt.xlim(0,3000)
    plt.axvline(200, c="r", lw =2)
    plt.savefig("plots/sn_vs_ell.png")

    plt.figure()

    comp_vs_bm = np.load("tests/tester_comp_FKEM.npz")
    ell = comp_vs_bm["ls"]
    err = np.concatenate([comp_vs_bm["cl_%s_err"%(t)]
                          for t in ["gg", "gs", "ss"]], axis=0)
    cl = np.concatenate([comp_vs_bm["cl_%s"%(t)]
                         for t in ["gg", "gs", "ss"]], axis=0)
    cl_bm = np.concatenate([comp_vs_bm["cl_%s_bm"%(t)]
                            for t in ["gg", "gs", "ss"]], axis=0)

    plt.imshow((cl-cl_bm)/err,
               extent=(ell[0], ell[-1], 0, cl.shape[0]),
               origin="lower")
    plt.colorbar(label="$(C_\\ell - C^{\\rm BM}_\\ell)/\\sigma$")

    plt.axhline(comp_vs_bm["cl_gg"].shape[0], c="k", lw=1)
    plt.axhline(comp_vs_bm["cl_gg"].shape[0] + comp_vs_bm["cl_gs"].shape[0],
                c="k", lw=1)

    plt.xscale("log")
    plt.xlabel("$\\ell$")

    plt.savefig("plots/err_vs_ell_vs_bin.png")
