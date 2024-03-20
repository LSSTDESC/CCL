"""
Compare halo model trispectra with Robert Reischke's code. We compare up to
k=10h/Mpc, which is safe enough. At smaller scales, there are 10-20%
differences coming mainly from the I_1_1 integral.

We did the following changes to his code:
 - We change the sampling of phi's for the trapz integrals as they were failing
   to converge with only 100 points (probably less would be enough, too) at
   small scales.
 - We normalize the Tinker bias with a delta function at the lowest Mass that
   absorbes all the difference with rho_bg. We do this only for I_1_1 where it
   is more important and leave the others with norm=1. This is enough for the
   scales considered.
We change the default configuration options:
 - No damping of the large scales for 1h
 - matter_klim and matter_mulim = 0. The default values were causing
   convergence issues in the integrals.
 - Increased the number of bins for the power spectrum log10k_bins = 1200 and
   changed its scale range to logk_min=1e-3 h/Mpc, logk_max=1.15 due to
   errors in the Tpt integrals at the largest scales (even with this we need to
   apply a large scale cut at k=5e-3 h/Mpc).

diff output:

onecov/cov_polyspectra.py:
1766c1766
<         phis = np.linspace(0, np.pi/2, 100)
---
>         phis = np.linspace(0+1e-5, .5*np.pi-1e-5, 8000)

onecov/cov_halo_model.py:
750c750,755
<                 bias = self.bias(bias_dict, hm_prec)
---
>                 bias = self.bias(bias_dict, hm_prec) * self.norm_bias
>                 norm = np.trapz(self.mass_func.dndm * bias *
>                             self.mass_func.m, self.mass_func.m) / self.rho_bg
>                 Abmin = (1-norm)*self.rho_bg/self.mass_func.m[0]
>
>                 integral_x_offset = Abmin*hurlyX[:, :, 0]
753c758
<
---
>                 integral_x += integral_x_offset
892c897
<                 bias = self.bias(bias_dict, hm_prec)
---
>                 bias = self.bias(bias_dict, hm_prec) * self.norm_bias
1011c1016
<             bias = self.bias(bias_dict, hm_prec)
---
>             bias = self.bias(bias_dict, hm_prec) * self.norm_bias

onecov/config.ini
206c206
< small_k_damping_for1h = damped
---
> small_k_damping_for1h = none
213,215c213,215
< log10k_bins = 200
< log10k_min = -3.49
< log10k_max = 2.15
---
> log10k_bins = 1000
> log10k_min = -3.
> log10k_max = 1.15
222,225c222,225
< log10k_min = -3.49
< log10k_max = 2
< matter_klim = 0.001
< matter_mulim = 0.001
---
> log10k_min = -3.
> log10k_max = 1
> matter_klim = 0
> matter_mulim = 0
227c227
< small_k_damping_for1h = damped
---
> small_k_damping_for1h = none
"""

import os
import numpy as np
import pyccl as ccl
import pytest


# CCL halo stuff
h = 0.7

cosmo = ccl.Cosmology(sigma8=0.8, h=h, Omega_c=0.25, Omega_b=0.05, w0=-1.0,
                      wa=0.0, n_s=0.96, Neff=3.046, m_nu=0.0, T_CMB=2.725,
                      transfer_function='eisenstein_hu')
hmf = ccl.halos.MassFuncTinker10(mass_def="200m")
hbf = ccl.halos.HaloBiasTinker10(mass_def="200m")
hmc = ccl.halos.HMCalculator(mass_function=hmf, halo_bias=hbf, log10M_max=17,
                             log10M_min=9)

c = ccl.halos.ConcentrationDuffy08(mass_def='200m')
nfw = ccl.halos.HaloProfileNFW(mass_def="200m", concentration=c)

p2pt = ccl.halos.Profile2pt()

aa = 1/(1+np.linspace(0, 0.3, 1)[::-1])

# Ref data cut to k<=10 h/Mpc
REF_DATA = np.load(os.path.join(os.path.dirname(__file__),
                                "data/trispectrum_terms.npz"))
KVALS = REF_DATA['k']


# CCL Trispectrum
ccl_trispec = {"trispec_1h": ccl.halos.halomod_trispectrum_1h,
               "trispec_2h_31": ccl.halos.halomod_trispectrum_2h_13,
               "trispec_2h_22": ccl.halos.halomod_trispectrum_2h_22,
               "trispec_3h": ccl.halos.halomod_trispectrum_3h,
               "trispec_4h": ccl.halos.halomod_trispectrum_4h}

ccl_tk3d = {"trispec_1h": ccl.halos.halomod_Tk3D_1h,
            "trispec_2h": ccl.halos.halomod_Tk3D_2h,
            "trispec_3h": ccl.halos.halomod_Tk3D_3h,
            "trispec_4h": ccl.halos.halomod_Tk3D_4h,
            "full": ccl.halos.halomod_Tk3D_cNG}


def test_integral_I_1_1():
    ref_I = REF_DATA['I11']
    ccl_I = hmc.I_1_1(cosmo, KVALS*h, 1., nfw)
    norm = nfw.get_normalization(cosmo, np.array([1.]), hmc=hmc)
    ccl_I = ccl_I / norm

    assert ccl_I == pytest.approx(ref_I, rel=1e-3)


def test_integral_I_1_2():
    ref_I = REF_DATA['I12']
    # Transpose because CCL outputs I(k', k) and the refernce data (k, k')
    ccl_I = hmc.I_1_2(cosmo, KVALS*h, 1., nfw, prof_2pt=p2pt,
                      diag=False).T
    norm = nfw.get_normalization(cosmo, np.array([1.]), hmc=hmc)
    ccl_I = ccl_I / norm ** 2 * h**3

    assert ccl_I.diagonal() == pytest.approx(ref_I.diagonal(), rel=1e-3)
    assert ccl_I.diagonal(1) == pytest.approx(ref_I.diagonal(1), rel=1e-3)
    assert ccl_I == pytest.approx(ref_I, rel=1e-3)


def test_integral_I_1_3():
    ref_I = REF_DATA['I13']
    # Transpose because CCL outputs I(k', k) and the refernce data (k, k')
    ccl_I = hmc.I_1_3(cosmo, KVALS*h, 1., nfw, prof_2pt=p2pt).T
    norm = nfw.get_normalization(cosmo, np.array([1.]), hmc=hmc)
    ccl_I = ccl_I / norm ** 3 * h**6

    assert ccl_I.diagonal() == pytest.approx(ref_I.diagonal(), rel=1e-3)
    assert ccl_I.diagonal(1) == pytest.approx(ref_I.diagonal(1), rel=1e-3)
    assert ccl_I == pytest.approx(ref_I, rel=1e-3)


@pytest.mark.parametrize("term", ["trispec_1h", "trispec_2h_31",
                                  "trispec_2h_22", "trispec_3h", "trispec_4h"])
def test_trispectrum_terms(term):
    ref_tk = REF_DATA[term]

    ccl_tk = \
        ccl_trispec[term](cosmo, hmc=hmc, k=KVALS*h, a=aa, prof=nfw)[0] * h**9
    # Transpose because CCL outputs T(k', k) and the refernce data (k, k')
    ccl_tk = ccl_tk.T

    if term == "trispec_4h":
        # We relax the relative difference and remove the largest scales due to
        # errors in the reference Tpt integrals at the largest scales
        assert ccl_tk.diagonal() == pytest.approx(ref_tk.diagonal(), rel=3e-2)
        assert ccl_tk.diagonal(1) == pytest.approx(ref_tk.diagonal(1),
                                                   rel=3e-2)
        sel = KVALS > 5e-3
        assert ccl_tk[sel][:, sel] == pytest.approx(ref_tk[sel][:, sel],
                                                    rel=3e-2)
    else:
        assert ccl_tk.diagonal() == pytest.approx(ref_tk.diagonal(), rel=1e-2)
        assert ccl_tk.diagonal(1) == pytest.approx(ref_tk.diagonal(1),
                                                   rel=1e-2)
        assert ccl_tk == pytest.approx(ref_tk, rel=1e-2)


@pytest.mark.parametrize("term", ["trispec_1h", "trispec_2h", "trispec_3h",
                                  "trispec_4h", "full"])
def test_Tk3D(term):
    ref_tk = REF_DATA[term]

    ccl_tk = ccl_tk3d[term](cosmo, hmc=hmc, lk_arr=np.log(KVALS*h), a_arr=aa,
                            prof=nfw)(KVALS*h, 1) * h**9
    # Transpose because CCL outputs T(k', k) and the refernce data (k, k')
    ccl_tk = ccl_tk.T

    if term in ["trispec_4h", "full"]:
        # We relax the relative difference and remove the largest scales due to
        # errors in the reference Tpt integrals at the largest scales
        assert ccl_tk.diagonal() == pytest.approx(ref_tk.diagonal(), rel=3e-2)
        assert ccl_tk.diagonal(1) == pytest.approx(ref_tk.diagonal(1),
                                                   rel=3e-2)
        sel = KVALS > 5e-3
        assert ccl_tk[sel][:, sel] == pytest.approx(ref_tk[sel][:, sel],
                                                    rel=3e-2)
    else:
        assert ccl_tk.diagonal() == pytest.approx(ref_tk.diagonal(), rel=1e-2)
        assert ccl_tk.diagonal(1) == pytest.approx(ref_tk.diagonal(1),
                                                   rel=1e-2)
        assert ccl_tk == pytest.approx(ref_tk, rel=1e-2)
