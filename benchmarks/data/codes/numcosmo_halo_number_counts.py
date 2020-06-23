import math
import pyccl as ccl

try:
    import gi
    gi.require_version('NumCosmo', '1.0')
except Exception:
    pass

from gi.repository import NumCosmo as Nc
from gi.repository import NumCosmoMath as Ncm
Ncm.cfg_init()

Omega_c = 0.25
Omega_b = 0.05
Omega_k = 0.0
h = 0.7
A_s = 2.1e-9
n_s = 0.96
Neff = 0.0
w0 = -1.0
wa = 0.0

cosmo = Nc.HICosmo.new_from_name(
    Nc.HICosmo, "NcHICosmoDECpl{'massnu-length':<0>}")
cosmo.omega_x2omega_k()
cosmo.param_set_by_name("H0", h * 100)
cosmo.param_set_by_name("Omegak", 0.0)
cosmo.param_set_by_name("w0", w0)
cosmo.param_set_by_name("w1", wa)
cosmo.param_set_by_name("Omegab", Omega_b)
cosmo.param_set_by_name("Omegac", Omega_c)
cosmo.param_set_by_name("ENnu", Neff)

# Set Omega_K in a consistent way
Omega_k = cosmo.Omega_k0()
Omega_v = cosmo.E2Omega_de(0.0)
T_CMB = cosmo.T_gamma0()
ccl_cosmo = ccl.Cosmology(
    Omega_c=Omega_c, Omega_b=Omega_b, Neff=Neff,
    h=h, n_s=n_s, Omega_k=Omega_k,
    w0=w0, wa=wa, Omega_g=0, sigma8=0.8,
    transfer_function='eisenstein_hu',
    T_CMB=T_CMB,
    matter_power_spectrum='linear')

hiprim = Nc.HIPrimPowerLaw.new()
hiprim.param_set_by_name("ln10e10ASA", math.log(1.0e10 * A_s))
hiprim.param_set_by_name("n_SA", n_s)
cosmo.add_submodel(hiprim)
dist = Nc.Distance.new(5.0)
dist.prepare(cosmo)
tf_eh = Nc.TransferFuncEH.new()
tf_eh.props.CCL_comp = True
psml = Nc.PowspecMLTransfer.new(tf_eh)
psml.require_kmin(1.0e-3)
psml.require_kmax(1.0e3)
psml.prepare(cosmo)

fact = (0.8 / psml.sigma_tophat_R(cosmo, 1.0e-7, 0.0, 8.0 / cosmo.h()))**2
hiprim.param_set_by_name("ln10e10ASA", math.log(1.0e10 * A_s * fact))
psml.require_kmin(1.0e-3)
psml.require_kmax(1.0e3)
psml.prepare(cosmo)

# now compute the mass function
psf = Ncm.PowspecFilter.new(psml, Ncm.PowspecFilterType.TOPHAT)
psf.set_best_lnr0()
psf.set_zf(2.5)
mulf = Nc.MultiplicityFunc.new_from_name("NcMultiplicityFuncTinkerMean")
mulf.set_Delta(200)
mf = Nc.HaloMassFunction.new(dist, psf, mulf)

psf.prepare(cosmo)
mf.set_area_sd(200.0)
mf.set_prec(1.0e-9)
mf.set_eval_limits(cosmo, math.log(1e14), math.log(1e16), 0.0, 2.5)
mf.prepare(cosmo)

gf = psml.peek_gf()
gf.prepare(cosmo)

cosmo.params_log_all()
print(cosmo.sigma8(psf))
print(cosmo.E(1.0/0.9 - 1))
print(psml.eval(cosmo, 1/0.9-1, 1.0))
print(gf.eval(cosmo, 1/0.9-1))

for m in [1e14, 5e14, 1e15, 5e15]:
    print(
        math.log10(m), mf.dn_dlnM(cosmo, math.log(m), 1/0.9-1) * math.log(10))

with open("../numcosmo_cluster_counts.txt", "w") as fp:
    fp.write("# counts  mmin  mmax  zmin  zmax\n")

    nc = mf.n(cosmo, math.log(1.0e14), math.log(2.0e14), 0, 2, True)
    fp.write(
        "%20.12g %20.12g %20.12g %20.12g %20.12g\n" % (nc, 1e14, 2e14, 0, 2))

    nc = mf.n(cosmo, math.log(2.0e14), math.log(1.0e15), 0, 2, True)
    fp.write(
        "%20.12g %20.12g %20.12g %20.12g %20.12g\n" % (nc, 2e14, 1e15, 0, 2))

    nc = mf.n(cosmo, math.log(1.0e15), math.log(1.0e16), 0, 2, True)
    fp.write(
        "%20.12g %20.12g %20.12g %20.12g %20.12g\n" % (nc, 1e15, 1e16, 0, 2))
