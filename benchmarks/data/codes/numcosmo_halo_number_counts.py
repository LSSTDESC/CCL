import math

try:
    import gi
    gi.require_version('NumCosmo', '1.0')
except Exception:
    pass

from gi.repository import NumCosmo as Nc
from gi.repository import NumCosmoMath as Ncm

# see https://numcosmo.github.io//examples/example_halo_mass_function/

Ncm.cfg_init()
cosmo = Nc.HICosmo.new_from_name(Nc.HICosmo, "NcHICosmoDEXcdm")
reion = Nc.HIReionCamb.new()
prim = Nc.HIPrimPowerLaw.new()

cosmo.add_submodel(reion)
cosmo.add_submodel(prim)

dist = Nc.Distance.new(2.0)

# EH98 transfer function
tf = Nc.TransferFunc.new_from_name("NcTransferFuncEH")
tf.set_CCL_comp(True)
psml = Nc.PowspecMLTransfer.new(tf)
psml.require_kmin(1.0e-3)
psml.require_kmax(1.0e3)

# now compute the mass function
psf = Ncm.PowspecFilter.new(psml, Ncm.PowspecFilterType.TOPHAT)
psf.set_best_lnr0()
mulf = Nc.MultiplicityFunc.new_from_name("NcMultiplicityFuncTinkerMean")
mulf.set_Delta(200)
mf = Nc.HaloMassFunction.new(dist, psf, mulf)

cosmo.props.H0 = 70.0
cosmo.props.Omegab = 0.05
cosmo.props.Omegac = 0.25
cosmo.props.Omegax = 0.70
cosmo.props.Tgamma0 = 2.725
cosmo.props.w = -1.0

nd = 2000
divfac = 1.0 / (nd - 1.0)

psml.prepare(cosmo)
psf.prepare(cosmo)
mf.set_area_sd(200.0)
mf.set_eval_limits(cosmo, math.log(1e14), math.log(1e16), 0.0, 2.0)
mf.prepare(cosmo)

gf = psml.peek_gf()
gf.prepare(cosmo)

# cosmo.params_log_all()
print(cosmo.sigma8(psf))
print(cosmo.E(1.0/0.9 - 1))
print(psml.eval(cosmo, 1/0.9-1, 1.0))
print(gf.eval(cosmo, 1/0.9-1))

print(mf.n(cosmo, math.log(1.0e14), math.log(1.0e16), 0, 2, True))
