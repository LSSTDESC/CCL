import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import mass_function, bias

h = 0.7
my_cosmo = {'flat': True, 'H0': 100 * h, 'Om0': 0.30, 'Ob0': 0.05, 'sigma8': 0.8, 'ns': 0.96}
cosmo_C = cosmology.setCosmology('my_cosmo', my_cosmo)

def get_mfs(model, mdef):
    m_arr=np.geomspace(1E11,1E15,9) 
    z_arr=np.array([0., 0.5, 1.])
    d_out = []
    d_out.append(m_arr)
    normfac = h**3 * np.log(10)
    for z in z_arr:
        d_out.append(mass_function.massFunction(m_arr * h, z,
                                                mdef = mdef,
                                                model = model,
                                                q_out = 'dndlnM') * normfac)
    np.savetxt("../hmf_" + model + ".txt",
               np.transpose(d_out),
               header = 'M nM(z=0) nM(z=0.5) nM(z=1)')

get_mfs('tinker08', '200m')
get_mfs('press74', 'fof')
get_mfs('sheth99', 'fof')
get_mfs('jenkins01', 'fof')
get_mfs('despali16', 'vir')
get_mfs('bocquet16', '200c')
get_mfs('watson13', '200m')
get_mfs('angulo12', 'fof')

def get_hbs(model, mdef):
    m_arr=np.geomspace(1E11,1E15,9) 
    z_arr=np.array([0., 0.5, 1.])
    d_out = []
    d_out.append(m_arr)
    normfac = h**3 * np.log(10)
    for z in z_arr:
        d_out.append(bias.haloBias(m_arr * h, z = z,
                                   mdef = mdef,
                                   model = model))
    np.savetxt("../hbf_" + model + ".txt",
               np.transpose(d_out),
               header = 'M bh(z=0) bh(z=0.5) bh(z=1)')

get_hbs('tinker10', '200m')
get_hbs('sheth01', 'fof')
get_hbs('bhattacharya11', 'fof')
