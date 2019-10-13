from colossus.halo import mass_so, concentration, mass_defs
from colossus.cosmology import cosmology
import numpy as np

h=0.7
params = {'flat': True, 'H0': 100*h , 'Om0': 0.3, 'Ob0': 0.05, 'sigma8': 0.8, 'ns': 0.96}
cosmo = cosmology.setCosmology('myCosmo', params)

Ms = np.array([1E12, 1E13, 1E14])
Rs_200m = mass_so.M_to_R(Ms * h, 0.0, '200m')/h*0.001
Rs_500c = mass_so.M_to_R(Ms * h, 0.0, '500c')/h*0.001
cs_200m = concentration.concentration(Ms * h, '200m',
                                      0.0, model = 'duffy08')
Ms_500c, _, _ = mass_defs.changeMassDefinition(Ms * h, cs_200m,
                                               0., '200m', '500c')
Ms_500c /= h

np.savetxt("../mdef_bm.txt",
           np.transpose([Ms, Rs_200m, Rs_500c, cs_200m, Ms_500c]))
