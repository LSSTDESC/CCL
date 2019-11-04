from colossus.halo import mass_so, concentration, mass_defs
from colossus.cosmology import cosmology
import numpy as np

h=0.7
params = {'flat': True, 'H0': 100*h , 'Om0': 0.3, 'Ob0': 0.05, 'sigma8': 0.8, 'ns': 0.96}
cosmo = cosmology.setCosmology('myCosmo', params)

Ms = np.array([1E12, 1E13, 1E14])
Rs_200m = mass_so.M_to_R(Ms * h, 0.0, '200m')/h*0.001
Rs_500c = mass_so.M_to_R(Ms * h, 0.0, '500c')/h*0.001
head_c = 'Mass '
cs_200m_d = concentration.concentration(Ms * h, '200m',
                                        0.0, model = 'duffy08')
head_c += 'c(duffy_200m)'
cs_200c_d = concentration.concentration(Ms * h, '200c',
                                        0.0, model = 'duffy08')
head_c += 'c(duffy_200c)'
cs_200m_b = concentration.concentration(Ms * h, '200m',
                                        0.0, model = 'bhattacharya13')
head_c += 'c(bhattacharya_200m)'
cs_200c_b = concentration.concentration(Ms * h, '200c',
                                        0.0, model = 'bhattacharya13')
head_c += 'c(bhattacharya_200c)'
cs_vir_k = concentration.concentration(Ms * h, 'vir',
                                       0.0, model = 'klypin11')
head_c += 'c(klypin_vir)'
cs_vir_b = concentration.concentration(Ms * h, 'vir',
                                       0.0, model = 'bhattacharya13')
head_c += 'c(bhattacharya_vir)'
cs_200c_p = concentration.concentration(Ms * h, '200c',
                                        0.0, model = 'prada12')
head_c += 'c(prada_200c)'
cs_200c_di = concentration.concentration(Ms * h, '200c',
                                         0.0, model = 'diemer15')
head_c += 'c(diemer_200c)'
Ms_500c, _, _ = mass_defs.changeMassDefinition(Ms * h, cs_200m_d,
                                               0., '200m', '500c')
Ms_500c /= h

np.savetxt("../mdef_bm.txt",
           np.transpose([Ms, Rs_200m, Rs_500c, Ms_500c]),
           header= 'Mass-(200m) Radius-(200m) Radius-(500c), Mass-(500c)')
np.savetxt("../conc_bm.txt",
           np.transpose([Ms,
                         cs_200m_d, cs_200c_d,
                         cs_200m_b, cs_200c_b,
                         cs_vir_k, cs_vir_b,
                         cs_200c_p, cs_200c_di]),
           header = head_c)
