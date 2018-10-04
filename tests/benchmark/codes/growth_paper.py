import numpy as np
import py_cosmo_mad as csm

TCMB = 2.725

z_arr = np.logspace(-2, 3, 10, endpoint=True)

cpar_model1={'om': 0.3,'ol': 0.7 ,'ob':0.05,'hh': 0.7,'s8': 0.8,'ns': 0.96,'w0': -1.0, 'wa': 0.0}
cpar_model2={'om': 0.3,'ol': 0.7 ,'ob':0.05,'hh': 0.7,'s8': 0.8,'ns': 0.96,'w0': -0.9, 'wa': 0.0}
cpar_model3={'om': 0.3,'ol': 0.7 ,'ob':0.05,'hh': 0.7,'s8': 0.8,'ns': 0.96,'w0': -0.9, 'wa': 0.1}
cpar_model4={'om': 0.3,'ol': 0.75,'ob':0.05,'hh': 0.7,'s8': 0.8,'ns': 0.96,'w0': -0.9, 'wa': 0.1}
cpar_model5={'om': 0.3,'ol': 0.65,'ob':0.05,'hh': 0.7,'s8': 0.8,'ns': 0.96,'w0': -0.9, 'wa': 0.1}

growth_factor = np.zeros((5+1, len(z_arr)))
growth_factor[0] = z_arr

for i, cpar in enumerate([cpar_model1, cpar_model2, cpar_model3, cpar_model5, cpar_model4]):
    pcs=csm.PcsPar()
    pcs.background_set(cpar['om'],cpar['ol'],cpar['ob'],cpar['w0'],cpar['wa'],cpar['hh'],TCMB)

    a_arr=1./(z_arr+1)
    gf_arr=np.array([pcs.growth_factor(a) for a in a_arr])

    growth_factor[i+1] = gf_arr

np.savetxt("../growth_allz_cosmomad_ccl1-5.txt", growth_factor.T ,header="z, D(z) CCL1, D(z) CCL2, D(z) CCL3, D(z) CCL4, D(z) CCL5")