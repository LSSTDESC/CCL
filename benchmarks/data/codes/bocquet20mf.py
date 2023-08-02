import numpy as np
import MiraTitanHMFemulator


emu = MiraTitanHMFemulator.Emulator()

hmcosmo = {'Ommh2': 0.147,
           'Ombh2': 0.02205,
           'Omnuh2': 0.005,
           'n_s': 0.96,
           'h': 0.7,
           'sigma_8': 0.81,
           'w_0': -1.05,
           'w_a': 0.1}

Ms = np.geomspace(1E13, 1E15, 32)
zs = np.array([0.0, 1.0])
mfp = emu.predict(hmcosmo, zs, Ms, get_errors=False)[0]
np.savez("../mf_bocquet20mf.py", m=Ms, z=zs, mf=mfp,
         **hmcosmo)

