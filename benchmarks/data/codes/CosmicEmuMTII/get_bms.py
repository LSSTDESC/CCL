import numpy as np
import os

cosmo1 = {'name': 'cosmo1', 'Omega_m': 0.3, 'Omega_b': 0.05, 'h': 0.67, 'n_s': 0.96, 'sigma8': 0.8, 'w0': -1.0, 'wa': 0.0, 'omega_nu': 0.0}
cosmo2 = {'name': 'cosmo2', 'Omega_m': 0.3, 'Omega_b': 0.05, 'h': 0.67, 'n_s': 0.96, 'sigma8': 0.8, 'w0': -1.0, 'wa': 0.0, 'omega_nu': 0.001}

def get_bm(cosmo, kind):
    cosmoname = cosmo['name']
    f = open("xstar.dat", "w")
    h = cosmo['h']
    ns = cosmo['n_s']
    s8 = cosmo['sigma8']
    w0 = cosmo['w0']
    wa = cosmo['wa']
    om = cosmo['Omega_m']*h**2
    ob = cosmo['Omega_b']*h**2
    onu = cosmo['omega_nu']
    f.write(f'{om} {ob} {s8} {h} {ns} {w0} {wa} {onu} 0.0\n')
    f.write(f'{om} {ob} {s8} {h} {ns} {w0} {wa} {onu} 1.0\n')
    f.close()
    os.system(f'./P_{kind}/emu.exe')
    for i in range(2):
        os.system(f'mv EMU{i}.txt ../../{cosmoname}_MTII_{kind}_{i}.txt')

get_bm(cosmo1, 'tot')
get_bm(cosmo1, 'cb')
get_bm(cosmo2, 'tot')
get_bm(cosmo2, 'cb')
