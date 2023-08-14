import sys
sys.path.append("fftlogx/")
import numpy as np
import time
import n5k
import yaml

config_name = sys.argv[1]
name = sys.argv[2]
lmax = int(sys.argv[3])
width = sys.argv[4]
n_bins_sh = int(sys.argv[5])
n_bins_cl = int(sys.argv[6])

with open(config_name) as f:
    conf = yaml.safe_load(f)

conf_test = {'output_prefix': 'outputs/tester',
             'benchmark_prefix': 'tests/benchmarks_nl_cut_sh2_cl7'}
if width != 'none':
    name_lc = name.lower()
    conf_test['benchmark_prefix'] = f'tests/benchmarks_nl_cut_sh2_cl7_{width}'
    conf['dndz_file'] = f'input/dNdzs_{width}width.npz'
    conf['kernel_file'] = f'input/kernels_{width}width.npz'
    conf['output_prefix'] = f'outputs/{name_lc}_{width}'
    conf['select_sh'] = list(range(5-n_bins_sh, 5))
    conf['select_cl'] = list(range(10-n_bins_cl, 10))
    conf_test['select_sh'] = list(range(5-n_bins_sh, 5))
    conf_test['select_cl'] = list(range(10-n_bins_cl, 10))


ct = n5k.N5KCalculatorTester(conf_test)
ct.setup_alt(conf)
sn = ct.compare_alt(name, conf)
#print('after compare')
fn_sn = conf_test['output_prefix']+'_comp_'+name+'.npz'
f_sn = np.load(fn_sn)
ids = f_sn['ls'] <= lmax
sn_lmax = np.sqrt(np.sum(f_sn['sn_per_l'][ids]**2))
print(f"Total S/N = {sn}")
print(f"S/N to ell={lmax} = {sn_lmax}")
