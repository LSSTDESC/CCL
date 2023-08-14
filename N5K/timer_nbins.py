import sys
sys.path.append("fftlogx/")
import numpy as np
import time
import n5k
import yaml

config_name = sys.argv[1]
class_name = sys.argv[2]
n_bins_cl = int(sys.argv[3])
n_bins_sh = int(sys.argv[4])

def time_run(cls, config):
    c = cls(config)
    c.setup()
    niter = 10
    ts = np.zeros(niter+1)
    for i in range(niter+1):
        t0 = time.time()
        c.run()
        tf = time.time()
        ts[i] = tf-t0
    tmean = np.mean(ts[1:])
    terr = np.std(ts[1:])/np.sqrt(niter)
    c.write_output()
    c.teardown()
    print("%d %d %f %f" % (n_bins_cl, n_bins_sh, tmean, terr))


with open(config_name) as f:
    conf = yaml.safe_load(f)

conf['select_sh'] = list(range(5-n_bins_sh, 5))
conf['select_cl'] = list(range(10-n_bins_cl, 10))
calc = n5k.n5k_calculator_from_name(class_name)
time_run(calc, conf)
