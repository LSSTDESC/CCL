import sys
sys.path.append("fftlogx/")
import numpy as np
import time
import n5k


def time_run(cls, config, niter):
    c = cls(config)
    c.setup()
    ts = np.zeros(niter+1)
    for i in range(niter+1):
        t0 = time.time()
        c.run()
        tf = time.time()
        ts[i] = tf-t0
        print('t=', ts[i])
    tmean = np.mean(ts[1:])
    terr = np.std(ts[1:])/np.sqrt(niter)
    c.write_output()
    c.teardown()
    print('%s: t=(%f+-%f)s'%(cls.name,tmean,terr))
    return ts


conf = sys.argv[1]
name = sys.argv[2]
niter = int(sys.argv[3])
fname_out = sys.argv[4]

calc = n5k.n5k_calculator_from_name(name)

times = time_run(calc, conf, niter)

if fname_out != 'none':
    np.savez(fname_out, times=times)
