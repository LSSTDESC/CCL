import sys
sys.path.append("fftlogx/")
import numpy as np
import time
import n5k

def time_run(cls, config):
    c = cls(config)
    c.setup()
    t0 = time.time()
    c.run()
    tf = time.time()
    c.write_output()
    c.teardown()
    print('%s: t=%fs'%(cls.name,tf-t0))

for cls, config in zip([n5k.N5KCalculatorBase,
                        n5k.N5KCalculatorCCL,
                        n5k.N5KCalculatorCCL,
                        n5k.N5KCalculatorFKEM],
                       ['tests/config.yml', 'tests/config.yml',
                        'tests/config_ccl_kernels.yml',
                        'tests/config_nonlim_fang.yml']):
    time_run(cls, config)
