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
    print('{cls.name}: t={tf-t0}s')

for cls, config in zip([n5k.N5KCalculatorBase,
                        n5k.N5KCalculatorCCL,
                        n5k.N5KCalculatorCCL],
                       ['tests/config.yml', 'tests/config.yml',
                        'tests/config_ccl_kernels.yml']):
    time_run(cls, config)
