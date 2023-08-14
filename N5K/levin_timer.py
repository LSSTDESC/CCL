import numpy as np
import time
import n5k

import n5k.calculator_levin


def time_run(cls, config):
    c = cls(config)
    c.setup()
    t0 = time.time()
    c.run()
    tf = time.time()
    c.write_output()
    c.teardown()
    print(f'{cls.name} ({config}): t={tf-t0}s')


for cls, config in zip([n5k.N5KCalculatorBase,
                        n5k.N5KCalculatorCCL,
                        n5k.calculator_levin.N5KCalculatorLevin,
                        n5k.calculator_levin.N5KCalculatorLevin,
                        n5k.calculator_levin.N5KCalculatorLevin,
                        n5k.calculator_levin.N5KCalculatorLevin],
                       ['tests/config.yml', 'tests/config_ccl_kernels.yml',
                        'tests/config_levin.yml',
                        'tests/config_levin_precompute.yml',
                        'tests/config_levin_parallelize_ell.yml',
                        'tests/config_levin_parallelize_ell_precompute.yml']):
    time_run(cls, config)
