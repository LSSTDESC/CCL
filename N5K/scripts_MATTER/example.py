import numpy as np
import time
import n5k


# The evaluation of challenge entries will also include accuracy and
# integratbility - this script is just to show an example of timing
# an entry.

def time_run(cls, config):
    c = cls(config)
    c.setup()
    t0 = time.time()
    c.run()
    tf = time.time()
    c.write_output()
    c.teardown()
    print(f'{cls.name}: t={tf-t0}s')

#for cls, config in zip([n5k.N5KCalculatorBase, n5k.N5KCalculatorCCL, n5k.N5KCalculatorCCL, n5k.N5KCalculatorMATTER], ['tests/config.yml', 'tests/config.yml', 'tests/config_ccl_kernels.yml', 'tests/config_matter.yml']):
for cls, config in zip([n5k.N5KCalculatorCCL, n5k.N5KCalculatorMATTER], ['tests/config_ccl_kernels.yml','tests/config_matter.yml']):
#for cls, config in zip([n5k.N5KCalculatorMATTER], ['tests/config_matter.yml']):
    time_run(cls, config)
