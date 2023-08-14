import sys
sys.path.append("fftlogx/")
import numpy as np
import time
import n5k
import os


# First, generate benchmarks if you haven't done so yet.
if not os.path.isfile("tests/ccl_nonlim_fang_clgg.npz"):
    cal_nl = n5k.N5KCalculatorFKEM("tests/config_nonlim_fang.yml")
    cal_nl.setup()
    cal_nl.run()
    cal_nl.write_output()
    cal_nl.teardown()
# exit()

# Generate C_ells for a given method (CCL with limber approximation, in this case)
# cal_lm = n5k.N5KCalculatorCCL("tests/config_ccl_limber.yml")
# cal_lm.setup()
# cal_lm.run()
# cal_lm.write_output()
# cal_lm.teardown()


# Now generate the tester calculator
# cal_test = n5k.N5KCalculatorTester('tests/config_tester_fang.yml')
cal_test = n5k.N5KCalculatorTester('tests/config_tester.yml')
cal_test.setup()
# Compute non-Limber S/N and plots
sn = cal_test.compare('FKEM', 'tests/config_nonlim_fang.yml',
                      plot_stuff=False)
print("Total S/N wrt benchmarks: ", sn)
print("Plots saved in " +
      cal_test.config['output_prefix'] +
      'clcomp*')
