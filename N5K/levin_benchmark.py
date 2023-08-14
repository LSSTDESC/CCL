import numpy as np
import time
import n5k
import os


# First, generate benchmarks if you haven't done so yet.
if not os.path.isfile("tests/benchmarks_nl_clgg.npz"):
    cal_nl = n5k.N5KCalculatorCCLNonLimber("tests/config_nl.yml")
    cal_nl.setup()
    cal_nl.run()
    cal_nl.write_output()
    cal_nl.teardown()


# Generate C_ells for a given method (CCL with limber approximation, in this case)
import n5k.calculator_levin
cal_lm = n5k.calculator_levin.N5KCalculatorLevin("tests/config_levin.yml")
cal_lm.setup()
cal_lm.run()
cal_lm.write_output()
cal_lm.teardown()


# Now generate the tester calculator
cal_test = n5k.N5KCalculatorTester('tests/config_tester.yml')
cal_test.setup()
# Compute non-Limber S/N and plots
sn = cal_test.compare('Levin', 'tests/config_levin.yml',
                      plot_stuff=True)
print("Total S/N wrt benchmarks: ", sn)
print("Plots saved in " +
      cal_test.config['output_prefix'] +
      'clcomp*')
