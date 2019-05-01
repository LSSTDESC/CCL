#!/usr/bin/env python
"""
Generate a set of CLASS power spectra across a set of sample points in 
cosmological parameter space, and compare with CCL.
"""
from param_space import *
import os, sys

# Need to specify directory containing 'class' executable
CLASS_ROOT = None
if len(sys.argv) > 1: CLASS_ROOT = sys.argv[1]
assert CLASS_ROOT is not None, \
    "Must specify CLASS_ROOT (as argument or in source file)."

PREFIX = "std" # Prefix to use for this run
NSAMP = 100 # No. of sample points in parameter space
SEED = 10 # Random seed to use for sampling
ZVALS = np.arange(0., 3., 0.5) # Redshifts to evaluate P(k) at

# Define parameter space to sample over
param_dict = {
    'h':            (0.55, 0.8),
    'Omega_cdm':    (0.15, 0.35),
    'Omega_b':      (0.018, 0.052),
    'A_s':          (1.5e-9, 2.5e-9),
    'n_s':          (0.94, 0.98)
}

# Check that expected output data directories exist
class_datadir = "%s/data/class" % os.path.abspath(".")
ccl_datadir = "%s/data/ccl" % os.path.abspath(".")
if not os.path.exists(class_datadir): os.makedirs(class_datadir)
if not os.path.exists(ccl_datadir): os.makedirs(ccl_datadir)

# Get root filename for CLASS and CCL filenames
root = "%s/%s" % (class_datadir, PREFIX)
ccl_root = "%s/%s" % (ccl_datadir, PREFIX)

# Generate sample points on Latin hypercube
sample_points = generate_latin_hypercube( samples=NSAMP, param_dict=param_dict,
                                          class_root=CLASS_ROOT, seed=SEED )
save_hypercube("%s_params.dat" % root, sample_points)


# Generate CLASS .ini files
print("Writing CLASS linear .ini files")
generate_class_ini(sample_points, root="%s_lin_std" % root, 
                   nonlinear=False, redshifts=ZVALS)
generate_class_ini(sample_points, root="%s_lin_pre" % root, 
                   nonlinear=False, redshifts=ZVALS)

print("Writing CLASS nonlinear .ini files")
generate_class_ini(sample_points, root="%s_nl_std" % root, 
                   nonlinear=True, redshifts=ZVALS)
generate_class_ini(sample_points, root="%s_nl_pre" % root, 
                   nonlinear=True, redshifts=ZVALS)


# Run CLASS on generated .ini files
print("Running CLASS on .ini files")
run_class(fname_pattern="%s_lin_std_?????.ini" % root, 
          class_root=CLASS_ROOT, precision=False)
run_class(fname_pattern="%s_lin_pre_?????.ini" % root, 
          class_root=CLASS_ROOT, precision=True)
run_class(fname_pattern="%s_nl_std_?????.ini" % root, 
          class_root=CLASS_ROOT, precision=False)
run_class(fname_pattern="%s_nl_pre_?????.ini" % root, 
          class_root=CLASS_ROOT, precision=True)


# Run CCL for the same sets of parameters
generate_ccl_pspec(sample_points, ccl_root, 
                   class_data_root="%s_lin_std" % root, 
                   zvals=ZVALS, default_params={'mnu': 0.}, mode='std')

generate_ccl_pspec(sample_points, ccl_root, 
                   class_data_root="%s_lin_pre" % root, 
                   zvals=ZVALS, default_params={'mnu': 0.}, mode='pre')

generate_ccl_pspec(sample_points, ccl_root, 
                   class_data_root="%s_nl_std" % root, 
                   zvals=ZVALS, default_params={'mnu': 0.}, 
                   nonlin=True, mode='std')

generate_ccl_pspec(sample_points, ccl_root, 
                   class_data_root="%s_nl_pre" % root, 
                   zvals=ZVALS, default_params={'mnu': 0.}, 
                   nonlin=True, mode='pre')

