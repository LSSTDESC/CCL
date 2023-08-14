# This script makes a version of a benchmark file with just a subset of the bins which would normally be included, for the purpose of testing accuracy in these modified configurations

import numpy as np
import sys

filename = sys.argv[1]
outputfile = sys.argv[2]
type_of_corr = sys.argv[3] # should be gg, gs or ss

# The default configuration has 10 clustering bins and 5 source bins.

# Set the number of bins you want to keep
# For now, these need to be starting from the HIGHEST redshift bin and not skipping any bins.
cl_keep = 7
sh_keep = 2
cl_tot = 10
sh_tot = 5

d = np.load(filename)

ls = d['ls']
cls =d['cls']

print(ls.shape)
print(cls.shape)

# Work out how many to cut:
if type_of_corr=='gg':
    # Correlations are symmetric so have only stored unique ones, need to account for this. I'm sure there is a better way to do this.
    num_spec = 0
    for i in range(0,cl_keep):
        for j in range(i, cl_keep):
            num_spec=num_spec+1
    # Now, cut the clgs to this:
    cls_cut = cls[-num_spec:,:]

elif type_of_corr=='gs':
    # Correlations are not symmetric, need to be a bit more careful.
    # This is a faff so I'm just going to put in by hand which elements I want.
    # Needs to be manually modified each time
    # cls_cut = cls[[32,33,34,37,38,39,42,43,44,47,48, 49]] : this is for shear 3-5 and clustering 7-10
    cls_cut = cls[[18,19,23,24,28,29,33,34,38,39,43,44,48,49]] # this is for shear 4-5 and clustering 4-10
    
elif type_of_corr=='ss':
    #Correlations are symmetric:
    num_spec=0
    for i in range(0,sh_keep):
        for j in range(i, sh_keep):
            num_spec=num_spec+1
    # Now, cut the clgs to this:
    cls_cut = cls[-num_spec:,:]

print(cls_cut.shape)

# Now save the cut benchmarks into a new file.
np.savez(outputfile, cls=cls_cut, ls=ls)

"""# First put the whole input thing in matrix form:
indices=[]

if type_of_corr== 'gg':
    for i1 in range(10):
        for i2 in range(i1, 10):
            indices.append((i1, i2))
elif type_of_corr=='gs':
    for i1 in range(10):    
        for i2 in range(5):
            indices.append((i1, i2))
elif type_of_corr=='ss':
    for i1 in range(5):
        for i2 in range(i1, 5):
            indices.append((i1, i2))

print(indices)"""
