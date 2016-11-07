import matplotlib.pyplot as plt
import numpy as np

#z, dNdz_clust, sigz_src, sigz_clust, bias_clust, dNdz_tomo = np.loadtxt('./specs_output_test.dat', unpack='True')
z, dNdz_tomo = np.loadtxt('./specs_test.out', unpack='True')


fig = plt.figure()
plt.plot(z, dNdz_tomo)
plt.show()
