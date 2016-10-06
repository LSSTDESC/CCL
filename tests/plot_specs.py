import matplotlib.pyplot as plt
import numpy as np

z, dNdzk2, dNdzk1, dNdzk0pt5, dNdz_clust, sigz_src, sigz_clust = np.loadtxt('./specs_output_test.dat', unpack='True')

fig = plt.figure()
plt.plot(z, sigz_clust)
plt.show()
