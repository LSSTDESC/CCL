import numpy as np
import matplotlib.pyplot as plt

for cltype,nspec in [("gg",55),("gs",50),("ss",15)]:

  plt.figure()

  m_file = np.load("tests/matter_cl{}.npz".format(cltype))
  c_file = np.load("tests/nl/benchmarks_nl_cl{}.npz".format(cltype))

  m_ls = m_file['ls']
  m_cls = m_file['cls']
  c_ls = c_file['ls']
  c_cls = c_file['cls']
  for i in range(nspec):
    #plt.semilogx(c_ls,c_cls[i],color="red")
    #plt.semilogx(c_ls,m_cls[i]-c_cls[i],color="black",linestyle="--")
    plt.semilogx(c_ls,(m_cls[i]-c_cls[i])/np.max(c_cls),label="{}".format(i))
    plt.legend()
  plt.show()
