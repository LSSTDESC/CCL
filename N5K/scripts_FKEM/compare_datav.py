import numpy as np
import os
import matplotlib.pyplot as plt 

cltype = 'ss'
index = 0
for i in range(10):
	for j in range(10-i):
# for index in range(55):
# index = 10
	# if(index==50):
		bench = np.load('tests/benchmarks_nl_cl%s.npz'%(cltype))

		fang = np.load('tests/ccl_nonlim_fang_cl%s.npz'%(cltype))

		limber = np.load('tests/ccl_limber_cl%s.npz'%(cltype))
		ls = fang['ls']

		plt.plot(bench['ls'], bench['cls'][index], label='bench, %d-%d'%(i,j))
		plt.plot(fang['ls'], fang['cls'][index], label='fang')
		# plt.plot(limber['ls'], limber['cls'][index], label='limber')
		plt.xscale('log')
		plt.yscale('log')
		plt.legend()
		plt.show()
		# if(i+j<6):
		# 	plt.plot(bench['ls'], fang['cls'][index]/bench['cls'][index]-1., label='gg %d-%d'%(i, i+j))
		# else:
		# 	plt.plot(bench['ls'], fang['cls'][index]/bench['cls'][index]-1.,'--', label='gg %d-%d'%(i, i+j))
		index+=1
	# plt.xscale('log')
	# # plt.ylim(-0.5,0.5)
	# plt.yscale('log')
	# plt.legend()
	plt.savefig('plots/gg_%d-x.png'%(i))
	plt.show()

exit()

# for i in range(10):
# # for index in range(55):
# # index = 10
# 	# if(index==50):
# 	bench = np.load('tests/benchmarks_nl_cl%s.npz'%(cltype))

# 	fang = np.load('tests/ccl_nonlim_fang_cl%s.npz'%(cltype))

# 	limber = np.load('tests/ccl_limber_cl%s.npz'%(cltype))
# 	ls = fang['ls']

# 	# plt.plot(bench['ls'], bench['cls'][index], label='bench')
# 	# plt.plot(fang['ls'], fang['cls'][index], label='fang')
# 	# plt.plot(limber['ls'], limber['cls'][index], label='limber')
# 	# plt.xscale('log')
# 	# # plt.yscale('log')
# 	# plt.legend()
# 	# plt.show()
# 	if(i<6):
# 		plt.plot(bench['ls'], fang['cls'][index]/bench['cls'][index]-1., label='gg %d-%d'%(i, i))
# 	else:
# 		plt.plot(bench['ls'], fang['cls'][index]/bench['cls'][index]-1.,'--', label='gg %d-%d'%(i, i))
# 	index+=(10-i)
# plt.xscale('log')
# plt.ylim(-0.005,0.005)
# # plt.yscale('log')
# plt.legend()
# plt.savefig('plots/gg_auto.png')
# plt.show()

for i in range(10):
# for index in range(55):
# index = 10
	# if(index==50):
	bench = np.load('tests/benchmarks_nl_cl%s.npz'%(cltype))

	fang = np.load('tests/ccl_nonlim_fang_cl%s.npz'%(cltype))

	limber = np.load('tests/ccl_limber_cl%s.npz'%(cltype))
	ls = fang['ls']

	# plt.plot(bench['ls'], bench['cls'][index], label='bench')
	# plt.plot(fang['ls'], fang['cls'][index], label='fang')
	# plt.plot(limber['ls'], limber['cls'][index], label='limber')
	# plt.xscale('log')
	# # plt.yscale('log')
	# plt.legend()
	# plt.show()
	if(i<6):
		plt.plot(bench['ls'], bench['cls'][index], label='gg %d-%d'%(i, i))
	else:
		plt.plot(bench['ls'], bench['cls'][index],'--', label='gg %d-%d'%(i, i))
	index+=(10-i)
plt.xscale('log')
# plt.ylim(-0.005,0.005)
plt.yscale('log')
plt.legend()
plt.savefig('plots/cl_gg_auto.png')
plt.show()