import os
import timeit
from timeit import default_timer as timer

import numpy as np
import PointCloud as pc
import matplotlib.pyplot as plt

# large uniform random points
box = 50
npar = 100000
points = np.random.uniform(low=0, high=50, size=(npar, 3))
rcut = box/10
boxsize = np.array([box, box, box])

rho = pc.rho(npar, boxsize)

start = timer()
pc.impose_pbc(points, boxsize)
end = timer()
print("impose_pbc ", end-start)

start = timer()
pairs = pc.get_pair(points, boxsize, rcut)
end = timer()
print("get_pair ", end-start)
# print(pairs)

start = timer()
rvec = pc.get_rvec(points, boxsize, rcut, pairs)
end = timer()
print("get_rvec ", end-start)

start = timer()
r, rdf = pc.gen_rdf(rvec, npar, pc.rho(npar, boxsize), rcut=rcut, nbins=200)
end = timer()
print("gen_rdf ", end-start)

plt.plot(r, rdf)
plt.savefig('rdf_uniform.png', dpi=300)
