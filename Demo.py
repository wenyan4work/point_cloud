import numpy as np
import os
import PointCloud as pc

import matplotlib as mpl
import matplotlib.pyplot as plt


def load_and_plot(filename, boxsize):
    points = np.loadtxt(filename, skiprows=2)
    pc.impose_pbc(points, boxsize)
    rcut = boxsize[0]/4
    npar = len(points)
    rho = pc.rho(npar, boxsize)
    pairs = pc.get_pair(points, boxsize, rcut)
    rvec = pc.get_rvec(points, boxsize, rcut, pairs)
    r, rdf = pc.gen_rdf(rvec, npar, pc.rho(
        npar, boxsize), rcut=rcut, nbins=200)
    # min q = 2*pi/boxsize[0]
    # max q = 500*(min q)
    q = np.array([(2*np.pi/boxsize[0])*(j+1) for j in range(500)])
    Sq = np.zeros(q.shape)
    if len(boxsize) == 3:
        for j in range(len(q)):
            Sq[j] = (pc.Sint3D(q[j], r, rdf, rho))
    else:
        for j in range(len(q)):
            Sq[j] = (pc.Sint2D(q[j], r, rdf, rho))

    plt.clf()
    plt.figure(figsize=(8, 4))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.plot(r, rdf)
    ax1.set_xlabel('$r$')
    ax1.set_ylabel('$g(r)$')
    ax2.plot(q, Sq)
    ax2.set_xlabel(r'$q/ \frac{2\pi k}{L}$')
    ax2.set_ylabel('$S(q)$')
    plt.tight_layout()
    name = os.path.splitext(os.path.basename(filename))[0]
    plt.savefig(name+'.png', dpi=300)


load_and_plot('TestData/FCC.xyz', boxsize=np.array([30, 30, 30]))
load_and_plot('TestData/Colloid2D.xyz', boxsize=np.array([256, 256]))
