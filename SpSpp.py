import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import codetiming as ct
from numba import njit, prange


@njit(parallel=True)
def SpSpp(xvec, pvec, kmax, k_unit):
    Sp_space = np.zeros((2*kmax+1, 2*kmax+1, 2*kmax+1))
    Spp_space = np.zeros((2*kmax+1, 2*kmax+1, 2*kmax+1))

    N = xvec.shape[0]//1000

    for a in prange(0, 2*kmax+1):
        for b in prange(0, 2*kmax+1):
            for c in prange(0, 2*kmax+1):
                kvec = np.array([a-kmax, b-kmax, c-kmax])*k_unit
                k2 = kvec.dot(kvec)
                if k2 > (k_unit*kmax)**2:
                    continue
                for i in range(N):
                    ci = xvec[i]
                    pi = pvec[i]
                    for j in range(N):
                        cj = xvec[j]
                        pj = pvec[j]
                        kpi = kvec.dot(pi)
                        kpj = kvec.dot(pj)
                        fac = np.cos(2*np.pi*kvec.dot(ci-cj))
                        Sp_space[a, b, c] += k2*fac*(kpi**2)*(kpj**2)
                        Spp_space[a, b, c] += k2*k2*fac*kpi*kpj*(pi.dot(pj))
    return (Sp_space, Spp_space)


def calc(file, box, kmax=5):
    data = np.loadtxt(file, skiprows=2, usecols=(3, 4, 5, 6, 7, 8))
    xvec = (data[:, :3]+data[:, 3:])/2
    vec = (data[:, 3:]-data[:, :3])
    vecnorm = np.linalg.norm(vec, axis=1)
    assert vecnorm.shape[0] == vec.shape[0]
    pvec = vec/vecnorm[:, np.newaxis]
    print(xvec, pvec)
    N = xvec.shape[0]

    k_unit = 1/box
    k2_set = set()
    # find unique values of k_norm:
    for a in range(0, 2*kmax+1):
        for b in range(0, 2*kmax+1):
            for c in range(0, 2*kmax+1):
                kvec = np.array([a-kmax, b-kmax, c-kmax])
                k2_set.add(kvec.dot(kvec))
    k2 = list(k2_set)
    k2.sort()
    print(k2)

    k2_Sp = dict()
    k2_Spp = dict()
    for x in k2:
        k2_Sp[x] = []
        k2_Spp[x] = []

    t = ct.Timer(name='calc')
    t.start()
    Sp_space, Spp_space = SpSpp(xvec, pvec, kmax, k_unit)
    t.stop()

    SpSpp.parallel_diagnostics(level=4)

    for a in range(0, 2*kmax+1):
        for b in range(0, 2*kmax+1):
            for c in range(0, 2*kmax+1):
                kvec = np.array([a-kmax, b-kmax, c-kmax])
                k2val = kvec.dot(kvec)
                kvec = kvec*k_unit
                k2_Sp[k2val].append(Sp_space[a, b, c])
                k2_Spp[k2val].append(Spp_space[a, b, c])

    k_Sp = []
    k_Spp = []
    for k2, v in k2_Sp.items():
        k_Sp.append([np.sqrt(k2)*k_unit, np.mean(v)])
    for k2, v in k2_Spp.items():
        k_Spp.append([np.sqrt(k2)*k_unit, np.mean(v)])
    k_Sp = np.array(k_Sp)
    k_Spp = np.array(k_Spp)

    np.savetxt('k_Sp.txt', k_Sp)
    np.savetxt('k_Spp.txt', k_Spp)

    # skip k=0 term
    plt.loglog(k_Sp[1:, 0]/k_unit, k_Sp[1:, 1] /
               k_Sp[1:, 0]**6, label="Sp/k^6")
    plt.loglog(k_Spp[1:, 0]/k_unit, k_Spp[1:, 1] /
               k_Sp[1:, 0]**6, label="Spp/k^6")
    plt.loglog(k_Sp[1:, 0]/k_unit, (k_Spp[1:, 1]-k_Sp[1:, 1]) /
               k_Sp[1:, 0]**6, label="(Spp-Sp)/k^6")
    plt.legend()
    plt.xlabel('k index')
    plt.ylabel('k^2<Sp,Spp>_k')
    plt.savefig('k_SpSpp.png', dpi=300)

    return


if __name__ == "__main__":
    file = '/mnt/ceph/users/wyan/ShareWithDrBlakesleyBurkhart/RPY/LD5AlignDipoleNBShort_RPY/result/result800-999/SylinderAscii_999.dat'
    box = 100
    calc(file, box)
