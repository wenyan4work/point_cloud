import numpy as np
import scipy as sp
import scipy.spatial as ss
import scipy.special as sp
import scipy.integrate as si
import numba as nb

# @nb.njit


def rho(npar, boxsize):
    vol = 1
    for l in boxsize:
        vol = vol*l
    return float(npar)/float(vol)


@nb.njit
def closest_point(target, points):
    target = np.array(target)
    points = np.array(points)
    distance = np.zeros(len(points))
    for j in range(len(points)):
        distance[j] = np.linalg.norm(points[j]-target)
    ind = np.argmin(distance)
    return ind


@nb.njit
def closest_point1d(target, points):
    distance = np.zeros(len(points))
    for j in range(len(points)):
        distance[j] = np.abs(points[j]-target)
    ind = np.argmin(distance)
    return ind


@nb.njit
def get_closetimage(target, source, boxsize):
    dim = target.shape[0]
    assert source.shape[0] == dim
    image = np.zeros(dim)
    for i in range(dim):
        candidates = np.array(
            [source[i], source[i]-boxsize[i], source[i]+boxsize[i]])
        ind = closest_point1d(target[i], candidates)
        image[i] = candidates[ind]
    return image


def get_pair(coords, boxsize, rcut=None):
    ''' generate pairs using cKDTree '''
    tree = ss.cKDTree(data=coords, boxsize=boxsize)
    boxsize = np.array(boxsize)
    if rcut == None:
        rcut = np.sqrt(boxsize.dot(boxsize))/10
    # this returns only pairs (i<j)
    pairs = np.array(list(tree.query_pairs(r=rcut)), dtype=np.int)
    npairs = len(pairs)
    pairs_array = np.zeros(shape=(npairs*2, 2), dtype=np.int)
    pairs_array[:npairs, :] = pairs
    pairs_array[npairs:, :] = pairs[:, [1, 0]]
    return pairs_array


@nb.njit
def impose_pbc(coords, boxsize):
    dim = len(boxsize)
    for p in coords:
        for i in range(dim):
            while p[i] < 0:
                p[i] = p[i]+boxsize[i]
            while p[i] > boxsize[i]:
                p[i] = p[i]-boxsize[i]

@nb.njit
def get_rvec(coords, boxsize, rcut, pairs):
    rvec = np.zeros((len(pairs), coords.shape[1]))
    index = 0
    for pair in pairs:
        id0 = pair[0]
        id1 = pair[1]
        pos0 = coords[id0]
        pos1 = coords[id1]
        vec01 = pos1-pos0
        if np.linalg.norm(vec01) < rcut:
            rvec[index] = vec01
        else:  # fix periodic image
            image = get_closetimage(pos0, pos1, boxsize)
            rvec[index] = image-pos0
        index = index+1
    return rvec

def gen_rdf(rvec, npar, density, rcut=None, nbins=20, print_msg=False):
    rnorm = np.linalg.norm(rvec, axis=1)
    lb = 0
    if rcut == None:
        ub = np.max(rnorm)
    else:
        ub = rcut
    dim = rvec.shape[1]
    bins = np.linspace(lb, ub, nbins)
    count, bins = np.histogram(rnorm, bins)
    if print_msg:
        print('count', count)
        print('bins', bins)
    # scale with vol and density
    vol = np.zeros(count.shape)
    if dim == 2:    # area = pi(r1^2-r0^2)
        for i in range(nbins-1):
            vol[i] = np.pi*(bins[i+1]**2-bins[i]**2)
    elif dim == 3:  # area = 4pi/3(r1^3-r0^3)
        for i in range(nbins-1):
            vol[i] = (4.0/3.0)*np.pi*(bins[i+1]**3-bins[i]**3)
    rdf = count/(npar*vol*density)
    r = 0.5*(bins[:-1]+bins[1:])
    return r, rdf

# @nb.njit


def Sint3D(q, r, gr, rho):
    f = np.sin(q*r)*r*(gr-1)
    return 1+4*np.pi*rho*si.trapz(f, r)/q


def Sint2D(q, r, gr, rho):
    f = sp.jv(0, q*r)*r*(gr-1)
    return 1+2*np.pi*rho*si.trapz(f, r)
