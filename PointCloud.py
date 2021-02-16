import numpy as np
import scipy as sp
import scipy.spatial as ss
import scipy.special as sp
import scipy.integrate as si
import numba as nb

@nb.njit
def rho(npar,boxsize):
    vol=1
    for l in boxsize:
        vol=vol*l
    return float(npar)/float(vol)

@nb.njit
def closest_point(target, points):
    target=np.array(target)
    points=np.array(points)
    distance=[]
    for p in points:
        distance.append(np.linalg.norm(p-target))
    distance=np.array(distance)
    ind=np.argmin(distance)
    return points[ind],ind

@nb.njit
def closest_point1d(target, points):
    distance=[]
    for p in points:
        distance.append(np.abs(p-target))
    distance=np.array(distance)
    ind=np.argmin(distance)
    return points[ind],ind

@nb.njit
def get_closetimage(target,source,boxsize):
    dim=target.shape[0]
    assert source.shape[0]==dim
    image=np.zeros(dim)
    for i in range(dim):
        pos, ind= closest_point1d(target[i],[source[i],source[i]-boxsize[i],source[i]+boxsize[i]])
        image[i]=pos
    return image

# generate pairs using cKDTree
def get_pair(coords, boxsize, rcut=None):
    tree = ss.cKDTree(data=coords,boxsize=boxsize)
    boxsize=np.array(boxsize)
    if rcut==None:
        rcut=np.sqrt(boxsize.dot(boxsize))/10
    pairs=tree.query_pairs(r=rcut) # this returns only pairs (i<j)
    pairs2=set()
    for p in pairs:
        pairs2.add((p[1],p[0]))
    pairs.update(pairs2)
    return pairs

@nb.njit
def impose_pbc(coords,boxsize):
    dim=len(boxsize)
    for p in coords:
        for i in range(dim):
            while p[i]<0:
                p[i]=p[i]+boxsize[i]
            while p[i]>boxsize[i]:
                p[i]=p[i]-boxsize[i]
                
#@nb.njit
def get_rvec(coords,boxsize,rcut,pairs):
    rvec=np.zeros((len(pairs),coords.shape[1]))
    index=0
    for pair in pairs:
        id0=pair[0]
        id1=pair[1]
        pos0=coords[id0]
        pos1=coords[id1]
        vec01=pos1-pos0
        if np.linalg.norm(vec01)<rcut:
            rvec[index]=vec01
        else: # fix periodic image
            image=get_closetimage(pos0,pos1,boxsize)
            rvec[index]=image-pos0
        index=index+1
    return rvec

#@nb.njit
def gen_rdf(rvec,npar,density,rcut=None,nbins=20,print_msg=False):
    rnorm=np.linalg.norm(rvec,axis=1)
    lb=0
    if rcut==None:
        ub=np.max(rnorm)
    else:
        ub=rcut
    dim=rvec.shape[1]
    bins=np.linspace(lb,ub,nbins)
    count,bins=np.histogram(rnorm,bins)
    if print_msg:
        print('count', count)
        print('bins', bins)
    # scale with vol and density
    vol=np.zeros(count.shape)
    if dim==2:    # area = pi(r1^2-r0^2)   
        for i in range(nbins-1):
            vol[i]=np.pi*(bins[i+1]**2-bins[i]**2)
    elif dim==3:  # area = 4pi/3(r1^3-r0^3) 
        for i in range(nbins-1):
            vol[i]=(4.0/3.0)*np.pi*(bins[i+1]**3-bins[i]**3)
    rdf=count/(npar*vol*density)
    r=0.5*(bins[:-1]+bins[1:])
    return r,rdf

@nb.njit
def Sint3D(q,r,gr,rho):
    f=np.sin(q*r)*r*(gr-1)
    return 1+4*np.pi*rho*si.trapz(f,r)/q

def Sint2D(q,r,gr,rho):
    f=sp.jv(0,q*r)*r*(gr-1)
    return 1+2*np.pi*rho*si.trapz(f,r)