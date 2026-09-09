"""Resolve a weakening NVF material seam with graded Gauss quadrature.

Research instrument only: no production dispatch, sampling floor, or tolerance is
changed. It compares the rejected contrast-field exponents after resolving the
narrow region which an ordinary uniform normal grid misses.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from wavebit._core import parametric as P
from wavebit._core.lattice import as_lattice
from contact_material_probe import material_contrast_field

CELL = as_lattice(period=(700., 700.))


def gauss_axis(edges, degree):
    z, w = leggauss(degree)
    lo, hi = np.asarray(edges[:-1]), np.asarray(edges[1:])
    half = (hi-lo)/2
    return ((lo+hi)[:,None]/2+half[:,None]*z).ravel(), (half[:,None]*w).ravel()


def seam_grid(delta, degree):
    if not np.isfinite(delta) or not 0. < abs(delta) <= .01:
        raise ValueError('the seam instrument requires 0 < abs(delta) <= 0.01')
    # Include both face endpoints and the medial line. The logarithmic cells
    # resolve distances down to O(delta), without sampling the seam itself.
    offsets = 700.*np.geomspace(max(abs(delta)*1e-3,1e-12),.1,40)
    edges = np.unique(np.r_[np.arange(0.,701.,25.),350.-offsets,350.+offsets])
    x, wx = gauss_axis(edges, degree)
    y, wy = gauss_axis(np.arange(0.,701.,25.), degree)
    return x,wx,y,wy


def coefficients(field, grid, order):
    x,wx,y,wy=grid
    modes=np.arange(-2*order,2*order+1)
    ex=np.exp(-2j*np.pi*modes[:,None]*x[None,:]/700.)*(wx[None,:]/700.)
    ey=np.exp(-2j*np.pi*modes[:,None]*y[None,:]/700.)*(wy[None,:]/700.)
    return ey@field@ex.T


def shapes(delta):
    return [dict(type='rectangle',lx=300.,ly=300.,cx=-150.,material=4.),
            dict(type='rectangle',lx=300.,ly=300.,cx=150.,material=4.+delta)]


def projector(delta, grid, exponent):
    field=material_contrast_field()
    field.__globals__['_BLEND_EXPONENT']=exponent
    x,wx,y,wy=grid
    X,Y=np.meshgrid(x,y)
    scene=shapes(delta)
    nx,ny=field(np,X,Y,scene,P.plan_gamma(scene,CELL),None,CELL,np.float64)
    return nx*nx,nx*ny,ny*ny


def transmission(delta, grid, exponent, order):
    import wavebit
    from unittest.mock import patch
    projectors=tuple(coefficients(f,grid,order) for f in projector(delta,grid,exponent))
    s=wavebit.rcwa(backend=0,period=(700.,700.),wavelength=900.,top=1.,bot=2.25,
        theta=.2,phi=0.,psi=0.,thickness=(300.,),truncation_order=(order,order),
        normal_grid=4*order+1,structure=[dict(background=1.,shapes=shapes(delta))])
    # Keep the existing NVF material transforms and Maxwell assembly. Only the
    # three projector integrals are supplied by the independent quadrature.
    with patch.object(P,'normal_projector_coeffs',return_value=projectors):
        return float(s.solve().T.sum())


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders',nargs='+',type=int,default=[3,5,7,9])
    parser.add_argument('--degrees',nargs='+',type=int,default=[8,16])
    parser.add_argument('--deltas',nargs='+',type=float,default=[1e-2,1e-4,1e-6])
    parser.add_argument('--exponent',type=float,default=1.)
    args=parser.parse_args()
    for order in args.orders:
        for degree in args.degrees:
            for delta in args.deltas:
                grid=seam_grid(delta,degree)
                zero=transmission(0.,grid,args.exponent,order)
                value=transmission(delta,grid,args.exponent,order)
                print(order,degree,delta,zero,(value-zero)/delta,flush=True)


if __name__=='__main__':
    main()
