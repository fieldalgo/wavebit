"""Experimental sign-independent complex NVF projector; no production dispatch.

For a positive weighted normal dyad M, J=M(1,i) has nonzero norm whenever
trace(M)>0. At a regular interface M is proportional to n n^T, so J J^H / |J|^2
is exactly n n^T there. Coincident opposite normals reinforce as dyads instead
of cancelling as vectors. Vanished interfaces remain as extension geometry;
whether their finite-order effect is acceptable must be tested, not assumed.
"""

from wavebit.compat import legacy as _legacy_wavebit
import inspect
import numpy as np
from wavebit.rcwa._engine import parametric as P
from wavebit.rcwa._engine import factorization as F


def field_builder():
    source=inspect.getsource(P._piece_field)
    marker='    # 3. The sum.'
    assert source.count(marker)==1
    source=source[:source.index(marker)]+'''
    a, b, c = zero, zero, zero
    for i, frame in frames.items():
        u = w[index[(i, '+u')]] + w[index[(i, '-u')]]
        v = w[index[(i, '+v')]] + w[index[(i, '-v')]]
        a = a + u * frame.cos**2 + v * frame.sin**2
        b = b + (u-v) * frame.cos * frame.sin
        c = c + u * frame.sin**2 + v * frame.cos**2
    norm2 = a*a + 2*b*b + c*c
    xx = (a*a+b*b)/norm2
    yy = (c*c+b*b)/norm2
    xy = ((a+c)*b + 1j*(b*b-a*c))/norm2
    return xx, xy, xp.conj(xy), yy
'''
    namespace=dict(P.__dict__)
    exec(compile(source,'<Jones contact instrument>','exec'),namespace)
    raw=namespace['_piece_field']
    def build(xp,X,Y,shapes,plan,slots,lattice,dtype):
        if not shapes or any(s.get('type','rectangle')!='rectangle' for s in shapes):
            raise ValueError('this Jones instrument currently requires rectangles')
        return raw(xp,X,Y,shapes,plan,slots,lattice,dtype)
    return build


def matrices(xp,shapes,background,hu,hv,lattice,dtype,shape_nvf_grid_size=256,plan=None):
    # Explicit fixture grid in the same internal Cartesian origin as _piece_field.
    from wavebit.lattice import is_axis_aligned
    from wavebit.rcwa._engine.boundary import _k_grids
    if not is_axis_aligned(lattice):
        raise ValueError('the probe grid requires an axis-aligned cell')
    real_dtype=xp.float64 if dtype==xp.complex128 else xp.float32
    x=(xp.arange(shape_nvf_grid_size,dtype=real_dtype)+.5)*lattice[0][0]/shape_nvf_grid_size
    y=(xp.arange(shape_nvf_grid_size,dtype=real_dtype)+.5)*lattice[1][1]/shape_nvf_grid_size
    X,Y=xp.meshgrid(x,y,indexing='xy')
    # Production builds the layer with its first shape at the centre of the cell and carries it
    # by a phase (2026-09-11); the instrument samples the same way, so what the comparison
    # reads is the field and not the sampling origin.
    ax,ay=shapes[0].get('cx',0.),shapes[0].get('cy',0.)
    built=[dict(shapes[0],cx=0.,cy=0.)]
    built+=[dict(s,cx=s.get('cx',0.)-ax,cy=s.get('cy',0.)-ay) for s in shapes[1:]]
    if plan is None:
        plan=P.plan_gamma(built,lattice)
    fields=field_builder()(xp,X,Y,built,plan,None,lattice,real_dtype)
    KY,KX=_k_grids(xp,hu,hv,real_dtype)
    phase=xp.exp(xp.astype(KY*(ay/lattice[1][1])+KX*(ax/lattice[0][0]),dtype)*(-2j*np.pi))
    cs=[P.conv_matrix_2d(xp,P._sampled_coeffs(xp,xp.astype(f,dtype),hu,hv)*phase,hu,hv)
        for f in fields]
    eps,inv=P.assemble_eps_coeffs(xp,shapes,background,hu,hv,lattice,dtype)
    E=P.conv_matrix_2d(xp,eps,hu,hv);I=P.conv_matrix_2d(xp,inv,hu,hv)
    return F._nvf_assemble(xp,E,I,cs[0],cs[1],cs[3],cs[2])


def main():
    import argparse
    from unittest.mock import patch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders',nargs='+',type=int,default=[3,7,11])
    parser.add_argument('--grids',nargs='+',type=int,default=[127,128,256])
    args=parser.parse_args()
    for order in args.orders:
        for grid in args.grids:
            def total(gap,delta,one=False):
                shapes=([dict(type='rectangle',lx=350.,ly=700.-gap,material=12.25)] if one else
                    [dict(type='rectangle',lx=300.,ly=300.,cx=-150.-gap/2,material=4.),
                     dict(type='rectangle',lx=300.,ly=300.,cx=150.+gap/2,material=4.+delta)])
                s=_legacy_wavebit.rcwa(backend='numpy',period=(700.,700.),wavelength=900.,top=1.,bot=2.25,
                    theta=.2,phi=0.,psi=0.,thickness=(300.,),truncation_order=(order,order),
                    shape_nvf_grid_size=grid,structure=[dict(background=1.,shapes=shapes)])
                with patch.object(P,'to_conv_mat_parametric_nvf',matrices):
                    return float(np.sum(s.solve().T))
            for name,one,contrast in [('period',True,0.),('pair',False,5.),('equal',False,0.)]:
                base=total(0.,contrast,one)
                changes=[((total(g,contrast,one)-base)/g if name!='equal' else
                          (total(0.,g,one)-base)/g) for g in (1e-4,1e-6)]
                print(order,grid,name,base,*changes,flush=True)


if __name__=='__main__':
    main()
