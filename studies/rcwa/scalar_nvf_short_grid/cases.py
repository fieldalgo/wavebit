"""Scalar shape-NVF layers for the short-grid study (docs/theory/rcwa/scalar-nvf-short-grid-study.md).

Every case is a scalar (isotropic) region layer on the shape NVF path. ``overlap`` paints two
overlapping regions, which hands the layer to the boundary (resolved) route.
"""
import contextlib

import numpy as np

import wavebit
from wavebit import Material, rcwa
from wavebit.rcwa._engine import factorization as fz
from wavebit.rcwa._engine import parametric


def _problem(regions, *, lattice=None, background=1.5, theta=.2, pol='s', wavelength=1.3):
    stack = dict(lattice=lattice) if lattice is not None else dict(period=(1., 1.))
    return rcwa.Problem(
        stack=rcwa.Stack(layers=(rcwa.Layer(thickness=.25, background=Material(background), regions=regions),),
                         bottom=Material(2.25), **stack),
        excitation=rcwa.Excitation(wavelength=wavelength, sources=(rcwa.PlaneWave(theta=theta, phi=.3, polarization=pol),)))


def R(geometry, eps):
    return rcwa.Region(geometry=geometry, material=Material(eps))


PENTAGON = [(.32*np.cos(2*np.pi*k/5 + .2), .3*np.sin(2*np.pi*k/5 + .2)) for k in range(5)]

# name -> (builder(x) for a scalar shape parameter x, nominal x, lossless?)
CASES = {
    'rectangle':  (lambda x: _problem((R(rcwa.Rectangle(size=(x, .4)), 4.),)), .3, True),
    'rect-tilted': (lambda x: _problem((R(rcwa.Rectangle(size=(x, .4), angle=.35), 4.),)), .3, True),
    'circle':     (lambda x: _problem((R(rcwa.Circle(radius=x), 12.25),)), .3, True),
    'ellipse':    (lambda x: _problem((R(rcwa.Ellipse(radii=(x, .2), angle=.4), 6.),)), .35, True),
    'polygon':    (lambda x: _problem((R(rcwa.Polygon(vertices=[(x*u/.32, v) for u, v in PENTAGON]), 4.),)), .32, True),
    'overlap':    (lambda x: _problem((R(rcwa.Circle(radius=x, center=(-.12, 0.)), 4.),
                                       R(rcwa.Circle(radius=.22, center=(.15, .05)), 9.))), .25, True),
    'hexagonal':  (lambda x: _problem((R(rcwa.Circle(radius=x), 4.),), lattice=wavebit.Lattice.hexagonal(1.)), .28, True),
    'oblique':    (lambda x: _problem((R(rcwa.Ellipse(radii=(x, .18), angle=.2), 5.),),
                                      lattice=wavebit.Lattice.oblique(1., 1.1, 1.3)), .3, True),
    'lossy':      (lambda x: _problem((R(rcwa.Circle(radius=x), 4. + 1.j),)), .3, False),
    'metallic':   (lambda x: _problem((R(rcwa.Circle(radius=x), -20. + 1.5j),), background=1.),
                   .3, False),
}


@contextlib.contextmanager
def arm(prototype):
    """``prototype=False`` is today's builder; ``True`` the interpolant coefficients."""
    old = fz._NVF_SHORT_GRID_INTERPOLANT
    fz._NVF_SHORT_GRID_INTERPOLANT = bool(prototype)
    parametric._PROJECTOR_CACHE.clear()
    try:
        yield
    finally:
        fz._NVF_SHORT_GRID_INTERPOLANT = old
        parametric._PROJECTOR_CACHE.clear()


def to_np(v):
    if hasattr(v, 'detach'):
        v = v.detach().cpu()
    return np.asarray(v)


@contextlib.contextmanager
def backend_ctx(backend):
    if backend == 'jax':
        import jax
        with jax.enable_x64(True):
            yield
    else:
        yield


def solve(case, grid, orders, backend='numpy', x=None, field=False):
    build, x0, _ = CASES[case]
    problem = build(x0 if x is None else x)
    numerics = rcwa.Numerics(orders=orders, shape_nvf_grid_size=grid)
    if field:
        import dataclasses
        problem = dataclasses.replace(problem, outputs=rcwa.Outputs(fields=True))
    with backend_ctx(backend):
        res = rcwa.run(problem, numerics=numerics, backend=backend)
        out = dict(R=to_np(res.R), T=to_np(res.T))
        if field:
            f = res.field(x=np.array([.11, -.3]), y=np.array([-.07, .2]), z=np.array([.12]))
            out['field'] = to_np(f.values).ravel()
    return out
