"""Isotropic region layers for the isotropic-integrated study.

The question: for an *isotropic* patterned layer under ``factorization='nvf'``, how does the
geometry-integrated Popov--Neviere construction of the tensor builder
(``wavebit.rcwa._engine.tensor_shapes``: ``B = I + (P - P e)/a`` per region, its trigonometric
interpolant integrated against the region's analytic indicator before truncation, closed as
``[[A]] [[B]]^-1``) compare with the scalar shape NVF builder (closed-form ``<eps>``, ``<1/eps>``,
a separately transformed projector, truncated matrices multiplied)?

``arm('scalar')`` is today's builder. ``arm('integrated')`` sets the private study switch
``tensor_shapes._ISOTROPIC_INTEGRATED``, which hands every patterned isotropic shape layer of an
ordinary scalar stack (and of a tensor stack) to the integrated builder; the rest of the solve is
the scalar stack's. A frozen geometry is refused under the other arm, so prepare and run under
the same arm (``solve`` does both).

Every case is one 0.25-thick region layer on a unit square cell (``hexagonal`` and ``oblique``:
the named lattices) over a substrate of permittivity 2.25, lit from air at wavelength 1.3,
``theta=0.2``, ``phi=0.3``, s polarization. ``CASES[name] = (build(x), x0, lossless)``: the
problem for one scalar shape parameter ``x``, its nominal value, and whether ``R + T = 1``
holds exactly for the scalar builder (the integrated one does not promise it at finite order).

Notes for measurements:

* ``overlap`` is painted (boundary route) on both arms. Its scalar arm costs about ``grid**2``
  times the polygonized circles' segment count: 9 s CPU at grid 64, 27 s at 128, and tens of
  minutes at the engine default (1024). Pass ``grid=`` explicitly for it.
* On the integrated arm a polygon layer takes the interface-plan (``'disjoint'``) route, and is
  built in place rather than re-centred on its first shape (a polygon refuses ``cx``/``cy``).
* On the integrated arm the floor ``shape_nvf_grid_size >= 4*max(orders)+1`` is not enforced,
  as for tensor layers: the interpolant reads any grid. The scalar arm still refuses below it.
"""
import contextlib

import numpy as np

import wavebit
from wavebit import Material, rcwa
from wavebit.rcwa._engine import parametric
from wavebit.rcwa._engine import tensor_shapes

ARMS = ('scalar', 'integrated')

PERIOD = 1.
WAVELENGTH = 1.3
THETA, PHI = .2, .3
SUBSTRATE = 2.25
THICKNESS = .25


def _problem(regions, *, lattice=None, background=1.5, pol='s'):
    cell = dict(lattice=lattice) if lattice is not None else dict(period=(PERIOD, PERIOD))
    return rcwa.Problem(
        stack=rcwa.Stack(layers=(rcwa.Layer(thickness=THICKNESS, background=Material(background),
                                            regions=regions),),
                         bottom=Material(SUBSTRATE), **cell),
        excitation=rcwa.Excitation(wavelength=WAVELENGTH,
                                   sources=(rcwa.PlaneWave(theta=THETA, phi=PHI, polarization=pol),)))


def R(geometry, eps):
    return rcwa.Region(geometry=geometry, material=Material(eps))


PENTAGON = [(.32*np.cos(2*np.pi*k/5 + .2), .3*np.sin(2*np.pi*k/5 + .2)) for k in range(5)]

CASES = {
    'rectangle':   (lambda x: _problem((R(rcwa.Rectangle(size=(x, .4)), 4.),)), .3, True),
    'rect-tilted': (lambda x: _problem((R(rcwa.Rectangle(size=(x, .4), angle=.35), 4.),)), .3, True),
    'circle':      (lambda x: _problem((R(rcwa.Circle(radius=x), 4.),)), .3, True),
    'ellipse':     (lambda x: _problem((R(rcwa.Ellipse(radii=(x, .2), angle=.4), 6.),)), .35, True),
    'polygon':     (lambda x: _problem((R(rcwa.Polygon(vertices=[(x*u/.32, v) for u, v in PENTAGON]),
                                          4.),)), .32, True),
    # Two overlapping circles of different permittivity: painted, the boundary route.
    'overlap':     (lambda x: _problem((R(rcwa.Circle(radius=x, center=(-.12, 0.)), 4.),
                                        R(rcwa.Circle(radius=.22, center=(.15, .05)), 9.))), .25, True),
    'hexagonal':   (lambda x: _problem((R(rcwa.Circle(radius=x), 4.),),
                                       lattice=wavebit.Lattice.hexagonal(PERIOD)), .28, True),
    'oblique':     (lambda x: _problem((R(rcwa.Ellipse(radii=(x, .18), angle=.2), 5.),),
                                       lattice=wavebit.Lattice.oblique(PERIOD, 1.1, 1.3)), .3, True),
    'lossy':       (lambda x: _problem((R(rcwa.Circle(radius=x), 4. + 1.j),)), .3, False),
    'metallic':    (lambda x: _problem((R(rcwa.Circle(radius=x), -20. + 1.5j),), background=1.),
                    .3, False),
    'high-contrast': (lambda x: _problem((R(rcwa.Circle(radius=x), 12.25),), background=1.), .3, True),
}


def _clear():
    parametric._PROJECTOR_CACHE.clear()


@contextlib.contextmanager
def arm(name):
    """Build isotropic shape layers with the ``'scalar'`` or ``'integrated'`` builder."""
    if name not in ARMS:
        raise ValueError(f'arm must be one of {ARMS}, not {name!r}')
    old = tensor_shapes._ISOTROPIC_INTEGRATED
    tensor_shapes._ISOTROPIC_INTEGRATED = name == 'integrated'
    _clear()
    try:
        yield
    finally:
        tensor_shapes._ISOTROPIC_INTEGRATED = old
        _clear()


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


def problem(case, x=None):
    build, x0, _ = CASES[case]
    return build(x0 if x is None else x)


def numerics(orders, grid=None):
    """``orders`` is ``(Mu, Mv)``, or one int for both."""
    if isinstance(orders, (int, np.integer)):
        orders = (int(orders), int(orders))
    return rcwa.Numerics(orders=tuple(orders), factorization='nvf', shape_nvf_grid_size=grid)


def solve(case, orders, grid=None, backend='numpy', x=None):
    """``(R, T)`` as NumPy arrays, under whichever :func:`arm` is active."""
    with backend_ctx(backend):
        res = rcwa.run(problem(case, x), numerics=numerics(orders, grid), backend=backend)
        return to_np(res.R), to_np(res.T)
