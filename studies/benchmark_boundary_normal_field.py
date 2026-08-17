"""benchmark_boundary_normal_field.py -- which normal field carries a position gradient?

A multi-shape *smooth* layer keeps the sampled analytic normal field
(``analytic.normal_projector_coeffs``), whose position gradient is poor for a reason that
module states in full: a rectangle's level-set gradient is piecewise constant, so ``cx``
reaches the field only through ``sign()``. The boundary path builds a different field -- the
gradient of the distance to the resolved boundary
(``boundary.distance_gradient_normal_field``) -- in which a vertex coordinate arrives as
arithmetic instead. A *disjoint* layer can be given that field today, without any code change,
by handing ``boundary_plans`` to ``to_conv_mat_smooth_layers``. So: is the swap a fix?

**Why autodiff against finite differences cannot answer it.** Both models make the loss
piecewise smooth in ``cx`` with jumps at pixel crossings. Autodiff returns the slope of the
smooth piece; a central difference with ``h`` below the pixel spacing stays inside that same
piece. The two therefore *cannot* disagree on either field, and they agree while both skip
whatever the jumps carry. Their agreement is self-consistency, not accuracy, and a study that
stops there will report a healthy gradient for a field that has none.

**What this measures instead.** If the loss is absolutely continuous in ``cx``, the integral of
``d/dcx`` across a window equals the change in the loss across it, whatever the derivative does
in between. That is the one comparison a rapidly fluctuating derivative cannot dodge. To keep
it from aliasing -- the derivative really does fluctuate on a sub-pixel scale -- the gradient is
sampled on the *same* grid as the loss, so the trapezoid and the endpoint difference are taken
over identical abscissae.

**The result, and it is one-sided.** On two tilted rectangles the sampled field integrates
high by about 22%, which is the known defect. The grad-of-distance field integrates to the
*wrong sign* -- at every position sampled, at fto 8 and again at fto 12, with a spread about
the mean far too small for sampling noise. On a circle pair the sampled field is already within
about 1% and the replacement is an order of magnitude worse, which is the same verdict the cost
table at the end reaches independently. The swap is not a fix; the field stays as it is.

This script does **not** settle forward accuracy, and deliberately does not try. The two
constructions are still ~2e-4 apart on a total at fto 24, so neither has reached the limit any
ratio between them would have to be measured against, and the ratio one computes is set by
which of them is called the limit. Write-up: ``docs/boundary-modeling.md``.

Runs in about 20 minutes on a laptop. From the repo root::

    conda run -n wavebit python studies/benchmark_boundary_normal_field.py
"""
import pathlib
import time

import numpy as np

import wavebit
from wavebit._core import analytic as _analytic
from wavebit._core import boundary as _boundary
from wavebit._core import clipper as _clipper
from wavebit._core.geometry import StaticGeometry

RESULTS = pathlib.Path(__file__).resolve().parent / 'results'

WL = 0.777
P = 2.0 * WL                    # wide enough that orders other than (0, 0) propagate
NORMAL_GRID = 512
PIX = P / NORMAL_GRID
CFG = dict(period=[P, P], thickness=[0.25], theta=0.35, phi=0.4, psi=0.0,
           wavelength=WL, n_top=1.0, n_bot=1.5)

#: Half-width of the window, in normal-grid pixels, and how many points span it. The window has
#: to be several pixels wide or the jumps are stepped over rather than averaged in; the point
#: count only has to resolve the trapezoid, since the gradient varies by ~1% across it.
HALF_PIXELS = 3.0
N_POINTS = 25


def _rect(cx, cy, lx, ly, angle, n):
    return dict(type='rectangle', cx=cx * P, cy=cy * P, lx=lx * P, ly=ly * P, angle=angle,
                n_index=n)


def _circle(cx, cy, d, n):
    return dict(type='circle', cx=cx * P, cy=cy * P, lx=d * P, n_index=n)


def two_rectangles(cx):
    """Two disjoint tilted rectangles; the first one's centre is the swept parameter."""
    return [dict(n_base=1.0, combine='paint', shapes=[
        dict(_rect(0.28, 0.30, 0.20, 0.15, 0.4, 3.0), cx=cx),
        _rect(0.68, 0.70, 0.20, 0.15, 0.9, 2.2)])]


def two_circles(cx):
    return [dict(n_base=1.0, combine='paint', shapes=[
        dict(_circle(0.28, 0.30, 0.22, 3.0), cx=cx),
        _circle(0.70, 0.72, 0.20, 2.2)])]


CASES = {'two tilted rectangles': (two_rectangles, 0.28 * P, (8, 12)),
         'two circles': (two_circles, 0.28 * P, (8,))}


def geometries(ucell):
    """The two frozen geometries the comparison needs, differing only in the normal field.

    A ``combine='paint'`` layer has to be frozen before it is traced -- whether the shapes
    overlap decides which structure it is, and a tracer cannot be asked. Both slots below are
    that freeze, taken once at the centre of the window since the topology does not change
    across it.

    ``boundary_plans[i] = None`` is not "unknown": it is what ``freeze_geometry`` records for a
    layer it has looked at and found disjoint, and it keeps the layer on the closed-form path.
    Building the plan anyway is what routes the same disjoint layer through the boundary path
    and so swaps its normal field -- the permittivity coefficients are untouched, agreeing to
    1e-15 between the two, so the field is the only thing that moves.
    """
    sampled = StaticGeometry(boundary_plans=[None for _ in ucell])
    swapped = StaticGeometry(boundary_plans=[
        _boundary.plan_layer(layer.get('shapes', []), (P, P)) for layer in ucell])
    return sampled, swapped


def total_reflectance(build, geometry, cx, fto, backend=0):
    solver = wavebit.rcwa(backend=backend, ucell=build(cx), fto=[fto, fto],
                            normal_grid=NORMAL_GRID, **CFG)
    result = solver.conv_solve(geometry=geometry)
    if backend == 1:
        import jax.numpy as jnp
        return jnp.real(jnp.sum(result.R))
    return float(np.real(np.sum(np.asarray(result.R))))


def integral_test(build, cx0, fto, use_boundary):
    """Trapezoid of d(sum R)/dcx across the window, against the change in sum R across it."""
    import jax
    import jax.numpy as jnp                                       # noqa: F401

    geometry = geometries(build(cx0))[1 if use_boundary else 0]
    xs = cx0 + np.linspace(-HALF_PIXELS * PIX, HALF_PIXELS * PIX, N_POINTS)
    grads = np.array([float(jax.grad(
        lambda c: total_reflectance(build, geometry, c, fto, backend=1))(
            jnp.asarray(x))) for x in xs])
    loss = np.array([total_reflectance(build, geometry, x, fto) for x in xs])
    return dict(integral=float(np.trapezoid(grads, xs)),
                change=float(loss[-1] - loss[0]),
                mean=float(grads.mean()), spread=float(grads.std() / abs(grads.mean())),
                wrong_sign=int((np.sign(grads) != np.sign(loss[-1] - loss[0])).sum()))


def normal_field_cost(build, cx0, grid, repeats=3):
    """Milliseconds to build each field once, at one grid, for one layer."""
    shapes = build(cx0)[0]['shapes']
    records, regions = _boundary.plan_layer(shapes, (P, P))
    verts = [_boundary.shape_vertices(np, s, (P, P)) for s in shapes]
    values = _clipper.eval_table(np, records, verts)
    loops = [_clipper.loop_verts(np, loop, values)
             for _shape, region in zip(shapes, regions) for loop in region]

    def timed(fn):
        fn()
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        return (time.perf_counter() - start) / repeats * 1e3

    sampled = timed(lambda: _analytic.normal_vector_field_analytical(
        np, (grid, grid), shapes, (P, P), np.float64))
    distance = timed(lambda: _boundary.distance_gradient_normal_field(
        np, loops, (grid, grid), (P, P), np.float64))
    return sampled, distance, sum(len(loop) for loop in loops)


def main():
    lines = []

    def say(text=''):
        print(text, flush=True)
        lines.append(text)

    say('window: +-%.1f normal-grid pixels (%.3e), %d points, normal_grid = %d'
        % (HALF_PIXELS, HALF_PIXELS * PIX, N_POINTS, NORMAL_GRID))
    say('a ratio of +1 would mean the gradient accounts for the loss exactly')
    say()
    header = ('%-24s %5s %-18s %12s %12s %8s %8s %6s'
              % ('case', 'fto', 'normal field', 'int d/dcx', 'loss change', 'ratio',
                 'spread', 'wrong'))
    say(header)
    say('-' * len(header))
    for name, (build, cx0, ftos) in CASES.items():
        for fto in ftos:
            for use_boundary in (False, True):
                start = time.perf_counter()
                out = integral_test(build, cx0, fto, use_boundary)
                say('%-24s %5d %-18s %+12.4e %+12.4e %8.2f %7.1f%% %3d/%d   (%.0fs)'
                    % (name, fto,
                       'grad-of-distance' if use_boundary else 'sampled',
                       out['integral'], out['change'], out['integral'] / out['change'],
                       out['spread'] * 100, out['wrong_sign'], N_POINTS,
                       time.perf_counter() - start))

    say()
    say('normal-field build cost, one layer, milliseconds')
    header = '%-24s %6s %10s %18s %8s %10s' % ('case', 'grid', 'sampled', 'grad-of-distance',
                                               'ratio', 'segments')
    say(header)
    say('-' * len(header))
    for name, (build, cx0, _ftos) in CASES.items():
        for grid in (256, NORMAL_GRID, 1024):
            sampled, distance, segments = normal_field_cost(build, cx0, grid)
            say('%-24s %6d %10.1f %18.1f %8.0fx %10d'
                % (name, grid, sampled, distance, distance / sampled, segments))
    say()
    say('1024 is what the solver defaults to (spec.RCWASpec.normal_grid).')

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'boundary_normal_field.log').write_text('\n'.join(lines) + '\n')
    print('\nwrote', RESULTS / 'boundary_normal_field.log')


if __name__ == '__main__':
    main()
