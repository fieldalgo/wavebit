"""Separate spherical surface moments from discarded material cross terms.

This is a research comparator, not a public tensor solver. The cross terms use
the shared-node, adjoint interpolation described by Oskooi, Kottke & Johnson,
Opt. Lett. 34, 2778 (2009), doi:10.1364/OL.34.002778, Fig. 1. No Meep source
is read. The production diagonal energy diagnostic is not valid for this arm
and is deliberately not included in the reported results.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
import numpy as np
from wavebit._core.backend import array_namespace
from wavebit.fd._shapes._bodies import _as
from wavebit.fd._shapes._exact import clipped_disc_area
from wavebit.fd._lower.occupancy import yee_points
from wavebit.fd._lower.coefficients import mix
from studies.fdtd_subpixel_mie_sphere import sphere_scene, read_result


def sphere_surface_moments(body, points, dx):
    """Exact diagonal surface moments from volume and six clipped face areas.

    In sphere-centred coordinates div(x_i e_i)=1 and x_i n_i=R n_i^2
    on the sphere. Therefore R M_i=V-(u_i A_i+ - l_i A_i-). This needs
    neither volume second moments nor disc second moments. A separate surface
    quadrature measures worst moment error / dx^2 = 1.33e-13 at R/dx<=20.
    This unoptimized research form repeats the volume evaluation.
    """
    xp = array_namespace(points)
    p = points - _as(xp, body.center, points)
    radius = _as(xp, body.radius, points)
    lo, hi = p-dx/2, p+dx/2
    volume = body.exact_fill(points, dx)*dx**3
    moments = []
    for c in range(3):
        a, b = [j for j in range(3) if j != c]
        faces = []
        for q in (lo[..., c], hi[..., c]):
            live = xp.abs(q) < radius
            r = xp.sqrt(xp.where(live, radius**2-q*q, xp.ones_like(q)))
            disk = clipped_disc_area(0., 0., r, lo[..., a], hi[..., a],
                                     lo[..., b], hi[..., b], xp)
            faces.append(xp.where(live, disk, xp.zeros_like(q)))
        moments.append((volume-(hi[..., c]*faces[1]-lo[..., c]*faces[0]))/radius)
    moment = xp.stack(moments, axis=-1)
    nearest = xp.maximum(xp.abs(p)-dx/2, xp.zeros_like(p))
    farthest = xp.abs(p)+dx/2
    # Geometric support, not a tolerance fitted to a cancellation residue.
    cut = ((xp.sum(nearest**2, axis=-1) < radius**2)
           & (xp.sum(farthest**2, axis=-1) > radius**2))
    moment = xp.where(cut[..., None], moment, xp.zeros_like(moment))
    return xp.sum(moment, axis=-1), moment


def cross_action(v, cross, xp):
    """Shared-node K_ij action on staggered components, cross ordered xy/xz/yz.

    Average each component to nodes, multiply, then apply the adjoint average
    back to its destination component. Periodic rolls are harmless here because
    every cross coefficient has compact support far from the domain boundary.
    """
    node = [0.5*(v[j]+xp.roll(v[j], 1, axis=2-j)) for j in range(3)]
    out = [xp.zeros_like(v[0]) for _ in range(3)]
    for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        a = cross[k]*node[j]
        b = cross[k]*node[i]
        out[i] = out[i]+0.5*(a+xp.roll(a, -1, axis=2-i))
        out[j] = out[j]+0.5*(b+xp.roll(b, -1, axis=2-j))
    return xp.stack(out)


def spectral_enclosure(diag, cross):
    """Gershgorin enclosure of the real symmetric constitutive matrix.

    A positive lower endpoint certifies positive definiteness; a negative one
    is inconclusive. This is not a PML stability proof. The stencil's distinct
    off-diagonal neighbours have nonnegative interpolation weights, so applying
    absolute cross coefficients to ones is its absolute row sum.
    """
    radius = cross_action(np.ones_like(diag), abs(cross), np)
    return [float(np.min(diag-radius)), float(np.max(diag+radius))]


class _TensorRuntime:
    """Research-only electric update, with all other runtime calls delegated."""

    def __init__(self, base, be, cross, gain):
        self.base, self.be, self.cross, self.gain = base, be, cross, gain

    def __getattr__(self, name):
        return getattr(self.base, name)

    def update(self, field, curl, alpha, beta):
        value = self.base.update(field, curl, alpha, beta)
        # Static object identity during tracing: the kernel passes params['be']
        # to E and params['minus_bh'] to H. No value-dependent JIT branch.
        if beta is self.be:
            value = value+self.gain*cross_action(curl, self.cross, self.xp)
        return value


def constitutive_arrays(scene, p, eps, exact_moment=False):
    """Return diagonal K at E points and cross K at common nodes, on the host."""
    from wavebit.fd._shapes import occupancy
    body = scene.structures[0].shape.body()
    grid = p.grid
    box = scene.structures[0].shape.support(grid)
    sl = tuple(slice(*box[a]) for a in (2, 1, 0))
    shape = tuple(reversed(grid.n))
    diag = np.ones((3,)+shape)
    for c in range(3):
        pts = yee_points(grid, box, 'E', c)
        f, w = occupancy(body, pts, grid.dx)
        if exact_moment:
            area, m = sphere_surface_moments(body, pts, grid.dx)
            # The prototype retains the production moment normalization rule.
            from wavebit.fd._shapes._cut import _mean_of_the_moment
            w = _mean_of_the_moment(area, m, grid.dx, np)
        diag[(c,)+sl] = 1/mix(np, np.ones_like(f), eps, f, w[..., c])
    pts = yee_points(grid, box, 'E', 0)
    pts[..., 0] -= grid.dx/2
    f = body.exact_fill(pts, grid.dx)
    _, normal = body.surface(pts)
    contrast = f/eps+(1-f)-1/(f*eps+(1-f))
    cross = np.zeros((3,)+shape)
    for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        cross[(k,)+sl] = contrast*normal[..., i]*normal[..., j]
    return diag, cross


def run_case(cpw, eps=2.25, offset=(0., 0., 0.), arm='tensor', periods=24.):
    """One public scene, with only the named constitutive substitution."""
    from wavebit.fd._lower import plan, lower
    from wavebit.fd._runtime import runtime
    from wavebit.fd._kernel.step import initial_state, step
    from wavebit.fd._kernel.sources import waveform_table
    from wavebit.fd._run import execute, _results
    if arm not in ('diagonal', 'moment', 'tensor', 'both'):
        raise ValueError(f'unknown constitutive study arm {arm!r}')
    start = time.perf_counter()
    scene = sphere_scene(cpw, eps, periods=periods, offset=offset)
    rt = runtime('jax')
    with warnings.catch_warnings(record=True) as notes:
        p = plan(scene)
        # Preserve the original controlled comparator after production gains
        # its own constitutive terms. Each arm below supplies its own update.
        p = p.with_static(terms=tuple(t for t in p.terms if t not in p.constitutive))
        params = lower(scene, p, rt)
        diag, cross = constitutive_arrays(scene, p, eps, arm in ('moment', 'both'))
        if arm in ('moment', 'both'):
            # Only E changes: the fixture is nondispersive and nonmagnetic.
            params['be'] = rt.array((p.dt/p.grid.dx)*diag, 'float64')
        if arm in ('tensor', 'both'):
            import jax
            params = dict(params, cross=rt.array(cross, 'float64'))
            @jax.jit
            def go(traced):
                traced = dict(traced, waveform=waveform_table(p, traced, rt))
                tensor_rt = _TensorRuntime(rt, traced['be'], traced['cross'], p.dt/p.grid.dx)
                state = initial_state(p, traced, rt)
                return rt.scan(lambda s, n: step(p, traced, s, n, tensor_rt),
                               state, p.steps, p.schedule)
            state = go(rt.to_device(params))
        else:
            state = execute(p, params, rt)
        result = _results(scene, p, state, rt, params)
    row = read_result(result, eps)
    active_cross = cross if arm in ('tensor', 'both') else np.zeros_like(cross)
    row.update(cpw=cpw, eps=eps, offset=offset, arm=arm, periods=periods,
               seconds=time.perf_counter()-start,
               constitutive_spectrum_enclosure=spectral_enclosure(diag, active_cross),
               warnings=list(dict.fromkeys(str(n.message) for n in notes)))
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cells', default='20,28,40')
    parser.add_argument('--eps', type=float, default=2.25)
    parser.add_argument('--arm', choices=('diagonal', 'moment', 'tensor', 'both'), default='tensor')
    parser.add_argument('--offset', default='0,0,0')
    parser.add_argument('--periods', type=float, default=24.)
    args = parser.parse_args()
    data = []
    for cpw in map(int, args.cells.split(',')):
        row = run_case(cpw, args.eps, tuple(map(float, args.offset.split(','))), args.arm, args.periods)
        data.append(row)
        args.output.write_text(json.dumps(data, indent=2), encoding='utf8')
        print(json.dumps({k:v for k,v in row.items() if k!='pattern'}), flush=True)
