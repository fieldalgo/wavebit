"""Does the certified edge/dual-face cylinder construction extend to a sphere?

Research only: Werner, Bauer and Cary (2013), sections 4 and 7. All public
scene checks run first; a new compact eight-triplet electric operator then
needs its own host-side energy/CFL certificate. Production is unchanged.
"""
import dataclasses
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from studies.fdtd_subpixel_triplet_cylinder import gather, scatter
from wavebit.fd._shapes import Sphere
from wavebit.fd._kernel.constitutive import ConstitutiveTerm
# The eight triplets, their local map and the per-node maps moved into the package unchanged
# (src/wavebit/fd/_lower/triplet.py, 2026-09-26), where the public stencil's baseline reads them.
# These names are the package's own functions, kept for this study and its tests.
from wavebit.fd._lower.triplet import TRIPLETS as ORIENTATIONS, sphere_maps  # noqa: F401
from wavebit.fd._lower.triplet import triplet_map as local_map  # noqa: F401


def edge_fraction(point, dx, axis, radius, xp=np):
    """The exact ball chord inside a component's edge, now public.

    This body moved to `src/wavebit/fd/_shapes/_bodies.py::Sphere.exact_edge_fill`
    unchanged; what is left here is the study's spelling of it. `point` is
    relative to the centre, which is a sphere at the origin. The namespace is
    read off `point` by the public form, so `xp` is kept only for the callers
    that still pass it. Since 2026-09-26 `sphere_maps` (moved to
    `src/wavebit/fd/_lower/triplet.py`) calls the public form itself, and nothing in
    this study calls this one: its tests do. Bit-identity against the form this replaces is
    held in `tests/test_fd_triplet_sphere.py::test_the_study_helpers_are_the_public_forms_to_the_bit`.
    """
    return Sphere(radius).exact_edge_fill(point, dx, axis)


def face_fraction(point, dx, axis, radius, xp=np):
    """Exact sphere cut on a component's dual square, with exact full/empty cases.

    The twin of `edge_fraction`, and the same move: the body is now
    `src/wavebit/fd/_shapes/_bodies.py::Sphere.exact_face_fill`.
    """
    return Sphere(radius).exact_face_fill(point, dx, axis)


def action(vector, matrices, xp=np, cross_only=False):
    out = [xp.zeros_like(vector[i]) if cross_only else vector[i] for i in range(3)]
    for k, orientation in enumerate(ORIENTATIONS):
        local = [gather(vector[i], orientation[i], i, xp) for i in range(3)]
        for i in range(3):
            delta = xp.zeros_like(local[i])
            for j in range(3):
                if i != j:
                    delta = delta+matrices[k, ..., i, j]*local[j]
                elif not cross_only:
                    delta = delta+(matrices[k, ..., i, i]-1)*local[j]
            # Eight permutations, each covering every component exactly once.
            out[i] = out[i]+scatter(delta, orientation[i], i, xp)/8
    return xp.stack(out)


def diagonal(matrices):
    out = [np.ones(matrices.shape[1:-2]) for _ in range(3)]
    for k, orientation in enumerate(ORIENTATIONS):
        for i in range(3):
            out[i] += scatter(matrices[k, ..., i, i]-1, orientation[i], i)/8
    return np.stack(out)


def certificate(scene, p, term, matrices, edges, faces, params):
    from wavebit.fd._lower.checks import cfl_axis_sum
    if term.field != 'E' or term.masks or len(p.constitutive) != 1:
        raise ValueError('SphereTripletScopeRefused: only one unmasked electric sphere')
    for array in (edges, faces):
        if not np.all(np.isfinite(array) & (array >= 0) & (array <= 1)):
            raise ValueError(f'SphereTripletGeometryRefused: fraction range {array.min()}..{array.max()}')
    layers = dict(p.pml_cells)
    for a, ((lo, hi), size) in enumerate(zip(term.box, (p.nz, p.ny, p.nx))):
        axis = 'zyx'[a]
        if not layers.get(axis+'-', 0) <= lo < hi <= size-layers.get(axis+'+', 0):
            raise ValueError('SphereTripletRegionRefused: patch exceeds lossless interior')
        for end in (0, -1):
            if np.any(np.take(matrices, end, axis=a+1) != np.eye(3)):
                raise ValueError('SphereTripletHaloRefused: patch ends are not exact vacuum')
    if not np.all(np.isfinite(matrices)) or not np.array_equal(matrices, np.swapaxes(matrices, -1, -2)):
        raise ValueError('SphereTripletSymmetryRefused: nonfinite or nonsymmetric block')
    eig = np.linalg.eigvalsh(matrices)
    lower, upper = min(1., float(eig.min())), max(1., float(eig.max()))
    rate = p.dt**2*cfl_axis_sum(scene, p.dx)*upper*float(np.max(params['bh']/term.ratio))
    if not (lower > 0 and np.isfinite(upper) and rate < 1):
        raise ValueError(f'SphereTripletEnergyCFLRefused: spectrum={lower,upper}, dt^2 rate={rate}')
    return dict(lower=lower, upper=upper, dt_squared_rate=rate, vacuum_halo_exact=True,
        method='average_of_eight_permuted_local_SPD_blocks')


@dataclasses.dataclass(frozen=True)
class SphereTripletTerm(ConstitutiveTerm):
    def prepare(self, curl, mem, params, rt):
        local = curl[self.view]
        return mem+self.ratio*local, self.ratio*action(local, params['sphere_triplets'], rt.xp, True)


def prepare_sphere(scene):
    from wavebit.fd._lower import plan, lower
    from wavebit.fd._lower.constitutive import common_points
    from wavebit.fd._kernel.masks import Region
    import wavebit.fd as fd
    if len(scene.structures) != 1 or not isinstance(scene.structures[0].shape, fd.Sphere):
        raise ValueError('SphereTripletScopeRefused: exactly one public Sphere is required')
    body, material = scene.structures[0].shape, scene.structures[0].material
    if (not isinstance(material, fd.Material) or material.eps != (material.eps[0],)*3
            or material.mu_value != (1.,)*3 or scene.background != fd.Material(eps=1.)):
        raise ValueError('SphereTripletScopeRefused: isotropic electric sphere only')
    p = plan(scene)  # Retain every public scene preparation check.
    params = lower(scene, p, 'numpy')
    old, = p.constitutive
    z, y, x = tuple((lo-1, hi+1) for lo, hi in old.box)
    new = SphereTripletTerm(old.name, old.field, Region(z=z, y=y, x=x), old.ratio, old.masks)
    points = common_points(p.grid, tuple(reversed(new.box)), 'E')
    _, matrices, edges, faces = sphere_maps(points, p.dx, body.center, body.radius,
        material.eps[0], geometry=True)
    params = dict(params, be=params['be'].copy(), sphere_triplets=matrices)
    params['be'][new.view] = new.ratio*diagonal(matrices)
    bounds = certificate(scene, p, new, matrices, edges, faces, params)
    params['stable'] = 1.  # All public checks and the replacement certificate succeeded.
    p = p.with_static(terms=tuple(new if t is old else t for t in p.terms))
    return p, params, bounds


def operator_audit(matrices):
    """Independently index the compact matrix; no shift helper is used."""
    from scipy.sparse import coo_matrix, eye
    from scipy.sparse.linalg import eigsh
    shape = matrices.shape[1:-2]
    size = int(np.prod(shape))
    rows, cols, values = [], [], []
    for k, orientation in enumerate(ORIENTATIONS):
        for node in np.ndindex(shape):
            cells = []
            for i in range(3):
                cell = list(node)
                if orientation[i] < 0:
                    cell[2-i] -= 1
                cells.append(cell)
            for i in range(3):
                for j in range(3):
                    if any(not 0 <= v < n for cell in (cells[i], cells[j]) for v, n in zip(cell, shape)):
                        continue
                    rows.append(i*size+np.ravel_multi_index(cells[i], shape))
                    cols.append(j*size+np.ravel_multi_index(cells[j], shape))
                    values.append((matrices[(k,)+node+(i,j)]-(1. if i == j else 0.))/8)
    operator = coo_matrix((values, (rows, cols)), shape=(3*size, 3*size)).tocsr()+eye(3*size)
    vector = np.sin(np.arange(3*size)).reshape((3,)+shape)
    computed = action(vector, matrices)
    difference = operator-operator.T
    return dict(indexed_action_gap=float(np.max(abs(computed.ravel()-operator@vector.ravel()))),
        split_gap=float(np.max(abs(computed-diagonal(matrices)*vector-action(vector, matrices, cross_only=True)))),
        assembled_asymmetry=float(np.max(abs(difference.data), initial=0.)),
        minimum_eigenvalue=float(eigsh(operator, k=1, which='SA', return_eigenvectors=False)[0]))


def main():
    import argparse
    import hashlib
    import json
    import os
    import time
    import warnings
    from studies.fdtd_subpixel_mie_sphere import sphere_scene, read_result
    from studies.fdtd_subpixel_mie_cylinder import convergence
    from wavebit.fd._runtime import runtime
    from wavebit.fd._run import execute, _results
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cells', default='20,28,40')
    parser.add_argument('--eps', type=float, default=2.25)
    parser.add_argument('--periods', type=float, default=24.)
    parser.add_argument('--radius', type=float, default=.25)
    parser.add_argument('--placements', default='0,1,2')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('DuplicateMeasurementRefused')
    placements = ((0., 0., 0.), (.5, 1/3, .2), (.25, 2/3, .4))
    cells = tuple(map(int, args.cells.split(',')))
    indices = tuple(map(int, args.placements.split(',')))
    watched = sorted((ROOT/'src/wavebit/fd').rglob('*.py'))+[
        Path(__file__), ROOT/'studies/fdtd_subpixel_triplet_cylinder.py',
        ROOT/'studies/fdtd_subpixel_mie_sphere.py']
    def hashes():
        return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    source = hashes()
    out = dict(status='running_research_sphere', pid=os.getpid(), rows=[],
        source_sha256=source, production_operator_changed=False, eps=args.eps,
        periods=args.periods, radius=args.radius, cells=cells, placements=indices)
    def save():
        args.output.write_text(json.dumps(out, indent=2)+'\n', encoding='utf8')
    save()
    try:
        errors = []
        for cpw in cells:
            group = []
            for index in indices:
                begin = time.perf_counter()
                out['current'] = dict(cpw=cpw, placement=index)
                save()
                scene = sphere_scene(cpw, args.eps, args.periods, placements[index], radius=args.radius)
                with warnings.catch_warnings(record=True) as notes:
                    p, params, bounds = prepare_sphere(scene)
                    rt = runtime('jax')
                    res = _results(scene, p, execute(p, params, rt), rt, params)
                row = read_result(res, args.eps, args.radius)
                times, energy = res.diagnostics['interior_energy']
                row.update(cpw=cpw, placement_index=index, placement=placements[index],
                    seconds=time.perf_counter()-begin, certificate=bounds,
                    warnings=[str(n.message) for n in notes],
                    interior_energy=dict(time=np.asarray(times).tolist(), value=np.asarray(energy).tolist()))
                out['rows'].append(row); group.append(row['error'])
                if hashes() != source:
                    raise ValueError('MeasurementSourceChangedRefused')
                save()
                print(json.dumps({k:row[k] for k in ('cpw','placement_index','qsca','qabs','error','decay','seconds')}), flush=True)
                del res, params
            errors.append(group)
        out.update(status='completed_research_measurements_not_C3_pass', source_unchanged=hashes()==source)
        if len(cells) >= 3:
            out['fit'] = convergence(cells, errors)
        out['current'] = None
        save()
    except Exception as err:
        out.update(status='stopped_for_review', failure=f'{type(err).__name__}: {err}', source_unchanged=hashes()==source)
        save()
        raise


if __name__ == '__main__':
    main()
