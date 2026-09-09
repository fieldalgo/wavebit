"""Independent H(curl) FEM study of the tilted NVF contact fixture.

This imports NGSolve, not Wavebit. It is an optional research reference, not an
NVF replacement or a converged golden value. Mesh, polynomial degree, and the
outgoing boundary must be refined before assigning a reference uncertainty.
All lengths here are micrometers (the Wavebit fixture uses nanometers).

Example, in an environment with ngsolve installed:
    python studies/nvf_fem_reference.py --hs .18 .14 --orders 3 4 --boundary rayleigh
    python studies/nvf_fem_reference.py --hs .18 --orders 3 4 --uniform 4
"""
import argparse
import json
import time

import numpy as np
import ngsolve as ng
import netgen.occ as occ
from netgen.meshing import IdentificationType

PERIOD = .7
THICKNESS = .3
WAVELENGTH = .9
THETA = .2
INCLUSION = 12.25
SUBSTRATE = 2.25


def slab_efficiencies(epsilon, theta=THETA):
    """Exact oblique p-polarized three-medium slab; independent of FEM."""
    k0 = 2*np.pi/WAVELENGTH
    lateral = np.sin(theta)
    kz = k0*np.sqrt(np.asarray([1., epsilon, SUBSTRATE], complex)-lateral**2)
    admittance = k0*np.asarray([1., epsilon, SUBSTRATE])/kz
    y0, yf, y1 = admittance
    inc, reflected = np.exp(-1j*kz[0]*THICKNESS), np.exp(1j*kz[0]*THICKNESS)
    forward, backward = np.exp(-1j*kz[1]*THICKNESS), np.exp(1j*kz[1]*THICKNESS)
    system = np.array([
        [reflected, -forward, -backward, 0],
        [y0*reflected, yf*forward, -yf*backward, 0],
        [0, 1, 1, -1],
        [0, -yf, yf, y1],
    ])
    amplitude = np.cos(theta)
    r, _, _, t = np.linalg.solve(system, [-amplitude*inc, amplitude*y0*inc, 0, 0])
    return dict(T=float(y1.real*abs(t)**2/amplitude),
                R=float(y0.real*abs(r)**2/amplitude))


def periodic_counts(mesh):
    """Check the generated point identifications, not just the geometry request.

    OCC's padded face bounding boxes can make a too-tight selection empty.
    Identify() on empty lists succeeds and produces a nonperiodic calculation.
    Requiring actual pairs in BOTH directions catches this failure.
    """
    counts = [0, 0]
    for a, b in mesh.ngmesh.GetIdentifications():
        delta = np.abs(np.asarray(mesh.ngmesh[b].p)-np.asarray(mesh.ngmesh[a].p))
        if np.allclose(delta, [PERIOD, 0, 0], rtol=0, atol=1e-9):
            counts[0] += 1
        elif np.allclose(delta, [0, PERIOD, 0], rtol=0, atol=1e-9):
            counts[1] += 1
        else:
            raise ValueError(f'unexpected periodic point displacement: {delta}')
    if not all(counts):
        raise ValueError(f'missing periodic point identifications: x={counts[0]}, y={counts[1]}')
    return dict(x=counts[0], y=counts[1])


def make_mesh(h, uniform=None, pad=.4, pml=.4, *, boundary='pml'):
    """Conforming, clipped periodic union, including the 13.9 nm contact run."""
    if not all(np.isfinite(value) and value > 0 for value in (h, pad, pml)):
        raise ValueError('h, pad and pml must be finite and positive')
    if uniform is not None and (not np.isfinite(uniform) or uniform <= 0):
        raise ValueError('the uniform control requires a positive real permittivity')
    if boundary not in ('pml', 'rayleigh'):
        raise ValueError('boundary must be pml or rayleigh')

    def box(lower, upper):
        return occ.Box(occ.Pnt(-PERIOD/2, -PERIOD/2, lower),
                       occ.Pnt(PERIOD/2, PERIOD/2, upper))

    film = box(0, THICKNESS)
    if uniform is not None:
        film.mat('inclusion')
        parts = [film]
    else:
        half_height = PERIOD/(2*np.sqrt(5))
        original = occ.Box(occ.Pnt(-.32, -half_height, 0),
                           occ.Pnt(.32, half_height, THICKNESS))
        original = original.Rotate(occ.Axis((0, 0, 0), occ.Z),
                                   np.rad2deg(np.arctan2(1, 2)))
        inclusion = original
        # Only the x edges are crossed by this fixed tilted rectangle. The
        # translated pieces are geometrically clipped and united before meshing.
        for shift in (-PERIOD, PERIOD):
            inclusion = inclusion + original.Move(occ.Vec(shift, 0, 0))
        inclusion = inclusion * film
        background = film - inclusion
        inclusion.mat('inclusion')
        background.mat('groove')
        parts = [background, inclusion]
    layers = [('substrate', -pad, 0), ('superstrate', THICKNESS, THICKNESS+pad)]
    if boundary == 'pml':
        layers += [('pml_bottom', -pad-pml, -pad),
                   ('pml_top', THICKNESS+pad, THICKNESS+pad+pml)]
    for name, lower, upper in layers:
        part = box(lower, upper)
        part.mat(name)
        parts.append(part)
    shape = occ.Glue(parts)
    shape.faces.Min(occ.Z).name = 'outer_bottom'
    shape.faces.Max(occ.Z).name = 'outer_top'
    for axis, label in ((occ.X, 'periodic_x'), (occ.Y, 'periodic_y')):
        # OCC bounds are padded by about 1e-7 in these micrometer coordinates.
        # This selects whole outer planes; it does not merge material interfaces.
        lower = shape.faces[axis < -PERIOD/2+1e-6]
        upper = shape.faces[axis > PERIOD/2-1e-6]
        if not len(lower) or len(lower) != len(upper):
            raise ValueError(f'{label}: unmatched outer faces')
        lower.Identify(upper, label, IdentificationType.PERIODIC,
                       occ.Translation(PERIOD*axis))
    for solid in shape.solids:
        solid.maxh = h/2 if solid.name in ('inclusion', 'groove') else h
    mesh = ng.Mesh(occ.OCCGeometry(shape).GenerateMesh(maxh=h))
    periodic_counts(mesh)
    actual = ng.Integrate(1, mesh, definedon=mesh.Materials('inclusion'))
    expected = (PERIOD**2 if uniform is not None else .64*PERIOD/np.sqrt(5))*THICKNESS
    if not np.isclose(actual, expected, rtol=0, atol=1e-11):
        raise ValueError(f'inclusion volume mismatch: {actual} != {expected}')
    return mesh


def annex_fields(theta=THETA):
    """Exact incident plus reflected/transmitted field of the bare substrate.

    The incident wave travels from z>0 toward negative z. The periodic unknown
    has exp(i*kx*x) removed, so its curl is curl(u)+i*k_cross_u. H uses e^-iwt.
    """
    k0 = 2*np.pi/WAVELENGTH
    kx = k0*np.sin(theta)
    g0 = k0*np.cos(theta)
    g1 = k0*np.sqrt(SUBSTRATE-np.sin(theta)**2)
    y0, y1 = k0/g0, k0*SUBSTRATE/g1
    reflection = (y0-y1)/(y0+y1)
    incident = ng.exp(-1j*g0*ng.z)
    reflected = reflection*ng.exp(1j*g0*ng.z)
    transmitted = (1+reflection)*ng.exp(-1j*g1*ng.z)
    amplitude = np.cos(theta)
    e = ng.IfPos(ng.z,
                 ng.CF((amplitude*(incident+reflected), 0,
                        amplitude*kx/g0*(incident-reflected))),
                 ng.CF((amplitude*transmitted, 0, amplitude*kx/g1*transmitted)))
    h = ng.IfPos(ng.z, ng.CF((0, amplitude*y0*(-incident+reflected), 0)),
                 ng.CF((0, -amplitude*y1*transmitted, 0)))
    return e, h


def solve(h=.14, order=3, uniform=None, pad=.4, pml=.4, stretch=2.,
          *, condense=True, progress=None):
    """One discretization, returning errors separately from the observable.

    Lossless energy balance and a small algebraic residual are necessary checks,
    not error estimates against the continuum Maxwell solution.
    """
    if not isinstance(order, int) or order < 1 or not np.isfinite(stretch) or stretch <= 0:
        raise ValueError('order must be a positive integer and stretch must be positive')
    start = time.monotonic()
    mesh = make_mesh(h, uniform, pad, pml)
    high = INCLUSION if uniform is None else uniform
    epsilon = mesh.MaterialCF({'substrate': SUBSTRATE, 'pml_bottom': SUBSTRATE,
                               'inclusion': high}, default=1.)
    jacobian = mesh.MaterialCF({'pml_top': 1+1j*stretch,
                                'pml_bottom': 1+1j*stretch}, default=1.)
    er = ng.CF((jacobian, 0, 0, 0, jacobian, 0, 0, 0, 1/jacobian), dims=(3, 3))*epsilon
    mui = ng.CF((1/jacobian, 0, 0, 0, 1/jacobian, 0, 0, 0, jacobian), dims=(3, 3))
    e_background, h_background = annex_fields()
    k0 = 2*np.pi/WAVELENGTH
    lateral = ng.CF((k0*np.sin(THETA), 0, 0))
    space = ng.Compress(ng.Periodic(ng.HCurl(
        mesh, order=order, complex=True, dirichlet='outer_top|outer_bottom')))
    u, v = space.TnT()
    curl_u = ng.curl(u)+1j*ng.Cross(lateral, u)
    curl_v = ng.curl(v)-1j*ng.Cross(lateral, v)
    a = ng.BilinearForm(space, condense=condense)
    a += (mui*curl_u*curl_v-k0**2*er*u*v)*ng.dx
    f = ng.LinearForm(space)
    f += k0**2*(high-1)*e_background*v*ng.dx('inclusion', bonus_intorder=6)
    metadata = dict(boundary='pml', h=h, order=order, uniform=uniform, pad=pad, pml=pml,
                    stretch=stretch, elements=mesh.ne, dofs=space.ndof,
                    periodic_pairs=periodic_counts(mesh))
    if progress is not None:
        progress(dict(stage='mesh', **metadata, seconds=time.monotonic()-start))
    with ng.TaskManager():
        a.Assemble()
        f.Assemble()
        field = ng.GridFunction(space)
        rhs = f.vec.CreateVector()
        rhs.data = f.vec
        if condense:
            rhs.data += a.harmonic_extension_trans*f.vec
        field.vec.data = a.mat.Inverse(space.FreeDofs(coupling=condense), inverse='umfpack')*rhs
        if condense:
            field.vec.data += a.harmonic_extension*field.vec
            field.vec.data += a.inner_solve*f.vec
    residual = f.vec.CreateVector()
    a.Apply(field.vec, residual)
    residual.data -= f.vec
    free = np.asarray(list(space.FreeDofs()), dtype=bool)
    norm_f = np.linalg.norm(f.vec.FV().NumPy()[free])
    norm_r = np.linalg.norm(residual.FV().NumPy()[free])
    relative_residual = norm_r/norm_f if norm_f > 0 else norm_r
    e = field+e_background
    magnetic = (ng.curl(field)+1j*ng.Cross(lateral, field))/(1j*k0)+h_background
    flux = e[0]*ng.Conj(magnetic[1])-e[1]*ng.Conj(magnetic[0])
    divisor = PERIOD**2*pad*np.cos(THETA)
    fluxes = [ng.Integrate(flux, mesh, definedon=mesh.Materials(name),
                           order=2*order+6).real/divisor
              for name in ('superstrate', 'substrate')]
    top, bottom = fluxes
    transmission, reflection = -bottom, 1+top
    if not np.isfinite([transmission, reflection, relative_residual]).all():
        raise FloatingPointError('nonfinite FEM result')
    if relative_residual > 1e-9:
        raise FloatingPointError(f'FEM algebraic residual too large: {relative_residual}')
    result = dict(stage='result', **metadata, T=transmission, R=reflection,
                  energy_error=abs(transmission+reflection-1), residual=relative_residual,
                  seconds=time.monotonic()-start)
    if uniform is not None:
        exact = slab_efficiencies(uniform)
        result['analytic_T'] = exact['T']
        result['analytic_T_error'] = abs(transmission-exact['T'])
    return result


def rayleigh_data(cut, epsilon):
    """Outgoing tangential DtN blocks, including every propagating order.

    The periodic unknown has the Bloch phase removed. For each reciprocal pair,
    Gamma = gamma I + k_parallel k_parallel.T/gamma, with Im(gamma)>=0.
    These blocks have inverse-length units; their quadratic form divided by k0
    is outgoing power. The fixed fixture has no exactly grazing order.
    """
    if not isinstance(cut, int) or cut < 0:
        raise ValueError('Rayleigh cut must be a nonnegative integer')
    k0 = 2*np.pi/WAVELENGTH
    # Check a larger square enclosing the propagating disk; a small cut must not
    # silently omit the near-grazing substrate (1,0) diffraction order.
    bound = int(np.ceil(PERIOD/WAVELENGTH*(np.sqrt(epsilon)+abs(np.sin(THETA)))))+1
    for m in range(-bound, bound+1):
        for n in range(-bound, bound+1):
            norm2 = (k0*np.sin(THETA)+2*np.pi*m/PERIOD)**2+(2*np.pi*n/PERIOD)**2
            if norm2 < k0*k0*epsilon and max(abs(m), abs(n)) > cut:
                raise ValueError('Rayleigh cut omits a propagating diffraction order')
    modes = [(m, n) for n in range(-cut, cut+1) for m in range(-cut, cut+1)]
    blocks = []
    for m, n in modes:
        k = np.array([k0*np.sin(THETA)+2*np.pi*m/PERIOD, 2*np.pi*n/PERIOD])
        gamma = np.sqrt(complex(k0*k0*epsilon-k@k))
        if abs(gamma) < 1e-12*k0:
            raise ValueError('this reference does not implement an exactly grazing order')
        blocks.append(gamma*np.eye(2)+np.outer(k, k)/gamma)
    return modes, np.asarray(blocks)


def solve_rayleigh(h=.18, order=3, cut=2, uniform=None, pad=.4,
                   *, condense=True, progress=None):
    """FEM with exact low-order Rayleigh DtN and a local Robin complement.

    A local impedance form is factored once. Fourier trace corrections are
    applied with a Woodbury solve, so no dense global boundary matrix is formed.
    The retained orders use the exact outgoing admittance; omitted evanescent
    orders still use local impedance. Cut and physical pad must therefore be
    varied independently before declaring this a converged reference.
    """
    if not isinstance(order, int) or order < 1:
        raise ValueError('order must be a positive integer')
    port_data = [rayleigh_data(cut, epsilon) for epsilon in (1., SUBSTRATE)]
    start = time.monotonic()
    mesh = make_mesh(h, uniform, pad, boundary='rayleigh')
    high = INCLUSION if uniform is None else uniform
    epsilon = mesh.MaterialCF({'substrate': SUBSTRATE, 'inclusion': high}, default=1.)
    e_background, h_background = annex_fields()
    k0 = 2*np.pi/WAVELENGTH
    lateral = ng.CF((k0*np.sin(THETA), 0, 0))
    space = ng.Compress(ng.Periodic(ng.HCurl(mesh, order=order, complex=True)))
    u, v = space.TnT()
    curl_u = ng.curl(u)+1j*ng.Cross(lateral, u)
    curl_v = ng.curl(v)-1j*ng.Cross(lateral, v)
    a = ng.BilinearForm(space, condense=condense)
    a += (curl_u*curl_v-k0*k0*epsilon*u*v)*ng.dx
    sides = [('outer_top', 1.), ('outer_bottom', SUBSTRATE)]
    for side, eps_half in sides:
        a += (-1j*k0*np.sqrt(eps_half)*u.Trace()*v.Trace())*ng.ds(side)
    f = ng.LinearForm(space)
    f += k0*k0*(high-1)*e_background*v*ng.dx('inclusion', bonus_intorder=6)
    modes = port_data[0][0]
    count = len(modes)
    rank = 4*count
    metadata = dict(boundary='rayleigh', h=h, order=order, cut=cut, uniform=uniform,
                    pad=pad, elements=mesh.ne, dofs=space.ndof, rank=rank,
                    periodic_pairs=periodic_counts(mesh))
    def report(stage):
        if progress is not None:
            progress(dict(stage=stage, **metadata, seconds=time.monotonic()-start))
    report('mesh')
    with ng.TaskManager():
        a.Assemble()
        f.Assemble()
        inverse = a.mat.Inverse(space.FreeDofs(coupling=condense), inverse='umfpack')
        report('factorized')
        rhs = f.vec.CreateVector()
        rhs.data = f.vec
        if condense:
            rhs.data += a.harmonic_extension_trans*f.vec
        base = f.vec.CreateVector()
        base.data = inverse*rhs
        traces = np.zeros((space.ndof, rank), complex)
        responses = np.zeros_like(traces)
        correction = np.zeros((rank, rank), complex)
        for side_index, (side, eps_half) in enumerate(sides):
            blocks = port_data[side_index][1]
            for mode_index, (m, n) in enumerate(modes):
                ids = [2*side_index*count+component*count+mode_index for component in range(2)]
                correction[np.ix_(ids, ids)] = -1j*(blocks[mode_index]-k0*np.sqrt(eps_half)*np.eye(2))
                phase = ng.exp(2j*np.pi*(m*ng.x+n*ng.y)/PERIOD)/PERIOD
                for component, column in enumerate(ids):
                    trace = ng.LinearForm(space)
                    trace += phase*v.Trace()[component]*ng.ds(side, bonus_intorder=2*cut+6)
                    trace.Assemble()
                    traces[:, column] = trace.vec.FV().NumPy()
                    response = trace.vec.CreateVector()
                    response.data = inverse*trace.vec
                    responses[:, column] = response.FV().NumPy()
        report('boundary')
        small = np.eye(rank)+correction@(traces.conj().T@responses)
        right = correction@(traces.conj().T@base.FV().NumPy())
        weights = np.linalg.solve(small, right)
        field = ng.GridFunction(space)
        field.vec.FV().NumPy()[:] = base.FV().NumPy()-responses@weights
        if condense:
            field.vec.data += a.harmonic_extension*field.vec
            field.vec.data += a.inner_solve*f.vec
    residual = f.vec.CreateVector()
    a.Apply(field.vec, residual)
    residual.data -= f.vec
    ports = traces.conj().T@field.vec.FV().NumPy()
    residual.FV().NumPy()[:] += traces@(correction@ports)
    free = np.asarray(list(space.FreeDofs()), dtype=bool)
    norm_f = np.linalg.norm(f.vec.FV().NumPy()[free])
    norm_r = np.linalg.norm(residual.FV().NumPy()[free])
    relative_residual = norm_r/norm_f if norm_f > 0 else norm_r

    # The volume flux and the outgoing Rayleigh flux are independent readouts of
    # the same numerical field. Keeping both exposes finite-element flux error.
    e = field+e_background
    magnetic = (ng.curl(field)+1j*ng.Cross(lateral, field))/(1j*k0)+h_background
    flux = e[0]*ng.Conj(magnetic[1])-e[1]*ng.Conj(magnetic[0])
    divisor = PERIOD**2*pad*np.cos(THETA)
    top, bottom = [ng.Integrate(flux, mesh, definedon=mesh.Materials(name),
                                order=2*order+6).real/divisor
                   for name in ('superstrate', 'substrate')]
    amplitudes = ports.reshape(2, 2, count).transpose(0, 2, 1).copy()
    center = count//2
    g0 = k0*np.cos(THETA)
    g1 = k0*np.sqrt(SUBSTRATE-np.sin(THETA)**2)
    y0, y1 = k0/g0, k0*SUBSTRATE/g1
    reflection = (y0-y1)/(y0+y1)
    amplitudes[0, center, 0] += PERIOD*np.cos(THETA)*reflection*np.exp(1j*g0*(THICKNESS+pad))
    amplitudes[1, center, 0] += PERIOD*np.cos(THETA)*(1+reflection)*np.exp(1j*g1*pad)
    powers = [float(np.einsum('mi,mij,mj->', amplitude.conj(), data[1], amplitude).real
                    /(PERIOD**2*k0*np.cos(THETA)))
              for amplitude, data in zip(amplitudes, port_data)]
    reflection, transmission = powers
    if not np.isfinite([transmission, reflection, top, bottom, relative_residual]).all():
        raise FloatingPointError('nonfinite Rayleigh FEM result')
    if relative_residual > 1e-9:
        raise FloatingPointError(f'FEM algebraic residual too large: {relative_residual}')
    result = dict(stage='result', **metadata, T=transmission, R=reflection,
                  volume_T=-bottom, volume_R=1+top,
                  flux_disagreement=max(abs(transmission+bottom), abs(reflection-1-top)),
                  energy_error=abs(transmission+reflection-1), residual=relative_residual,
                  seconds=time.monotonic()-start)
    if uniform is not None:
        result.update(analytic_T=slab_efficiencies(uniform)['T'],
                      analytic_T_error=abs(transmission-slab_efficiencies(uniform)['T']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hs', nargs='+', type=float, default=[.18, .14])
    parser.add_argument('--orders', nargs='+', type=int, default=[3, 4])
    parser.add_argument('--uniform', type=float)
    parser.add_argument('--pad', type=float, default=.4)
    parser.add_argument('--pml', type=float, default=.4)
    parser.add_argument('--stretch', type=float, default=2.)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--boundary', choices=['pml', 'rayleigh'], default='rayleigh')
    parser.add_argument('--cut', type=int, default=2)
    args = parser.parse_args()
    ng.SetNumThreads(args.threads)
    def report(result):
        print(json.dumps(result, allow_nan=False), flush=True)
    for h in args.hs:
        for order in args.orders:
            if args.boundary == 'rayleigh':
                report(solve_rayleigh(h, order, args.cut, args.uniform, args.pad, progress=report))
            else:
                report(solve(h, order, args.uniform, args.pad, args.pml, args.stretch, progress=report))


if __name__ == '__main__':
    main()
