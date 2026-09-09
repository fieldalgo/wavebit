"""Independent H(curl) FEM studies of tilted and affine NVF interfaces.

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


def periodic_counts(mesh, lattice=None):
    """Check the generated point identifications, not just the geometry request.

    OCC's padded face bounding boxes can make a too-tight selection empty.
    Identify() on empty lists succeeds and produces a nonperiodic calculation.
    Requiring actual pairs in BOTH directions catches this failure.
    ``lattice`` contains two physical primitive columns in micrometers; None
    retains the original square cell for both 2D and 3D reference meshes.
    Pair orientations may be reversed, but componentwise sign changes may not.
    """
    lattice = np.asarray(np.eye(2)*PERIOD if lattice is None else lattice)
    if (lattice.shape != (2, 2) or lattice.dtype.kind not in 'iuf'
            or not np.isfinite(lattice).all()
            or np.linalg.det(lattice.astype(float)) == 0):
        raise ValueError('periodic lattice must be a finite nonsingular real 2-by-2 matrix')
    vectors = np.vstack((lattice, np.zeros(2))).T
    counts = [0, 0]
    for a, b in mesh.ngmesh.GetIdentifications():
        delta = np.asarray(mesh.ngmesh[b].p)-np.asarray(mesh.ngmesh[a].p)
        matches = [i for i, vector in enumerate(vectors)
                   if np.allclose(delta, vector, rtol=0, atol=1e-9)
                   or np.allclose(delta, -vector, rtol=0, atol=1e-9)]
        if len(matches) != 1:
            raise ValueError(f'unexpected periodic point displacement: {delta}')
        counts[matches[0]] += 1
    if not all(counts):
        raise ValueError(f'missing periodic point identifications: x={counts[0]}, y={counts[1]}')
    return dict(x=counts[0], y=counts[1])


def make_affine_profile_mesh(h=.2, *, transform=None, period=PERIOD,
                             width=.350, gap=.017, seam=.031,
                             thickness=THICKNESS, pad=.1, uniform=None, edge_h=None):
    """Fit the affine two-material gap profile directly in physical space.

    All lengths are in micrometers. ``transform`` maps reference x/y positions
    to physical x/y positions, with default [[1,.45],[0,.7]]. The centered
    reference envelope has ``width`` and height ``period-gap``; ``seam`` splits
    epsilon 4 from epsilon 9. The film occupies 0 < z < thickness, with epsilon
    1 above and 2.25 below. ``uniform`` replaces the entire film by that positive
    scalar permittivity for slab controls. Only physical Rayleigh padding is
    meshed; this helper supplies no PML or Maxwell scattering solution.
    ``edge_h`` requests a physical edge mesh-size target on the 20 dielectric edges
    of the patterned film. It must be positive and no larger than ``h``; it
    cannot accompany ``uniform``. None retains the global mesh-size control.
    Netgen treats this as a sizing target, not a strict segment-length bound.

    Returns the mesh and its material-permittivity dictionary. The existing
    fixed-fixture solve functions do not consume this mesh.
    """
    settings = (h, period, width, gap, seam, thickness, pad)
    if any(np.asarray(value).shape != () or np.asarray(value).dtype.kind not in 'iuf'
           for value in settings):
        raise ValueError('affine geometry lengths must be finite real scalars')
    if (not np.isfinite(settings).all() or min(h, period, thickness, pad) <= 0
            or not 0 < width < period or not 0 < gap < period
            or not -width/2 < seam < width/2):
        raise ValueError('invalid affine material profile or mesh settings')
    if uniform is not None and (np.asarray(uniform).shape != ()
                               or np.asarray(uniform).dtype.kind not in 'iuf'
                               or not np.isfinite(uniform) or uniform <= 0):
        raise ValueError('the uniform control requires a positive real permittivity')
    if edge_h is not None:
        edge_value = np.asarray(edge_h)
        if (edge_value.shape != () or edge_value.dtype.kind not in 'iuf'
                or not np.isfinite(edge_value) or not 0 < edge_value <= h):
            raise ValueError('edge_h must be a positive finite real scalar no larger than h')
        if uniform is not None:
            raise ValueError('edge_h requires the patterned film, not a uniform control')
        edge_h = float(edge_value)
    transform = np.asarray([[1., .45], [0., .7]] if transform is None else transform)
    if (transform.shape != (2, 2) or transform.dtype.kind not in 'iuf'
            or not np.isfinite(transform).all()):
        raise ValueError('transform must be a finite real 2-by-2 matrix')
    transform = transform.astype(float)
    determinant = np.linalg.det(transform)
    if not np.isfinite(determinant) or determinant <= 0:
        raise ValueError('transform must preserve orientation and have positive determinant')
    inverse = np.linalg.inv(transform)
    # Opposite face distances, not primitive-vector lengths, constrain whether
    # the periodic cell can contain more than one element across each direction.
    if h >= period/(2*np.linalg.norm(inverse, axis=1).max()):
        raise ValueError('h must be less than half each physical cell width')
    lattice = period*transform

    def prism(left, right, lower_y, upper_y, lower_z, upper_z):
        corners = np.array([[left, lower_y], [right, lower_y],
                            [right, upper_y], [left, upper_y]])@transform.T
        plane = occ.WorkPlane().MoveTo(*corners[0])
        for corner in corners[1:]:
            plane = plane.LineTo(*corner)
        return plane.Close().Face().Extrude(upper_z-lower_z).Move(occ.Vec(0, 0, lower_z))

    def cell(lower_z, upper_z):
        return prism(-period/2, period/2, -period/2, period/2, lower_z, upper_z)

    film = cell(0, thickness)
    materials, volumes, parts = {}, {}, []
    area = determinant*period**2
    if uniform is not None:
        film.mat('film')
        parts.append(film)
        materials['film'], volumes['film'] = float(uniform), area*thickness
    else:
        remaining = film
        height = period-gap
        for i, (left, right, epsilon) in enumerate(
                ((-width/2, seam, 4.), (seam, width/2, 9.))):
            part = prism(left, right, -height/2, height/2, 0, thickness)
            name = f'piece_{i}'
            part.mat(name)
            if edge_h is not None:
                # Each physical prism has 12 edges; the common interface has
                # four shared edges, leaving 20 distinct dielectric edges.
                # Tag before Glue so selection follows the physical prisms
                # under affine transforms without coordinate-box tolerances.
                for edge in part.edges:
                    edge.maxh = edge_h
            materials[name] = epsilon
            volumes[name] = determinant*(right-left)*height*thickness
            parts.append(part)
            remaining = remaining-part
        remaining.mat('background')
        parts.append(remaining)
        materials['background'] = 1.
        volumes['background'] = area*thickness-sum(volumes.values())
    for name, lower, upper, epsilon in (('substrate', -pad, 0, SUBSTRATE),
                                        ('superstrate', thickness, thickness+pad, 1.)):
        part = cell(lower, upper)
        part.mat(name)
        materials[name], volumes[name] = epsilon, area*pad
        parts.append(part)
    shape = occ.Glue(parts)
    shape.faces.Min(occ.Z).name = 'outer_bottom'
    shape.faces.Max(occ.Z).name = 'outer_top'
    for i, label in enumerate(('periodic_x', 'periodic_y')):
        axis = occ.Vec(*inverse[i], 0.)
        # Filter in reference coordinates while retaining the physical faces.
        # OCC pads its bounds; requests must select nonempty whole outer planes.
        lower = shape.faces[axis < -period/2+1e-6]
        upper = shape.faces[axis > period/2-1e-6]
        if not len(lower) or len(lower) != len(upper):
            raise ValueError(f'{label}: unmatched affine outer faces')
        lower.name = upper.name = label
        lower.Identify(upper, label, IdentificationType.PERIODIC,
                       occ.Translation(occ.Vec(*lattice[:, i], 0.)))
    mesh = ng.Mesh(occ.OCCGeometry(shape).GenerateMesh(maxh=h))
    periodic_counts(mesh, lattice)
    for name, volume in volumes.items():
        actual = ng.Integrate(1, mesh, definedon=mesh.Materials(name))
        if not np.isclose(actual, volume, rtol=0, atol=1e-11):
            raise ValueError(f'{name}: affine material volume mismatch')
    if not np.isclose(ng.Integrate(1, mesh), area*(thickness+2*pad), rtol=0, atol=1e-11):
        raise ValueError('affine cell volume mismatch')
    for side in ('outer_top', 'outer_bottom'):
        actual = ng.Integrate(1, mesh, definedon=mesh.Boundaries(side))
        if not np.isclose(actual, area, rtol=0, atol=1e-11):
            raise ValueError(f'{side}: affine outer face area mismatch')
    return mesh, materials


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


def rayleigh_data(cut, epsilon, *, lattice=None, theta=THETA, phi=0.):
    """Outgoing tangential DtN blocks, including every propagating order.

    The periodic unknown has the Bloch phase removed. For each reciprocal pair,
    Gamma = gamma I + k_parallel k_parallel.T/gamma, with Im(gamma)>=0.
    These blocks have inverse-length units; their quadratic form divided by k0
    is outgoing power. ``lattice`` has physical primitive vectors as columns,
    in micrometers; None retains the original square fixture. The incident
    lateral vector uses the unit-index superstrate and angles in radians.

    The original square-fixture solves retain their fixed lattice/incidence;
    solve_affine_rayleigh connects these data to a physical affine mesh.
    Exactly grazing retained orders are not implemented by this instrument.
    """
    if isinstance(cut, (bool, np.bool_)) or not isinstance(cut, (int, np.integer)) or cut < 0:
        raise ValueError('Rayleigh cut must be a nonnegative integer')

    def real_scalar(value, name):
        array = np.asarray(value)
        if array.shape != () or array.dtype.kind not in 'iuf' or not np.isfinite(array):
            raise ValueError(f'{name} must be a finite real scalar')
        return float(array)

    epsilon = real_scalar(epsilon, 'epsilon')
    theta, phi = real_scalar(theta, 'theta'), real_scalar(phi, 'phi')
    if epsilon <= 0:
        raise ValueError('epsilon must be positive')
    lattice = np.asarray(np.eye(2)*PERIOD if lattice is None else lattice)
    if (lattice.shape != (2, 2) or lattice.dtype.kind not in 'iuf'
            or not np.isfinite(lattice).all()):
        raise ValueError('lattice must be a finite real 2-by-2 matrix')
    lattice = lattice.astype(float)
    try:
        reciprocal = 2*np.pi*np.linalg.inv(lattice).T
    except np.linalg.LinAlgError as error:
        raise ValueError('lattice must be nonsingular') from error
    if not np.isfinite(reciprocal).all():
        raise ValueError('lattice reciprocal vectors must be finite')
    k0 = 2*np.pi/WAVELENGTH
    incident = k0*np.sin(theta)*np.array([np.cos(phi), np.sin(phi)])
    # For a physical propagating vector k, each integer label satisfies
    # label_i = a_i dot (k-incident)/(2*pi), |k| < k0*sqrt(epsilon).
    # Bounding these projections encloses the full disk even for a sheared
    # primitive basis. The extra integer on each side protects rounded bounds.
    center = -lattice.T@incident/(2*np.pi)
    radius = np.linalg.norm(lattice, axis=0)*np.sqrt(epsilon)/WAVELENGTH
    if not np.isfinite(center).all() or not np.isfinite(radius).all():
        raise ValueError('lattice propagating-order bounds must be finite')
    ranges = [range(int(np.floor(c-r))-1, int(np.ceil(c+r))+2)
              for c, r in zip(center, radius)]
    for m in ranges[0]:
        for n in ranges[1]:
            k = incident+reciprocal@np.array([m, n])
            if k@k < k0*k0*epsilon and max(abs(m), abs(n)) > cut:
                raise ValueError('Rayleigh cut omits a propagating diffraction order')
    modes = [(m, n) for n in range(-cut, cut+1) for m in range(-cut, cut+1)]
    blocks = []
    for m, n in modes:
        k = incident+reciprocal@np.array([m, n])
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


def _affine_incidence(theta, phi):
    """Validate real incidence angles before constructing any affine mesh."""
    values = []
    for name, value in (('theta', theta), ('phi', phi)):
        array = np.asarray(value)
        if array.shape != () or array.dtype.kind not in 'iuf' or not np.isfinite(array):
            raise ValueError(f'{name} must be a finite real scalar')
        values.append(float(array))
    theta, phi = values
    if not -np.pi/2 < theta < np.pi/2:
        raise ValueError('theta must lie strictly between -pi/2 and pi/2')
    return theta, phi


def slab_efficiencies_polarized(epsilon, theta=THETA):
    """Independent p/s Fabry--Perot control for the fixed three-medium slab.

    Exterior permittivities are 1 and 2.25, wavelength .9 micrometers, and film
    thickness .3 micrometers. Reflection amplitudes use tangential electric
    components; efficiencies are independent of that amplitude convention.
    """
    theta, _ = _affine_incidence(theta, 0.)
    value = np.asarray(epsilon)
    if (value.shape != () or value.dtype.kind not in 'iuf'
            or not np.isfinite(value) or value <= 0):
        raise ValueError('epsilon must be a positive finite real scalar')
    permittivities = np.array([1., float(value), SUBSTRATE])
    q = np.sqrt(permittivities.astype(complex)-np.sin(theta)**2)
    if np.any(abs(q) < 1e-12):
        raise ValueError('the slab control does not implement an exactly grazing layer')
    phase = np.exp(2j*np.pi/WAVELENGTH*q[1]*THICKNESS)
    result = {}
    for polarization, admittance in (('p', permittivities/q), ('s', q)):
        y0, yf, y1 = admittance
        r01, r12 = (y0-yf)/(y0+yf), (yf-y1)/(yf+y1)
        denominator = 1+r01*r12*phase**2
        r = (r01+r12*phase**2)/denominator
        t = (1+r01)*(1+r12)*phase/denominator
        result[polarization] = dict(T=float(y1.real/y0.real*abs(t)**2),
                                    R=float(abs(r)**2))
    return result


def _affine_background_fields(theta, phi):
    """Unit-electric-amplitude p/s incidence on the bare physical interface.

    The Bloch factor has been removed. Each tuple contains E, H, and the two
    outgoing tangential electric amplitudes evaluated at the interface z=0.
    H is scaled by vacuum impedance, with the e^-iwt convention.
    """
    k0 = 2*np.pi/WAVELENGTH
    kt, g0 = k0*np.sin(theta), k0*np.cos(theta)
    g1 = k0*np.sqrt(SUBSTRATE-np.sin(theta)**2)
    tangent = np.array([np.cos(phi), np.sin(phi)])
    transverse = np.array([-np.sin(phi), np.cos(phi)])
    t, s, z = ng.CF((*tangent, 0.)), ng.CF((*transverse, 0.)), ng.CF((0., 0., 1.))
    incident = ng.exp(-1j*g0*ng.z)
    result = {}
    for polarization, y0, y1, amplitude, direction in (
            ('p', k0/g0, k0*SUBSTRATE/g1, np.cos(theta), tangent),
            ('s', g0/k0, g1/k0, 1., transverse)):
        reflection = (y0-y1)/(y0+y1)
        reflected = reflection*ng.exp(1j*g0*ng.z)
        transmitted = (1+reflection)*ng.exp(-1j*g1*ng.z)
        if polarization == 'p':
            upper_e = amplitude*((incident+reflected)*t+kt/g0*(incident-reflected)*z)
            lower_e = amplitude*transmitted*(t+kt/g1*z)
            upper_h = amplitude*y0*(-incident+reflected)*s
            lower_h = -amplitude*y1*transmitted*s
        else:
            upper_e = (incident+reflected)*s
            lower_e = transmitted*s
            upper_h = y0*(incident-reflected)*t+kt/k0*(incident+reflected)*z
            lower_h = transmitted*(y1*t+kt/k0*z)
        result[polarization] = (
            ng.IfPos(ng.z, upper_e, lower_e), ng.IfPos(ng.z, upper_h, lower_h),
            amplitude*reflection*direction, amplitude*(1+reflection)*direction)
    return result


def robin_complement_power(mesh, scattered_field, ports, *, area, theta,
                           quadrature_order):
    """Outgoing power assigned to the omitted trace by the local Robin term.

    ``ports`` has shape (2, retained_modes, 2): top/bottom, orthonormal Fourier
    mode, physical tangential x/y component. Its basis is exp(i*G.r)/sqrt(area).
    ``scattered_field`` has the lateral Bloch phase removed. The analytic
    interface field is a retained constant trace, so it has no complement.

    On each horizontal side this is sqrt(epsilon) times the difference between
    the integrated tangential squared norm and the retained coefficient norm,
    divided by area*cos(theta). Epsilon is 1 above and 2.25 below. This is a
    numerical outgoing sink from the finite Robin complement, not material
    absorption. Quadrature/projection error may make it negative; return that
    signed observation without clamping or treating it as an optical bound.
    """
    theta, _ = _affine_incidence(theta, 0.)
    area_value = np.asarray(area)
    if (area_value.shape != () or area_value.dtype.kind not in 'iuf'
            or not np.isfinite(area_value) or area_value <= 0):
        raise ValueError('area must be a positive finite real scalar')
    if (isinstance(quadrature_order, (bool, np.bool_))
            or not isinstance(quadrature_order, (int, np.integer))
            or quadrature_order < 0):
        raise ValueError('quadrature_order must be a nonnegative integer')
    ports = np.asarray(ports)
    if (ports.ndim != 3 or ports.shape[0] != 2 or ports.shape[1] < 1 or ports.shape[2] != 2
            or ports.dtype.kind not in 'iufc' or not np.isfinite(ports).all()):
        raise ValueError('ports must be finite coefficients with shape (2, retained_modes, 2)')
    tangential_norm = sum(scattered_field[i]*ng.Conj(scattered_field[i]) for i in (0, 1))
    divisor = float(area_value)*np.cos(theta)
    result = {}
    for side_index, (name, side, epsilon) in enumerate(
            (('top', 'outer_top', 1.), ('bottom', 'outer_bottom', SUBSTRATE))):
        whole = ng.Integrate(tangential_norm, mesh, definedon=mesh.Boundaries(side),
                             order=int(quadrature_order)).real
        retained = np.sum(abs(ports[side_index])**2)
        value = float(np.sqrt(epsilon)*(whole-retained)/divisor)
        if not np.isfinite(value):
            raise FloatingPointError('nonfinite Robin complement power')
        result[name] = value
    result['total'] = result['top']+result['bottom']
    return result


def solve_affine_rayleigh(h=.2, order=2, cut=1, uniform=None, pad=.1,
                          *, transform=None, theta=THETA, phi=0., condense=True,
                          quadrature_bonus=6, edge_h=None, progress=None):
    """Independent affine H(curl) FEM with shared p/s Rayleigh boundary solves.

    The physical profile is fixed to the .7-micrometer reference cell, width
    .350, gap .017, seam .031, and .3-thick film. ``transform`` changes its
    physical affine image; all lengths and lattice columns are in micrometers.
    ``uniform`` replaces the patterned film by a positive scalar permittivity.
    ``edge_h`` requests a physical edge mesh-size target on the 20 dielectric edges
    of the patterned film, as in ``make_affine_profile_mesh``.

    One mesh, sparse factorization, Fourier trace response and two-column
    Woodbury solve serve both polarizations. At finite cut, omitted evanescent
    traces retain a local Robin condition. Mesh/order, cut and physical padding
    remain separate reference uncertainties; small residual and energy error
    do not qualify a continuum Maxwell oracle.

    Trace and inverse-response arrays retain only outer-boundary rows. Two
    final inverse applications recover the full p/s corrections using the same
    factorization. ``boundary_array_bytes`` reports those two retained arrays,
    excluding the sparse factorization, dense port matrices and work vectors.

    ``quadrature_bonus`` is a nonnegative integer for numerical integration
    studies, not mesh or FEM polynomial refinement. It sets the source bonus,
    the trace bonus to 2*cut+bonus, and flux/norm readout order to 2*order+bonus.
    The polynomial bulk and local Robin forms retain their default quadrature.
    """
    start = time.monotonic()
    if (isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer))
            or order < 1):
        raise ValueError('order must be a positive integer')
    if not isinstance(condense, (bool, np.bool_)):
        raise ValueError('condense must be boolean')
    if (isinstance(quadrature_bonus, (bool, np.bool_))
            or not isinstance(quadrature_bonus, (int, np.integer))
            or quadrature_bonus < 0):
        raise ValueError('quadrature_bonus must be a nonnegative integer')
    order, condense = int(order), bool(condense)
    quadrature_bonus = int(quadrature_bonus)
    theta, phi = _affine_incidence(theta, phi)
    transform = np.asarray([[1., .45], [0., .7]] if transform is None else transform)
    if (transform.shape != (2, 2) or transform.dtype.kind not in 'iuf'
            or not np.isfinite(transform).all()):
        raise ValueError('transform must be a finite real 2-by-2 matrix')
    transform = transform.astype(float)
    determinant = np.linalg.det(transform)
    if not np.isfinite(determinant) or determinant <= 0:
        raise ValueError('transform must preserve orientation and have positive determinant')
    lattice = PERIOD*transform
    area = float(np.linalg.det(lattice))
    port_data = [rayleigh_data(cut, epsilon, lattice=lattice, theta=theta, phi=phi)
                 for epsilon in (1., SUBSTRATE)]
    mesh, materials = make_affine_profile_mesh(
        h, transform=transform, uniform=uniform, pad=pad, edge_h=edge_h)
    edge_h = None if edge_h is None else float(edge_h)
    epsilon = mesh.MaterialCF(materials)
    background = _affine_background_fields(theta, phi)
    polarizations = ('p', 's')
    k0 = 2*np.pi/WAVELENGTH
    lateral = ng.CF((k0*np.sin(theta)*np.cos(phi), k0*np.sin(theta)*np.sin(phi), 0.))
    space = ng.Compress(ng.Periodic(ng.HCurl(mesh, order=order, complex=True)))
    boundary = np.flatnonzero(list(space.GetDofs(mesh.Boundaries('outer_top|outer_bottom'))))
    u, v = space.TnT()
    curl_u = ng.curl(u)+1j*ng.Cross(lateral, u)
    curl_v = ng.curl(v)-1j*ng.Cross(lateral, v)
    a = ng.BilinearForm(space, condense=condense)
    a += (curl_u*curl_v-k0*k0*epsilon*u*v)*ng.dx
    sides = [('outer_top', 1.), ('outer_bottom', SUBSTRATE)]
    for side, eps_half in sides:
        a += (-1j*k0*np.sqrt(eps_half)*u.Trace()*v.Trace())*ng.ds(side)
    sources = []
    for polarization in polarizations:
        f = ng.LinearForm(space)
        for name, value in materials.items():
            if name not in ('substrate', 'superstrate'):
                f += (k0*k0*(value-1)*background[polarization][0]*v
                      *ng.dx(name, bonus_intorder=quadrature_bonus))
        sources.append(f)
    modes = port_data[0][0]
    count, rank = len(modes), 4*len(modes)
    settings = dict(h=h, order=order, cut=int(cut), uniform=uniform, pad=pad,
                    condense=condense, quadrature_bonus=quadrature_bonus, edge_h=edge_h,
                    period=PERIOD, width=.350, gap=.017,
                    seam=.031, thickness=THICKNESS)
    metadata = dict(fixture='affine_partitioned_gap', boundary='rayleigh',
                    lattice=lattice.tolist(), transform=transform.tolist(),
                    area=area, theta=theta, phi=phi, settings=settings,
                    units='micrometers', wavelength=WAVELENGTH, thickness=THICKNESS,
                    period=PERIOD, width=.350, gap=.017, seam=.031, values=[4., 9.],
                    h=h, order=order, cut=int(cut), uniform=uniform, pad=pad,
                    quadrature_bonus=quadrature_bonus, edge_h=edge_h,
                    elements=mesh.ne, dofs=space.ndof,
                    coupling_dofs=sum(space.FreeDofs(coupling=True)), rank=rank,
                    boundary_dofs=len(boundary),
                    boundary_array_bytes=2*len(boundary)*rank*np.dtype(complex).itemsize,
                    periodic_pairs=periodic_counts(mesh, lattice),
                    modes=[list(mode) for mode in modes])

    def report(stage):
        if progress is not None:
            progress(dict(stage=stage, **metadata, seconds=time.monotonic()-start))

    report('mesh')
    with ng.TaskManager():
        a.Assemble()
        for f in sources:
            f.Assemble()
        inverse = a.mat.Inverse(space.FreeDofs(coupling=condense), inverse='umfpack')
        report('factorized')
        base = np.empty((space.ndof, 2), complex)
        for column, f in enumerate(sources):
            rhs = f.vec.CreateVector()
            rhs.data = f.vec
            if condense:
                rhs.data += a.harmonic_extension_trans*f.vec
            response = rhs.CreateVector()
            response.data = inverse*rhs
            base[:, column] = response.FV().NumPy()
        traces = np.zeros((len(boundary), rank), complex)
        responses = np.zeros_like(traces)
        correction = np.zeros((rank, rank), complex)
        reciprocal = 2*np.pi*np.linalg.inv(lattice).T
        for side_index, (side, eps_half) in enumerate(sides):
            blocks = port_data[side_index][1]
            for mode_index, (m, n) in enumerate(modes):
                ids = [2*side_index*count+component*count+mode_index for component in range(2)]
                correction[np.ix_(ids, ids)] = -1j*(blocks[mode_index]-k0*np.sqrt(eps_half)*np.eye(2))
                g = reciprocal@np.array([m, n])
                phase = ng.exp(1j*(g[0]*ng.x+g[1]*ng.y))/np.sqrt(area)
                for component, column in enumerate(ids):
                    trace = ng.LinearForm(space)
                    trace += phase*v.Trace()[component]*ng.ds(
                        side, bonus_intorder=2*cut+quadrature_bonus)
                    trace.Assemble()
                    traces[:, column] = trace.vec.FV().NumPy()[boundary]
                    response = trace.vec.CreateVector()
                    response.data = inverse*trace.vec
                    responses[:, column] = response.FV().NumPy()[boundary]
        report('boundary')
        small = np.eye(rank)+correction@(traces.conj().T@responses)
        right = correction@(traces.conj().T@base[boundary])
        weights = np.linalg.solve(small, right)
        # Linearity gives (inverse U) weights = inverse (U weights). Keeping
        # only boundary rows of inverse U suffices for the Gram matrix above;
        # these two extra applications recover every active correction row.
        boundary_rhs = traces@weights
        del responses
        fields = []
        for column, f in enumerate(sources):
            rhs = f.vec.CreateVector()
            rhs[:] = 0
            rhs.FV().NumPy()[boundary] = boundary_rhs[:, column]
            response = rhs.CreateVector()
            response.data = inverse*rhs
            field = ng.GridFunction(space)
            field.vec.FV().NumPy()[:] = base[:, column]-response.FV().NumPy()
            if condense:
                field.vec.data += a.harmonic_extension*field.vec
                field.vec.data += a.inner_solve*f.vec
            fields.append(field)

    free = np.asarray(list(space.FreeDofs()), dtype=bool)
    g0 = k0*np.cos(theta)
    g1 = k0*np.sqrt(SUBSTRATE-np.sin(theta)**2)
    zero_mode = modes.index((0, 0))
    result = dict(stage='result', **metadata)
    analytic = slab_efficiencies_polarized(uniform, theta) if uniform is not None else None
    for polarization, f, field in zip(polarizations, sources, fields):
        residual = f.vec.CreateVector()
        a.Apply(field.vec, residual)
        residual.data -= f.vec
        ports = traces.conj().T@field.vec.FV().NumPy()[boundary]
        residual.FV().NumPy()[boundary] += traces@(correction@ports)
        norm_f = np.linalg.norm(f.vec.FV().NumPy()[free])
        norm_r = np.linalg.norm(residual.FV().NumPy()[free])
        relative_residual = norm_r/norm_f if norm_f > 0 else norm_r
        e_background, h_background, reflected, transmitted = background[polarization]
        e = field+e_background
        magnetic = (ng.curl(field)+1j*ng.Cross(lateral, field))/(1j*k0)+h_background
        flux = e[0]*ng.Conj(magnetic[1])-e[1]*ng.Conj(magnetic[0])
        divisor = area*pad*np.cos(theta)
        top, bottom = [ng.Integrate(flux, mesh, definedon=mesh.Materials(name),
                                    order=2*order+quadrature_bonus).real/divisor
                       for name in ('superstrate', 'substrate')]
        amplitudes = ports.reshape(2, 2, count).transpose(0, 2, 1).copy()
        amplitudes[0, zero_mode] += np.sqrt(area)*reflected*np.exp(1j*g0*(THICKNESS+pad))
        amplitudes[1, zero_mode] += np.sqrt(area)*transmitted*np.exp(1j*g1*pad)
        powers = [np.einsum('mi,mij,mj->m', amplitude.conj(), data[1], amplitude).real
                  /(area*k0*np.cos(theta))
                  for amplitude, data in zip(amplitudes, port_data)]
        reflection, transmission = (float(np.sum(power)) for power in powers)
        if (not np.isfinite([transmission, reflection, top, bottom, relative_residual]).all()
                or not np.isfinite(powers).all()):
            raise FloatingPointError('nonfinite affine Rayleigh FEM result')
        if relative_residual > 1e-9:
            raise FloatingPointError(f'affine FEM algebraic residual too large: {relative_residual}')
        record = dict(T=transmission, R=reflection, volume_T=-bottom, volume_R=1+top,
                      flux_disagreement=max(abs(transmission+bottom), abs(reflection-1-top)),
                      energy_error=abs(transmission+reflection-1), residual=float(relative_residual),
                      order_R=powers[0].tolist(), order_T=powers[1].tolist())
        complement = robin_complement_power(
            mesh, field, ports.reshape(2, 2, count).transpose(0, 2, 1),
            area=area, theta=theta, quadrature_order=2*order+quadrature_bonus)
        record['robin_complement_power'] = complement
        # The remaining budget can contain FEM, quadrature, source and linear
        # solve error; it is not a pure Galerkin defect or an optical error bound.
        record['energy_budget_residual'] = float(1-reflection-transmission-complement['total'])
        if analytic is not None:
            record.update(analytic_T=analytic[polarization]['T'],
                          analytic_T_error=abs(transmission-analytic[polarization]['T']))
        result[polarization] = record
    result['seconds'] = time.monotonic()-start
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
