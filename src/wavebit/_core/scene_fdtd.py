"""Differentiable rectangular Stack adapter for the shared Yee time-domain core.

The mesh is static; internal layer boundaries, incidence theta/phi, raster/half-space
epsilon, mu and electric pole values retain their backend graph. A uniform
incident-medium run calibrates each current-sheet basis.
No external implementation is used here.
"""
import math

import numpy as np

from .backend import _concrete
from .conventions import boundary_kz, incidence_clashes, incident_sp_amplitudes, parse_order_pair, parse_period_mapping, real_order_frame, time_delay_phase, yee_longitudinal_phase
from .domain import as_material
from .grid_fdtd import _validate_pml_reflection, fdtd_grid
from .result import Result, ResultSub
from ..backends.time_domain import time_loop_backend


def _notched_pulse(pulse, dt, frequencies, notches, epsilon, carrier, record=False):
    """Static causal FIR shaping; finite-sum spectra guard incident calibration.

    Each zero adds one delayed sample. Reserve those samples before filtering,
    rather than dropping a convolution tail and filling the intended zero.
    Host NumPy is intentional: scene source/mesh parameters are static.
    """
    values = np.asarray(notches)
    if (values.ndim > 1 or not values.size or values.dtype.kind not in 'fiu'
            or not np.isfinite(values).all() or np.any(values <= 0)):
        raise ValueError('pulse_notches must be a positive finite static frequency or nonempty frequency vector.')
    values = np.atleast_1d(values).astype(float)
    if len(values) > 2:
        raise ValueError('pulse_notches supports at most two zeros; repeated differences amplify pulse truncation and roundoff.')
    if not np.isfinite(values).all() or np.any(values >= .5/dt):
        raise ValueError('pulse_notches must lie below temporal Nyquist.')
    if len(values) >= len(pulse)-1:
        raise ValueError('pulse_notches requires more time samples; increase n_periods.')
    checked = np.r_[frequencies, carrier] if record else frequencies
    if np.any(abs(checked[:, None]-values) <= 32*np.finfo(float).eps*np.maximum(checked[:, None], values)):
        raise ValueError('pulse_notches must avoid requested frequencies and the time-record carrier.')
    result = pulse[:-len(values)].copy()
    for frequency in values:
        result = np.convolve(result, [1., -time_delay_phase(np, frequency, dt)])
        scale = np.max(abs(result))
        if not np.isfinite(scale) or scale == 0:
            raise ValueError('pulse_notches produced an unresolved pulse; reduce the number of notches.')
        result /= scale
    # Check the actual source precision, including rounding after cancellation.
    result = result.astype(np.complex64 if epsilon > 1e-10 else np.complex128)
    samples = result.astype(np.complex128)
    norm = np.sum(abs(samples))
    time = np.arange(len(samples))*dt
    spectrum = np.array([abs(np.sum(samples*np.exp(2j*np.pi*f*time)))/norm
                         for f in np.r_[frequencies, carrier]])
    required = spectrum if record else spectrum[:-1]
    if not np.isfinite(spectrum).all() or np.any(required <= 100*epsilon):
        raise ValueError('pulse_notches leaves a required frequency in the numerical source tail; move or remove notches.')
    return result, dict(notches=values, relative_spectrum=spectrum[:-1],
                        spectrum_kind='sampled_l1', carrier_fraction=spectrum[-1],
                        sampled_peak_time=float(time[np.argmax(abs(samples))]))


def _scattering_change_bound(xp, columns):
    """Maximum component change for unit Jones inputs; zero subgradient at zero."""
    squared = xp.sum(xp.abs(columns)**2, axis=-1)
    zero = squared == 0
    return xp.where(zero, xp.zeros_like(squared),
                    xp.sqrt(xp.where(zero, xp.ones_like(squared), squared)))


def _require_scene(valid, message):
    """Reject concrete invalid inputs; the scene accumulates abstract validity."""
    if _concrete(valid) is False:
        raise ValueError(message)


def _diffraction_channels(band, kx, ky, index, dt, dx, dy, *, xp=np, check=_require_scene):
    """Continuum and Yee transverse-cutoff data for a lossless half space.

    Positive Yee cutoff parameter is the lower edge of the longitudinal
    passband, not a guarantee against longitudinal Nyquist or mesh reflections.
    """
    k0 = 2*np.pi/band[:, None, None]
    transverse = (kx/k0)**2+(ky/k0)**2
    q2 = index**2-(kx/k0)**2-(ky/k0)**2
    grazing = ((abs(q2) < 1e-16)
               | (abs(q2) <= 32*np.finfo(float).eps*(index**2+transverse)))
    check(~xp.any(grazing), 'A retained diffraction order is exactly grazing; adjust wavelength/window.')
    q = boundary_kz(xp, index**2, transverse+0j)
    yee = (2/k0)**2*((index*xp.sin(k0*dt/2)/dt)**2
                       -(xp.sin(kx*dx/2)/dx)**2-(xp.sin(ky*dy/2)/dy)**2)
    squared = kx*kx+ky*ky
    # Avoid an unused sqrt(0) derivative at normal incidence.
    length = xp.sqrt(xp.where(squared == 0, xp.ones_like(squared), squared))
    cutoff = xp.where(squared == 0, xp.full_like(length, np.inf), 2*np.pi*index/length)
    return dict(q=q, propagating=q2 > 0, cutoff_wavelength=cutoff,
                yee_cutoff_parameter=yee, cutoff_resolved=(q2 > 0) == (yee > 0),
                p_projection_gain=index/abs(q))


def _directional_projection(band, kx, ky, epsilon, mu, dt, dx, dy, dz, *, xp=np, check=_require_scene):
    """Map co-located tangential H to E+ - E- on a uniform Yee cell pair.

    Transverse interpolation is already removed. H retains its z interpolation;
    this matrix removes that factor and uses discrete Maxwell admittance. Both
    the angle-dependent matrix and its application preserve backend graphs.
    """
    omega = 2*np.pi/band[:, None, None]
    check(~xp.any(omega*dt >= np.pi), 'Directional diffraction requires frequencies below temporal Nyquist.')
    frequency = 2*xp.sin(omega*dt/2)/dt
    x, y = 2*xp.sin(kx*dx/2)/dx, 2*xp.sin(ky*dy/2)/dy
    transverse = (x/frequency)**2+(y/frequency)**2
    square = epsilon*mu-transverse
    check(~xp.any(abs(square) <= 32*np.finfo(float).eps*(epsilon*mu+transverse)),
          'Directional diffraction is singular at a discrete cutoff; refine the grid or shift wavelength.')
    z = frequency*boundary_kz(xp, epsilon*mu, transverse+0j)
    check(~xp.any(dz*xp.real(z)/2 >= 1-32*np.finfo(float).eps),
          'Directional diffraction reaches longitudinal Nyquist; refine the grid.')
    interpolation = xp.cos(xp.asin(dz*z/2))
    scale = epsilon*frequency*z*interpolation
    return xp.stack((xp.stack((xp.broadcast_to(x*y, scale.shape), z*z+y*y), axis=1),
                     xp.stack((-(z*z+x*x), xp.broadcast_to(-x*y, scale.shape)), axis=1)), axis=1)/scale[:, None, None]


def _split_directions(xp, electric, magnetic, projection):
    """Tangential E travelling toward +z and -z, including evanescent branches."""
    derived = xp.stack(tuple(sum(projection[:, :, c, h]*magnetic[:, :, h]
                                  for h in range(2)) for c in range(2)), axis=2)
    return xp.stack(((electric+derived)/2, (electric-derived)/2), axis=2)


def _weights(centres, width, edges, periodic=False, subpixel=True):
    """Static cell overlaps; material arithmetic is performed by the backend."""
    if not subpixel:
        x = np.mod(centres, edges[-1]) if periodic else centres
        return np.eye(len(edges)-1)[np.clip(np.searchsorted(edges, x, side='right')-1, 0, len(edges)-2)]
    width = np.asarray(width)
    width = width[:, None] if width.ndim else width
    left, right = centres[:, None]-width/2, centres[:, None]+width/2
    shifts = (-edges[-1], 0., edges[-1]) if periodic else (0.,)
    weights = sum(np.maximum(0., np.minimum(right, edges[None, 1:]+s)
                             - np.maximum(left, edges[None, :-1]+s)) for s in shifts)/width
    return weights/weights.sum(axis=1, keepdims=True)


def _parametric_recipe(structure):
    return isinstance(structure, (list, tuple)) and any(isinstance(layer, dict) for layer in structure)


def _shape_partition(structure, periods, *, xp, array, check, integration='separable', lattice=None):
    """Resolve shared recipes into a separable partition or live area geometry.

    Sorting every periodic face keeps the partition size static. Face events
    define membership, including zero-width intervals between coincident faces.
    Midpoint classification there would choose an arbitrary material and lose
    derivatives even for disjoint shapes whose projected edges coincide.
    Face coordinates retain AD; actual paint topology changes can have kinks.
    The partition is not the Yee mesh and is never detached or rasterized on
    a fixed auxiliary grid. The area route returns live boundary events for
    multiple painted rectangles or one polygon/disk/ellipse per layer,
    for integration against fixed Yee cells. Shared shape readers
    own lengths and cell origin on both routes.

    ``lattice`` is the cell the recipe is drawn on, ``diag(periods)`` when omitted. A shape recipe
    means the same thing on any cell, so the area route reads a skewed one for what it is: the
    centre folds into the cell on the Lagrange-reduced basis and the size checks compare
    projected extents with the cell's width across each pair of its edges
    (:func:`_fits_across_cell`). On an axis-aligned cell both are the rectangular expressions,
    kept verbatim. The separable route is a tensor product of x and y face intervals, which a
    skewed cell has none of, so it refuses one; so do several painted rectangles, whose fold
    stands on the rectangular Yee box (:func:`_yee_box`).
    """
    from .lattice import as_lattice, is_axis_aligned
    from .parametric import check_layer_spec, check_shape_spec, shape_geometry
    lattice = as_lattice(period=periods, lattice=lattice)
    if integration != 'area' and not is_axis_aligned(lattice):
        raise ValueError('Separable FDTD partitions the cell into x and y face intervals, which a skewed '
                         'cell does not have; use shape_integration=area.')
    def scalar(value, name, *, positive=False):
        v = array(value)
        if v.shape != () or not xp.isdtype(v.dtype, ('real floating', 'integral', 'complex floating')):
            raise ValueError(f'{name} must be a real scalar.')
        real = xp.real(v) if xp.isdtype(v.dtype, 'complex floating') else v
        valid = xp.isfinite(v) & ((xp.imag(v) == 0) if xp.isdtype(v.dtype, 'complex floating') else True)
        valid = valid & (real > 0) if positive else valid
        check(valid, f'{name} must be finite and real'+(' and positive.' if positive else '.'))
        return xp.astype(real, xp.result_type(real, array(1.)))
    def material(value):
        m = as_material(value)
        return scalar(m.epsilon, 'Shape epsilon', positive=True), scalar(m.mu_value, 'Shape mu', positive=True)
    layers, area_layers, faces = [], [], [[array(0.), array(periods[0])], [array(0.), array(periods[1])]]
    y_invariant = True
    for i, layer in enumerate(structure):
        if not isinstance(layer, dict):
            raise ValueError('Parametric FDTD structure must contain only layer dictionaries.')
        check_layer_spec(layer, i)
        base = material(layer.get('background', 1.))
        shapes = []
        area_shapes = []
        for j, shape in enumerate(layer.get('shapes', [])):
            if not isinstance(shape, dict):
                raise ValueError('Parametric FDTD shapes must be dictionaries.')
            check_shape_spec(shape, j, i)
            kind, cx, cy, lx, ly, angle = shape_geometry(shape, lattice)
            y_invariant &= isinstance(ly, (int, float, np.number)) and np.isreal(ly) and ly == periods[1]
            circular = kind in ('circle', 'disk')
            if kind not in ('rectangle', 'square', 'circle', 'disk', 'ellipse'):
                raise ValueError('Unsupported parametric FDTD shape.')
            if (circular or kind == 'ellipse') and integration != 'area':
                raise ValueError('Circles and ellipses require shape_integration=area.')
            if integration == 'separable' and (not isinstance(angle, (int, float, np.number)) or not np.isreal(angle) or angle != 0):
                raise ValueError('Separable FDTD requires static angle=0; use shape_integration=area for rotation.')
            centres = [scalar(v, 'Shape centre') for v in (cx, cy)]
            sizes = [scalar(v, 'Shape length', positive=True) for v in (lx, ly)]
            if is_axis_aligned(lattice):
                centres = [xp.remainder(c, p) for c, p in zip(centres, periods)]
            else:
                centres = _fold_into_reduced_cell(centres, lattice, xp=xp)
            if integration == 'area':
                from .polygon import rect_verts
                angle = scalar(angle, 'Shape angle')
                if circular:
                    if is_axis_aligned(lattice):
                        check((sizes[0] <= periods[0]) & (sizes[0] <= periods[1]),
                              'Circle diameter must not exceed either rectangular period.')
                    else:
                        check(_fits_across_cell(lattice, 'circle', sizes, angle, xp=xp),
                              'Circle diameter must not exceed the cell width across either pair of '
                              'cell edges (a1, a2).')
                    # The shared circle/disk contract uses lx (or l) as its
                    # diameter. A finite angle has no effect on a disk.
                    area_shapes.append((xp.stack((*centres, sizes[0]/2)), material(shape.get('material', 1.))))
                    continue
                co, si = xp.abs(xp.cos(angle)), xp.abs(xp.sin(angle))
                if kind == 'ellipse':
                    if is_axis_aligned(lattice):
                        check(((co*sizes[0])**2+(si*sizes[1])**2 <= periods[0]**2)
                              & ((si*sizes[0])**2+(co*sizes[1])**2 <= periods[1]**2),
                              'Ellipse projected x/y extents must not exceed their periods.')
                    else:
                        check(_fits_across_cell(lattice, 'ellipse', sizes, angle, xp=xp),
                              'Ellipse projected extents across each pair of cell edges (a1, a2) must '
                              'not exceed the cell width there.')
                    area_shapes.append((xp.stack((*centres, sizes[0]/2, sizes[1]/2, angle)),
                                        material(shape.get('material', 1.))))
                    continue
                if is_axis_aligned(lattice):
                    check((co*sizes[0]+si*sizes[1] <= periods[0]) & (si*sizes[0]+co*sizes[1] <= periods[1]),
                          'Area integration requires rotated x/y extents no larger than the periods.')
                else:
                    check(_fits_across_cell(lattice, 'rectangle', sizes, angle, xp=xp),
                          'Area integration requires rotated extents across each pair of cell edges '
                          '(a1, a2) no larger than the cell width there.')
                vertices = rect_verts(xp, *centres, *sizes, angle)
                area_shapes.append((vertices, material(shape.get('material', 1.))))
                continue
            face_ids = []
            for axis, (c, size, period) in enumerate(zip(centres, sizes, periods)):
                check(size <= period, 'Shape lengths must not exceed their rectangular periods.')
                face_ids.append((len(faces[axis]), len(faces[axis])+1))
                faces[axis].extend((xp.remainder(c-size/2, period), xp.remainder(c+size/2, period)))
            shapes.append((sizes, face_ids, material(shape.get('material', 1.))))
        layers.append((base, shapes))
        area_shape = area_shapes[0] if area_shapes else None
        if len(area_shapes) > 1:
            if any(v.ndim != 2 for v, _ in area_shapes):
                raise ValueError('Multiple area shapes currently require rectangles or squares; circles and ellipses require a single-shape layer.')
            area_shape = {'edges': _painted_rectangle_edges(area_shapes, base, periods, xp=xp, array=array,
                                                            lattice=lattice)}
        area_layers.append((base, area_shape))
    if integration == 'area':
        # Two placeholder rows select the full, static y mesh. Their values
        # are never sampled: area_layers supplies each Yee-cell coefficient.
        epsilon, mu = [xp.stack([xp.ones((2, 1), dtype=b[k].dtype, device=getattr(b[k], 'device', None))*b[k]
                                 for b, _ in area_layers]) for k in range(2)]
        return epsilon, mu, area_layers
    edges, ranks = [], []
    for axis in faces:
        values = xp.stack(axis)
        order = xp.argsort(values, stable=True)
        edges.append(values[order])
        ranks.append(xp.argsort(order, stable=True))
    def membership(axis, ids, size):
        start, end = ranks[axis][ids[0]], ranks[axis][ids[1]]
        cells = xp.arange(len(edges[axis])-1, device=getattr(start, 'device', None))
        inside = xp.where(start < end, (cells >= start) & (cells < end),
                          (cells >= start) | (cells < end))
        return inside | (size == periods[axis])
    epsilon, mu = [], []
    for base, shapes in layers:
        shape = (len(edges[1])-1, len(edges[0])-1)
        eps, permeability = [xp.ones(shape, dtype=v.dtype, device=getattr(v, 'device', None))*v for v in base]
        for sizes, ids, values in shapes:
            inside = membership(0, ids[0], sizes[0])[None, :] & membership(1, ids[1], sizes[1])[:, None]
            eps, permeability = xp.where(inside, values[0], eps), xp.where(inside, values[1], permeability)
        epsilon.append(eps); mu.append(permeability)
    epsilon, mu = xp.stack(epsilon), xp.stack(mu)
    if y_invariant:
        # A statically full-height recipe is the same one-row grating the
        # raster path accepts. Live ly values never choose this shortcut.
        epsilon, mu = epsilon[:, :1], mu[:, :1]
        edges[1] = array([0., periods[1]])
    return epsilon, mu, edges


def _clipped_rectangle_area(vertices, left, right, bottom, top, *, xp, moment_weights=None):
    """Exact clipped area or bilinearly weighted integral with live vertices.

    Four half-plane clips can produce at most eight vertices. Stable packing
    keeps their cyclic order without a host geometry plan or detached values.
    Safe inactive denominators prevent NaN derivatives on parallel edges.
    Combinatorics are piecewise; edge-on-edge contacts may have distinct slopes.
    """
    shape = left.shape
    polygon = xp.broadcast_to(xp.concat((vertices, vertices), axis=-2), (*shape, 8, 2))
    count = xp.full(shape, 4, dtype=xp.int64, device=getattr(vertices, 'device', None))
    index = xp.arange(8, device=getattr(vertices, 'device', None))
    for axis, bound, sign in ((0, left, 1), (0, right, -1), (1, bottom, 1), (1, top, -1)):
        previous = xp.where(index == 0, count[..., None]-1, index-1)
        previous = xp.maximum(previous, xp.zeros_like(previous))
        a = xp.take_along_axis(polygon, xp.broadcast_to(previous[..., None], polygon.shape), axis=-2)
        b = polygon
        da, db = sign*(a[..., axis]-bound[..., None]), sign*(b[..., axis]-bound[..., None])
        active = index < count[..., None]
        inside_a, inside_b = da >= 0, db >= 0
        crossing = active & (inside_a != inside_b)
        denominator = xp.where(crossing, da-db, xp.ones_like(da))
        t = xp.where(crossing, da/denominator, xp.zeros_like(da))
        intersection = a+t[..., None]*(b-a)
        candidates = xp.stack((intersection, b), axis=-2).reshape(*shape, 16, 2)
        valid = xp.stack((crossing, active & inside_b), axis=-1).reshape(*shape, 16)
        order = xp.argsort(xp.astype(~valid, xp.int32), axis=-1, stable=True)[..., :8]
        polygon = xp.take_along_axis(candidates, xp.broadcast_to(order[..., None], (*shape, 8, 2)), axis=-2)
        count = xp.sum(xp.astype(valid, xp.int64), axis=-1)
    following = xp.where(index+1 < count[..., None], index+1, xp.zeros_like(index))
    other = xp.take_along_axis(polygon, xp.broadcast_to(following[..., None], polygon.shape), axis=-2)
    # Cell-relative coordinates reduce cancellation for tiny clipped pieces.
    origin = xp.stack((left, bottom), axis=-1)[..., None, :]
    a, b = polygon-origin, other-origin
    if moment_weights is not None:
        extent = xp.stack((right-left, top-bottom), axis=-1)
        scale = xp.where(extent > 0, extent, xp.ones_like(extent))[..., None, :]
        active = (index < count[..., None])[..., None]
        a = xp.where(active, a/scale, xp.zeros_like(a))
        b = xp.where(active, b/scale, xp.zeros_like(b))
    cross = a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
    cross = xp.where(index < count[..., None], cross, xp.zeros_like(cross))
    area = .5*xp.sum(cross, axis=-1)
    if moment_weights is not None:
        # Exact integrals of 1, u, v, uv on the clipped polygon, in normalized
        # box coordinates. Integrate (a0+a1*u)*(b0+b1*v) without quadrature.
        mx = xp.sum((a[..., 0]+b[..., 0])*cross, axis=-1)/6
        my = xp.sum((a[..., 1]+b[..., 1])*cross, axis=-1)/6
        mxy = xp.sum((2*a[..., 0]*a[..., 1]+a[..., 0]*b[..., 1]
                       +b[..., 0]*a[..., 1]+2*b[..., 0]*b[..., 1])*cross, axis=-1)/24
        a0, a1, b0, b1 = [moment_weights[..., k] for k in range(4)]
        area = (a0*b0*area+a1*b0*mx+a0*b1*my+a1*b1*mxy)*extent[..., 0]*extent[..., 1]
    return xp.where(area < 0, xp.zeros_like(area), area)


def _translation_windows(centres, width, smoothing):
    """Static linear pieces of a cell indicator averaged over translations."""
    centres = np.asarray(centres)
    if not smoothing:
        return [(centres-width/2, centres+width/2, 1., 0.)]
    cuts = np.sort([-(width+smoothing)/2, (smoothing-width)/2,
                    (width-smoothing)/2, (width+smoothing)/2])
    height = min(width, smoothing)/smoothing
    values = [0., height, height, 0.]
    return [(centres+left, centres+right, low, high-low)
            for left, right, low, high in zip(cuts[:-1], cuts[1:], values[:-1], values[1:]) if right > left]


def _yee_box(periods, lattice):
    """The Cartesian Yee grid's periodic box ``(Lx, Ly)`` -- which only an axis-aligned cell has.

    The grid tiles ``[0, Lx) x [0, Ly)`` and closes its last cell onto its first by ``(Lx, 0)``
    and ``(0, Ly)``, which are lattice vectors only when the cell is ``diag(Lx, Ly)``. A skewed
    cell has its images at ``m*a1 + n*a2``, and a painted fold, a wrapped face or a one-ring image
    sum written on this box would describe a rectangle. So the geometry helpers that stand on the
    box ask for it here, beside their arithmetic, and a skewed ``lattice`` is refused there rather
    than answered for a rectangle. The entries returned are the caller's own ``periods`` objects,
    so a rectangular cell computes exactly what it did before the box had a name.

    Not everything on the box goes through here. :func:`solve_scene_fdtd` builds the grid itself,
    and its Bloch shift, order window and monitor coordinates, from ``lengths`` -- the same pair --
    with only the refusal of any ``lattice`` at its door in front of them. Relaxing that door
    means guarding those too; the cell-shape gate cannot see them, because ``lengths`` is not a
    period alias.

    Written so the cell-shape gate (``tests/test_cell_shape_gate.py``) sees no period pair in the
    arithmetic that follows: that is honest only because this refusal comes first. A caller that
    drops it would be laundering a rectangle past the gate.
    """
    from .lattice import is_axis_aligned
    if not is_axis_aligned(lattice):
        raise ValueError(
            'The FDTD Yee grid is a Cartesian periodic box: it closes its last cell onto its first by '
            '(Lx, 0) and (0, Ly), and only an axis-aligned cell has those as lattice vectors. A skewed '
            'cell has its images at m*a1 + n*a2; refused rather than answered for a rectangle.')
    cell = np.asarray(lattice)
    if not (cell[0, 0] == periods[0] and cell[1, 1] == periods[1]):
        raise ValueError('The Yee box must be the cell it is built on, but periods and lattice disagree.')
    return periods[0], periods[1]


def _image_shifts(periods, lattice, rings=(1, 1), *, array):
    """A shape's periodic images, ``(m*Lx, n*Ly)`` for ``|m| <= rings[0]``, ``|n| <= rings[1]``.

    ``m`` outer and ``n`` inner, the order the image sums in this file reduce over, so a change of
    spelling here cannot regroup a floating sum. It replaces six copies of the same literal, so the
    moment function and the fill fraction now stand on one ring by construction rather than by a
    docstring's promise (``tests/test_fdtd_interface_moments.py`` pins that pairing).

    The box index set is enough only on the rectangular argument its callers make: a centre
    folded per axis and a projected extent no wider than the period. It is refused on a skewed
    cell (:func:`_yee_box`) rather than extended to one. That would need the images enumerated on
    the reduced basis to a reach, as ``boundary.periodic_segments`` and
    ``parametric._ring_offsets`` do, and a Yee grid that closes on the same parallelogram, which a
    Cartesian grid cannot.
    """
    Lx, Ly = _yee_box(periods, lattice)
    return array([(m*Lx, n*Ly) for m in range(-rings[0], rings[0]+1) for n in range(-rings[1], rings[1]+1)])


def _fold_into_reduced_cell(centres, lattice, *, xp):
    """A shape centre carried into the cell on the Lagrange-reduced basis (``boundary.fold_into_cell``).

    The general form of the per-axis ``remainder`` the rectangular route keeps. Reduced first,
    because only on the reduced basis is one ring of images enough, which is the lesson of the
    RCWA boundary path (ae4c770): the cell a point is folded into and the cell the images are
    enumerated on have to be the same one.
    """
    from .boundary import fold_into_cell
    from .lattice import reduced_basis
    from .parametric import _cell_floats
    a1x, a1y, a2x, a2y = _cell_floats(reduced_basis(lattice))
    folded = fold_into_cell(xp, xp.stack(centres)[None, :], ((a1x, a2x), (a1y, a2y)))[0]
    return [folded[0], folded[1]]


def _fits_across_cell(lattice, kind, sizes, angle, *, xp):
    """Whether a shape's projected extent across each pair of cell edges fits the cell there.

    On the Lagrange-reduced basis ``(a1, a2)`` the cell is ``|det|/|a2|`` wide across the edges
    parallel to ``a2`` and ``|det|/|a1|`` across the edges parallel to ``a1``. A shape no wider
    than that in either direction spans at most one cell in each fractional coordinate. That is
    what the rectangular checks in :func:`_shape_partition` say on ``diag(Px, Py)``, and it is
    enough for what they guard: a centre folded into the cell stays within one ring of images,
    and no image overlaps its own neighbour. Enough, not needed: on the hexagonal cell of side
    ``a`` the width is ``a*sqrt(3)/2``, so a disc with ``a*sqrt(3)/2 < lx <= a`` is refused though
    it overlaps none of its images -- as a rotated bar that is wider than the period in ``x`` but
    overlaps nothing is refused on a rectangle.

    Along a unit normal ``n`` a box is ``lx*|u.n| + ly*|v.n|`` wide (``u``, ``v`` the shape's own
    axes), an ellipse ``sqrt((lx*u.n)**2 + (ly*v.n)**2)``, a circle ``lx``. The basis is static
    (FDTD periods are), so it is read as floats; the shape's sizes and angle stay live, and the
    result feeds ``check`` like the rectangular checks do.
    """
    from .lattice import reduced_basis
    from .parametric import _cell_floats
    a1x, a1y, a2x, a2y = _cell_floats(reduced_basis(lattice))
    det = abs(a1x*a2y-a2x*a1y)
    fits = None
    for (nx, ny), length in (((a2y, -a2x), math.hypot(a2x, a2y)), ((-a1y, a1x), math.hypot(a1x, a1y))):
        nx, ny, width = nx/length, ny/length, det/length
        if kind == 'circle':
            ok = sizes[0] <= width
        else:
            pu = xp.cos(angle)*nx+xp.sin(angle)*ny
            pv = -xp.sin(angle)*nx+xp.cos(angle)*ny
            ok = ((sizes[0]*pu)**2+(sizes[1]*pv)**2 <= width**2 if kind == 'ellipse'
                  else sizes[0]*xp.abs(pu)+sizes[1]*xp.abs(pv) <= width)
        fits = ok if fits is None else fits & ok
    return fits


def _painted_rectangle_edges(shapes, base, periods, *, xp, array, lattice=None):
    """Piecewise material jumps on a fundamental cell's straight boundaries.

    Rectangle/rectangle intersections split each oriented edge. Infinitesimal
    left/right membership supplies the last-painted material on both sides;
    coincident boundaries belong to their last owner. Cell-border segments
    retain the periodic constant mode even when shapes cover the whole cell.
    Static event arrays permit topology changes within one traced function.
    Contacts retain only piecewise first derivatives, not a unique shape slope.

    The cell-border loop, the face cuts and the final fold all stand on the rectangular Yee box:
    the pieces are folded into the box the Cartesian windows tile. :func:`_image_shifts` refuses a
    skewed ``lattice`` through :func:`_yee_box`, which is the guard for all of them.
    """
    from .lattice import as_lattice
    lattice = as_lattice(period=periods, lattice=lattice)
    vertices = xp.stack([s[0] for s in shapes])
    shifts = _image_shifts(periods, lattice, array=array)
    images = vertices[:, None, :, :]+shifts[None, :, None, :]
    directions = xp.roll(images, -1, axis=-2)-images
    normals = xp.stack((-directions[..., 1], directions[..., 0]), axis=-1)
    normals = normals/xp.sqrt(xp.sum(normals**2, axis=-1, keepdims=True))
    tol = 32*xp.finfo(vertices.dtype).eps*max(periods)
    cell = array(np.array([[0., 0.], [periods[0], 0.], periods, [0., periods[1]]]))
    result = []
    for owner, polygon in [*enumerate(vertices), (-1, cell)]:
        for edge in range(4):
            a, b = polygon[edge], polygon[(edge+1)%4]
            direction = b-a
            normal = xp.stack((-direction[1], direction[0]))
            normal = normal/xp.sqrt(xp.sum(normal**2))
            # Clip this segment to every periodic rectangle using half planes.
            d = xp.sum((a-images)*normals, axis=-1)
            slope = xp.sum(direction*normals, axis=-1)
            moving = xp.abs(slope) > tol
            t = -d/xp.where(moving, slope, xp.ones_like(slope))
            lower = xp.max(xp.where(moving & (slope > 0), t, xp.zeros_like(t)), axis=-1)
            upper = xp.min(xp.where(moving & (slope < 0), t, xp.ones_like(t)), axis=-1)
            valid = (upper > lower) & xp.all(moving | (d >= -tol), axis=-1)
            lower = xp.where(valid, xp.maximum(lower, xp.zeros_like(lower)), xp.zeros_like(lower))
            upper = xp.where(valid, xp.minimum(upper, xp.ones_like(upper)), xp.zeros_like(upper))
            lower = xp.minimum(lower, xp.ones_like(lower)); upper = xp.maximum(upper, xp.zeros_like(upper))
            cuts = [lower.reshape(-1), upper.reshape(-1)]
            for axis, period in enumerate(periods):
                active = xp.abs(direction[axis]) > tol
                face = (array([0., period])-a[axis])/xp.where(active, direction[axis], 1.)
                cuts.append(xp.where(active, xp.clip(face, 0., 1.), xp.zeros_like(face)))
            # Internal events can coincide with an endpoint while retaining a
            # live slope. They must not replace the segment's fixed 0/1 limits.
            cuts = xp.concat((array([0.]), xp.sort(xp.concat(cuts)), array([1.])))
            lo, hi = cuts[:-1], cuts[1:]
            midpoint = a+((lo+hi)/2)[:, None]*direction
            distance = xp.sum((midpoint[None, None, :, None, :]-images[:, :, None, :, :])
                              *normals[:, :, None, :, :], axis=-1)
            # Roundoff-close contacts use the one-sided limiting half plane.
            on = xp.abs(distance) <= tol
            side = xp.sum(normals*normal, axis=-1)[:, :, None, :]
            left = xp.any(xp.all((distance > tol) | (on & (side >= 0)), axis=-1), axis=1)
            right = xp.any(xp.all((distance > tol) | (on & (side <= 0)), axis=-1), axis=1)
            closed = xp.all(distance >= -tol, axis=-1)
            boundary = xp.any(closed & xp.any(on, axis=-1), axis=1)
            inside = xp.broadcast_to(xp.stack(base), (len(lo), 2))
            outside = inside
            for j, (_, values) in enumerate(shapes):
                inside = xp.where(left[j, :, None], xp.stack(values), inside)
                outside = xp.where(right[j, :, None], xp.stack(values), outside)
            if owner < 0:
                jump = inside-xp.stack(base)
                shift = xp.zeros_like(midpoint)
            else:
                jump = inside-outside
                hidden = xp.any(boundary[owner+1:], axis=0) if owner+1 < len(shapes) else xp.zeros_like(lo, dtype=xp.bool)
                # Fold only physical boundaries. The cell-border loop stays
                # closed in its original coordinates and replaces seam edges.
                shift = xp.floor(midpoint/array(periods))*array(periods)
                local = midpoint-shift
                seam = xp.any((xp.abs(direction) <= tol) & ((local < tol) | (array(periods)-local < tol)), axis=-1)
                jump = xp.where((hidden | seam)[:, None], xp.zeros_like(jump), jump)
            result.append((a+lo[:, None]*direction-shift, a+hi[:, None]*direction-shift, jump))
    return result


def _painted_cell_materials(edges, base, x, y, dx, dy, periods, *, xp, array, smoothing=(0., 0.),
                            lattice=None):
    """Integrate painted boundaries against static separable cell kernels.

    Green's theorem converts the cell mass into integral(CDF_x * weight_y dy).
    Translation windows are piecewise linear; splitting at every window knot
    leaves a cubic polynomial along each straight edge, exact with two Gauss
    nodes. No intermediate raster, frozen host clipping plan or custom VJP.

    The windows are Yee cells, the pieces are folded into the Yee box, and the ring rule compares a
    Cartesian kernel with the box: all of it is the rectangular box :func:`_yee_box` names and
    refuses a skewed ``lattice`` for.
    """
    from .lattice import as_lattice
    lattice = as_lattice(period=periods, lattice=lattice)
    box = _yee_box(periods, lattice)
    xwindows = [tuple(array(v) if np.ndim(v) else v for v in row) for row in _translation_windows(x, dx, smoothing[0])]
    ywindows = [tuple(array(v) if np.ndim(v) else v for v in row) for row in _translation_windows(y, dy, smoothing[1])]
    rings = (2, 2) if smoothing[0]+dx > box[0] or smoothing[1]+dy > box[1] else (1, 1)
    shifts = _image_shifts(periods, lattice, rings, array=array)
    total = xp.broadcast_to(xp.stack(base), (len(y), len(x), 2))
    for start, end, jump in edges:
        a = start[:, None, None, None, :]+shifts[None, :, None, None, :]
        delta = (end-start)[:, None, None, None, :]
        shape = (len(start), len(shifts), len(y), len(x))
        lo = xp.zeros(shape, dtype=start.dtype, device=getattr(start, 'device', None))
        cuts = []
        for axis, windows in enumerate([xwindows, ywindows]):
            for face in [windows[0][0], *(w[1] for w in windows)]:
                f = face[None, :] if axis == 0 else face[:, None]
                active = xp.abs(delta[..., axis]) > 32*xp.finfo(start.dtype).eps*box[axis]
                t = (f-a[..., axis])/xp.where(active, delta[..., axis], xp.ones_like(delta[..., axis]))
                cuts.append(xp.broadcast_to(xp.where(active, xp.clip(t, 0., 1.), xp.zeros_like(t)), shape))
        cuts = xp.concat((lo[..., None], xp.sort(xp.stack(cuts, axis=-1), axis=-1),
                          xp.ones_like(lo)[..., None]), axis=-1)
        half, mid = (cuts[..., 1:]-cuts[..., :-1])/2, (cuts[..., 1:]+cuts[..., :-1])/2
        nodes = mid[..., None]+half[..., None]*array([-1/np.sqrt(3), 1/np.sqrt(3)])
        points = a[..., None, None, :]+nodes[..., None]*delta[..., None, None, :]
        cdf, weight = xp.zeros_like(nodes), xp.zeros_like(nodes)
        for l, r, a0, a1 in xwindows:
            width = r-l
            u = xp.clip((points[..., 0]-l[None, None, None, :, None, None])/xp.where(width > 0, width, 1.)[None, None, None, :, None, None], 0., 1.)
            cdf = cdf+width[None, None, None, :, None, None]*(a0*u+a1*u*u/2)
        for l, r, a0, a1 in ywindows:
            width = r-l
            u = (points[..., 1]-l[None, None, :, None, None, None])/xp.where(width > 0, width, 1.)[None, None, :, None, None, None]
            # A horizontal edge can lie on a shared knot for a whole segment.
            # Count that knot once, even though dy=0: its angle derivative need
            # not vanish. Use coordinates for membership, not rounded ratios.
            inside = ((points[..., 1] >= l[None, None, :, None, None, None])
                      & (points[..., 1] < r[None, None, :, None, None, None])
                      & (width > 0)[None, None, :, None, None, None])
            weight = weight+xp.where(inside, a0+a1*u, xp.zeros_like(u))
        value = delta[..., 1]*xp.sum(half*xp.sum(cdf*weight, axis=-1), axis=-1)
        total = total+xp.sum(value[..., None]*jump[:, None, None, None, :], axis=(0, 1))/(dx*dy)
    return xp.moveaxis(total, -1, 0)


def _weighted_circle_area(circle, left, right, bottom, top, weights, *, xp):
    """Stable bilinear disk integral using flat strips and angular arc strips.

    Direct differences of four global moments lose digits on narrow smoothing
    windows. Here each strip is integrated in its own box coordinates. On an
    arc, x=r*sin(theta) makes the integrand a trigonometric polynomial of degree
    at most four. A fixed 16-point Gauss rule on spans <= pi/2 resolves it to
    double precision without a polygonal boundary or a tunable sampling grid.
    """
    cx, cy, radius = (circle[..., k] for k in range(3))
    a0, a1, b0, b1 = (weights[..., k] for k in range(4))
    l, r, b, t = left-cx, right-cx, bottom-cy, top-cy
    wx, wy = right-left, top-bottom
    nonempty = (wx > 0) & (wy > 0)
    # A positive static knot interval can round to zero after its centre is
    # added. It has no measure; guard every inactive normalized coordinate.
    wx, wy = xp.where(wx > 0, wx, xp.ones_like(wx)), xp.where(wy > 0, wy, xp.ones_like(wy))
    def positive(x):
        return xp.where(x > 0, x, xp.zeros_like(x))
    def root(v):
        active = v < radius
        return xp.where(active, xp.sqrt(xp.where(active, (radius-v)*(radius+v), xp.ones_like(v))), xp.zeros_like(v))
    def angle(v):
        active = v < radius
        return xp.where(active, xp.asin(xp.where(active, v/radius, xp.zeros_like(v))), np.pi/2)
    nodes, gauss = [xp.asarray(v, dtype=radius.dtype, device=getattr(radius, 'device', None))
                    for v in np.polynomial.legendre.leggauss(16)]
    result = xp.zeros_like(left)
    for sx in (-1., 1.):
        lo_x, hi_x = (l, r) if sx == 1 else (-r, -l)
        lo_x, hi_x = xp.minimum(positive(lo_x), radius), xp.minimum(positive(hi_x), radius)
        for sy in (-1., 1.):
            lo_y, hi_y = (b, t) if sy == 1 else (-t, -b)
            lo_y, hi_y = xp.minimum(positive(lo_y), radius), xp.minimum(positive(hi_y), radius)
            flat_end = xp.minimum(hi_x, root(hi_y))
            flat_width = positive(flat_end-lo_x)
            fx = a0+a1*((sx*(flat_end+lo_x)/2-l)/wx)
            fy = b0+b1*((sy*(hi_y+lo_y)/2-b)/wy)
            flat = flat_width*(hi_y-lo_y)*fx*fy
            arc_start = xp.maximum(lo_x, root(hi_y))
            arc_end = xp.minimum(hi_x, root(lo_y))
            active = (arc_end > arc_start) & (hi_y > lo_y)
            theta0 = angle(xp.where(active, arc_start, xp.zeros_like(arc_start)))
            theta1 = angle(xp.where(active, arc_end, xp.zeros_like(arc_end)))
            half = (theta1-theta0)/2
            theta = (theta0+half)[..., None]+half[..., None]*nodes
            x, y = radius[..., None]*xp.sin(theta), radius[..., None]*xp.cos(theta)
            weight_x = a0[..., None]+a1[..., None]*(sx*x-l[..., None])/wx[..., None]
            weight_y = b0[..., None]+b1[..., None]*(sy*(y+lo_y[..., None])/2-b[..., None])/wy[..., None]
            arc = half*xp.sum(gauss*y*weight_x*(y-lo_y[..., None])*weight_y, axis=-1)
            result = result+flat+xp.where(active, arc, xp.zeros_like(arc))
    return xp.where(nonempty & (result >= 0), result, xp.zeros_like(result))


def _clipped_circle_area(circle, left, right, bottom, top, *, xp, moment_weights=None):
    """Analytic disk/box area, dispatching weighted windows to stable arc strips.

    Integrate vertical disk chords on each signed quadrant. The four corner
    primitives give the area in disk-centred coordinates. No
    polygon, auxiliary raster or numerical quadrature defines the boundary.
    Inactive square roots/arcsines are guarded *before* evaluation for AD.
    Choosing a signed branch at zero preserves centre derivatives on axes.
    """
    if moment_weights is not None:
        return _weighted_circle_area(circle, left, right, bottom, top, moment_weights, xp=xp)
    cx, cy, radius = (circle[..., k] for k in range(3))
    r2 = radius*radius
    def primitive(x, y):
        sx, sy = xp.where(x >= 0, 1., -1.), xp.where(y >= 0, 1., -1.)
        a, b = xp.minimum(sx*x, radius), xp.minimum(sy*y, radius)
        inside = b < radius
        split = xp.where(inside, xp.sqrt(xp.where(inside, r2-b*b, xp.ones_like(b))), xp.zeros_like(b))
        s = xp.where(a < split, a, split)
        def arc(t):
            # The elementary primitive subtracts divergent derivatives near
            # t=r. Integrating sqrt(2*z-z*z) as a series at that endpoint
            # removes this cancellation (the omitted term is < 1e-22).
            near = t > .99*radius
            u = xp.where(near, xp.zeros_like(t), t/radius)
            z = xp.where(near, (radius-t)/radius, xp.zeros_like(t))
            cap = np.sqrt(2.)*z**1.5*(2/3+z*(-1/10+z*(-1/112+z*(-1/576
                +z*(-5/11264+z*(-7/53248+z*(-7/163840-z*33/2228224)))))))
            return r2*xp.where(near, np.pi/4-cap, .5*(u*xp.sqrt(1-u*u)+xp.asin(u)))
        area = b*s+arc(a)-arc(s)
        return sx*sy*area
    l, r, b, t = left-cx, right-cx, bottom-cy, top-cy
    area = primitive(r, t)-primitive(l, t)-primitive(r, b)+primitive(l, b)
    return xp.where(area < 0, xp.zeros_like(area), area)


def _clipped_ellipse_area(ellipse, left, right, bottom, top, *, xp, moment_weights=None):
    """Integrate rotated ellipse chords without a polygonal approximation.

    Split vertical strips at the ellipse's intersections with horizontal box
    faces and at x=0. On each strip, x=A*sin(theta) makes the weighted chord
    integral a trigonometric polynomial of degree <=4 on a span <=pi/2. A
    fixed 16-point Gauss rule resolves both sharp and bilinear cell kernels.
    All cut coordinates remain live; inactive tangent roots are guarded.
    """
    cx, cy, a, b, angle = (ellipse[..., k] for k in range(5))
    co, si = xp.cos(angle), xp.sin(angle)
    xx, yy, xy = (a*co)**2+(b*si)**2, (a*si)**2+(b*co)**2, (a*a-b*b)*co*si
    ax, ay = xp.sqrt(xx), xp.sqrt(yy)
    tilt, chord = xy/ax, a*b/ax
    l, r, low, high = left-cx, right-cx, bottom-cy, top-cy
    l, r = xp.maximum(l, -ax), xp.minimum(r, ax)
    nonempty = (r > l) & (top > bottom)
    r = xp.where(nonempty, r, l)
    # Use in-domain zeroes for empty boxes before any inverse sine is taken.
    l, r = xp.where(nonempty, l, xp.zeros_like(l)), xp.where(nonempty, r, xp.zeros_like(r))
    cuts = [l, r, xp.zeros_like(l)]
    for face in (low, high):
        active = xp.abs(face) < ay
        v = xp.where(active, face/ay, xp.zeros_like(face))
        spread = a*b/ay*xp.sqrt(1-v*v)
        centre = xy/ay*v
        cuts.extend([xp.where(active, centre-spread, l), xp.where(active, centre+spread, l)])
    cuts = xp.sort(xp.stack([xp.maximum(l, xp.minimum(r, v)) for v in cuts], axis=-1), axis=-1)
    u = cuts/ax[..., None]
    # Projected contacts can land a few ulps inside the endpoint. Inverse-sine
    # slopes there amplify quadrature roundoff by 1/sqrt(eps). Use the limiting
    # tangent for roundoff-close contacts (area change is O(eps**1.5)).
    interior = xp.abs(u) < 1-4*xp.finfo(u.dtype).eps
    theta = xp.where(interior, xp.asin(xp.where(interior, u, xp.zeros_like(u))),
                     xp.where(u >= 0, np.pi/2, -np.pi/2))
    start, end = theta[..., :-1], theta[..., 1:]
    half, mid = (end-start)/2, (end+start)/2
    nodes, gauss = [xp.asarray(v, dtype=a.dtype, device=getattr(a, 'device', None))
                    for v in np.polynomial.legendre.leggauss(16)]
    angles = mid[..., None]+half[..., None]*nodes
    sine, cosine = xp.sin(angles), xp.cos(angles)
    x = ax[..., None, None]*sine
    y = tilt[..., None, None]*sine
    h = chord[..., None, None]*cosine
    lower = xp.maximum(y-h, low[..., None, None])
    upper = xp.minimum(y+h, high[..., None, None])
    height = xp.where(upper > lower, upper-lower, xp.zeros_like(upper))
    integrand = ax[..., None, None]*cosine*height
    if moment_weights is not None:
        wx, wy = right-left, top-bottom
        wx, wy = xp.where(wx > 0, wx, xp.ones_like(wx)), xp.where(wy > 0, wy, xp.ones_like(wy))
        a0, a1, b0, b1 = (moment_weights[..., k, None, None] for k in range(4))
        fx = a0+a1*(x+(cx-left)[..., None, None])/wx[..., None, None]
        fy = b0+b1*((upper+lower)/2-low[..., None, None])/wy[..., None, None]
        integrand = integrand*fx*fy
    area = xp.sum(half*xp.sum(gauss*integrand, axis=-1), axis=-1)
    return xp.where(nonempty & (area >= 0), area, xp.zeros_like(area))


def _shape_cell_fractions(vertices, x, y, dx, dy, periods, *, xp, array, smoothing=(0., 0.), lattice=None):
    """Periodic shape fractions, optionally averaged over box translations.

    A translated cell has a trapezoidal weight on each axis. Split its support
    into linear pieces and integrate their bilinear product with polygon
    moments or disk/ellipse arc strips. Wide kernels need a second periodic image ring near cell seams.
    Projected shape extents and smoothing widths must not exceed periods.

    The cells are Yee cells and the last one closes onto the first across the Yee box, so this
    stands on the rectangular box :func:`_yee_box` names and refuses a skewed ``lattice`` for.
    """
    from .lattice import as_lattice
    lattice = as_lattice(period=periods, lattice=lattice)
    box = _yee_box(periods, lattice)
    curved = vertices.ndim == 1
    clip = (_clipped_circle_area if vertices.shape[0] == 3 else _clipped_ellipse_area) if curved else _clipped_rectangle_area
    def images(shifts, smoothed):
        if curved:
            shifts = xp.concat((shifts, xp.broadcast_to(xp.zeros_like(shifts[:, :1]),
                                                       (len(shifts), vertices.shape[0]-2))), axis=-1)
            return (vertices+shifts)[:, None, None, None, :] if smoothed else (vertices+shifts)[:, None, None, :]
        return (vertices[None, None, None, None, :, :]+shifts[:, None, None, None, None, :] if smoothed else
                vertices[None, None, None, :, :]+shifts[:, None, None, None, :])
    if any(smoothing):
        windows = [(a, b) for a in _translation_windows(x, dx, smoothing[0])
                   for b in _translation_windows(y, dy, smoothing[1])]
        bounds = [np.broadcast_arrays(a[0][None, :], a[1][None, :], b[0][:, None], b[1][:, None])
                  for a, b in windows]
        rings = [2 if w+h > p else 1 for w, h, p in zip((dx, dy), smoothing, box)]
        shifts = _image_shifts(periods, lattice, rings, array=array)
        left, right, bottom, top = [array(np.stack([b[k] for b in bounds]))[None, ...]
                                    +xp.zeros_like(shifts[:, 0, None, None, None]) for k in range(4)]
        weights = array([[*a[2:], *b[2:]] for a, b in windows])[None, :, None, None, :]
        mass = clip(images(shifts, True), left, right, bottom, top, xp=xp, moment_weights=weights)
        fraction = xp.sum(mass, axis=(0, 1))/(dx*dy)
        return xp.where(fraction > 1, xp.ones_like(fraction), fraction)
    if curved:
        left, bottom = xp.broadcast_arrays(array(np.asarray(x)-dx/2)[None, :],
                                           array(np.asarray(y)-dy/2)[:, None])
    else:
        left, bottom = xp.broadcast_arrays(array(x)[None, :]-dx/2, array(y)[:, None]-dy/2)
    shifts = _image_shifts(periods, lattice, array=array)
    left, bottom = xp.broadcast_arrays(left[None, ...]+xp.zeros_like(shifts[:, 0, None, None]), bottom[None, ...])
    if curved:
        # Reuse identical stored faces on adjoining cells. Tiny independent
        # rounding gaps produce sqrt(gap) changes in tangent chord slopes.
        xleft, ybottom = np.asarray(x)-dx/2, np.asarray(y)-dy/2
        right = array(np.r_[xleft[1:], xleft[:1]+box[0]])[None, None, :]+xp.zeros_like(left)
        top = array(np.r_[ybottom[1:], ybottom[:1]+box[1]])[None, :, None]+xp.zeros_like(bottom)
    else:
        right, top = left+dx, bottom+dy
    area = clip(images(shifts, False), left, right, bottom, top, xp=xp)
    fraction = xp.sum(area, axis=0)/(dx*dy)
    return xp.where(fraction > 1, xp.ones_like(fraction), fraction)


def _rectangle_edge_moments(vertices, x, y, dx, dy, periods, *, xp, array, check=_require_scene,
                            lattice=None):
    """Per-cell interface arclength and second normal moment of a straight-sided shape.

    Returns (L, M_xx, M_xy, M_yy), each of shape (len(y), len(x)): arrays are v-first in this
    file, so the row index runs over y, while periods stays the x-first typed pair every caller
    passes. M_cd integrates n_c n_d over the shape's own boundary inside a cell, so M/L is the
    mean of n n^T over that interface and M_xx + M_yy == L identically.

    The second moment, not the first. A first moment N, normalised as N N^T/|N|^2, collapses to
    rank one at a symmetric 90-degree corner -- [[1/4, 1/4], [1/4, 1/4]], a spurious diagonal
    normal, where n n^T reads diag(1/2, 1/2) -- and cancels to zero between the antiparallel
    faces of a feature thinner than a cell, which is exactly the laminate for which
    eps = <eps> I - Delta n n^T is exact. It is not a factor of two waiting to be simplified away.

    It is also not liftable out of _clipped_rectangle_area: that routine's eight-vertex repack is
    what lets a topology change happen inside one trace and equally what forgets which vertices
    came from a shape edge and which from a box face, and the rotated chords of the closed clipped
    polygon sum to zero identically, so no identity recovers the shape's own edges afterwards.
    """
    if vertices.ndim == 1 or vertices.shape[-2] != 4:
        raise ValueError('_rectangle_edge_moments walks the four oriented edges of a straight-sided '
                         'shape. A circle/ellipse recipe carries axes rather than vertices and has '
                         'no polygon to walk; refused rather than approximated by a chord.')
    from .lattice import as_lattice
    lattice = as_lattice(period=periods, lattice=lattice)
    box = _yee_box(periods, lattice)
    tol = 32*xp.finfo(vertices.dtype).eps*max(periods)
    # The same 3x3 image ring and the same closed cell boxes as the *sharp* branch of
    # _shape_cell_fractions -- a fraction built with nonzero smoothing is a different geometry
    # (wider ring, trapezoidal windows) and pairing the two mixes two structures. This sums each
    # image's boundary, not the boundary of their union, so touching images double a shared face
    # that a sign-blind moment cannot cancel, and a wider shape outruns the ring and loses
    # arclength; one refusal excludes both, exactly rather than conservatively, since images a
    # lattice vector apart overlap along a face only when a projected extent spans a period. Both
    # are rectangle arguments -- images at (m*Lx, n*Ly) -- which is why the box is asked for above,
    # and why the ring comes from the same helper as the fill fraction's.
    for axis, period in enumerate(box):
        check(xp.max(vertices[:, axis])-xp.min(vertices[:, axis]) < period,
              'Interface moments require projected shape extents strictly smaller than their periods; '
              'a touching or oversized periodic image shares a face with its neighbour, which this sum '
              'counts twice rather than cancels.')
    shifts = _image_shifts(periods, lattice, array=array)
    start = vertices[None, :, :]+shifts[:, None, :]
    edge = xp.roll(start, -1, axis=-2)-start
    edge_x, edge_y = edge[..., 0], edge[..., 1]
    # Guarded normalisation in the idiom of _clipped_rectangle_area's inactive denominators: a
    # zero-length edge leaves a zero normal and a zero length, so it contributes nothing and
    # differentiates without the infinite slope of sqrt at the origin. Normalising the whole edge
    # once rather than each clipped piece needs one such guard instead of one per cell and 36
    # square roots rather than 9*4*len(y)*len(x), for the same direction either way.
    squared = edge_x*edge_x+edge_y*edge_y
    extended = squared > 0
    safe = xp.where(extended, squared, xp.ones_like(squared))
    length = xp.where(extended, xp.sqrt(safe), xp.zeros_like(squared))
    inverse = 1/xp.sqrt(safe)
    normal_x, normal_y = -edge_y*inverse, edge_x*inverse
    # Cell boxes carry x on the last axis and y on the second to last, so that the
    # accumulated result is (len(y), len(x)) without a transpose anywhere.
    left_x = array(x)-dx/2
    bottom_y = array(y)-dy/2
    left_x, right_x = left_x[None, None, None, :], (left_x+dx)[None, None, None, :]
    bottom_y, top_y = bottom_y[None, None, :, None], (bottom_y+dy)[None, None, :, None]
    start_x, start_y = start[..., 0][..., None, None], start[..., 1][..., None, None]
    edge_x, edge_y = edge_x[..., None, None], edge_y[..., None, None]
    # Liang-Barsky parametric clipping (Liang and Barsky, ACM TOG 3, 1 (1984), 1-22):
    # each half plane is one inequality slope*t <= offset on the edge's own parameter, so
    # the four faces only ever move the entry/exit limits and never rebuild a polygon.
    enter = xp.zeros_like(start_x+left_x+bottom_y)
    leave = xp.ones_like(enter)
    outside = xp.zeros_like(enter, dtype=xp.bool)
    for slope, offset in ((-edge_x, start_x-left_x), (edge_x, right_x-start_x),
                          (-edge_y, start_y-bottom_y), (edge_y, top_y-start_y)):
        moving = xp.abs(slope) > tol
        t = offset/xp.where(moving, slope, xp.ones_like(slope))
        enter = xp.where(moving & (slope < 0), xp.maximum(enter, t), enter)
        leave = xp.where(moving & (slope > 0), xp.minimum(leave, t), leave)
        # A face the edge runs parallel to admits the whole segment or none of it. The closed box
        # keeps an exactly coincident edge (offset == 0) in both neighbours: harmless where that
        # face is a cell's only interface, a genuine step in L where a transverse face also cuts it.
        outside = outside | (~moving & (offset < 0))
    span = xp.where(outside, xp.zeros_like(enter), xp.maximum(leave-enter, xp.zeros_like(enter)))
    # The clipped fraction of the parameter interval times the FULL edge length, and the
    # full edge's normal: nothing here is read off the clipped piece.
    arclength = span*length[..., None, None]
    return (xp.sum(arclength, axis=(0, 1)),
            xp.sum(arclength*(normal_x*normal_x)[..., None, None], axis=(0, 1)),
            xp.sum(arclength*(normal_x*normal_y)[..., None, None], axis=(0, 1)),
            xp.sum(arclength*(normal_y*normal_y)[..., None, None], axis=(0, 1)))


def _periodic_shape_weights(centres, width, edges, period, smoothing, *, xp, array):
    """Integrate a translated periodic interval partition over fixed Yee cells.

    Positive smoothing additionally averages a common translation uniformly
    over its full physical width. Three periodic images suffice for cells no
    wider than period/2 and smoothing <= period. Saturated step fractions are
    exact, avoiding remote cancellation when intervals disappear at a seam.
    """
    left = array(centres)[:, None]-width/2
    right = left+width
    def fraction(face):
        l, r = left-face, right-face
        if not smoothing:
            return xp.clip(r/width, 0., 1.)
        half = smoothing/2
        def primitive(v):
            u = (xp.clip(v, -half, half)+half)/smoothing
            return xp.where(v <= -half, xp.zeros_like(v), xp.where(v >= half, v, .5*smoothing*u*u))
        f = (primitive(r)-primitive(l))/width
        return xp.where(l >= half, xp.ones_like(f), xp.where(r <= -half, xp.zeros_like(f), f))
    weights = sum(fraction(edges[None, :-1]+shift)-fraction(edges[None, 1:]+shift)
                  for shift in (-period, 0., period))
    # A zero-width partition interval can carry a nonzero face derivative.
    # max(0,0) would split/drop that derivative on JAX/Torch at coincident
    # projected edges. Clip only negative roundoff; preserve the zero branch.
    weights = xp.where(weights < 0, xp.zeros_like(weights), weights)
    return weights/xp.sum(weights, axis=1, keepdims=True)


def _smoothed_layer_weights(centres, width, edges, smoothing, *, xp, array, moving_face=None):
    """C1 cell fractions: average internal interface positions over a box.

    The box has full physical width ``smoothing``. Integrating its linear
    step response over each Yee cell gives a piecewise-quadratic primitive.
    Exterior faces stay sharp unless moving_face selects one for averaging.
    Callers keep the supports clear of fixed faces and extraction stencils.
    """
    centres, width = array(centres), array(width)
    width = width[:, None] if width.ndim else width
    left = centres[:, None]-width/2-edges[None, :]
    right = centres[:, None]+width/2-edges[None, :]
    half = smoothing/2
    def primitive(x):
        # Clip the unused polynomial to avoid squaring remote coordinates.
        u = (xp.clip(x, -half, half)+half)/smoothing
        return xp.where(x <= -half, xp.zeros_like(x),
                        xp.where(x >= half, x, (.5*smoothing)*u*u))
    smooth = (primitive(right)-primitive(left))/width
    # Exact saturated fractions avoid cancellation and spurious far-away fills.
    smooth = xp.where(left >= half, xp.ones_like(smooth),
                      xp.where(right <= -half, xp.zeros_like(smooth), smooth))
    sharp = xp.clip(right/width, 0., 1.)
    internal = np.zeros(len(edges), bool)
    internal[2:-2] = True  # outside extent, top face, internal faces, bottom, extent
    if moving_face is not None:
        internal[1 if moving_face == 'top' else -2] = True
    fraction = xp.where(array(internal)[None, :], smooth, sharp)
    weights = xp.maximum(array(0.), fraction[:, :-1]-fraction[:, 1:])
    return weights/xp.sum(weights, axis=1, keepdims=True)


def _layer_weights(centres, width, edges, *, xp, array, smoothing=0., moving_face=None):
    """Exact cell fractions for moving planar interfaces on a fixed z mesh.

    At zero smoothing the one-sided derivatives can differ at cell faces.
    Positive smoothing uses C1 fractions. Neither uses a straight-through rule.
    """
    if smoothing:
        return _smoothed_layer_weights(centres, width, edges, smoothing, xp=xp, array=array, moving_face=moving_face)
    centres, width = array(centres), array(width)
    width = width[:, None] if width.ndim else width
    left, right = centres[:, None]-width/2, centres[:, None]+width/2
    weights = xp.maximum(array(0.), xp.minimum(right, edges[None, 1:])
                         - xp.maximum(left, edges[None, :-1]))/width
    return weights/xp.sum(weights, axis=1, keepdims=True)


def _static_thickness(value, layers, name):
    message = f'{name} must contain one positive finite static real length per layer.'
    # Never silently detach an eager backend tensor while selecting a mesh.
    if not (isinstance(value, np.ndarray) or
            isinstance(value, (list, tuple)) and all(isinstance(v, (int, float, np.number)) for v in value)):
        raise ValueError(message+' Backend thickness values require explicit mesh_thickness.')
    value = np.asarray(value)
    if (value.shape != (layers,) or value.dtype.kind not in 'fiu'
            or not np.isfinite(value).all() or np.any(value <= 0)):
        raise ValueError(message)
    value = value.astype(float)
    if not np.isfinite(value.sum()):
        raise ValueError(message)
    return value


def _validate_layer_smoothing(value):
    message = 'layer_smoothing must be a nonnegative finite static real scalar.'
    if not isinstance(value, (int, float, np.number, np.ndarray)):
        raise ValueError(message)
    value = np.asarray(value)
    if value.shape != () or value.dtype.kind not in 'fiu' or not np.isfinite(value) or value < 0:
        raise ValueError(message)
    value = float(value)
    if not np.isfinite(value) or value > 0 and value/2 == 0:
        raise ValueError(message)
    return value


def _layer_geometry(thickness, mesh_thickness, layers, *, xp, array, check=_require_scene, smoothing=0.,
                    variable_depth=False, anchor_bottom=False):
    """Separate static mesh selection from live physical boundaries.

    By default pin both outer faces exactly, including when a low-precision
    sum rounds. variable_depth releases the physical exit while retaining the
    nominal mesh; anchor_bottom implements a fixed-coordinate z reflection.
    """
    if mesh_thickness is None:
        if variable_depth:
            raise ValueError('variable_depth requires explicit mesh_thickness.')
        nominal = _static_thickness(thickness, layers, 'thickness')
        return nominal, np.r_[0., np.cumsum(nominal)]
    nominal = _static_thickness(mesh_thickness, layers, 'mesh_thickness')
    actual = array(thickness)
    if actual.shape != (layers,) or not xp.isdtype(actual.dtype, ('real floating', 'integral')):
        raise ValueError('thickness must contain one positive finite real length per layer.')
    actual = xp.astype(actual, xp.result_type(actual, array(1.)))
    check(xp.all(xp.isfinite(actual) & (actual > 0)),
          'thickness must contain one positive finite real length per layer.')
    depth = float(nominal.sum())
    if variable_depth:
        planes = xp.concat((array([0.]), xp.cumulative_sum(actual)))
        if anchor_bottom:
            # Reflection is about the nominal depth, keeping original-coordinate
            # nodes, source and monitor locations independent of actual depth.
            planes = depth-xp.concat((xp.flip(xp.cumulative_sum(xp.flip(actual, axis=0)), axis=0), array([0.])))
    else:
        check(xp.abs(xp.sum(actual)-depth) <= 32*xp.finfo(actual.dtype).eps*depth,
              'thickness must preserve the fixed total depth of mesh_thickness.')
        planes = xp.concat((array([0.]), xp.cumulative_sum(actual)[:-1], array([depth])))
    check(xp.all(xp.isfinite(planes)) & xp.all(planes[1:] > planes[:-1]),
          'thickness boundaries must remain finite, distinct and ordered.')
    if smoothing and (layers > 1 or variable_depth):
        # A bound such as D-h/2 can round to a clearance just below h/2,
        # especially after reflection. Use the same scale as the total check.
        tolerance = 32*xp.finfo(actual.dtype).eps*depth
        check((planes[1]-planes[0] >= smoothing/2-tolerance) & (planes[-1]-planes[-2] >= smoothing/2-tolerance),
              'layer_smoothing requires end-layer lengths at least half its width.')
    return nominal, planes


def _phase_between_planes(nodes, widths, cell_phases, start, stop, *, xp, array):
    """Integrate discrete propagation to live planes on a static nodal mesh.

    A plane inside a cell uses that cell's phase per unit length. At a node
    the right cell defines the one-sided position derivative consistently
    across backends; uniform meshes have no such derivative discontinuity.
    """
    nodes, widths = array(nodes), array(widths)
    # Share represented endpoints. Reconstructing each right edge as node+dz
    # can overlap its neighbor by roundoff and count two position derivatives.
    ends = xp.concat((nodes[1:], nodes[-1:]+widths[-1:]))
    spans = ends-nodes
    def fractions(position):
        offset = position-nodes
        length = xp.where(offset < 0, xp.zeros_like(offset), xp.where(position >= ends, spans, offset))
        return length/spans
    return xp.tensordot(fractions(stop)-fractions(start), cell_phases, axes=((0,), (0,)))


def _validate_record_bounds(bounds):
    if bounds is None:
        return None
    if not isinstance(bounds, dict) or not bounds or set(bounds)-{'x', 'y', 'z'}:
        raise ValueError('record_bounds must map x/y/z to finite (lower, upper) pairs.')
    result = {}
    for axis, pair in bounds.items():
        values = np.asarray(pair)
        if (values.shape != (2,) or values.dtype.kind not in 'fiu'
                or not np.isfinite(values).all() or values[0] > values[1]):
            raise ValueError('record_bounds requires finite real lower <= upper pairs.')
        result[axis] = tuple(float(v) for v in values)
    return result


def _bounded_indices(coordinates, bounds):
    if bounds is None:
        return np.arange(len(coordinates))
    lower, upper = bounds
    tolerance = 8*np.finfo(float).eps*max(1., abs(lower), abs(upper), np.max(np.abs(coordinates)))
    return np.flatnonzero((coordinates >= lower-tolerance) & (coordinates <= upper+tolerance))


def _solve_bottom_scene(options):
    """Reflect z about the stack midplane; E is polar and H is axial.

    Diffraction amplitudes are incidence-relative, as in RCWA, so their s/p
    coefficients do not change. Field coordinates and Cartesian components do.
    """
    runtime = time_loop_backend(options['backend'], device=options['device'],
                                reference=options['structure'])
    xp, array = runtime.xp, runtime.array
    parametric = _parametric_recipe(options['structure'])
    raster = options['structure'] if parametric else array(options['structure'])
    if not parametric and (raster.ndim != 3 or min(raster.shape) < 1):
        raise ValueError('structure must have shape (layers, y, x).')
    geometry_valid = array(True)
    def check_geometry(valid, message):
        nonlocal geometry_valid
        _require_scene(valid, message)
        geometry_valid = geometry_valid & valid
    nominal, layer_planes = _layer_geometry(options['thickness'], options['mesh_thickness'], len(raster),
                                             xp=xp, array=array, check=check_geometry, smoothing=options['layer_smoothing'],
                                             variable_depth=options['variable_depth'])
    depth = float(nominal.sum())
    thickness = xp.flip(array(options['thickness']), axis=0) if options['mesh_thickness'] is not None else nominal[::-1]
    if options['mesh_thickness'] is not None:
        # Preserve original-coordinate invalidity through the reflected solve.
        # Roundoff may collapse a final layer only on one side of the reflection.
        thickness = xp.astype(thickness, xp.result_type(thickness, array(1.)))
        thickness = xp.where(geometry_valid, thickness, xp.full_like(thickness, float('nan')))
    mirrored = dict(options, incidence_side='top', top=options['bot'], bot=options['top'],
                    thickness=thickness, structure=raster[::-1] if parametric else xp.flip(raster, axis=0),
                    _anchor_bottom=options['variable_depth'])
    if options['mesh_thickness'] is not None:
        mirrored['mesh_thickness'] = nominal[::-1]
    if options['mesh_indices'] is not None:
        mirrored['mesh_indices'] = _validate_mesh_indices(options['mesh_indices'])[::-1]
    if options['record_bounds'] is not None:
        mirrored['record_bounds'] = dict(options['record_bounds'])
        if 'z' in mirrored['record_bounds']:
            lower, upper = mirrored['record_bounds']['z']
            mirrored['record_bounds']['z'] = (depth-upper, depth-lower)
    if options['mu'] is not None:
        permeability = array(options['mu'])
        if permeability.shape != raster.shape:
            raise ValueError('mu must have the same raster shape as structure.')
        mirrored['mu'] = xp.flip(permeability, axis=0)
    if options['dispersion'] is not None:
        mirrored['dispersion'] = np.flip(np.asarray(options['dispersion']), axis=0)
    if options['field_monitors'] is not None:
        planes = options['field_monitors']
        if not isinstance(planes, dict) or not planes:
            raise ValueError('field_monitors must be a nonempty mapping of names to z coordinates.')
        for coordinate in planes.values():
            if np.ndim(coordinate) or np.iscomplexobj(coordinate) or not np.isfinite(coordinate):
                raise ValueError('Monitor z must be a finite real scalar.')
        mirrored['field_monitors'] = {name: depth-z for name, z in planes.items()}
    answer = solve_scene_fdtd(**mirrored)
    if not options['return_diagnostics']:
        return answer
    result, diagnostics = answer
    signs = dict(ex=1, ey=1, ez=-1, hx=-1, hy=-1, hz=1)
    for plane in diagnostics['monitors'].values():
        plane['z'] = depth-plane['z']
        plane['fields'] = {name: signs[name]*value for name, value in plane['fields'].items()}
        plane['flux_density'] = plane['flux_density']*array([1., 1., -1.])
        plane['power_z'] = -plane['power_z']
        plane['normalization'] = 'unit incident E at bottom interface'
    for name, field in diagnostics.get('fields', {}).items():
        field['values'] = signs[name]*xp.flip(field['values'], axis=1)
        field['z'] = depth-xp.flip(field['z'], axis=0)
    if 'fields' in diagnostics:
        first = diagnostics['fields'][diagnostics['field_e_component']]
        diagnostics.update(field_e=first['values'], field_z=first['z'])
    diagnostics['z_nodes'] = depth-xp.flip(diagnostics['z_nodes'], axis=0)
    # Reversed nodal grid: closing cell remains last; reverse the interior widths.
    diagnostics['z_spacing'] = xp.concat((xp.flip(diagnostics['z_spacing'][:-1], axis=0), diagnostics['z_spacing'][-1:]))
    diagnostics.update(source_z=depth-diagnostics['source_z'],
                       pml_bounds=tuple(depth-z for z in reversed(diagnostics['pml_bounds'])),
                       layer_planes=layer_planes,
                       incidence_side='bottom', reflection_z=layer_planes[-1], transmission_z=0.)
    for key in ('reflection_monitor_z', 'transmission_monitor_z'):
        diagnostics['diffraction'][key] = depth-diagnostics['diffraction'][key]
    diagnostics['mesh_indices'] = diagnostics['mesh_indices'][::-1]
    diagnostics['mesh_thickness'] = diagnostics['mesh_thickness'][::-1]
    diagnostics['background'] = xp.flip(diagnostics['background'], axis=0)
    return result, diagnostics


def _validate_mesh_indices(value):
    """Nominal top/bottom indices select geometry, never material coefficients."""
    message = 'mesh_indices must be two positive finite static refractive indices (top, bottom).'
    try:
        value = np.asarray(value)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(message) from error
    if (value.shape != (2,) or value.dtype.kind not in 'fiu'
            or not np.isfinite(value).all() or np.any(value <= 0)):
        raise ValueError(message)
    result = tuple(float(v) for v in value)
    if not np.isfinite(result).all():
        raise ValueError(message)
    return result


def _uniform_reference(epsilon, eps_top, mu_top, background, settings, kx0, ky0, dx, dy, *,
                       xp, array):
    """The uniform-medium reference run, stepped on a two-cell transverse column.

    The reference medium is uniform, so what this run produces is the lattice plane wave: every
    transverse cell is the first one times ``exp(i(kx0 x + ky0 y))``. The column keeps ``dz``,
    ``dy`` and ``dx``, so the discrete transverse operator is the one the full grid steps -- only
    the cell *count* falls -- and the plane is continued from it by that phase, which is why the
    column's Bloch phase is the full one scaled to two cells. ``kx0`` and ``ky0`` are live where
    ``theta`` and ``phi`` are, so the continuation is built with ``xp`` and carries their
    derivative.

    Measured against the full-grid reference on 2026-09-16, worst relative difference over the
    monitor spectra: 0 at normal incidence, on a graded z mesh and from below; 3.9e-15 oblique
    TE and TM; 6.4e-15 on a conical crossed grating; 4.4e-15 on a Drude raster and on jax.

    This is a change of answer, not a rewrite: of 581 amplitudes over a battery of scenes, 439
    came back bit-identical and the rest moved by at most 3.7e-11 relative -- on the
    cross-polarised channels of a conical crossed grating, where the reference is subtracted
    from a nearly equal actual run and the cancellation lifts a 1e-15 input to that. A jax
    gradient and its value were unchanged to twelve digits. The saving is the reference's share
    of the solve: measured 3.90 s -> 3.20 s on a slab, 8.19 s -> 6.10 s on a grating, and
    11.52 s -> 6.03 s on a crossed grating, where the transverse grid is largest.

    On jax it is a trade, because the column is a second shape to compile: a first solve went
    7.69 s -> 8.71 s and a new size 2.97 s -> 3.72 s, while the same size again went
    0.84 s -> 0.58 s. A sweep of one-shot sizes pays; an optimization loop, which is what the
    differentiable route is for, is paid back from the second iteration on.

    The run's own dict comes back with the spectra continued and the arrays that stay two cells
    wide removed, so a later reader who wants one gets a ``KeyError`` rather than an array that
    is silently the wrong width. Everything else passes through untouched, which is what the
    instruments that stub this call out rely on (``studies/fdtd_lattice_reference.py``).
    """
    ny, nx = np.shape(epsilon)[-2:]
    column = ny > 2 or nx > 2
    if column:
        wide = settings['bloch_phase']
        settings = dict(settings, source=settings['source'][..., :2, :2],
                        bloch_phase=xp.stack((wide[0], wide[1]*2/ny, wide[2]*2/nx)))
        epsilon = epsilon[..., :2, :2]
    run = fdtd_grid(xp.ones_like(epsilon)*eps_top, mu=mu_top,
                    pml_background=(background, background), **settings)

    def spread(value):
        if not column:
            return value
        ys = array(np.arange(ny, dtype=float))[:, None]
        xs = array(np.arange(nx, dtype=float))[None, :]
        return value[..., :1, :1]*xp.exp(1j*(ky0*dy*ys + kx0*dx*xs))

    out = dict(run)
    for key in ('electric_spectrum', 'magnetic_spectrum'):
        if key in out:
            out[key] = spread(out[key])
    if isinstance(out.get('convergence'), dict):
        convergence = dict(out['convergence'])
        for key in ('electric_spectrum_before_window', 'magnetic_spectrum_before_window'):
            if key in convergence:
                convergence[key] = spread(convergence[key])
        out['convergence'] = convergence
    if column:
        # Fields and flux stay the column's own; nothing here reads them, and a silent
        # two-cell plane is worse than a KeyError for whoever reads them next.
        for key in ('E', 'H', 'fields', 'flux_density', 'h_time'):
            out.pop(key, None)
    return out


def solve_scene_fdtd(structure, thickness, top, bot, period=None, wavelength=None, *,
                     backend='numpy', device=None, resolution=None, n_pml=12, n_periods=40,
                     courant=.8, subpixel=True, theta=0., phi=None, tetm=None,
                     psi=None, chi=None, polarization=None, mu=None,
                     truncation_order=None, dispersion=None, poles=None,
                     pad_scale=1., incidence_side='top', lattice=None,
                     checkpoint_steps=32, return_diagnostics=False,
                     field_monitors=None, record_field=None, record_components=None, grading=None,
                     convergence_window=None, convergence_tol=1e-4, dft_subsample=1, record_bounds=None,
                     pulse_width=None, align_layers=False, diffraction_distance=None,
                     diffraction_method='electric', pml_reflection=1e-8, pulse_notches=None,
                     source_distance=None, transverse_resolution=None, mesh_indices=None,
                     mesh_thickness=None, layer_smoothing=0., shape_smoothing=0., shape_integration='separable',
                     variable_depth=False, _anchor_bottom=False, _public_order_frame=False):
    """Solve a rectangular scene with fixed-mesh layer/angle/material gradients.

    Explicit resolution is required: optimizing materials must not resize the
    grid. Wavelength and periods are static. Axis-aligned rectangle/square
    recipes use the shared RCWA shape convention and retain centre, size and
    scalar Material epsilon/mu gradients. Later shapes paint over earlier
    shapes, including periodic wrapping. They require two rectangular periods,
    subpixel=True and sizes in (0,period]; raster pole labels are unsupported
    on recipes. shape_smoothing is a static scalar or
    (x,y) pair of full physical averaging widths in [0,period]. Positive widths
    average transverse translations before sampling; moving contacts between
    shapes can still have topology kinks. The Yee mesh never follows a shape.
    shape_integration='area' instead accepts multiple rectangles or one circle/ellipse
    per layer. Later rectangles overwrite earlier ones, including periodic overlaps;
    live boundary events permit changing overlap topology. Multiple/mixed curved
    shapes are refused. Rectangle/ellipse angles retain derivatives in radians; ellipse lx/ly
    are full principal-axis lengths, and ellipse cells use fixed 16-point angular
    integration. For circle/disk,
    lx (or l) is the diameter, with no ly. Projected extents must fit the periods.
    Positive shape_smoothing averages translations using exact polygon
    moments or stable disk/ellipse arc strips with fixed 16-point Gaussian integration.
    No user sampling grid or quadrature order is introduced. Cell integrals give arithmetic transverse
    material averages. This is not a full rotated effective tensor and does not promise second-order interface accuracy.
    Area integration always retains the full y-period mesh, including angle=0.
    Half-space epsilon/mu can be live
    positive real scalars with explicit mesh_indices and truncation_order.
    mesh_indices=(n_top,n_bottom) fixes padding/grading using nominal static
    refractive indices; actual materials still determine all physics and CPML.
    Without it, host scalar half spaces retain automatic mesh selection.
    mesh_thickness fixes nominal layer lengths for the z mesh while actual
    thickness values move internal interfaces. All lengths must remain positive
    and their sum must equal the nominal total. Backend thickness values require
    mesh_thickness and subpixel=True. Gradients are defined along fixed-total
    paths and are piecewise smooth, with possible kinks at cell-face crossings.
    align_layers fits nominal boundaries only; outer reference planes stay fixed.
    Opt-in variable_depth=True also moves the physical exit and retains its
    de-embedding phase derivative. It requires mesh_thickness and subpixel=True.
    The laboratory-coordinate mesh, current sheets, extraction planes and
    records stay fixed for both incidence sides. All interfaces and averaging
    supports must remain clear of extraction stencils. Positive layer_smoothing
    also averages the moving exterior face. On nonuniform meshes, fractional
    reference positions use the local cell's discrete phase; its slope can
    change at nodes even with material smoothing.
    Positive static layer_smoothing averages internal interface positions over
    a box of that full physical width, giving continuous first derivatives.
    It requires mesh_thickness and subpixel=True; internal supports must stay
    inside the fixed outer faces. Zero preserves exact cell-overlap fractions.
    Refining the grid at a fixed positive width retains a regularization bias.
    theta and phi can
    be backend real scalars; either array angle requires truncation_order to
    fix the diffraction window. Array phi always samples the full y period,
    including at zero, and requires Jones/psi polarization rather than tetm.
    Angle derivatives hold the mesh fixed and change the one Bloch wavevector
    shared by the whole wavelength band.
    Bottom incidence uses a z-reflected scene and returns fields in original
    coordinates. Optional grading builds a static nonuniform z mesh; material
    gradients do not resize it. pulse_width is the static Gaussian envelope's
    1/e amplitude half-width in c=1 time units; None uses the central wavelength.
    align_layers places every layer boundary on a z node using equal cells per
    layer, each no wider than 1/resolution. It can be combined with grading.
    diffraction_distance optionally places the extraction planes near their
    reference interfaces without resizing the padding, source or PML mesh.
    diffraction_method='directional' separates E/H into travelling directions
    using the discrete admittance; both planes need locally uniform z cells.
    pml_reflection sets a static nominal normal-incidence round-trip amplitude
    target in (0,1) on both ends; smaller means stronger absorption. It is not
    a finite-grid error bound and does not change the PML thickness.
    pulse_notches optionally suppresses selected positive frequencies using
    causal delayed copies of the Gaussian. Required frequencies must remain
    resolved, including the carrier when time records are requested.
    source_distance optionally selects a source sheet a fixed physical distance
    from the incident interface, independently of padding and monitor positions.
    transverse_resolution optionally sets static x/y cells per unit length,
    as a scalar or (x,y) pair. The z mesh continues to use resolution.
    """
    layer_smoothing = _validate_layer_smoothing(layer_smoothing)
    if not isinstance(variable_depth, (bool, np.bool_)):
        raise ValueError('variable_depth must be a static boolean.')
    if variable_depth and (mesh_thickness is None or not subpixel):
        raise ValueError('variable_depth requires explicit mesh_thickness and subpixel=True.')
    parametric = _parametric_recipe(structure)
    if shape_integration not in ('separable', 'area'):
        raise ValueError('shape_integration must be separable or area.')
    if shape_integration == 'area' and not parametric:
        raise ValueError('Area integration requires parametric rectangle layers.')
    shape_width = np.asarray(shape_smoothing)
    if (shape_width.shape not in ((), (2,)) or shape_width.dtype.kind not in 'fiu'
            or not np.isfinite(shape_width).all() or np.any(shape_width < 0)):
        raise ValueError('shape_smoothing must be a nonnegative finite static scalar or (x,y) pair.')
    if np.any(shape_width) and not parametric:
        raise ValueError('shape_smoothing requires parametric rectangle layers.')
    if parametric and (not subpixel or mu is not None or dispersion is not None or poles is not None):
        raise ValueError('Parametric FDTD requires subpixel=True and Material epsilon/mu; raster mu and dispersion labels are unsupported.')
    if layer_smoothing and (mesh_thickness is None or not subpixel):
        raise ValueError('Positive layer_smoothing requires explicit mesh_thickness and subpixel=True.')
    record_bounds = _validate_record_bounds(record_bounds)
    if record_bounds is not None and record_field is None:
        raise ValueError('record_bounds requires record_field.')
    if incidence_side == 'bottom':
        return _solve_bottom_scene({k: v for k, v in locals().items() if k not in ('parametric', 'shape_width')})
    if (field_monitors is not None or record_field is not None) and not return_diagnostics:
        raise ValueError('field_monitors/record_field require return_diagnostics=True.')
    if convergence_window is not None and not return_diagnostics:
        raise ValueError('convergence_window requires return_diagnostics=True.')
    if field_monitors is not None and (not isinstance(field_monitors, dict) or not field_monitors):
        raise ValueError('field_monitors must be a nonempty mapping of names to z coordinates.')
    if record_components is not None and record_field is None:
        raise ValueError('record_components requires record_field.')
    if incidence_side != 'top':
        raise ValueError("incidence_side must be 'top' or 'bottom'.")
    if lattice is not None:
        raise ValueError('The common scene engine requires rectangular period.')
    if not isinstance(align_layers, (bool, np.bool_)):
        raise ValueError('align_layers must be a static boolean.')
    if diffraction_method not in ('electric', 'directional'):
        raise ValueError("diffraction_method must be 'electric' or 'directional'.")
    pml_reflection = _validate_pml_reflection(pml_reflection)
    if source_distance is not None:
        value = np.asarray(source_distance)
        if (value.shape != () or value.dtype.kind not in 'fiu'
                or not np.isfinite(value) or value <= 0):
            raise ValueError('source_distance must be a positive finite static scalar.')
        source_distance = float(value)
        if not np.isfinite(source_distance):
            raise ValueError('source_distance must be representable as a finite static scalar.')
    if diffraction_distance is not None:
        distances = np.asarray(diffraction_distance)
        if (distances.shape not in ((), (2,)) or distances.dtype.kind not in 'fiu'
                or not np.isfinite(distances).all() or np.any(distances <= 0)):
            raise ValueError('diffraction_distance must be a positive finite scalar or (reflection, transmission) pair.')
        distances = np.broadcast_to(distances, (2,))
    if grading is True:
        grading = 1.06
    elif grading is False:
        grading = None
    if grading is not None and (not np.isfinite(grading) or not 1 < grading < 2):
        raise ValueError('grading must be True or a static growth ratio between 1 and 2.')
    if resolution is None or not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('The common scene engine requires a positive explicit resolution (cells/length).')
    transverse = np.asarray(resolution if transverse_resolution is None else transverse_resolution)
    if (transverse.shape not in ((), (2,)) or transverse.dtype.kind not in 'fiu'
            or not np.isfinite(transverse).all() or np.any(transverse <= 0)):
        raise ValueError('transverse_resolution must be a positive finite static scalar or (x,y) pair.')
    rx, ry = np.broadcast_to(transverse, (2,)).astype(float)
    if not np.isfinite([rx, ry]).all():
        raise ValueError('transverse_resolution must be representable as finite static values.')
    if not 0 < courant < 1 or not np.isfinite(n_periods) or n_periods <= 0:
        raise ValueError('Require 0 < courant < 1 and positive n_periods.')
    if int(n_pml) != n_pml or n_pml < 2 or not np.isfinite(pad_scale) or pad_scale <= 0:
        raise ValueError('Require integer n_pml >= 2 and positive pad_scale.')
    conflicts = incidence_clashes(tetm, psi, chi, polarization)
    if conflicts:
        raise ValueError('; '.join(conflicts))
    if polarization is not None and len(polarization) != 2:
        raise ValueError('polarization must contain exactly two scalar amplitudes.')
    if tetm not in (None, 'te', 'tm'):
        raise ValueError("tetm must be 'te' or 'tm'.")
    phi = 0. if phi is None else phi
    scalar_phi = isinstance(phi, (int, float, np.number))
    in_plane = scalar_phi and phi == 0
    # Inferred order counts may depend on a host scalar, never on a live array.
    if truncation_order is None and (not scalar_phi or not isinstance(theta, (int, float, np.number))):
        raise ValueError('Array theta or phi requires explicit truncation_order to keep the diffraction window fixed.')
    host_theta, host_phi = theta, phi
    if tetm is not None and not in_plane:
        raise ValueError('Use psi or polarization for conical incidence or array phi instead of tetm.')
    band = np.atleast_1d(np.asarray(wavelength, float))
    if band.ndim != 1 or not len(band) or not np.isfinite(band).all() or np.any(band <= 0):
        raise ValueError('wavelength must be a positive scalar or vector.')
    centre = float((band.min()+band.max())/2)
    if pulse_width is None:
        width = centre
    else:
        value = np.asarray(pulse_width)
        if value.shape != () or value.dtype.kind not in 'fiu' or not np.isfinite(value) or value <= 0:
            raise ValueError('pulse_width must be a positive finite static scalar in c=1 time units.')
        width = float(value)
    span = float(np.max(np.abs(1/band-1/centre)))
    runtime = time_loop_backend(backend, device=device, reference=structure)
    xp, array = runtime.xp, runtime.array
    theta = array(theta)
    if theta.shape != () or not xp.isdtype(theta.dtype, ('real floating', 'integral')):
        raise ValueError('theta must be a real scalar, 0 <= theta < pi/2.')
    theta = xp.astype(theta, xp.result_type(theta, array(1.)))
    phi = array(phi)
    if phi.shape != () or not xp.isdtype(phi.dtype, ('real floating', 'integral')):
        raise ValueError('phi must be a finite real scalar.')
    phi = xp.astype(phi, xp.result_type(phi, array(1.)))
    scene_valid = array(True)
    def check(valid, message):
        nonlocal scene_valid
        _require_scene(valid, message)
        scene_valid = scene_valid & valid
    check(xp.isfinite(theta) & (theta >= 0) & (theta < np.pi/2),
          'Require finite incidence angles, 0 <= theta < pi/2.')
    check(xp.isfinite(phi), 'phi must be a finite real scalar.')
    cos_phi, sin_phi = xp.cos(phi), xp.sin(phi)
    # A requested frequency in the Gaussian's numerical tail cannot provide
    # reliable incident calibration. This is a roundoff guard, not an accuracy
    # guarantee for weakly excited frequencies.
    source_epsilon = xp.finfo(xp.real(array(np.array(0j, dtype=np.complex128))).dtype).eps
    if span and width > np.sqrt(-np.log(100*source_epsilon))/(np.pi*span):
        raise ValueError('pulse_width leaves a requested wavelength in the numerical source tail; reduce pulse_width or narrow the wavelength band.')
    transverse_edges, area_layers = None, None
    if parametric:
        periods = np.atleast_1d(np.asarray(parse_period_mapping(period), float))
        if periods.shape != (2,) or not np.isfinite(periods).all() or np.any(periods <= 0):
            raise ValueError('Parametric FDTD requires two positive static rectangular periods.')
        shape_width = np.broadcast_to(shape_width, (2,)).astype(float)
        if np.any(shape_width > periods):
            raise ValueError('shape_smoothing must not exceed either rectangular period.')
        # The cell the recipe is drawn on. `lattice` is None here -- the door above refused any other --
        # so this is diag(periods), stated rather than defaulted: a stated None is a decision.
        from .lattice import as_lattice
        cell = as_lattice(period=periods, lattice=lattice)
        raster, mu, partition = _shape_partition(structure, periods, xp=xp, array=array, check=check,
                                                     integration=shape_integration, lattice=cell)
        if shape_integration == 'area':
            area_layers = partition
        else:
            transverse_edges = partition
    else:
        raster = array(structure)
    if raster.ndim != 3 or min(raster.shape) < 1:
        raise ValueError('structure must have shape (layers, y, x).')
    def real_material(value):
        value = array(value)
        real = xp.real(value) if xp.isdtype(value.dtype, 'complex floating') else value
        imaginary_ok = xp.all(xp.imag(value) == 0) if xp.isdtype(value.dtype, 'complex floating') else True
        if _concrete(xp.all(xp.isfinite(value)) & imaginary_ok & xp.all(real > 0)) is False:
            raise ValueError('Use positive real epsilon-infinity/mu; loss is specified by electric poles.')
        return xp.astype(real, xp.result_type(real, array(1.)))
    raster = real_material(raster)
    muraster = xp.ones_like(raster) if mu is None else real_material(mu)
    if muraster.shape != raster.shape:
        raise ValueError('mu must have the same raster shape as structure.')
    background = []
    host_indices = []
    array_background = False
    for value in (top, bot):
        material = as_material(value)
        raw = (material.epsilon, 1. if material.mu is None else material.mu)
        host = all(isinstance(v, (int, float, complex, np.number)) for v in raw)
        array_background |= not host
        pair = []
        for v in raw:
            v = array(v)
            if v.shape != () or not xp.isdtype(v.dtype, ('real floating', 'complex floating', 'integral')):
                raise ValueError('Half spaces must have positive real scalar epsilon and mu.')
            real = xp.real(v) if xp.isdtype(v.dtype, 'complex floating') else v
            imaginary_ok = xp.imag(v) == 0 if xp.isdtype(v.dtype, 'complex floating') else True
            check(xp.isfinite(v) & imaginary_ok & (real > 0),
                  'Half spaces must have positive real scalar epsilon and mu.')
            pair.append(xp.astype(real, xp.result_type(real, array(1.))))
        if host:
            host_indices.append(np.sqrt(complex(raw[0]).real*complex(raw[1]).real))
        background.append(tuple(pair))
    if array_background and (mesh_indices is None or truncation_order is None):
        raise ValueError('Array half spaces require explicit mesh_indices and explicit truncation_order to keep the grid fixed.')
    mesh_indices = _validate_mesh_indices(host_indices if mesh_indices is None else mesh_indices)
    et, mt = background[0]
    eb, mb = background[1]
    nt, nb = xp.sqrt(et*mt), xp.sqrt(eb*mb)
    if mesh_thickness is not None and not subpixel:
        raise ValueError('mesh_thickness requires subpixel=True for moving layer interfaces.')
    thick, layer_planes = _layer_geometry(thickness, mesh_thickness, raster.shape[0], xp=xp, array=array,
                                          check=check, smoothing=layer_smoothing, variable_depth=variable_depth,
                                          anchor_bottom=_anchor_bottom)
    lengths = np.atleast_1d(np.asarray(parse_period_mapping(period), float))
    if lengths.ndim != 1 or len(lengths) not in (1, 2) or not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError('period must contain one or two positive lengths.')
    if raster.shape[1] > 1 and len(lengths) != 2:
        raise ValueError('A crossed raster requires both x and y periods.')
    lx = float(lengths[0])
    dz = 1/float(resolution)
    ly = float(lengths[1]) if len(lengths) == 2 else 2/ry
    nx = max(2, math.ceil(lx*rx))
    # Array azimuths must not collapse/expand the mesh while crossing zero.
    # Scalar zero retains the established two-cell y-invariant shortcut.
    ny = max(2, math.ceil(ly*ry)) if raster.shape[1] > 1 or not in_plane else 2
    dx, dy = lx/nx, ly/ny
    pad = max(6, math.ceil(pad_scale*centre/min(mesh_indices)/dz))
    source_z, reflect_z = int(n_pml)+2, int(n_pml)+2+pad
    interface = reflect_z+pad
    transmit_z = interface+math.ceil(thick.sum()/dz)+pad
    nz = transmit_z+pad+int(n_pml)+2
    z = (np.arange(nz)-interface)*dz
    widths = np.full(nz, dz)
    bottom_interface = interface+math.ceil(thick.sum()/dz)
    nonuniform = grading is not None or align_layers
    if nonuniform:
        if align_layers:
            counts = np.ceil(thick*resolution).astype(int)
            core = np.concatenate([np.full(count, t/count) for t, count in zip(thick, counts)])
        else:
            core = np.full(bottom_interface-interface, thick.sum()/(bottom_interface-interface))
        if grading is not None:
            from .fdtd import graded_cells
            coarse = lambda fine, index: max(fine, min(2*fine, band.min()/(20*index)))
            up = graded_cells(core[0], coarse(core[0], mesh_indices[0]), interface*dz, grading)[::-1]
            down = graded_cells(core[-1], coarse(core[-1], mesh_indices[1]), (nz-bottom_interface)*dz, grading)
        else:
            up = np.full(interface, dz)
            down = np.full(nz-bottom_interface, dz)
        widths = np.r_[up, core, down]
        interface, bottom_interface = len(up), len(up)+len(core)
        nz = len(widths)
        z = np.r_[0., np.cumsum(widths[:-1])]-up.sum()
        if align_layers:
            # Keep represented faces exact even after long padding accumulation.
            # Point sampling must not pick a different side due to cancellation.
            z[interface+np.r_[0, np.cumsum(counts)]] = np.r_[0., np.cumsum(thick)]
        source_z = int(n_pml)+2
        reflect_z = int(np.argmin(abs(z+pad*dz)))
        transmit_z = int(np.argmin(abs(z-(thick.sum()+pad*dz))))
        if not source_z+2 < reflect_z < interface or transmit_z >= nz-int(n_pml)-2:
            raise ValueError('Nonuniform padding cannot fit source/monitors/PML; increase pad_scale.')
    if source_distance is not None:
        source_z = int(np.argmin(abs(z+source_distance)))
        if source_z < int(n_pml)+2 or source_z >= interface:
            raise ValueError('source_distance must place the source in the incident half space outside the PML; increase padding or adjust distance.')
    if diffraction_distance is not None:
        reflect_z = int(np.argmin(abs(z+distances[0])))
        transmit_z = int(np.argmin(abs(z-(thick.sum()+distances[1]))))
    if source_distance is not None and not source_z+2 < reflect_z:
        raise ValueError('source_distance must leave at least three cells before the reflection plane; adjust source or diffraction_distance.')
    if diffraction_distance is not None:
        if (not source_z+2 < reflect_z or z[reflect_z]+widths[reflect_z]/2 > 0
                or z[transmit_z]-widths[transmit_z-1]/2 < thick.sum()
                or transmit_z >= nz-int(n_pml)-2):
            raise ValueError('diffraction_distance must select exterior planes between the source/stack and stack/PML; adjust distance, resolution or pad_scale.')
    if diffraction_method == 'directional':
        indices = np.array([reflect_z, transmit_z])
        if not np.allclose(widths[indices], widths[indices-1], rtol=1e-12, atol=0):
            raise ValueError('Directional diffraction requires locally uniform z cells at both planes; adjust diffraction_distance or grading.')
    if variable_depth:
        # Both extraction stencils must remain entirely in their half spaces,
        # including the support of an averaged moving exterior interface.
        check((layer_planes[0]-(layer_smoothing/2 if _anchor_bottom else 0.) >= z[reflect_z]+max(widths[reflect_z-1:reflect_z+1]))
              & (layer_planes[-1]+(0. if _anchor_bottom else layer_smoothing/2) <= z[transmit_z]-max(widths[transmit_z-1:transmit_z+1])),
              'variable_depth must keep layer boundaries and smoothing supports clear of the fixed diffraction planes; increase padding or adjust nominal thickness.')
    dual = .5*(widths+np.roll(widths, 1))
    cuts = (np.r_[z[0]-2*widths.max(), layer_planes, z[-1]+2*widths.max()] if mesh_thickness is None else
            xp.concat((array([z[0]-2*widths.max()]), layer_planes, array([z[-1]+2*widths.max()]))))
    shape = (nz, ny, nx)
    area_fractions = {}
    if area_layers is not None:
        for ox, oy in ((0., 0.), (.5, 0.), (0., .5), (.5, .5)):
            area_fractions[ox, oy] = [None if s is None else
                _painted_cell_materials(s['edges'], base, (np.arange(nx)+ox)*dx, (np.arange(ny)+oy)*dy,
                                       dx, dy, lengths, xp=xp, array=array, smoothing=shape_width,
                                       lattice=cell) if isinstance(s, dict) else
                _shape_cell_fractions(s[0], (np.arange(nx)+ox)*dx, (np.arange(ny)+oy)*dy,
                                     dx, dy, lengths, xp=xp, array=array, smoothing=shape_width,
                                     lattice=cell) for base, s in area_layers]
    def sample(values, upper, lower, magnetic=False, harmonic=True):
        components = []
        for c in range(3):
            offsets = np.ones(3)*.5 if magnetic else np.zeros(3)
            offsets[c] = 0. if magnetic else .5  # physical x,y,z offsets
            v = values
            transverse_axes = [
                (2, (np.arange(nx)+offsets[0])*dx, dx,
                 np.linspace(0, lx, raster.shape[2]+1) if transverse_edges is None else transverse_edges[0], True),
                (1, (np.arange(ny)+offsets[1])*dy, dy,
                 np.linspace(0, ly, raster.shape[1]+1) if transverse_edges is None else transverse_edges[1], True)]
            # Normal harmonic averaging precedes tangential arithmetic averaging
            # in the transverse plane. A fixed x-then-y order treated Ex and Ey
            # differently at crossed raster corners and broke x/y covariance.
            if harmonic and c == 1:
                transverse_axes.reverse()
            for axis, centres, width, edges, periodic in (() if area_layers is not None else transverse_axes):
                weights = array(_weights(centres, width, edges, periodic, subpixel), dtype=v.dtype) if transverse_edges is None else \
                    _periodic_shape_weights(centres, width, edges, lengths[2-axis], shape_width[2-axis], xp=xp, array=array)
                use_harmonic = harmonic and c == 2-axis
                v = 1/v if use_harmonic else v
                v = xp.moveaxis(xp.tensordot(weights, v, axes=((1,), (axis,))), 0, axis)
                v = 1/v if use_harmonic else v
            if area_layers is not None:
                k = int(magnetic)
                fractions = area_fractions[offsets[0], offsets[1]]
                v = xp.stack([xp.ones((ny, nx), dtype=base[k].dtype, device=getattr(base[k], 'device', None))*base[k]
                              if item is None else fraction[k] if isinstance(item, dict) else base[k]+fraction*(item[1][k]-base[k])
                              for (base, item), fraction in zip(area_layers, fractions)])
            v = xp.concat((xp.ones_like(v[:1])*upper, v, xp.ones_like(v[:1])*lower), axis=0)
            if not nonuniform:
                centres, cell_widths = z+offsets[2]*dz, dz
            elif offsets[2]:
                centres, cell_widths = z+widths/2, widths
            else:
                centres = z+(widths-np.roll(widths, 1))/4
                cell_widths = dual
            if not subpixel and nonuniform:
                centres = z+offsets[2]*widths
            weights = array(_weights(centres, cell_widths, cuts, subpixel=subpixel) if mesh_thickness is None else
                            _layer_weights(centres, cell_widths, cuts, xp=xp, array=array,
                                            smoothing=layer_smoothing,
                                            moving_face=('top' if _anchor_bottom else 'bottom') if variable_depth else None), dtype=v.dtype)
            v = 1/(xp.tensordot(weights, 1/v, axes=((1,), (0,)))) if harmonic and c == 2 else xp.tensordot(weights, v, axes=((1,), (0,)))
            # Exact half-space values also make CPML material validation exact.
            if variable_depth:
                guard = widths.max()+layer_smoothing/2
                v = xp.where((array(z) < layer_planes[0]-guard)[:, None, None], upper, v)
                v = xp.where((array(z) > layer_planes[-1]+guard)[:, None, None], lower, v)
            else:
                v = xp.where(array((z < -dz)[:, None, None]), upper, v)
                v = xp.where(array((z > thick.sum()+dz)[:, None, None]), lower, v)
            components.append(v)
        return xp.stack(components)
    epsilon = sample(raster, et, eb)
    permeability = sample(muraster, mt, mb, magnetic=True)
    native_poles = []
    if (dispersion is None) != (poles is None):
        raise ValueError('dispersion labels and poles must be provided together.')
    if dispersion is not None:
        labels = np.asarray(dispersion)
        if labels.shape != raster.shape:
            raise ValueError('dispersion labels must match structure.')
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError('dispersion labels must be integers; zero is non-dispersive.')
        if set(np.unique(labels))-{0}-set(poles):
            raise ValueError('Every nonzero dispersion label must name a pole model.')
        if 0 in poles:
            raise ValueError('Dispersion label zero is reserved for non-dispersive cells.')
        for label, terms in poles.items():
            fill = sample(array((labels == label).astype(float)), 0., 0., harmonic=False)
            for omega, gamma, strength in terms:
                native_poles.append((omega, gamma, array(strength)*fill))
    # A static vacuum-speed CFL leaves the optimization grid/time step unchanged.
    dt = courant/np.sqrt(widths.min()**-2+dy**-2+dx**-2)
    if width < dt:
        raise ValueError('pulse_width must span at least one time step; increase pulse_width or resolution.')
    required_duration = 10*width+2*(z[transmit_z]-z[source_z])*xp.maximum(nt, nb)
    check(n_periods*centre >= required_duration,
          'n_periods is too short for the pulse and a round trip across the scene.')
    steps = math.ceil(n_periods*centre/dt)
    time = np.arange(steps)*dt
    pulse = np.exp(-((time-5*width)/width)**2)*np.exp(-2j*np.pi*time/centre)
    pulse_info = dict(width=width, peak_time=5*width, frequency=1/centre,
                      notches=np.array([], dtype=float), spectrum_kind='ideal_gaussian',
                      relative_spectrum=np.exp(-(np.pi*width*(1/band-1/centre))**2))
    if pulse_notches is not None:
        pulse, shaped = _notched_pulse(pulse, dt, 1/band, pulse_notches, source_epsilon,
                                       1/centre, record=record_field is not None)
        check((steps-len(shaped['notches']))*dt >= required_duration,
              'n_periods is too short for pulse_notches and the scene round trip.')
        pulse_info.update(shaped)
    kx0, ky0 = 2*np.pi/centre*nt*xp.sin(theta)*xp.stack((cos_phi, sin_phi))
    check(xp.all((array(band)/centre)*xp.sin(theta) < 1),
          'The fixed Bloch wavevector must propagate throughout the wavelength band.')
    if truncation_order is None:
        host_kx, host_ky = 2*np.pi/centre*host_indices[0]*np.sin(host_theta)*np.array([np.cos(float(host_phi)), np.sin(float(host_phi))])
        halfx = math.floor(lx*(max(host_indices)/band.min()+abs(host_kx)/(2*np.pi)))
        halfy = math.floor(ly*(max(host_indices)/band.min()+abs(host_ky)/(2*np.pi))) if raster.shape[1] > 1 else 0
    else:
        (halfx, halfy), _ = parse_order_pair(truncation_order)
    if 2*halfx+1 > nx or 2*halfy+1 > ny:
        raise ValueError('Diffraction window exceeds the spatial Nyquist limit; increase resolution.')
    mx, my = np.meshgrid(np.arange(-halfx, halfx+1), np.arange(-halfy, halfy+1))
    kx, ky = kx0+array(2*np.pi*mx/lx), ky0+array(2*np.pi*my/ly)
    # Refuse singular retained channels before any of the four time-domain runs.
    reflection_channels = _diffraction_channels(array(band), kx, ky, nt, dt, dx, dy, xp=xp, check=check)
    transmission_channels = _diffraction_channels(array(band), kx, ky, nb, dt, dx, dy, xp=xp, check=check)
    # The public adapter uses each order's s/p frame. Keep the private fixed
    # TE/TM frame for archived experiments that call this kernel directly.
    co, si = ((xp.ones_like(kx), xp.zeros_like(kx))
              if not _public_order_frame and tetm is not None and raster.shape[1] == 1 and in_plane
              else real_order_frame(xp, ky, kx, phi))
    # Raster coordinates start at zero, as in the RCWA Fourier coefficients.
    transform = xp.exp(-1j*(kx[..., None, None]*array(np.arange(nx)[None, :]*dx)
                            + ky[..., None, None]*array(np.arange(ny)[:, None]*dy)))
    correction = xp.stack((xp.cos(kx*dx/2), xp.cos(ky*dy/2)))
    check(xp.min(xp.abs(correction)) >= 1e-8,
          'A retained order lies at the Yee interpolation Nyquist zero.')
    def modes(spectrum, magnetic=False):
        tangential = xp.sum(spectrum[:, :, :2, None, None]*transform[None, None, None], axis=(-2, -1))/(nx*ny)
        divisor = xp.flip(correction, axis=0) if magnetic else correction
        return tangential/divisor[None, None]
    monitor_indices = [reflect_z, transmit_z]
    monitor_brackets = {}
    for name, coordinate in (field_monitors or {}).items():
        if not isinstance(name, str) or not name:
            raise ValueError('Monitor names must be nonempty strings.')
        if np.ndim(coordinate) or np.iscomplexobj(coordinate) or not np.isfinite(coordinate):
            raise ValueError('Monitor z must be a finite real scalar.')
        if coordinate < z[int(n_pml)+1] or coordinate > z[-int(n_pml)-2]:
            raise ValueError('Monitor must lie in the interior outside the PML.')
        hi = int(np.searchsorted(z, coordinate))
        lo = hi if z[hi] == coordinate else hi-1
        if lo <= source_z <= hi:
            raise ValueError('Monitor interpolation must not touch the current sheet.')
        for index in (lo, hi):
            if index not in monitor_indices:
                monitor_indices.append(index)
        monitor_brackets[name] = (monitor_indices.index(lo), monitor_indices.index(hi),
                                  0. if lo == hi else (coordinate-z[lo])/(z[hi]-z[lo]))
    # Capture the smallest common native box, then trim each staggered component.
    # The union differs from individual bounds by at most the staggering cells.
    record_region, record_trims = None, {}
    if record_bounds is not None:
        names = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')
        selected = names if record_components is None or (isinstance(record_components, str) and record_components == 'all') else record_components
        selected = (selected,) if isinstance(selected, str) else tuple(selected)
        if not selected or len(set(selected)) != len(selected) or any(name not in names for name in selected):
            raise ValueError('record_components must name unique ex/ey/ez/hx/hy/hz components.')
        positions = {}
        for name in selected:
            offsets = np.ones(3)*.5 if name.startswith('h') else np.zeros(3)
            offsets[names.index(name)%3] = 0. if name.startswith('h') else .5
            coordinates = (z+offsets[2]*widths,
                           (np.arange(ny)+offsets[1])*dy-ly/2 if raster.shape[1] > 1 else np.array([0.]),
                           (np.arange(nx)+offsets[0])*dx-lx/2)
            positions[name] = tuple(_bounded_indices(v, record_bounds.get(axis))
                                    for axis, v in zip(('z', 'y', 'x'), coordinates))
            if any(len(v) == 0 for v in positions[name]):
                raise ValueError(f'record_bounds contains no native samples for {name}; widen bounds or refine resolution.')
        region_starts = [min(v[axis][0] for v in positions.values()) for axis in range(3)]
        record_region = tuple(slice(start, max(v[axis][-1] for v in positions.values())+1)
                              for axis, start in enumerate(region_starts))
        record_trims = {name: tuple(slice(v[0]-start, v[-1]-start+1)
                                   for v, start in zip(indices, region_starts))
                        for name, indices in positions.items()}
    # Include the carrier frequency for causal, centre-calibrated time records.
    run_band = np.r_[band, centre] if record_field is not None else band
    projections = {}
    if diffraction_method == 'directional':
        for kind, materials in (('reference', (background[0], background[0])), ('actual', background)):
            planes = []
            for index, (eps, mu_value) in zip((reflect_z, transmit_z), materials):
                matrix = _directional_projection(array(band), kx, ky, eps, mu_value, dt, dx, dy,
                                                  widths[index], xp=xp, check=check)
                if record_field is not None:
                    # Only the incident order at the extra carrier is used for
                    # time-record calibration. Unrequested orders may be grazing.
                    carrier = array(np.zeros((1, 2, 2)+kx.shape, complex))
                    if kind == 'reference' and index == reflect_z:
                        value = _directional_projection(array([centre]), kx[halfy:halfy+1, halfx:halfx+1],
                                                         ky[halfy:halfy+1, halfx:halfx+1], eps, mu_value,
                                                         dt, dx, dy, widths[index], xp=xp, check=check)
                        carrier = value*array((mx == 0) & (my == 0))
                    matrix = xp.concat((matrix, carrier))
                planes.append(matrix)
            projections[kind] = xp.stack(planes, axis=1)
    def run_modes(run, kind, prefix=False):
        data = run['convergence'] if prefix else run
        suffix = '_before_window' if prefix else ''
        electric = modes(data['electric_spectrum'+suffix])
        if diffraction_method == 'electric':
            return electric
        magnetic = modes(data['magnetic_spectrum'+suffix], magnetic=True)
        return _split_directions(xp, electric[:, :2], magnetic[:, :2], projections[kind])
    native_runs = []
    convergence_runs = []
    earlier_runs = []
    runs = []
    stable = scene_valid
    source_plane = array((np.arange(shape[0]) == source_z)[:, None, None])
    for vector in ((-sin_phi, cos_phi), (-cos_phi, -sin_phi)):
        components = []
        for c in range(2):
            xx = array((np.arange(nx)+(.5 if c == 0 else 0))*dx)
            yy = array((np.arange(ny)+(.5 if c == 1 else 0))*dy)
            components.append(source_plane*vector[c]*xp.exp(1j*(kx0*xx[None, :]+ky0*yy[:, None])))
        source = xp.stack((*components, xp.zeros_like(components[0])))
        settings = dict(spacing=(dz, dy, dx), dt=dt, steps=steps, source=source, pulse=pulse,
                        wavelengths=run_band, monitor_z=monitor_indices, backend=backend,
                        device=device, checkpoint_steps=checkpoint_steps, pml_cells=int(n_pml),
                        convergence_window=convergence_window, convergence_tol=convergence_tol, dft_subsample=dft_subsample,
                        pml_wavelength=centre, pml_reflection=pml_reflection,
                        z_spacing=widths if nonuniform else None, bloch_phase=xp.stack((array(0.), ky0*ly, kx0*lx)))
        reference = _uniform_reference(epsilon, et, mt, background[0], settings, kx0, ky0,
                                       dx, dy, xp=xp, array=array)
        actual = fdtd_grid(epsilon, mu=permeability, poles=native_poles,
                           pml_background=background, record_field=record_field,
                           record_components=record_components, record_region=record_region, **settings)
        stable = stable & reference['stable'] & actual['stable']
        runs.append((run_modes(reference, 'reference'), run_modes(actual, 'actual')))
        native_runs.append(actual)
        if convergence_window is not None:
            convergence_runs.append(dict(reference=reference['convergence'], actual=actual['convergence']))
            earlier_runs.append(tuple(run_modes(run, kind, prefix=True)
                                      for run, kind in ((reference, 'reference'), (actual, 'actual'))))
    ResultType, SubType = Result, ResultSub
    if backend in (1, 'jax'):
        from ..backends.jax_backend import ResultJax, ResultSubJax
        ResultType, SubType = ResultJax, ResultSubJax
    psi = (np.pi/2 if tetm == 'te' else 0.) if tetm else (np.pi/4 if psi is None else psi)
    a_s, a_p = incident_sp_amplitudes(xp, runs[0][0].dtype, psi, chi, polarization)
    if _concrete(xp.isfinite(a_s) & xp.isfinite(a_p)) is False:
        raise ValueError('Incident polarization must be finite and nonzero.')
    results = []
    weights = []
    amplitude_changes, calibration_valid = [], []
    basis_changes, basis_calibration_valid = [], []
    incoming_columns = []
    for f, wl in enumerate(run_band):
        k0 = 2*np.pi/wl
        transverse = (kx/k0)**2+(ky/k0)**2
        qt = boundary_kz(xp, nt**2, transverse+0j)
        qb = boundary_kz(xp, nb**2, transverse+0j)
        qi = xp.real(boundary_kz(xp, nt**2, (kx0/k0)**2+(ky0/k0)**2+0j))
        frequency = 2*np.sin(k0*dt/2)/dt
        def propagation(index, kx_value, ky_value, spacing):
            transverse = (2*xp.sin(kx_value*dx/2)/(dx*frequency))**2+(2*xp.sin(ky_value*dy/2)/(dy*frequency))**2
            q = boundary_kz(xp, index**2, transverse+0j)
            return yee_longitudinal_phase(xp, q, frequency, spacing)
        kz_top, kz_bot = propagation(nt, kx, ky, dz)/dz, propagation(nb, kx, ky, dz)/dz
        kz_inc = propagation(nt, kx0, ky0, dz)/dz
        phase_ref, phase_trn = kz_top*z[reflect_z], kz_bot*(z[transmit_z]-thick.sum())
        phase_inc = kz_inc*z[reflect_z]
        if nonuniform:
            def path_phase(index, kx_value, ky_value, start, stop):
                segment = widths[min(start, stop):max(start, stop)]
                angles = propagation(index, kx_value, ky_value, array(segment.reshape((-1,)+(1,)*kx_value.ndim)))
                return np.sign(stop-start)*xp.sum(angles, axis=0)
            phase_ref = path_phase(nt, kx, ky, interface, reflect_z)
            phase_trn = path_phase(nb, kx, ky, bottom_interface, transmit_z)
            phase_inc = path_phase(nt, kx0, ky0, interface, reflect_z)
        if variable_depth:
            def live_phase(index, kx_value, ky_value, start, stop):
                cells = array(widths.reshape((-1,)+(1,)*kx_value.ndim))
                angles = propagation(index, kx_value, ky_value, cells)
                return _phase_between_planes(z, widths, angles, start, stop, xp=xp, array=array)
            phase_ref = live_phase(nt, kx, ky, layer_planes[0], array(z[reflect_z]))
            phase_trn = live_phase(nb, kx, ky, layer_planes[-1], array(z[transmit_z]))
            phase_inc = live_phase(nt, kx0, ky0, layer_planes[0], array(z[reflect_z]))
        def extract(run_values, reference_scale=None):
            incident_columns, outgoing, incoming = [], [], []
            for ref, act in run_values:
                reference = ref[f, 0, 0] if diffraction_method == 'directional' else ref[f, 0]
                inc = reference[:, halfy, halfx]*xp.exp(-1j*phase_inc)
                incident_columns.append(xp.stack((-sin_phi*inc[0]+cos_phi*inc[1],
                                                  -nt/qi*(cos_phi*inc[0]+sin_phi*inc[1]))))
                if f < len(band):
                    if diffraction_method == 'directional':
                        reflected = (act[f, 0, 1]-ref[f, 0, 1])*xp.exp(1j*phase_ref)
                        transmitted = act[f, 1, 0]*xp.exp(-1j*phase_trn)
                        returned_r = (act[f, 0, 0]-ref[f, 0, 0])*xp.exp(-1j*phase_ref)
                        returned_t = act[f, 1, 1]*xp.exp(1j*phase_trn)
                        incoming.append(xp.stack((-si*returned_r[0]+co*returned_r[1],
                                                  -nt/qt*(co*returned_r[0]+si*returned_r[1]),
                                                  -si*returned_t[0]+co*returned_t[1],
                                                  nb/qb*(co*returned_t[0]+si*returned_t[1]))))
                    else:
                        reflected = (act[f, 0]-ref[f, 0])*xp.exp(1j*phase_ref)
                        transmitted = act[f, 1]*xp.exp(-1j*phase_trn)
                    outgoing.append(xp.stack((-si*reflected[0]+co*reflected[1],
                                              nt/qt*(co*reflected[0]+si*reflected[1]),
                                              -si*transmitted[0]+co*transmitted[1],
                                              -nb/qb*(co*transmitted[0]+si*transmitted[1]))))
            incident = xp.stack(incident_columns, axis=1)
            valid = xp.all(xp.isfinite(incident)) & (xp.abs(xp.linalg.det(incident))
                     > 100*xp.finfo(xp.real(incident).dtype).eps*xp.sum(xp.abs(incident)**2))
            scale = xp.sum(xp.abs(incident)**2)
            if reference_scale is not None:
                valid = valid & (scale > (100*xp.finfo(xp.real(incident).dtype).eps)**2*reference_scale)
                incident = xp.where(valid, incident, array(np.eye(2), dtype=incident.dtype))
            calibration = xp.linalg.inv(incident)
            scattering = (xp.tensordot(xp.stack(outgoing, axis=-1), calibration, axes=((-1,), (0,)))
                          if f < len(band) else None)
            returned = (xp.tensordot(xp.stack(incoming, axis=-1), calibration, axes=((-1,), (0,)))
                        if incoming else None)
            return scattering, calibration, valid, scale, returned
        scattering, calibration, valid, scale, returned = extract(runs)
        weights.append(xp.where(stable, calibration @ xp.stack((a_s, a_p)), array(complex(float('nan'), float('nan')))))
        if f == len(band):
            continue
        if returned is not None:
            incoming_columns.append(xp.where(stable, returned, array(complex(float('nan'), float('nan')))))
        if convergence_window is not None:
            earlier, _, earlier_valid, _, _ = extract(earlier_runs, reference_scale=scale)
            difference = scattering-earlier
            basis_valid = valid & earlier_valid & xp.all(xp.isfinite(difference), axis=(0, 1, 2))
            basis_changes.append(xp.where(basis_valid[None, None, None, :], difference,
                                           array(complex(float('nan'), float('nan')))))
            basis_calibration_valid.append(basis_valid)
            change = difference[..., 0]*a_s+difference[..., 1]*a_p
            valid = valid & earlier_valid & xp.all(xp.isfinite(change))
            amplitude_changes.append(xp.where(valid, change, array(complex(float('nan'), float('nan')))))
            calibration_valid.append(valid)
        def pack(amplitudes):
            rs, rp, ts, tp = xp.where(stable, amplitudes, array(complex(float("nan"), float("nan"))))
            fr, ft = xp.real(qt)/qi, xp.real(qb/mb)/(qi/mt)
            r_s, r_p = fr*xp.abs(rs)**2, fr*xp.abs(rp)**2
            t_s, t_p = ft*xp.abs(ts)**2, ft*xp.abs(tp)**2
            return SubType(rs, rp, ts, tp, r_s+r_p, r_s, r_p, t_s+t_p, t_s, t_p)
        results.append(ResultType(pack(scattering[..., 0]*a_s+scattering[..., 1]*a_p),
                                  pack(scattering[..., 0]), pack(scattering[..., 1]),
                                  order_window=((0, halfx), (0, halfy))))
    results = results[:len(band)]
    result = results[0] if np.ndim(wavelength) == 0 else results
    if return_diagnostics:
        diagnostics = dict(incidence_side='top', reflection_z=layer_planes[0], transmission_z=layer_planes[-1],
                           stable=stable, dt=dt, steps=steps, n_steps=steps, dft_subsample=dft_subsample,
                           dft_samples=native_runs[0]['dft_samples'],
                           spacing=(float(widths.min()), dy, dx), shape=shape, wavelengths=array(band),
                           transverse_resolution=(float(rx), float(ry)),
                           mesh_indices=mesh_indices, mesh_thickness=tuple(thick), background=array(background),
                           pulse={key: array(value) if isinstance(value, np.ndarray) else value
                                  for key, value in pulse_info.items()},
                           z_nodes=array(z), z_spacing=array(widths), grading=grading, align_layers=bool(align_layers),
                           n_pml=int(n_pml), pml_reflection=pml_reflection,
                           source_z=z[source_z], source_distance=source_distance,
                           layer_planes=layer_planes, layer_smoothing=layer_smoothing, variable_depth=bool(variable_depth),
                           geometry_kind='parametric' if parametric else 'raster',
                           invariant_y=raster.shape[1] == 1,
                           shape_integration=shape_integration,
                           shape_smoothing=tuple(shape_width) if parametric else (0., 0.),
                           # The final closing cell also belongs to the lower PML.
                           pml_bounds=(z[0], z[int(n_pml)], z[-int(n_pml)], z[-1]+widths[-1]),
                           angles=xp.asin(array(band)/centre*xp.sin(theta)), azimuth=phi,
                           normalization='uniform-medium reference; two electric-current bases')
        if transverse_edges is not None:
            diagnostics['shape_edges'] = tuple(edge-period/2 for edge, period in zip(transverse_edges, lengths))
        weights = xp.stack(weights)
        diagnostics['diffraction'] = dict(
            method=diffraction_method,
            order_x=array(mx), order_y=array(my), kx=array(kx), ky=array(ky),
            reflection_monitor_z=float(z[reflect_z]), transmission_monitor_z=float(z[transmit_z]),
            reflection={key: array(value) for key, value in reflection_channels.items()},
            transmission={key: array(value) for key, value in transmission_channels.items()},
            cutoff_resolved=xp.all(reflection_channels['cutoff_resolved'])
                            & xp.all(transmission_channels['cutoff_resolved']))
        if incoming_columns:
            incoming = xp.stack(incoming_columns)
            diagnostics['diffraction']['incoming'] = {
                name: {key: value[:, i] for i, key in enumerate(('r_s', 'r_p', 't_s', 't_p'))}
                for name, value in (('incident', incoming[..., 0]*a_s+incoming[..., 1]*a_p),
                                    ('s_inc', incoming[..., 0]), ('p_inc', incoming[..., 1]))}
        if convergence_window is not None:
            change = xp.stack(amplitude_changes)
            valid = xp.stack(calibration_valid)
            inactive = xp.all(xp.stack([run[kind]['source_inactive']
                                       for run in convergence_runs for kind in ('reference', 'actual')]))
            diagnostics['convergence'] = dict(
                bases=convergence_runs, calibration_valid=valid,
                comparison_time=(steps-convergence_window)*dt,
                amplitude_change={key: change[:, i] for i, key in enumerate(('r_s', 'r_p', 't_s', 't_p'))},
                amplitudes_settled=stable & inactive & xp.all(valid) & xp.all(xp.abs(change) <= convergence_tol),
                field_decayed=xp.all(xp.stack([run[kind]['field_decayed']
                                              for run in convergence_runs for kind in ('reference', 'actual')])))
            for key in ('material_decayed', 'state_decayed'):
                diagnostics['convergence'][key] = xp.all(xp.stack(
                    [run[kind][key] for run in convergence_runs for kind in ('reference', 'actual')]))
            columns = xp.stack(basis_changes)
            column_valid = xp.stack(basis_calibration_valid)
            for leg, name in enumerate(('s_inc', 'p_inc')):
                diagnostics['convergence'][name] = dict(
                    calibration_valid=column_valid[:, leg],
                    amplitude_change={key: columns[:, i, ..., leg]
                                      for i, key in enumerate(('r_s', 'r_p', 't_s', 't_p'))},
                    amplitudes_settled=stable & inactive & xp.all(column_valid[:, leg])
                    & xp.all(xp.abs(columns[..., leg]) <= convergence_tol))
            # A row's Euclidean norm bounds its change for every unit Jones
            # vector, including constructive interference between both bases.
            row_bound = _scattering_change_bound(xp, columns)
            diagnostics['convergence']['scattering_change_bound'] = {
                key: row_bound[:, i] for i, key in enumerate(('r_s', 'r_p', 't_s', 't_p'))}
            diagnostics['convergence']['scattering_settled'] = (
                stable & inactive & xp.all(column_valid) & xp.all(row_bound <= convergence_tol))
        monitors = {}
        for name, (lo, hi, fraction) in monitor_brackets.items():
            fields = {}
            for family, key in (('e', 'electric_spectrum'), ('h', 'magnetic_spectrum')):
                combined = sum(native[key][:len(band)]*weights[:len(band), leg, None, None, None, None]
                               for leg, native in enumerate(native_runs))
                plane = (1-fraction)*combined[:, lo]+fraction*combined[:, hi]
                if raster.shape[1] == 1:
                    plane = plane[:, :, :1]
                for component, axis in enumerate('xyz'):
                    fields[family+axis] = plane[:, component]
            electric = xp.stack([fields['e'+axis] for axis in 'xyz'], axis=-1)
            magnetic = xp.stack([fields['h'+axis] for axis in 'xyz'], axis=-1)
            density = .5*xp.real(xp.linalg.cross(electric, xp.conj(magnetic), axis=-1))
            area = lx*(ly if raster.shape[1] > 1 else 1.)
            monitors[name] = dict(fields=fields, z=float(field_monitors[name]),
                                  x=array(np.arange(nx)*dx-lx/2),
                                  y=array(np.arange(ny)*dy-ly/2 if raster.shape[1] > 1 else [0.]),
                                  wavelengths=array(band), flux_density=density,
                                  power_z=xp.mean(density[..., 2], axis=(-2, -1))*area,
                                  cell_area=area, per_unit_y=raster.shape[1] == 1,
                                  region='total', normalization='unit incident E at top interface')
        diagnostics['monitors'] = monitors
        if record_field is not None:
            # Constant centre-frequency coefficients preserve a causal waveform.
            # Frequency-by-frequency calibration cannot be applied to time samples.
            carrier_spectrum = array(np.sum(pulse*np.exp(2j*np.pi*(time+dt)/centre))*dt)
            coefficients = weights[-1]*carrier_spectrum
            fields = {}
            for name, first in native_runs[0]['fields'].items():
                values = sum(coefficients[leg]*native['fields'][name]['values']
                             for leg, native in enumerate(native_runs))
                if raster.shape[1] == 1:
                    values = values[:, :, :1]
                fields[name] = dict(first, values=values, x=first['x']-lx/2,
                                    y=first['y']-ly/2 if raster.shape[1] > 1 else array([0.]),
                                    z=first['z']+z[0])
            for name, (zs, ys, xs) in record_trims.items():
                field = fields[name]
                field.update(values=field['values'][:, zs, ys, xs],
                             z=field['z'][zs], y=field['y'][ys], x=field['x'][xs])
            primary = next((name for name in ('ey', 'ex') if name in fields), next(iter(fields)))
            first = fields[primary]
            diagnostics.update(field_e=first['values'], field_e_component=primary,
                               field_t=first['t'], field_z=first['z'], field_x=first['x'], field_y=first['y'])
            diagnostics.update(fields=fields, field_temporal_representation='positive_frequency_carrier',
                               field_normalization='unit incident E at carrier frequency',
                               field_carrier_wavelength=centre)
        return result, diagnostics
    return result
