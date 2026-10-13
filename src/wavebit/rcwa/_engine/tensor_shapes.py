"""Geometry-integrated Popov--Neviere coefficients for tensor shape layers.

Each homogeneous region has its own smooth-extension PN matrix B(e, n).
We integrate its trigonometric interpolant against the *continuous* region
indicator, using the indicator's analytic Fourier coefficients. A = e B is
formed per region before the final convolution matrices are closed. This is
not a product of two truncated RCWA convolution matrices.

Primitive indicators are exact. Painted overlaps use the existing polygon
contours (curves are polygonized there). The normal extension and its spectral
interpolation remain numerical, controlled by shape_nvf_grid_size.
"""
import wavebit.rcwa._engine.factorization as fac
import wavebit.rcwa._engine.parametric as P
import wavebit.rcwa._engine.boundary as boundary
from wavebit._common.backend import as_array
from wavebit.materials.response import as_material
from wavebit.rcwa._engine.geometry import StaticGeometry
from wavebit.rcwa._engine.geometry import StaleGeometryError
from wavebit._common.geometry.overlap import _ELLIPSE_TYPES


def _frequencies(xp, n, h, dtype, device=None):
    return xp.arange(-(n//2) - 2*h, n//2 + 2*h + 1, dtype=dtype, device=device)


def _interpolant(xp, field):
    """Centred coefficients of the cell-centre trigonometric interpolant.

    Split each even-grid Nyquist mode between its two signs. The half-pixel
    phase makes these copies opposite in sign. Keeping only one sign would
    turn a real sampled field into a complex function between sample points.
    """
    ny, nx = field.shape
    dev = getattr(field, 'device', None)
    dtype = xp.real(field).dtype
    qy = xp.arange(-(ny//2), (ny-1)//2+1, dtype=dtype, device=dev)
    qx = xp.arange(-(nx//2), (nx-1)//2+1, dtype=dtype, device=dev)
    c = xp.fft.fftshift(xp.fft.fft2(field)) / (ny*nx)
    c = c * xp.exp(-1j*3.141592653589793*(qy[:,None]/ny + qx[None,:]/nx))
    if ny % 2 == 0:
        c = xp.concat([c[:1]/2, c[1:], -c[:1]/2], axis=0)
    if nx % 2 == 0:
        c = xp.concat([c[:,:1]/2, c[:,1:], -c[:,:1]/2], axis=1)
    return c


def weighted_coefficients(xp, field, indicator_fft, hu, hv):
    """Exact integral of a finite interpolant times a continuous indicator.

    indicator_fft is FFT(ifftshift(F)) on frequencies [-N//2-2h, N//2+2h].
    Padding by 2h per side makes cyclic convolution equal linear convolution
    at every retained difference frequency, including odd and even grids.
    """
    c = _interpolant(xp, field)
    dev = getattr(c, 'device', None)
    if hu:
        z = xp.zeros((c.shape[0], 2*hu), dtype=c.dtype, device=dev)
        c = xp.concat([z,c,z], axis=1)
    if hv:
        z = xp.zeros((2*hv,c.shape[1]), dtype=c.dtype, device=dev)
        c = xp.concat([z,c,z], axis=0)
    out = xp.fft.fftshift(xp.fft.ifft2(
        xp.fft.fft2(xp.fft.ifftshift(c))*indicator_fft))
    my,mx = out.shape[0]//2,out.shape[1]//2
    return out[my-2*hv:my+2*hv+1,mx-2*hu:mx+2*hu+1]


def _shifted(layer):
    shapes = layer.get('shapes', [])
    if not shapes:
        return dict(layer), (0.,0.)
    shift = shapes[0].get('cx',0.), shapes[0].get('cy',0.)
    return {**layer, 'shapes': [dict(s, cx=s.get('cx',0.)-shift[0],
                                   cy=s.get('cy',0.)-shift[1]) for s in shapes]}, shift


def _settings(solver):
    return (tuple(tuple(v) for v in solver.order_window), solver.shape_nvf_grid_size)


def _plan(layer, lattice, index):
    P.check_layer_shapes(layer,lattice,index)
    shapes = layer.get('shapes', [])
    if not shapes:
        return None
    overlap = P.layer_needs_resolving(layer,lattice)
    if len(shapes) == 1:
        gamma = P.plan_gamma(shapes,lattice)
        if gamma is None:
            raise ValueError('freeze_geometry() with concrete tensor shape geometry before tracing')
        return ('primitive', gamma)
    # Even disjoint shapes need an actual-interface plan for tensor contacts:
    # all nine components participate in deciding whether a seam survives.
    interfaces = boundary.plan_layer(shapes,lattice)
    if not overlap and any(s.get('type', 'rectangle') in _ELLIPSE_TYPES for s in shapes):
        gamma = P.plan_gamma(shapes,lattice)
        if gamma is None:
            raise ValueError('freeze_geometry() with concrete tensor shape geometry before tracing')
        return ('disjoint', interfaces, _curve_plan(gamma))
    return ('painted' if overlap else 'disjoint', interfaces)


def freeze(solver):
    plans = tuple(_plan(_shifted(layer)[0],solver.lattice,i)
                  for i,layer in enumerate(solver.structure))
    return StaticGeometry(tensor_geometry={'mode':'integrated', 'plans':plans,
                                          'settings':_settings(solver)})


def _polygons(xp, shapes, lattice, plan):
    import wavebit._common.geometry.clipper as clipper
    vertices = [boundary.shape_vertices(xp,s,lattice,boundary.POLYGON_VERTICES) for s in shapes]
    records,regions,_,_ = plan
    values = clipper.eval_table(xp,records,vertices)
    return [[clipper.loop_verts(xp,loop,values) for loop in region] for region in regions]


def _polygon_ft(xp, vertices, ky, kx, lattice):
    """Bound contour temporaries by chunking frequencies, including under JIT."""
    from wavebit._common.geometry.polygon import polygon_ft
    gy,gx = xp.broadcast_arrays(ky,kx)
    shape = gx.shape
    pairs = xp.stack([xp.reshape(gy,(-1,)),xp.reshape(gx,(-1,))],axis=-1)
    size = max(1, min(pairs.shape[0], 262144//max(1,vertices.shape[0])))
    if 'jax' in type(pairs).__module__:
        import jax
        remainder = (-pairs.shape[0]) % size
        padded = xp.concat([pairs,xp.zeros((remainder,2),dtype=pairs.dtype)],axis=0)
        def evaluate(chunk):
            return polygon_ft(xp,vertices,chunk[:,0],chunk[:,1],lattice)
        out = jax.lax.map(jax.checkpoint(evaluate),xp.reshape(padded,(-1,size,2)))
        out = xp.reshape(out,(-1,))[:pairs.shape[0]]
    else:
        out = xp.concat([polygon_ft(xp,vertices,pairs[i:i+size,0],pairs[i:i+size,1],lattice)
                         for i in range(0,pairs.shape[0],size)])
    return xp.reshape(out,shape)


#: The instances a single-image ellipse is searched over, around its parallelogram fold.
_NEIGHBOURS = tuple((m, n) for m in (-1, 0, 1) for n in (-1, 0, 1))

#: The round types whose two radii are one (``shape_geometry`` reads ``ly`` as ``lx``), and
#: whose nearest centre is therefore their nearest boundary. Every other round type -- an
#: ellipse, or an alias added to ``overlap._ELLIPSE_TYPES`` later -- is searched over
#: :data:`_NEIGHBOURS`, which is right for any convex shape inside its Wigner-Seitz cell.
_CIRCLES = ('circle', 'disk')


def _curve_plan(gamma):
    """What the curve candidates read of a :class:`~wavebit.rcwa._engine.parametric.GammaPlan`.

    The basis reduction, and each round shape's ``'arc'`` piece as ``(shape, imgs)``:
    nothing of the box faces, pairs, dead flags or ring, which no curve candidate reads.
    This projection is the frozen record, and all its staleness check compares
    (:func:`_check_curve_plan_still_fits`).
    """
    return (gamma.reduction,
            tuple((pc.shape, pc.imgs) for pc in gamma.pieces if pc.face == 'arc'))


def _searched(imgs):
    """The instances an arc's candidate searches, as offsets around its parallelogram fold.

    A ring is searched as planned and a single image over :data:`_NEIGHBOURS`
    (:func:`_curve_candidate`). A single-image circle reads its nearest centre instead,
    which on the reduced basis is one of those nine
    (:func:`~wavebit.rcwa._engine.parametric._fold_to_nearest_image`), so the nine stand for it.
    """
    return frozenset(imgs if len(imgs) > 1 else _NEIGHBOURS)


def _check_curve_plan_still_fits(layer, lattice, held, index):
    """Refuse a frozen curve image plan that searches less than a fresh one, and nothing else.

    A re-plan compared through :func:`_curve_plan`, arc by arc as the sets of instances
    searched (:func:`_searched`). Each fresh set holds its curve's nearest boundary at every
    sample, so a record searching a superset finds the same one, by the same arithmetic per
    instance, and in the same order among the instances both search: a box growing across
    its own image-ring threshold, which changes the full plan but no curve candidate, an
    ellipse shrinking back into its Wigner-Seitz cell, whose planned ring holds the 3x3, and
    a thin one leaving it into a ring that is the 3x3, all replay as a fresh build does. An
    ellipse growing out of its cell into a wider ring, or a ring outgrowing the recorded one,
    is refused by name. Another basis reduction, and a shape turning round or not, never
    reach this check: :func:`~wavebit.rcwa._engine.boundary.check_plan_still_fits`, which
    :func:`checked_layers` runs first on the same layer, refuses both, so the record's
    reduction and its arcs' shapes are the fresh plan's here. Skipped when the geometry is
    traced, as every re-plan is.
    """
    shapes = layer.get('shapes', [])
    try:
        P._cell_floats(lattice)
        concrete = P._primitives(shapes, lattice) is not None
    except Exception:                                  # noqa: BLE001 - a traced basis
        return
    fresh = P.plan_gamma(shapes, lattice) if concrete else None
    if fresh is None or _curve_plan(fresh) == held:
        return
    arcs = dict(held[1])
    clauses = []
    for i, new in _curve_plan(fresh)[1]:
        old = arcs[i]
        if not _searched(new) <= _searched(old):
            why = ('its support crossed the edge of its Wigner-Seitz cell'
                   if 1 in (len(old), len(new)) else 'its size crossed a ring boundary')
            clauses.append(f"shape {P._shape_name(shapes, i)}'s image list changed "
                           f"({len(old)} -> {len(new)} instances: {why})")
    if not clauses:
        return
    raise StaleGeometryError(
        f"the frozen geometry no longer fits this structure: {'; '.join(clauses)}. The "
        f'recorded curve image plan of layer {index} decides which periodic instances each '
        f'curved interface is searched over, and a fresh plan searches instances this record '
        f'does not, so a replay could miss the nearest boundary. Call freeze_geometry() '
        f'again, or drop geometry= to plan eagerly.')


def _curve_direction(xp, xr, yr, a, b, c, s, on, cos_a, sin_a, circle=False):
    """The unit vector one instance's curve candidate carries, in Cartesian axes.

    The level-set rule: ``sign(f) grad f / |grad f|`` for ``f = (xr/a)^2 + (yr/b)^2 - 1``,
    which points from the curve toward the sample on both sides -- inward inside the shape,
    as a segment candidate's ``p - foot`` does
    (:func:`~wavebit.rcwa._engine.boundary.material_projector_fields`) -- and where ``on``, the
    sample within the blend's on-interface threshold, the outward normal, which is what a
    segment's ``n_edge`` carries there. ``(xr, yr)`` is the sample relative to the instance's
    centre in the shape's own frame, ``a`` and ``b`` its semi-axes, ``(c, s)`` the angle of
    its nearest boundary point (:func:`~wavebit.rcwa._engine.parametric._ellipse_nearest_point`) and
    ``(cos_a, sin_a)`` the rotation. The side is the level-set sign, read only off the
    interface, where rounding cannot flip it; on it the side is outward.

    The gradient is spelled ``(xr b / a, yr a / b)``: the direction of
    ``(xr / a^2, yr / b^2)``, in units of a length as the samples are. On the curve it is
    read at the sample too, not at the nearest point: the two normals agree there but their
    derivatives do not, and the gradient's is the derivative of the field around the sample
    (on an ellipse sized to put a sample on its curve, the nearest point's normal moved the
    angle slope of total transmission 8.5% off a central difference; the gradient's agreed to
    1e-5). It vanishes only at the centre, where the level set has no direction and the
    candidate carries the normal at the nearest point instead -- along ``(b c, a s)``, never
    shorter than ``min(a, b)``; an ellipse's nearest point there is an end of its minor axis
    -- signed as everywhere off the curve, from the curve toward the sample: inward, the
    limit of the level-set direction along the minor axis from that end's side. A ``circle``
    carries the nearest point's normal everywhere: its level-set gradient is radial, the
    same vector and, the two fields being one, the same derivative; one arithmetic serves
    every circle, inside the curve, on it and at its centre.

    For an ellipse the level-set gradient is not the nearest-point vector the polygon route
    converges to as its facets refine: off the curve the two differ by up to 60-80 degrees.
    It is chosen for what a gradient sees. The nearest point jumps across the major axis
    between the evolute's cusps, so under the nearest-point rule a normal-grid sample that a
    rotation or a translation carries across that segment steps the blend, autodiff sees
    none of the steps, and they do not average away as the grid refines (each shrinks as
    ``1 / n**2`` in ``shape_nvf_grid_size`` while the samples crossed grow as ``n**2``). The
    level-set gradient is continuous everywhere but at the centre. A sample that a motion
    carries past the centre at a distance ``d`` turns through a half-turn, continuously, over
    a stretch of the motion of about ``d (b / a)**2`` along the minor axis (``d (a / b)**2``
    along the major), and the gradient of a loss peaks there roughly as ``1 / d``: without
    bound as the pass nears the centre, so a gradient step taken beside such a pass can be
    arbitrarily large, and a gradient integrated over a window resolves the pass only on
    abscissae finer than that stretch. It converges more slowly in the truncation order.
    ``docs/guide/rcwa/tensor-shape-integration.md`` has the measurements on both sides of the choice,
    and ``tests/rcwa/test_tensor_curve_normals.py`` holds the window check.
    """
    one = xp.ones_like(xr)
    nu = xp.where(xr < 0., -one, one) * (b * c)
    nv = xp.where(yr < 0., -one, one) * (a * s)
    if not circle:
        gu, gv = xr * (b / a), yr * (a / b)
        level = gu * gu + gv * gv > 0.
        nu, nv = xp.where(level, gu, nu), xp.where(level, gv, nv)
    side = xp.where(((xr / a) ** 2 + (yr / b) ** 2 < 1.) & ~on, -one, one)
    scale = side / xp.sqrt(nu * nu + nv * nv)
    nu, nv = nu * scale, nv * scale
    return nu * cos_a - nv * sin_a, nu * sin_a + nv * cos_a


def _curve_candidate(xp, X, Y, shape, lattice, reduced, dtype, imgs, threshold2):
    """``(distance, nx, ny)``: the nearest boundary among a round shape's periodic instances.

    The distance is exact (:func:`~wavebit.rcwa._engine.parametric._ellipse_distance`) and taken to
    the nearest instance's *boundary*, on every lattice; the direction is that instance's
    level-set direction (:func:`_curve_direction`), with ``threshold2`` the blend's
    on-interface threshold. A circle is read in an unrotated frame: its ``angle`` describes
    nothing, and where a sample has no radial direction -- its centre, which an odd grid
    samples when the circle is the layer's first shape -- the frame alone would fix the
    vector carried there, and the answer would turn with an angle that should not reach it.
    Which instances are searched is decided by the plan's ``imgs`` and the shape type, all
    static, so the build stays jit-static and differentiable through the hard minimum:

    * a ring (``len(imgs) > 1``, a shape leaving the Wigner-Seitz cell of its centre) is
      searched as planned, around the parallelogram fold that
      :func:`~wavebit.rcwa._engine.parametric._arc_offsets` bounds it around;
    * a single-image circle (:data:`_CIRCLES`) is read at the nearest *centre*
      (:func:`~wavebit.rcwa._engine.parametric._nearest_centre_displacement`, the fold the primitive
      path uses), which for equal radii is the nearest boundary: one exact distance;
    * any other single-image shape -- an ellipse -- is searched over the 3x3 cells around
      its parallelogram fold. Its nearest centre is **not** its nearest boundary: off the
      curve, a neighbour's tip is often nearer (measured on 127 of 129 random single-image
      ellipses on a square cell, up to 280 nm too far on a 600 nm period). The 3x3 is
      enough for any convex shape inside the Wigner-Seitz cell ``V`` on a Lagrange-reduced
      basis. A sample ``p`` outside every instance reads ``f(p - t)`` with ``f`` the
      distance to the shape, and whenever ``(t - p) . w >= |w|^2`` for a lattice vector
      ``w``, instance ``t - w`` is no farther (the shape lies in the slab
      ``|x . w| <= |w|^2 / 2``) and ``t - w`` is nearer ``p``. So the minimum sits at a
      ``t`` with ``t - p`` inside ``2 V``. On a reduced basis every
      point of ``V`` has fractional coordinates within 2/3 (the hexagonal vertex attains
      it), the fold leaves ``p`` within 1/2, so ``|t_i| < 1/2 + 4/3 < 2``. A sample inside an
      instance is in that instance's cell, which the same bound puts in the 3x3, and no other
      boundary is nearer than that cell's edge.
    """
    stype, cx, cy, lx, ly, angle = P.shape_geometry(shape, lattice)
    circle = stype in _CIRCLES
    if circle:
        cos_a, sin_a = 1.0, 0.0
    else:
        cos_a = xp.cos(as_array(xp, angle, dtype=dtype))
        sin_a = xp.sin(as_array(xp, angle, dtype=dtype))
    a, b = lx / 2.0, ly / 2.0
    if len(imgs) > 1:
        instances = P._instance_displacements(xp, X, Y, cx, cy, reduced, imgs)
    elif circle:
        instances = (P._nearest_centre_displacement(xp, X, Y, cx, cy, reduced),)
    else:
        instances = P._instance_displacements(xp, X, Y, cx, cy, reduced, _NEIGHBOURS)
    best = nx_best = ny_best = None
    for dx, dy in instances:
        xr = dx * cos_a + dy * sin_a
        yr = -dx * sin_a + dy * cos_a
        distance, c, s = P._ellipse_nearest_point(xp, xr, yr, a, b)
        nx, ny = _curve_direction(xp, xr, yr, a, b, c, s, distance * distance <= threshold2,
                                  cos_a, sin_a, circle)
        if best is None:
            best, nx_best, ny_best = distance, nx, ny
        else:
            take = distance < best
            best = xp.where(take, distance, best)
            nx_best = xp.where(take, nx, nx_best)
            ny_best = xp.where(take, ny, ny_best)
    return best, nx_best, ny_best


def _curve_fields(xp, shapes, lattice, dtype, curves):
    """Analytic candidates for the uncut curves of a disjoint layer, keyed by region pair.

    ``curves`` is the layer's :func:`_curve_plan`. Each candidate returns the squared
    distance to the curve's nearest periodic instance and that instance's level-set
    direction, oriented as a segment candidate's vector (:func:`_curve_candidate`,
    :func:`_curve_direction`), under the blend's own on-interface threshold. The
    material-pair blend still decides how much each interface contributes; only the curve's
    polygon candidate is replaced.
    """
    from wavebit.lattice import reduced_basis
    reduction, arcs = curves
    reduced = reduced_basis(lattice, reduction)
    threshold2 = boundary._on_interface_threshold2(xp, lattice, dtype)
    fields = {}
    for index, imgs in arcs:
        def evaluate(points, shape=shapes[index], imgs=imgs):
            distance, nx, ny = _curve_candidate(xp, points[:, 0], points[:, 1], shape, lattice,
                                                reduced, dtype, imgs, threshold2)
            return distance*distance, xp.stack([nx, ny], axis=-1)
        fields[(index, len(shapes))] = evaluate
    return fields


def _projector(solver, shapes, plan, loops, values):
    xp,dtype = solver.xp,solver.float_dtype
    n = solver.shape_nvf_grid_size
    if plan[0] == 'primitive':
        # No inter-material seam exists within one primitive. Its geometric
        # normal is independent of epsilon, mu, and their complex phases.
        geometric = [dict(shapes[0],material=1.)]
        nx,ny = P.normal_vector_field_analytical(xp,(n,n),geometric,dtype,
                                                solver.lattice,plan=plan[1])
        return nx*nx,nx*ny,ny*ny
    from wavebit._common.geometry.interfaces import replay_interfaces
    _,_,transform,interfaces = plan[1]
    regions = [(1.,poly) for poly in loops]
    ordered_values = xp.stack(values[1:]+values[:1])
    segments = replay_interfaces(xp,regions,1.,solver.lattice,transform,interfaces,
                                 material_values=ordered_values)
    if segments is None:
        one = xp.ones((n,n),dtype=dtype,device=solver._array_device)
        return one,xp.zeros_like(one),xp.zeros_like(one)
    left = as_array(xp,[p.left for p in interfaces.pieces],dtype=xp.int64,device=solver._array_device)
    right = as_array(xp,[p.right for p in interfaces.pieces],dtype=xp.int64,device=solver._array_device)
    a,b = ordered_values[left],ordered_values[right]
    def norm(t):
        square = xp.sum(xp.real(t*xp.conj(t)),axis=(-1,-2))
        return xp.where(square>0,xp.sqrt(xp.where(square>0,square,xp.ones_like(square))),0.)
    scale = norm(a)+norm(b)
    contrast = norm(a-b)/xp.where(scale>0,scale,xp.ones_like(scale))
    curves = (_curve_fields(xp,shapes,solver.lattice,dtype,plan[2])
              if len(plan) == 3 else None)
    return boundary.material_projector_fields(xp,(n,n),solver.lattice,dtype,
        segments,contrast,interfaces,transform=transform,candidate_fields=curves)


def _layer(solver, layer, plan, quantity):
    from wavebit.materials.tensors import as_tensor
    from wavebit.rcwa._engine.anisotropy import full_tensor
    xp,dtype,dev = solver.xp,solver.complex_dtype,solver._array_device
    (_,hu),(_,hv) = solver.order_window
    n = solver.shape_nvf_grid_size
    shapes = layer['shapes']
    materials = [layer.get('background',1.)]+[s['material'] for s in shapes]
    values = [as_tensor(xp, as_material(m).epsilon if quantity=='epsilon'
                        else as_material(m).mu_value,dtype,dev) for m in materials]
    all_values = xp.stack(values)
    size = 3 if full_tensor(xp, all_values) else 2
    loops = None if plan[0]=='primitive' else _polygons(xp,shapes,solver.lattice,plan[1])
    ky = _frequencies(xp,n,hv,solver.float_dtype,dev)[:,None]
    kx = _frequencies(xp,n,hu,solver.float_dtype,dev)[None,:]
    indicators = []
    for i,s in enumerate(shapes):
        if plan[0] == 'painted':
            f = xp.zeros((ky.shape[0],kx.shape[1]),dtype=dtype,device=dev)
            for vertices in loops[i]:
                f = f + _polygon_ft(xp,vertices,ky,kx,solver.lattice)
        else:
            f = P.shape_ft(xp,s,ky,kx,solver.lattice)
        indicators.append(f)
    delta = xp.astype((ky==0)&(kx==0),dtype)
    indicators.insert(0,delta-sum(indicators))
    pxx,pxy,pyy = _projector(solver,shapes,plan,loops,values)
    zero = xp.zeros_like(pxx)
    projector = ((pxx,pxy,zero),(pxy,pyy,zero),(zero,zero,zero))
    A = [[xp.zeros((4*hv+1,4*hu+1),dtype=dtype,device=dev) for j in range(size)] for i in range(size)]
    B = [[xp.zeros_like(A[0][0]) for j in range(size)] for i in range(size)]
    zz = xp.zeros_like(A[0][0])
    my,mx = indicators[0].shape[0]//2,indicators[0].shape[1]//2
    for e,f in zip(values,indicators):
        f_fft = xp.fft.fft2(xp.fft.ifftshift(f))
        a = e[0,0]*pxx+(e[0,1]+e[1,0])*pxy+e[1,1]*pyy
        a = xp.where(pxx+pyy==0,xp.ones_like(a),a)
        # B = I + (P - P e)/a, with P = n n^T. Compute one component
        # at a time, avoiding any N x N x 3 x 3 material raster.
        c = [[weighted_coefficients(xp,(1. if i==j else 0.)+
                                    (projector[i][j]-projector[i][0]*e[0,j]-
                                     projector[i][1]*e[1,j])/a,f_fft,hu,hv)
              for j in range(size)] for i in range(size)]
        for i in range(size):
            for j in range(size):
                B[i][j] = B[i][j]+c[i][j]
                A[i][j] = A[i][j]+sum(e[i,k]*c[k][j] for k in range(size))
        zz = zz+e[2,2]*f[my-2*hv:my+2*hv+1,mx-2*hu:mx+2*hu+1]
    return A,B,zz


def checked_layers(solver, geometry):
    """Yield the live layers and validated plans, without Fourier assembly."""
    record = None if geometry is None else geometry.tensor_geometry
    if geometry is not None and (record is None or record.get('mode')!='integrated' or
            record['settings']!=_settings(solver) or len(record['plans'])!=len(solver.structure)):
        raise StaleGeometryError('tensor shape integration settings changed; freeze_geometry() again')
    for index,original in enumerate(solver.structure):
        layer,shift = _shifted(original)
        P.check_layer_shapes(layer,solver.lattice,index)
        plan = _plan(layer,solver.lattice,index) if record is None else record['plans'][index]
        shapes = layer.get('shapes',[])
        if not shapes:
            if plan is not None:
                raise StaleGeometryError('tensor shapes changed; freeze_geometry() again')
            yield layer, shift, plan
            continue
        if plan is None:
            raise StaleGeometryError('tensor shapes changed; freeze_geometry() again')
        if record is not None:
            if plan[0]=='primitive':
                if len(shapes)!=1:
                    raise StaleGeometryError('tensor shapes changed; freeze_geometry() again')
                P.check_gamma_plan_still_fits(layer,solver.lattice,plan[1],index)
            else:
                boundary.check_plan_still_fits(shapes,solver.lattice,plan[1])
                if plan[0]=='disjoint':
                    P.check_no_plan_still_fits(layer,solver.lattice,index)
                    if len(plan) == 3:
                        _check_curve_plan_still_fits(layer,solver.lattice,plan[2],index)
        yield layer, shift, plan


def build(solver, geometry=None, quantity='epsilon'):
    from wavebit.rcwa._engine.anisotropy import _uniform_slots
    from wavebit.materials.tensors import as_tensor
    from wavebit.materials.tensors import tensor_shape
    from wavebit.rcwa._engine.anisotropy import TensorConvMatrices
    from wavebit.rcwa._engine.anisotropy import full_tensor
    from wavebit.rcwa._engine.anisotropy import tensor_slots
    xp,dtype,dev = solver.xp,solver.complex_dtype,solver._array_device
    (_,hu),(_,hv) = solver.order_window
    count = (2*hu+1)*(2*hv+1)
    output = []
    for layer, shift, plan in checked_layers(solver, geometry):
        if not layer.get('shapes', []):
            m = as_material(layer.get('background',1.))
            raw = m.epsilon if quantity=='epsilon' else m.mu_value
            e = as_tensor(xp,raw,dtype,dev)
            full = tensor_shape(raw)==(3,3) and full_tensor(xp,e)
            output.append(_uniform_slots(xp,e,count,full));continue
        A,B,zz = _layer(solver,layer,plan,quantity)
        ky = xp.arange(-2*hv,2*hv+1,dtype=solver.float_dtype,device=dev)[:,None]
        kx = xp.arange(-2*hu,2*hu+1,dtype=solver.float_dtype,device=dev)[None,:]
        a,b = solver.lattice.u,solver.lattice.v
        det = a[0]*b[1]-a[1]*b[0]
        phase = xp.exp(-2j*3.141592653589793*((b[1]*kx-a[1]*ky)*shift[0]+
                                             (-b[0]*kx+a[0]*ky)*shift[1])/det)
        # Geometry quadrature/FFT may promote a backend's intermediates. The
        # RCWA operators must honor the solver precision at this boundary.
        def conv(c): return P.conv_matrix_2d(xp,xp.astype(c*phase,dtype),hu,hv)
        def block(c): return xp.concat([xp.concat([conv(v) for v in row],axis=1) for row in c],axis=0)
        e = block(A) @ xp.linalg.inv(block(B))
        if len(A) == 3:
            output.append(list(tensor_slots(xp, [[e[i*count:(i+1)*count,j*count:(j+1)*count]
                            for j in range(3)] for i in range(3)])))
        else:
            output.append([e[:count,:count],e[count:,count:],xp.linalg.inv(conv(zz)),
                           -e[:count,count:],-e[count:,:count]])

    full = any(len(row)==9 for row in output)
    if full:
        for row in output:
            if len(row)==5: row.extend([xp.zeros_like(row[0]) for _ in range(4)])
    cls = TensorConvMatrices if full else fac.ConvMatrices
    return cls(*(xp.stack([row[j] for row in output]) for j in range(9 if full else 5)))
