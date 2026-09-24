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
from . import factorization as fac, parametric as P, boundary as boundary
from .backend import as_array
from .domain import as_material
from .geometry import StaticGeometry, StaleGeometryError


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
    return ('painted' if overlap else 'disjoint', boundary.plan_layer(shapes,lattice))


def freeze(solver):
    plans = tuple(_plan(_shifted(layer)[0],solver.lattice,i)
                  for i,layer in enumerate(solver.structure))
    return StaticGeometry(tensor_geometry={'mode':'integrated', 'plans':plans,
                                          'settings':_settings(solver)})


def _polygons(xp, shapes, lattice, plan):
    from . import clipper
    vertices = [boundary.shape_vertices(xp,s,lattice,boundary.POLYGON_VERTICES) for s in shapes]
    records,regions,_,_ = plan
    values = clipper.eval_table(xp,records,vertices)
    return [[clipper.loop_verts(xp,loop,values) for loop in region] for region in regions]


def _polygon_ft(xp, vertices, ky, kx, lattice):
    """Bound contour temporaries by chunking frequencies, including under JIT."""
    from .polygon import polygon_ft
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
    from .interfaces import replay_interfaces
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
    return boundary.material_projector_fields(xp,(n,n),solver.lattice,dtype,
        segments,contrast,interfaces,transform=transform)


def _layer(solver, layer, plan, quantity):
    from .anisotropy import as_tensor, full_tensor
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


def build(solver, geometry=None, quantity='epsilon'):
    from .anisotropy import _uniform_slots, as_tensor, tensor_shape, TensorConvMatrices, full_tensor, tensor_slots
    xp,dtype,dev = solver.xp,solver.complex_dtype,solver._array_device
    (_,hu),(_,hv) = solver.order_window
    count = (2*hu+1)*(2*hv+1)
    record = None if geometry is None else geometry.tensor_geometry
    if geometry is not None and (record is None or record.get('mode')!='integrated' or
            record['settings']!=_settings(solver) or len(record['plans'])!=len(solver.structure)):
        raise StaleGeometryError('tensor shape integration settings changed; freeze_geometry() again')
    output = []
    for index,original in enumerate(solver.structure):
        layer,shift = _shifted(original)
        P.check_layer_shapes(layer,solver.lattice,index)
        plan = _plan(layer,solver.lattice,index) if record is None else record['plans'][index]
        shapes = layer.get('shapes',[])
        if not shapes:
            if plan is not None:
                raise StaleGeometryError('tensor shapes changed; freeze_geometry() again')
            m = as_material(layer.get('background',1.))
            raw = m.epsilon if quantity=='epsilon' else m.mu_value
            e = as_tensor(xp,raw,dtype,dev)
            full = tensor_shape(raw)==(3,3) and full_tensor(xp,e)
            output.append(_uniform_slots(xp,e,count,full));continue
        if plan is None:
            raise StaleGeometryError('tensor shapes changed; freeze_geometry() again')
        if record is not None:
            if plan[0]=='primitive':
                if len(shapes)!=1:
                    raise StaleGeometryError('tensor shapes changed; freeze_geometry() again')
                P.check_gamma_plan_still_fits(layer,solver.lattice,plan[1],index)
            else:
                boundary.check_plan_still_fits(shapes,solver.lattice,plan[1])
                if plan[0]=='disjoint': P.check_no_plan_still_fits(layer,solver.lattice,index)
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
