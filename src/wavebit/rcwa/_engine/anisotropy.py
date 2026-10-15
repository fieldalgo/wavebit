"""Material tensors at the scene boundary, separate from scalar factorization.

Tensor rasters contain physical constitutive tensors, never convolution slots.
Uniform tensors use exact scaled identities. Patterned tensors use
Popov--Neviere NVF by default, or successive Cartesian Li factorization.
Shape tensors default to continuous geometry integration (tensor_shapes);
an explicit sampled NVF route also remains available.
"""

from wavebit.materials.tensors import as_tensor
from wavebit.materials.tensors import material_is_tensor
from wavebit.materials.tensors import tensor_shape
from typing import NamedTuple

import numpy as np

from wavebit._common.backend import array_namespace
from wavebit._common.backend import as_array
from wavebit._common.backend import _concrete
from wavebit._common.backend import carries_autodiff
from wavebit._common.backend import torch_autodiff
from wavebit.materials.response import as_material
from wavebit.rcwa._engine.geometry import StaticGeometry
from wavebit.rcwa._engine.geometry import StaleGeometryError
import wavebit.rcwa._engine.factorization as fac
from wavebit._common.backend import eager_under_torch_compile


#: ``tetm`` beside a tensor material. One sentence for the two places that refuse it:
#: :func:`validate_options`, where the convolution matrices are built, and
#: ``CoreSolveMixin._refuse_tetm_off_the_pure_path``, for matrices built before ``tetm`` was set.
TETM_REFUSAL = ('tensor materials require the coupled polarization solve; use polarization '
                'instead of tetm')


@eager_under_torch_compile
class TensorConvMatrices(NamedTuple):
    """Nine constitutive blocks (epsilon or mu), in ``solve_conv`` slot order.

    Public inputs and outputs use the cell centre. Like ``ConvMatrices``, this
    container also carries private corner-frame operators and performs no
    origin conversion itself.
    """
    epx: object
    epy: object
    epz_i: object
    epxy: object
    epyx: object
    epxz: object
    epyz: object
    epzx: object
    epzy: object


def full_tensor(xp, tensor):
    """Whether all nine blocks are needed, including possible material tangents.

    A zero coupling at the evaluation point can have a nonzero derivative. Do
    not discard it under autodiff; an abstract tensor also keeps all blocks.
    """
    if _geometry_derivative(tensor):
        return True
    clear = xp.all(tensor[..., 2, :2] == 0) & xp.all(tensor[..., :2, 2] == 0)
    return _concrete(clear) is not True


def tensor_slots(xp, blocks):
    """Physical 3 by 3 operator blocks to nine slots, preserving their Fourier frame."""
    return TensorConvMatrices(blocks[0][0], blocks[1][1], xp.linalg.inv(blocks[2][2]),
                              -blocks[0][1], -blocks[1][0], -blocks[0][2],
                              -blocks[1][2], -blocks[2][0], -blocks[2][1])






def scene_is_tensor(structure, mu=None):
    if isinstance(structure, list):
        return any(material_is_tensor(layer.get('background', 1.)) or
                   any(material_is_tensor(s.get('material', 1.)) for s in layer.get('shapes', []))
                   for layer in structure)
    return any(getattr(a, 'ndim', 0) == 5 for a in (structure, mu))






def _uniform_slots(xp, e, count, full):
    I = xp.eye(count, dtype=e.dtype, device=getattr(e, 'device', None))
    values = [e[0,0]*I, e[1,1]*I, xp.linalg.inv(e[2,2]*I), -e[0,1]*I, -e[1,0]*I]
    if full:
        values += [-e[0,2]*I, -e[1,2]*I, -e[2,0]*I, -e[2,1]*I]
    return values


def _labels(tensor):
    """Identify all tensor phases; equal traces do not imply equal materials."""
    a = fac._to_numpy(tensor)
    _, labels = np.unique(a.reshape(-1, 9), axis=0, return_inverse=True)
    return labels.reshape(a.shape[:2])


def _shape_values(shape, basis):
    """A sampled shape's kind and ``(cx, cy, lx, ly, angle)``, refused where no mask can be painted."""
    from wavebit._common.geometry.recipes import shape_geometry
    kind, *values = shape_geometry(shape, basis)
    if kind not in ('circle', 'disk', 'ellipse', 'rectangle', 'square'):
        raise NotImplementedError('tensor shapes support circles, ellipses and rectangles')
    values = np.asarray(values, dtype=complex)
    if not np.all(np.isfinite(values)) or np.any(values.imag != 0):
        raise ValueError('shape positions, lengths and angles must be finite and real')
    cx, cy, lx, ly, angle = values.real
    if min(lx, ly) <= 0:
        raise ValueError('shape lengths must be positive')
    return kind, cx, cy, lx, ly, angle


def _periodic_shape_mask(shape, X, Y, basis):
    """Paint all images intersecting the cell, including an unreduced lattice.

    Bounding each shape in fractional coordinates avoids a fixed image ring,
    which misses translated shapes and long/sheared cells.
    """
    kind, cx, cy, lx, ly, angle = _shape_values(shape, basis)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    inverse = np.linalg.inv(basis)
    center = (inverse @ np.array([cx, cy])) % 1.
    extents = (inverse @ rotation) * np.array([lx/2, ly/2])
    half = (np.sum(np.abs(extents), axis=1) if kind in ('rectangle', 'square')
            else np.linalg.norm(extents, axis=1))
    low = np.ceil(-center-half).astype(int)
    high = np.floor(1-center+half).astype(int)
    mask = np.zeros(X.shape, dtype=bool)
    for m in range(low[0], high[0]+1):
        for k in range(low[1], high[1]+1):
            ox, oy = basis @ (center + [m,k])
            dx, dy = X-ox, Y-oy
            u, v = c*dx+s*dy, -s*dx+c*dy
            mask |= ((np.abs(u) <= lx/2) & (np.abs(v) <= ly/2) if kind in ('rectangle','square')
                     else (u/(lx/2))**2+(v/(ly/2))**2 <= 1.)
            if mask.all():
                return mask
    return mask


def _shape_snapshot(solver, basis=None, n=None, paint=True):
    """Read only geometry, never material values, to make a reusable sampling record.

    ``paint=False`` meets every refusal of the record and paints no mask -- one is
    ``shape_nvf_grid_size`` squared per shape, 1024 x 1024 by default -- and returns None:
    the admission check's call (:func:`check_layers`).
    """
    import wavebit.rcwa._engine.parametric as parametric
    from wavebit.lattice import sample_grid
    xp = array_namespace(np.asarray(0.))
    try:
        basis = np.asarray(fac._to_numpy(solver.lattice) if basis is None else basis, dtype=float)
        n = solver.shape_nvf_grid_size if n is None else n
        X,Y = sample_grid(xp, (n,n), basis, np.float64) if paint else (None, None)
        masks, keys = [], []
        for i, layer in enumerate(solver.structure):
            parametric.check_layer_spec(layer, i)
            current, key = [], []
            for j, shape in enumerate(layer.get('shapes', [])):
                parametric.check_shape_spec(shape, j, i)
                # Reject geometry tangents rather than return a plausible zero gradient.
                geometry = {k:v for k,v in shape.items() if k != 'material'}
                for v in geometry.values():
                    if carries_autodiff(v):
                        raise ValueError('sampled tensor shapes do not support geometry derivatives; '
                                         'keep geometry fixed and differentiate material values')
                key.append(tuple(sorted((k, repr(fac._to_numpy(v)) if hasattr(v, 'shape') else repr(v))
                                        for k,v in geometry.items())))
                if paint:
                    current.append(_periodic_shape_mask(shape, X, Y, basis))
                else:
                    _shape_values(shape, basis)
            masks.append(current);keys.append(tuple(key))
        return (masks, (n, basis.tobytes(), tuple(keys))) if paint else None
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and 'geometry derivatives' in str(exc):
            raise
        raise ValueError(f'tensor shape sampling requires fixed, valid geometry: {exc}') from exc


def _field(solver, structure, quantity, snapshot=None):
    xp, dtype, dev = solver.xp, solver.complex_dtype, solver._array_device
    if not isinstance(structure, list):
        a = as_array(xp, structure, dtype=dtype, device=dev)
        if a.ndim == 3:
            return a[...,None,None] * xp.eye(3, dtype=dtype, device=dev)
        if a.ndim != 5 or a.shape[-2:] != (3,3):
            raise ValueError('tensor rasters have shape (layers, Ny, Nx, 3, 3)')
        if _concrete(xp.all(xp.isfinite(a))) is False:
            raise ValueError('material tensors must contain finite values')
        return a
    fields = []
    for i, layer in enumerate(structure):
        def value(v):
            m = as_material(v)
            return as_tensor(xp, m.epsilon if quantity == 'epsilon' else m.mu_value, dtype, dev)
        bg = value(layer.get('background', 1.))
        shapes = layer.get('shapes', [])
        if not shapes:
            fields.append(bg[None,None,...]);continue
        masks = snapshot[0][i]
        n = masks[0].shape[0]
        t = xp.broadcast_to(bg, (n,n,3,3))
        for mask, shape in zip(masks, shapes):
            t = xp.where(as_array(xp, mask, device=dev)[...,None,None], value(shape['material']), t)
        fields.append(t)
    return fields


def _geometry_derivative(value):
    if isinstance(value, (tuple, list)):
        return any(_geometry_derivative(v) for v in value)
    # JIT's constant tracers are allowed; a JVP/linearization of geometry is not, and neither
    # is either PyTorch mode: `requires_grad` or a `forward_ad` dual, which does not set it --
    # under `torch.func.vmap` as well, where both sit under a `BatchedTensor` that reads
    # neither, and the value went on to a raw "Cannot access data pointer" error instead of
    # this refusal (`torch_autodiff`).
    return (torch_autodiff(value)
            or type(value).__name__ in ('JVPTracer', 'LinearizeTracer'))


def validate_options(solver):
    from wavebit.rcwa._engine.enums import Factorization
    from wavebit.rcwa._engine.enums import FourierCoefficients
    solver._reject_layer_count_mismatch()
    if isinstance(solver.structure, list):
        from wavebit._common.geometry.recipes import check_layer_spec
        from wavebit._common.geometry.recipes import holds_polygon
        for i, layer in enumerate(solver.structure):
            check_layer_spec(layer, i)
            if holds_polygon(layer):
                # By type, before either tensor builder reads a shape: both are written over the
                # primitives' parameters (the sampled masks and the integrated plan's centre
                # shift), and neither has an outline to read. The scalar boundary route does.
                raise NotImplementedError(
                    f'layer {i} holds a polygon, and tensor materials are built for rectangles, '
                    f'circles and ellipses only: neither tensor builder reads an outline given '
                    f'by vertices. A polygon solves with scalar materials (on the boundary '
                    f'route); for a tensor region, sample the layer onto a tensor raster.')
    if solver._factorization == Factorization.LAURENT:
        raise NotImplementedError(
            "factorization='laurent' is not built for tensor materials. A tensor layer is "
            "factorized by tensor Li (wavebit.rcwa._engine.tensor_li) or by the Popov-Neviere NVF "
            "rearrangement; Laurent's rule on a tensor would be a third tensor builder -- "
            "[[eps_ij]] in all nine slots, the z row included -- with no reference checked "
            "against it here, so it is refused rather than answered by the scalar build. A "
            "uniform tensor layer is the same under every rule, so a scene of uniform layers "
            "loses nothing by leaving factorization unset.")
    if solver._factorization == Factorization.LI:
        import wavebit.rcwa._engine.rectilinear as rectilinear
        if not solver.lattice_is_axis_aligned:
            raise NotImplementedError("tensor factorization='li' requires an axis-aligned rectangular cell; use 'nvf' for oblique cells")
        solver._reject_unread_normal_method(Factorization.LI)
        solver._reject_unread_projector(Factorization.LI)
        # Tensor Li builds the fixed nesting only: an explicit 'symmetric' is refused by name.
        solver._reject_unread_nesting(Factorization.LI)
        if solver._normal_grid is not None:
            raise NotImplementedError("shape_nvf_grid_size has no reader with tensor factorization='li'")
        if solver._tetm is not None:
            raise ValueError(TETM_REFUSAL)
        if isinstance(solver.structure, list):
            rectilinear.validate(solver.structure, solver.lattice)
            if solver._fourier_coefficients == FourierCoefficients.SAMPLED:
                raise NotImplementedError("tensor Li shapes use fourier_coefficients='integrated'")
        return
    # NVF nests no rules, so any explicit nesting is refused here.
    solver._reject_unread_nesting(Factorization.NVF)
    if solver._projector not in (None, 'linear'):
        raise NotImplementedError("tensor material patterns currently use nvf_projector='linear'")
    if solver._tetm is not None:
        raise ValueError(TETM_REFUSAL)
    patterned = not isinstance(solver.structure, list) or any(l.get('shapes') for l in solver.structure)
    if patterned and not integrated_shapes(solver) and any(_geometry_derivative(v) for v in (solver.lattice.u, solver.lattice.v)):
        raise ValueError('sampled tensor patterns do not support lattice derivatives; keep the cell fixed')
    if patterned and not isinstance(solver.structure,list) and solver._fourier_coefficients not in (None, FourierCoefficients.SAMPLED):
        raise NotImplementedError("tensor material patterns currently require fourier_coefficients='sampled'")
    if isinstance(solver.structure, list) and solver._normal_method is not None:
        raise NotImplementedError('raster_normal_method is only for tensor raster inputs')
    if not isinstance(solver.structure, list):
        solver._reject_unread_normal_grid()


def integrated_shapes(solver):
    from wavebit.rcwa._engine.enums import FourierCoefficients
    return isinstance(solver.structure,list) and solver._fourier_coefficients != FourierCoefficients.SAMPLED


def _settings(solver):
    return (tuple(tuple(v) for v in solver.order_window),
            repr(solver.raster_normal_method))


def _at_floor(solver, t):
    (_, hu), (_, hv) = solver.order_window
    return fac._tensor_at_the_normal_field_floor(solver.xp, t, hu, hv)


def freeze(solver):
    validate_options(solver)
    if solver._factorization == 'li':
        import wavebit.rcwa._engine.tensor_li as tensor_li
        return tensor_li.freeze(solver)
    if integrated_shapes(solver):
        import wavebit.rcwa._engine.tensor_shapes as tensor_shapes
        return tensor_shapes.freeze(solver)
    snapshot = _shape_snapshot(solver) if isinstance(solver.structure, list) and any(l.get('shapes') for l in solver.structure) else None
    record = {'snapshot': snapshot, 'epsilon': [], 'mu': [], 'settings': _settings(solver),
              'basis': fac._to_numpy(solver.lattice).copy()}
    for quantity, structure in [('epsilon',solver.structure), ('mu', solver.mu if not isinstance(solver.structure,list) else solver.structure)]:
        if structure is None: continue
        fields = _field(solver, structure, quantity, snapshot)
        for t in fields:
            uniform = t.shape[:2] == (1,1)
            t = t if uniform else _at_floor(solver, t)
            labels = _labels(t)
            normal = None if uniform else fac.normal_vector_field(
                labels.astype(float), lattice=solver.lattice, method=solver.raster_normal_method)
            record[quantity].append((labels, normal))
    return StaticGeometry(tensor_geometry=record)


def check_layers(solver):
    """The layer refusals :func:`freeze` meets, without the record it takes.

    Each builder refuses a shape layer as it records it, so this makes the call its freeze
    makes, on the same arguments, and keeps nothing: Cartesian Li's plans, which refuse a
    rectangle that leaves the cell; the integrated shapes' layer checks
    (:func:`~wavebit.rcwa._engine.tensor_shapes.check_layers`); and the sampled snapshot,
    unpainted. The sampled masks wrap every shape periodically, so a shape larger than the
    period is sampled like any other there; the snapshot refuses a length that is not positive.
    """
    if not isinstance(solver.structure, list):
        return
    if solver._factorization == 'li':
        import wavebit.rcwa._engine.rectilinear as rectilinear
        for layer in solver.structure:
            rectilinear.plan_layer(layer.get('shapes', []), solver.lattice)
    elif integrated_shapes(solver):
        import wavebit.rcwa._engine.tensor_shapes as tensor_shapes
        tensor_shapes.check_layers(solver)
    elif any(layer.get('shapes') for layer in solver.structure):
        _shape_snapshot(solver, paint=False)


def _sampled_fields(solver, geometry, quantity):
    """Validate the sampled record and return its live fields, without Fourier assembly."""
    if geometry is not None and geometry.tensor_geometry is None:
        raise StaleGeometryError('freeze_geometry() for this tensor structure before replay')
    record = None if geometry is None else geometry.tensor_geometry
    if record is not None and record.get('mode') is not None:
        raise StaleGeometryError('tensor factorization or coefficient mode changed; freeze_geometry() again')
    if record is not None and _settings(solver) != record['settings']:
        raise StaleGeometryError('tensor lattice, orders or normal method changed; freeze_geometry() again')
    shaped = isinstance(solver.structure, list)
    snapshot = None
    if shaped and any(l.get('shapes') for l in solver.structure):
        snapshot = _shape_snapshot(solver, None if record is None else record['basis'])
        if record is not None and (record['snapshot'] is None or snapshot[1] != record['snapshot'][1]):
            raise StaleGeometryError('tensor shape geometry changed; freeze_geometry() again')
    elif record is not None and record['snapshot'] is not None:
        raise StaleGeometryError('tensor shape geometry changed; freeze_geometry() again')
    structure = solver.structure if quantity == 'epsilon' or shaped else solver.mu
    fields = _field(solver, structure, quantity, snapshot)
    if record is not None and len(fields) != len(record[quantity]):
        raise StaleGeometryError('tensor layer count changed; freeze_geometry() again')
    xp, dtype, dev = solver.xp, solver.complex_dtype, solver._array_device
    same_basis = as_array(xp, True, device=dev) if record is None else xp.all(as_array(
        xp, solver.lattice.columns, dtype=solver.float_dtype, device=dev) == as_array(
        xp, record['basis'], dtype=solver.float_dtype, device=dev))
    if _concrete(same_basis) is False:
        raise StaleGeometryError('tensor lattice changed; freeze_geometry() again')
    same_geometry = same_basis
    prepared = []
    for i,t in enumerate(fields):
        single = t.shape[:2] == (1,1)
        t = t if single else _at_floor(solver, t)
        stored = None if record is None else record[quantity][i]
        try:
            labels = _labels(t)
        except (TypeError, ValueError):
            if stored is None and not single:
                raise ValueError('freeze_geometry() before tracing a tensor raster') from None
            labels = np.zeros(t.shape[:2],int) if stored is None else stored[0]
        def interfaces(a):
            return np.stack((a != np.roll(a,1,0), a != np.roll(a,1,1)))
        if stored is not None and (labels.shape != stored[0].shape or
                                   not np.array_equal(interfaces(labels), interfaces(stored[0]))):
            raise StaleGeometryError('tensor material boundaries changed; freeze_geometry() again')
        if stored is not None:
            live_interfaces = xp.stack([xp.any(t != xp.roll(t, 1, axis=a), axis=(-1,-2))
                                         for a in (0,1)])
            same_geometry = same_geometry & xp.all(live_interfaces == as_array(
                xp, interfaces(stored[0]), device=dev))
        prepared.append((t, stored, labels))
    return shaped, structure, prepared, same_geometry


def check_geometry(solver, geometry):
    """Run the builders' record checks before a compiled call hides concrete inputs."""
    validate_options(solver)
    if solver._factorization == 'li':
        import wavebit.rcwa._engine.tensor_li as tensor_li
        tensor_li.check_geometry(solver, geometry)
    elif integrated_shapes(solver):
        import wavebit.rcwa._engine.tensor_shapes as tensor_shapes
        list(tensor_shapes.checked_layers(solver, geometry))
    else:
        _sampled_fields(solver, geometry, 'epsilon')
        if solver._layers_are_magnetic():
            _sampled_fields(solver, geometry, 'mu')


def build(solver, geometry=None, quantity='epsilon'):
    validate_options(solver)
    if solver._factorization == 'li':
        import wavebit.rcwa._engine.tensor_li as tensor_li
        return tensor_li.build(solver, geometry, quantity)
    if integrated_shapes(solver):
        import wavebit.rcwa._engine.tensor_shapes as tensor_shapes
        return tensor_shapes.build(solver,geometry,quantity)
    shaped, structure, prepared, same_geometry = _sampled_fields(solver, geometry, quantity)
    xp, dtype, dev = solver.xp, solver.complex_dtype, solver._array_device
    (_, hu), (_, hv) = solver.order_window
    count = (2*hu+1)*(2*hv+1)
    layers, any_full = [], False
    for i, (t, stored, labels) in enumerate(prepared):
        # A constant-valued multi-pixel raster still has independent pixel
        # derivatives. Collapsing it to t[0,0] would silently discard them.
        uniform = t.shape[:2] == (1,1)
        if uniform:
            value = (as_material(structure[i].get('background',1.)) if shaped else None)
            raw = (value.epsilon if quantity == 'epsilon' else value.mu_value) if shaped else None
            declared_diagonal = shaped and not structure[i].get('shapes') and tensor_shape(raw) in ((),(3,))
            full = not declared_diagonal and full_tensor(xp, t)
            layers.append(_uniform_slots(xp,t[0,0],count,full));any_full |= full
        else:
            normal = stored[1] if stored is not None else fac.normal_vector_field(
                labels.astype(float), lattice=solver.lattice, method=solver.raster_normal_method)
            nx,ny = [as_array(xp,a,dtype=solver.float_dtype,device=dev) for a in normal[:2]]
            slots = list(fac.pn_eps_conv_matrices(xp,t,nx,ny,hu,hv))
            layers.append(slots)
            any_full |= len(slots) == 9
    if any_full:
        for layer in layers:
            if len(layer) == 5: layer.extend([xp.zeros_like(layer[0]) for _ in range(4)])
    cls = TensorConvMatrices if any_full else fac.ConvMatrices
    return cls(*(xp.where(same_geometry, xp.stack([layer[j] for layer in layers]),
                         as_array(xp, complex('nan'), dtype=dtype, device=dev))
                 for j in range(9 if any_full else 5)))
