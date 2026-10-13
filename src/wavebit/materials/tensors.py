"""Constitutive tensor inspection and rotation."""
import numpy as np
from wavebit._common.backend import array_namespace, as_array, _concrete
from wavebit.materials.response import Material, as_material

def tensor_shape(value):
    shape = getattr(value, 'shape', None)
    if shape is None and isinstance(value, (list, tuple)):
        shape = (len(value),) + (tensor_shape(value[0]) if value else ())
    return tuple(shape or ())


def material_is_tensor(material):
    if isinstance(material, Material) and material._wavelength_shape is not None:
        return False
    m = as_material(material)
    return tensor_shape(m.epsilon) in ((3,), (3, 3)) or tensor_shape(m.mu) in ((3,), (3, 3))


def as_tensor(xp, value, dtype, device=None):
    def lift(v):
        # torch.asarray([live_scalar, ...]) copies the values and detaches them.
        # Stack tensor components so a natural list spelling retains derivatives.
        if isinstance(v, (list, tuple)):
            return xp.stack([lift(item) for item in v])
        return as_array(xp, v, dtype=dtype, device=device)
    a = lift(value)
    if _concrete(xp.all(xp.isfinite(a))) is False:
        raise ValueError('material tensors must contain finite values')
    eye = xp.eye(3, dtype=dtype, device=device)
    if a.ndim == 0:
        return a * eye
    if a.shape == (3,):
        return a[None, :] * eye
    if a.shape != (3, 3):
        raise ValueError('a region material must be scalar, three principal values, or a 3x3 tensor')
    return a


def rotated_material(material, angle, axis):
    """Active right-handed rotation in radians, preserving backend derivatives."""
    if axis not in ('x', 'y', 'z'):
        raise ValueError("rotation axis must be 'x', 'y', or 'z'")
    if material._wavelength_shape is not None:
        raise ValueError('rotate the material table before evaluating a wavelength array')
    values = (angle, material.epsilon, material.mu)
    def leaves(v):
        if isinstance(v, (list, tuple)):
            return [leaf for item in v for leaf in leaves(item)]
        return [v]
    arrays = [v for v in leaves(values) if hasattr(v, 'dtype') and hasattr(v, 'shape')]
    # Prefer the live backend over host constants.
    live = [v for v in arrays if not isinstance(v, np.ndarray)]
    xp = array_namespace(*(live or arrays or [np.asarray(0.)]))
    ref = (live or arrays or [None])[0]
    dev = getattr(ref, 'device', None)
    dtype = xp.complex64 if str(getattr(ref, 'dtype', '')) in ('float32', 'complex64', 'torch.float32', 'torch.complex64') else xp.complex128
    a = as_array(xp, angle, device=dev)
    if a.ndim != 0 or xp.isdtype(a.dtype, 'complex floating'):
        raise ValueError('rotation angle must be a real scalar in radians')
    a = xp.astype(a, xp.float32 if dtype == xp.complex64 else xp.float64)
    if _concrete(xp.isfinite(a)) is False:
        raise ValueError('rotation angle must be finite')
    c, s = xp.cos(a), xp.sin(a)
    z, o = c * 0, c * 0 + 1
    rows = {'x': ((o,z,z),(z,c,-s),(z,s,c)),
            'y': ((c,z,s),(z,o,z),(-s,z,c)),
            'z': ((c,-s,z),(s,c,z),(z,z,o))}[axis]
    R = xp.astype(xp.stack([xp.stack(r) for r in rows]), dtype)
    def rotate(v):
        if tensor_shape(v) == ():
            return v
        t = as_tensor(xp, v, dtype, dev)
        return R @ t @ xp.permute_dims(R, (1, 0))
    return Material(rotate(material.epsilon), None if material.mu is None else rotate(material.mu))
