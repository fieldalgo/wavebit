"""Shared validation primitives."""
from wavebit.materials import tensors as _anisotropy
from wavebit._common.names import near_match_hint

def _refuse_shaped_half_space(material, where):
    """A half-space is one uniform material, so its numbers must be scalars.

    Checked here rather than in :func:`~wavebit.materials.response.as_material`, which is
    deliberately permissive: it cannot tell a vmapped tracer from a raster grid, and refusing
    both to catch one would break the design-variable case (``tests/materials/test_material.py::
    test_arrays_and_tracers_are_not_refused``). This slot can be precise where that one cannot,
    because *half-space* is a stronger claim than *material* -- and because the test is a
    **shape**, never a value. A shape is static under ``jit``, ``grad`` and ``vmap`` alike, and
    a vmapped scalar reads ``()`` inside the trace, so nothing traced is caught by this.

    Without it a shaped half-space travelled as far as ``_separable_kxy``, where it met the
    order range and raised ``operands could not be broadcast together with shapes (1, 8) and
    (5,)`` -- loud, but naming neither the argument at fault nor what a half-space is.
    """
    for name, value in (('epsilon', material.epsilon), ('mu', material.mu)):
        shape = _anisotropy.tensor_shape(value)
        size = 1
        for dim in shape:
            size *= int(dim)
        if size > 1:
            raise ValueError(
                f'`{where}` is a half-space, so its {name} must be a single value, but it has '
                f'shape {shape}. A half-space is semi-infinite and homogeneous: a patterned or '
                'graded medium is a *layer*, which goes in `structure` with a thickness of its '
                'own. An anisotropic half-space is not supported either -- its modes are the '
                'plane waves that `sqrt(eps*mu)` alone defines.')


def _unknown_attribute(owner, name, known, kind='conditions'):
    """The refusal of an assignment to a public name nothing reads, as an exception to raise.

    One function because there are three attribute doors and they must refuse the same
    mistake the same way: the solver's own ``__setattr__`` (``CoreFactorizationMixin``), the
    source-excitation facade's (``illumination.CoherentSolver``), which forwards the scene's
    conditions and used to store every other public name on itself in silence --
    ``s.thicknes = [900.0]`` there was accepted and changed nothing, when the solver had
    refused it since 2026-08-27 -- and the spec dataclasses' (``domain._DeclaredFieldsOnly``),
    where ``inc.thet = 0.2`` was stored and ``as_kwargs`` still wrote ``theta=0``. ``owner`` is
    the class name the caller assigned on; ``known`` is what that class can set, and the
    near-match hint (:func:`~wavebit._common.names.near_match_hint`) is drawn from it alone.
    ``kind`` names what ``known`` holds: a solver's conditions, a dataclass's fields.
    """
    known = sorted(known)
    hint = near_match_hint([name], known)
    return AttributeError(
        f'{owner} has no attribute {name!r} to set.{hint} Refused rather '
        f'than stored, because storing it is silent: the solve reads the attribute it '
        f'knows about, so the answer comes back for the value you meant to change and '
        f'nothing says otherwise.\n'
        f'  Settable {kind}: {", ".join(known)}.\n'
        f'  To attach state of your own, use a name starting with an underscore, or '
        f'declare it as a class attribute on a subclass.')
