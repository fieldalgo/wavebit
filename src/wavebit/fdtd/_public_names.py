"""The public spelling of each engine declaration whose public class has another name.

The engine keeps its own class names: a static plan records a declaration by its class name
(``_prepare/plan.py``), so renaming an engine class would move every plan skeleton and the
frozen references. A message that names a declaration it formats from an engine object goes
through :func:`public_name` instead, so a public caller reads the name they wrote.
"""

#: Engine class name -> public class name, for every declaration renamed in the public API.
PUBLIC_NAMES = {
    'Record': 'TimeSeriesMonitor', 'Field': 'FieldMonitor', 'Flux': 'FluxMonitor',
    'Diffraction': 'DiffractionMonitor', 'FarField': 'FarFieldMonitor',
    'Region': 'SnappedBox', 'Slab': 'SnappedSlab', 'Dipole': 'CurrentDipole',
    'Axis1D': 'InvariantAxis', 'Symmetry': 'MirrorSymmetry',
    'Absorber': 'ConductiveAbsorber', 'Decay': 'DecayStop',
}


#: Where each renamed engine class is defined. A class is relabelled only if it IS that engine
#: class: a caller's own ``Region`` subclass of ``Shape``, or ``fdtd.periodic.Snapshots``, keeps
#: its name.
_ENGINE_MODULES = {
    'Record': 'monitors', 'Field': 'monitors', 'Flux': 'monitors', 'Diffraction': 'monitors',
    'FarField': 'monitors', 'Decay': 'monitors', 'Region': 'geometry', 'Slab': 'geometry',
    'Dipole': 'sources', 'Axis1D': 'boundaries', 'Symmetry': 'boundaries', 'Absorber': 'boundaries',
}


def _renamed(cls):
    """The public name of ``cls`` if it is one of the renamed engine classes, else ``None``."""
    home = _ENGINE_MODULES.get(cls.__name__)
    if home and cls.__module__ == f'wavebit.fdtd._legacy_{home}':
        return PUBLIC_NAMES[cls.__name__]
    return None


def public_name(declaration):
    """The public class name of ``declaration`` (a class or an instance).

    A renamed engine class returns the public spelling; any other class, a public subclass or
    a caller's own class included, returns its own name.
    """
    cls = declaration if isinstance(declaration, type) else type(declaration)
    return _renamed(cls) or cls.__name__


#: ``id`` of an engine shape the public lowering extended -> the shape before the extension.
#: An entry lives exactly as long as the extended shape (:func:`remember_given`).
_GIVEN = {}


def remember_given(extended, given):
    """Let :func:`public_description` describe ``extended`` as ``given``.

    A ``Box`` whose ``Structure`` declares ``continuation`` is lowered with its continued faces
    moved through the padding (``_lower_public._continue_box``), so its own ``describe()``
    prints a size and centre the caller never wrote. The table is keyed by identity and the
    entry is dropped when ``extended`` is collected, so it neither touches the shape -- its
    fields and ``describe()`` are plan data -- nor answers for an equal shape the caller gave.
    """
    if extended is given:
        return
    import weakref
    key = id(extended)
    _GIVEN[key] = given
    weakref.finalize(extended, _GIVEN.pop, key, None)


#: ``id`` of an engine shape the public lowering continued -> the faces its Structure named.
_CONTINUED = {}


def remember_continued(shape, faces):
    """Record that a public ``Structure`` declared ``shape`` to run on through ``faces``.

    The declaration is the caller's, and the lowering checked it (``_lower_public``): a
    substrate ``Box``, a cell-snapped core or a ``SnappedSlab`` running through those faces'
    absorbing layers. A note that measures the gap between the body and such a face's layer
    reads :func:`declared_continuation`. Keyed and dropped as :func:`remember_given` is.
    """
    import weakref
    key = id(shape)
    _CONTINUED[key] = tuple(faces)
    weakref.finalize(shape, _CONTINUED.pop, key, None)


def declared_continuation(shape):
    """The faces a public ``Structure`` declared ``shape`` runs through, or ``()``."""
    return _CONTINUED.get(id(shape), ())


def public_description(shape):
    """``shape.describe()`` with the public class name, for a message a public caller reads.

    ``describe()`` itself stays the engine's spelling: a flush pair's junction signature records
    it, so it is plan data. Only a spelling that starts with a renamed engine class name changes
    (``Region(size=...)`` reads ``SnappedBox(size=...)``); a slab's ``z=(...)`` names no class.
    A turned Box or Prism holds its ``angle`` in the engine's degrees; inside a public call it
    reads in the radians the caller gave (``_spelling.angle``), as a PlaneWave's angles do.
    A continued substrate reads at the size and centre the caller gave (:func:`remember_given`).
    """
    shape = _GIVEN.get(id(shape), shape)
    text = shape.describe()
    turned = getattr(shape, 'angle', None)
    if turned and type(shape).__module__ == 'wavebit.fdtd._legacy_geometry':
        from wavebit.fdtd._spelling import angle, angle_unit
        if angle_unit() == 'radians':
            text = text.replace(f', angle={turned}', f', angle={angle(turned)}', 1)
    for cls in type(shape).__mro__:
        public = _renamed(cls)
        if public and text.startswith(cls.__name__ + '('):
            return public + text[len(cls.__name__):]
    return text


# The tombstone helpers live with the other declaration refusals; fdtd re-exports them.
from wavebit._common.declarations import Answered, removed  # noqa: E402,F401
