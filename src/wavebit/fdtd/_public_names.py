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
#: class: a caller's own ``Region`` subclass of ``Shape``, or ``fdtd.periodic.Record``, keeps
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


def public_description(shape):
    """``shape.describe()`` with the public class name, for a message a public caller reads.

    ``describe()`` itself stays the engine's spelling: a flush pair's junction signature records
    it, so it is plan data. Only a spelling that starts with a renamed engine class name changes
    (``Region(size=...)`` reads ``SnappedBox(size=...)``); a slab's ``z=(...)`` names no class.
    """
    text = shape.describe()
    for cls in type(shape).__mro__:
        public = _renamed(cls)
        if public and text.startswith(cls.__name__ + '('):
            return public + text[len(cls.__name__):]
    return text
