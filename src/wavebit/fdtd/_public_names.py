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


def public_name(declaration):
    """The public class name of ``declaration`` (a class or an instance).

    An engine class whose public twin is spelled differently returns the public spelling; any
    other class, a public subclass included, returns its own name.
    """
    cls = declaration if isinstance(declaration, type) else type(declaration)
    return PUBLIC_NAMES.get(cls.__name__, cls.__name__)


def public_description(shape):
    """``shape.describe()`` with the public class name, for a message a public caller reads.

    ``describe()`` itself stays the engine's spelling: a flush pair's junction signature records
    it, so it is plan data. Only a spelling that starts with a renamed engine class name changes
    (``Region(size=...)`` reads ``SnappedBox(size=...)``); a slab's ``z=(...)`` names no class.
    """
    text = shape.describe()
    for cls in type(shape).__mro__:
        public = PUBLIC_NAMES.get(cls.__name__)
        if public and text.startswith(cls.__name__ + '('):
            return public + text[len(cls.__name__):]
    return text
