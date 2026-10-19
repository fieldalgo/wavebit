"""The module path a plan signature records for a declaration's class.

Signatures record ``type(value).__module__`` so that two classes with one name do not collide.
The material modules moved to private paths on 2026-10-19 (#57); a signature still records the
path they had, so a moved module changes no plan skeleton.
"""

#: Current private module -> the path signatures recorded before the move.
_RECORDED_AS = {
    'wavebit.materials._time_domain': 'wavebit.materials.time_domain',
    'wavebit.materials._tabulated': 'wavebit.materials.tabulated',
}


def recorded_module(cls):
    """The module path a plan signature records for ``cls``."""
    return _RECORDED_AS.get(cls.__module__, cls.__module__)
