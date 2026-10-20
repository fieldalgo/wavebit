"""Declarations shared by the explicit-grid modules ``fdtd.native`` and ``fdtd.periodic``.

Each module keeps its own public classes (``native.Snapshots`` takes integer ``slices=``,
``periodic.Snapshots`` physical ``bounds=``); the rules they share live here once, so the two
check a component name, a snapshot interval and a decay window the same way.
"""
import math
from contextlib import contextmanager

import numpy as np

from wavebit.errors import UnsupportedFeatureError, ValidationError, WavebitError

#: The public component names, in the engine's order. The engine spells them 'ex'..'hz'.
COMPONENTS = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')


def snapshot_every(value, path):
    """The snapshot interval: a positive whole number of time steps."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValidationError(f'every={value!r}: a positive whole number of time steps is needed',
                              code='snapshots.every', path=path)
    return int(value)


def snapshot_components(value, path):
    """The recorded components as a tuple of 'Ex'..'Hz'; a single name or 'all' is accepted."""
    if isinstance(value, str):
        value = COMPONENTS if value == 'all' else (value,)
    try:
        names = tuple(value)
    except TypeError:
        raise ValidationError(f'components={value!r}: give component names from {list(COMPONENTS)}',
                              code='snapshots.components', path=path) from None
    lower = [n for n in names if isinstance(n, str) and n.capitalize() in COMPONENTS and n not in COMPONENTS]
    if lower:
        raise ValidationError(f'components={names!r}: components are spelled {list(COMPONENTS)}, '
                              f'as every other Wavebit output spells them (for example '
                              f'{lower[0].capitalize()!r}, not {lower[0]!r})', code='snapshots.components', path=path)
    if not names or len(set(names)) != len(names) or any(n not in COMPONENTS for n in names):
        raise ValidationError(f'components={names!r}: name unique components from '
                              f'{list(COMPONENTS)}', code='snapshots.components', path=path)
    return names


def engine_components(names):
    """The engine's spelling of public component names."""
    return tuple(n.lower() for n in names)


def decay_check_fields(threshold, window_seconds, path):
    """A decay check's ``threshold`` (a ratio in (0, 1)) and ``window_seconds`` (positive)."""
    def real(value, name):
        if isinstance(value, (bool, np.bool_)) or np.ndim(value) or np.iscomplexobj(value):
            raise ValidationError(f'DecayCheck({name}={value!r}): a real number is needed',
                                  code=f'decay_check.{name}', path=f'{path}.{name}')
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValidationError(f'DecayCheck({name}={value!r}): a real number is needed',
                                  code=f'decay_check.{name}', path=f'{path}.{name}') from None
    threshold = real(threshold, 'threshold')
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValidationError(f'DecayCheck(threshold={threshold!r}): a finite ratio between zero '
                              f'and one is needed', code='decay_check.threshold', path=f'{path}.threshold')
    window = real(window_seconds, 'window_seconds')
    if not math.isfinite(window) or window <= 0:
        raise ValidationError(f'DecayCheck(window_seconds={window!r}): a finite positive duration '
                              f'in seconds is needed', code='decay_check.window_seconds', path=f'{path}.window_seconds')
    return threshold, window


def engine_message(exc, spelling, area):
    """``exc`` reworded for a public caller, or ``exc`` itself if it names no engine keyword.

    ``spelling`` pairs an engine keyword with the public declaration that set it. The engine
    keeps its own wording, which its own callers and tests read; only the public door rewords.
    A reworded refusal carries the module's catch-all code, ``f'{area}.invalid'``.
    """
    text = str(exc)
    public = text
    for engine, declared in spelling:
        public = public.replace(engine, declared)
    if public == text:
        return exc
    return ValidationError(public, code=getattr(exc, 'code', None) or f'{area}.invalid',
                           path=getattr(exc, 'path', '') or 'problem')


def public_refusal(exc, area):
    """``exc`` as a structured refusal of module ``area`` (``'native'`` or ``'periodic'``).

    A ``WavebitError`` is returned as it is. Any other engine refusal keeps its message and gains
    the module's catch-all code: ``f'{area}.unsupported'`` (an ``UnsupportedFeatureError``) for a
    ``NotImplementedError``, ``f'{area}.invalid'`` (a ``ValidationError``) otherwise.
    """
    if isinstance(exc, WavebitError):
        return exc
    path = getattr(exc, 'path', '') or 'problem'
    if isinstance(exc, NotImplementedError):
        return UnsupportedFeatureError(str(exc), code=f'{area}.unsupported', path=path)
    return ValidationError(str(exc), code=f'{area}.invalid', path=path)


@contextmanager
def refusals(area):
    """Raise every ``TypeError``/``ValueError``/``NotImplementedError`` as ``public_refusal``.

    ``check`` reports these as issues; ``prepare``, ``run`` and ``Prepared.run`` raise them, so
    every public refusal of a module carries a code, whichever entry point found it.
    """
    try:
        yield
    except (TypeError, ValueError, NotImplementedError) as exc:
        public = public_refusal(exc, area)
        if public is exc:
            raise
        raise public from exc
