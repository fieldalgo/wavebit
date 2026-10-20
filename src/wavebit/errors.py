"""Structured failures shared by solver lifecycle entry points."""
from wavebit._common.backend import eager_under_torch_compile
from dataclasses import dataclass
from typing import Any


@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class Issue:
    code: str
    path: str
    message: str
    expected: Any = None
    actual: Any = None
    severity: str = 'error'


class WavebitError(Exception):
    """The root of every structured Wavebit failure: ``code``, ``path`` and an ``Issue``.

    ``except wavebit.WavebitError`` catches both families below; catch one of them to tell a
    bad input (``ValidationError``) from a valid input the numerics could not answer
    (``NumericalError``).

    ``code`` is a lowercase dotted ``area.item`` name (for example ``numerics.orders``); a
    construction that names none gets its class's ``default_code``.
    """
    default_code = 'input.invalid'

    def __init__(self, message, *, code=None, path='', expected=None, actual=None):
        if code is None:
            code = type(self).default_code
        self.code = code
        self.path = path
        self.message = message
        self.expected = expected
        self.actual = actual
        self.issue = Issue(code=code, path=path, message=message, expected=expected, actual=actual)
        super().__init__(f'{path}: {message}' if path else message)


class ValidationError(WavebitError, ValueError):
    """An input was refused: it is malformed, out of range, or not supported here."""
    default_code = 'input.invalid'


class UnsupportedFeatureError(ValidationError):
    """A meaningful physical or numerical request is unsupported by this path."""
    default_code = 'feature.unsupported'


class IncompatibleProblemError(ValidationError):
    """A live problem violates a prepared calculation's explicit reuse contract."""
    default_code = 'problem.incompatible'


class NumericalError(WavebitError):
    """A valid input whose calculation could not produce valid numerical outputs.

    Not a ``ValidationError``: ``except ValidationError`` meant for a bad input does not also
    swallow a numerical failure on a good one.
    """
    default_code = 'numerical.failure'
