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


class ValidationError(ValueError):
    def __init__(self, message, *, code='invalid_input', path='', expected=None, actual=None):
        self.code = code
        self.path = path
        self.message = message
        self.expected = expected
        self.actual = actual
        self.issue = Issue(code=code, path=path, message=message, expected=expected, actual=actual)
        super().__init__(f'{path}: {message}' if path else message)


class UnsupportedFeatureError(ValidationError):
    """A meaningful physical or numerical request is unsupported by this path."""


class IncompatibleProblemError(ValidationError):
    """A live problem violates a prepared calculation's explicit reuse contract."""


class NumericalError(ValidationError):
    """A requested calculation could not produce valid numerical outputs."""
