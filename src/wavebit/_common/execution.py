"""Execution vocabulary and structured validation reports, without eager backends."""
from wavebit._common.backend import eager_under_torch_compile
from dataclasses import dataclass, field
from contextlib import contextmanager
from types import MappingProxyType
from .declarations import Declaration
from .enums import backend_of
from wavebit.errors import ValidationError, UnsupportedFeatureError


def frozen_mapping(value):
    return MappingProxyType(dict(value))


@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class Execution(Declaration):
    backend: str = 'numpy'
    device: str = 'cpu'
    precision: str = 'double'

    @contextmanager
    def _scope(self):
        """Prevent legacy implementation doors from mutating global JAX precision.

        Private: every solver entry point opens it itself.
        """
        self.validate()
        from .backend import ALLOW_LEGACY_PRECISION_LIFT
        token = ALLOW_LEGACY_PRECISION_LIFT.set(False)
        try:
            yield self
        finally:
            ALLOW_LEGACY_PRECISION_LIFT.reset(token)

    def __getattr__(self, name):
        # Reached only when normal lookup fails, so it never shadows a field or method.
        if name == 'scope':
            from .declarations import removed
            raise removed('Execution', name, 'private', 'run, check and prepare open it themselves')
        raise AttributeError(f'{type(self).__name__!r} object has no attribute {name!r}')

    def validate(self):
        try:
            backend = backend_of(self.backend).value
        except ValueError as exc:
            raise ValidationError(str(exc), code='execution.backend', path='execution.backend') from exc
        if self.precision not in ('single', 'double'):
            raise ValidationError("precision must be 'single' or 'double'", code='execution.precision', path='execution.precision')
        if not isinstance(self.device, str):
            raise ValidationError('device must be an explicit string', code='execution.device', path='execution.device')
        if backend == 'numpy' and self.device != 'cpu':
            raise UnsupportedFeatureError('NumPy execution requires cpu', code='execution.device', path='execution.device', actual=self.device)
        if backend == 'jax':
            try:
                import jax
            except ImportError as exc:
                raise UnsupportedFeatureError('install the jax extra to use this backend', code='execution.backend_unavailable', path='execution.backend') from exc
            if self.precision == 'double' and not jax.config.x64_enabled:
                raise UnsupportedFeatureError('JAX double precision requires jax.config.update("jax_enable_x64", True) before creating arrays', code='execution.jax_x64', path='execution.precision')
            platform, sep, index = self.device.partition(':')
            if platform not in ('cpu', 'gpu', 'tpu') or (sep and not index.isdigit()):
                raise UnsupportedFeatureError('JAX device must be cpu, gpu, or tpu, optionally followed by :index', code='execution.device', path='execution.device')
            try:
                devices = jax.devices(platform)
            except RuntimeError as exc:
                raise UnsupportedFeatureError(f'JAX {platform} device is unavailable', code='execution.device_unavailable', path='execution.device') from exc
            if not devices or int(index or 0) >= len(devices):
                raise UnsupportedFeatureError('requested JAX device is unavailable', code='execution.device_unavailable', path='execution.device')
        if backend == 'torch':
            try:
                import torch
                dev = torch.device(self.device)
            except ImportError as exc:
                raise UnsupportedFeatureError('install the pytorch extra to use this backend', code='execution.backend_unavailable', path='execution.backend') from exc
            except (RuntimeError, ValueError) as exc:
                raise ValidationError('invalid Torch device', code='execution.device', path='execution.device') from exc
            if dev.type not in ('cpu', 'cuda'):
                raise UnsupportedFeatureError('supported Torch devices are cpu and cuda', code='execution.device', path='execution.device')
            if dev.type == 'cuda' and (not torch.cuda.is_available() or (dev.index or 0) >= torch.cuda.device_count()):
                raise UnsupportedFeatureError('requested CUDA device is unavailable', code='execution.device_unavailable', path='execution.device')
        return self


@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class CheckReport(Declaration):
    issues: tuple = ()
    estimates: object = field(default_factory=dict)
    capabilities: object = field(default_factory=dict)
    resolved: object = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, 'issues', tuple(self.issues))
        for name in ('estimates', 'capabilities', 'resolved'):
            object.__setattr__(self, name, frozen_mapping(getattr(self, name)))

    @property
    def ok(self):
        return not any(getattr(i, 'severity', 'error') == 'error' for i in self.issues)

    def raise_if_invalid(self, error_type=ValidationError):
        for issue in self.issues:
            if getattr(issue, 'severity', 'error') == 'error':
                raise error_type(issue.message, code=issue.code, path=issue.path, expected=issue.expected, actual=issue.actual)
        return self
