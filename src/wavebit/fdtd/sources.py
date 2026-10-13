"""Physical time-domain sources: radians, seconds, and SI amplitudes."""
from wavebit._common.backend import eager_under_torch_compile
from dataclasses import dataclass
from wavebit._common.declarations import Declaration
from ._structure import tuples

__all__ = ['GaussianPulse', 'PlaneWave', 'Dipole', 'TFSF', 'InitialField']

@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class GaussianPulse(Declaration):
    """Gaussian amplitude spectrum; frequency_sigma_hz is its standard deviation."""
    wavelength: object
    frequency_sigma_hz: object
    cutoff: float = 1e-3

@eager_under_torch_compile
@dataclass(frozen=True, kw_only=True)
class InitialField(Declaration):
    """Initial Yee fields: electric V/m and magnetic A/m, optionally scaled."""
    E: object = None
    H: object = None
    amplitude: object = 1.

@eager_under_torch_compile
@dataclass(frozen=True)
class PlaneWave(Declaration):
    name: str
    plane: object
    pulse: GaussianPulse
    direction: str = '-z'
    theta: float = 0.
    phi: float = 0.
    polarization: str = 's'
    amplitude: object = 1.
    phase: object = 0.
    delay: object = 0.

@eager_under_torch_compile
@dataclass(frozen=True)
class Dipole(Declaration):
    """Point-current moment: electric A*m, or equivalent magnetic V*m.

    This waveform directly drives current; it is not an electric dipole moment
    p(t) in C*m, whose derivative would instead drive the Maxwell equations."""
    name: str
    center: tuple
    pulse: GaussianPulse
    component: str = 'Ey'
    kind: str = 'electric'
    current_moment: object = 1.
    phase: object = 0.
    delay: object = 0.
    stamp: str = 'bspline'

    def __post_init__(self):
        object.__setattr__(self, 'center', tuples(self.center))

@eager_under_torch_compile
@dataclass(frozen=True)
class TFSF(Declaration):
    """Axis-aligned total-field/scattered-field source; electric amplitude V/m."""
    name: str
    box: object
    pulse: GaussianPulse
    direction: str = '-z'
    polarization: str = 'x'
    amplitude: object = 1.
    phase: object = 0.
    delay: object = 0.
