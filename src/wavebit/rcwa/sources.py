"""Monochromatic source declarations in the shared +z-up Cartesian frame."""
from .._common.declarations import renamed_attributes, spec
from ..errors import ValidationError


@spec
class PlaneWave:
    """One incident electric field referenced at the illuminated stack interface.

    amplitude is in V/m. polarization is 's', 'p', or a complex (s,p) Jones
    pair, normalized to unit electric magnitude. Its phase remains meaningful.
    Use either theta/phi (radians) or kx/ky (inverse scene lengths).
    """
    side: str = 'top'
    polarization: object = 's'
    amplitude: object = 1.
    theta: object = None
    phi: object = None
    kx: object = None
    ky: object = None

    def __post_init__(self):
        if self.side not in ('top', 'bottom'):
            raise ValidationError("side must be 'top' or 'bottom'.", code='source.side', path='source.side')
        if (self.theta is not None or self.phi is not None) and (self.kx is not None or self.ky is not None):
            raise ValidationError('Use theta/phi or kx/ky, not both.',
                                  code='source.direction_ambiguous', path='source')
        if isinstance(self.polarization, str) and self.polarization not in ('s', 'p'):
            raise ValidationError("polarization must be 's', 'p', or an (s,p) Jones pair.",
                                  code='source.polarization', path='source.polarization')
        if isinstance(self.polarization, list):
            object.__setattr__(self, 'polarization', tuple(self.polarization))


@spec(renamed={'orders': 'weights'})
class IncidentOrders:
    """Incident diffraction orders, lit coherently around a reference plane-wave direction.

    weights is a tuple of ((m,n), complex_weight) entries: each order's relative
    electric amplitude at x = y = 0 on the lit boundary, the cell-centered reference.
    """
    weights: tuple
    side: str = 'top'
    polarization: object = 's'
    amplitude: object = 1.
    theta: object = None
    phi: object = None
    kx: object = None
    ky: object = None

    def __post_init__(self):
        if isinstance(self.polarization, list):
            object.__setattr__(self, 'polarization', tuple(self.polarization))
        PlaneWave(side=self.side, polarization=self.polarization, amplitude=self.amplitude,
                  theta=self.theta, phi=self.phi, kx=self.kx, ky=self.ky)
        if isinstance(self.weights, dict):
            raise ValidationError('Use tuple (((m,n),weight), ...), not a mutable order mapping.',
                                  code='source.weights', path='source.weights')
        object.__setattr__(self, 'weights', tuple((tuple(k), v) for k, v in self.weights))
        if not self.weights or any(len(k) != 2 or any(type(v) is not int for v in k) for k, _ in self.weights):
            raise ValidationError('Each order needs two integer indices and a weight.',
                                  code='source.weights', path='source.weights')
        if len({k for k, _ in self.weights}) != len(self.weights):
            raise ValidationError('Order labels must be unique.', code='source.weights_duplicate',
                                  path='source.weights')

    __getattr__ = renamed_attributes({'orders': 'weights'}, prefix='rcwa.')


@spec(renamed={'spectral_extent': 'angular_spectrum_extent'})
class GaussianBeam:
    """Propagating Gaussian angular spectrum; waist is the 1/e amplitude radius.

    sampling sets the midpoint-quadrature points on the two axes transverse to the
    central direction; angular_spectrum_extent is the half-width of that quadrature,
    in units of 1/waist.
    """
    waist: object
    focus: tuple = (0., 0., 0.)
    sampling: tuple = (9, 9)
    angular_spectrum_extent: float = 4.
    side: str = 'top'
    polarization: object = 's'
    amplitude: object = 1.
    theta: object = None
    phi: object = None
    kx: object = None
    ky: object = None

    def __post_init__(self):
        if isinstance(self.polarization, list):
            object.__setattr__(self, 'polarization', tuple(self.polarization))
        PlaneWave(side=self.side, polarization=self.polarization, amplitude=self.amplitude,
                  theta=self.theta, phi=self.phi, kx=self.kx, ky=self.ky)
        object.__setattr__(self, 'focus', tuple(self.focus))
        object.__setattr__(self, 'sampling', tuple(self.sampling))
        if len(self.focus) != 3 or len(self.sampling) != 2:
            raise ValidationError('focus needs three components and sampling needs two.',
                                  code='source.dimension', path='source')

    __getattr__ = renamed_attributes({'spectral_extent': 'angular_spectrum_extent'}, prefix='rcwa.')


@spec
class ElectricDipole:
    """A prescribed electric dipole moment in C m, using exp(-i omega t)."""
    position: tuple
    moment: tuple
    repetition: str = 'periodic'
    bloch_k: object = None
    sampling: tuple = (5, 5)

    def __post_init__(self):
        object.__setattr__(self, 'position', tuple(self.position))
        object.__setattr__(self, 'moment', tuple(self.moment))
        object.__setattr__(self, 'sampling', tuple(self.sampling))
        if len(self.position) != 3 or len(self.moment) != 3:
            raise ValidationError('position and moment need three components.',
                                  code='source.dimension', path='source')
        if self.repetition not in ('periodic', 'isolated'):
            raise ValidationError("repetition must be 'periodic' or 'isolated'.",
                                  code='source.repetition', path='source.repetition')
