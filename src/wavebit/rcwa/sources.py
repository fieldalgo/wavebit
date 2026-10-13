"""Monochromatic source declarations in the shared +z-up Cartesian frame."""
from .._common.declarations import spec
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


@spec
class OrderExcitation:
    """Coherent incident orders around a reference plane-wave direction.

    orders is a tuple of ((m,n), complex_weight) entries. Values are relative
    electric amplitudes at the cell-centered reference plane.
    """
    orders: tuple
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
        if isinstance(self.orders, dict):
            raise ValidationError('Use tuple (((m,n),weight), ...), not a mutable order mapping.',
                                  code='source.orders', path='source.orders')
        object.__setattr__(self, 'orders', tuple((tuple(k), v) for k, v in self.orders))
        if not self.orders or any(len(k) != 2 or any(type(v) is not int for v in k) for k, _ in self.orders):
            raise ValidationError('Each order needs two integer indices and a weight.',
                                  code='source.orders', path='source.orders')
        if len({k for k, _ in self.orders}) != len(self.orders):
            raise ValidationError('Order labels must be unique.', code='source.orders_duplicate', path='source.orders')


@spec
class GaussianBeam:
    """Propagating Gaussian angular spectrum; waist is the 1/e amplitude radius."""
    waist: object
    focus: tuple = (0., 0., 0.)
    sampling: tuple = (9, 9)
    spectral_extent: float = 4.
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
