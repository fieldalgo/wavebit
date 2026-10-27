"""Physical time-domain sources: radians, seconds, and SI amplitudes."""
from wavebit._common.declarations import spec
from ._structure import tuples

__all__ = ['GaussianPulse', 'PlaneWave', 'CurrentDipole', 'TFSF', 'ModeSource', 'GaussianBeam',
           'InitialField']


@spec
class GaussianPulse:
    """A Gaussian pulse, by its centre wavelength and the standard deviation of its spectrum.

    ``fdtd.GaussianPulse(wavelength=0.8, frequency_sigma_hz=2e13)``. ``wavelength`` is the
    vacuum centre wavelength in ``Problem.units``; ``frequency_sigma_hz`` is the standard
    deviation of the amplitude spectrum in Hz (not a fractional bandwidth or an intensity
    FWHM). ``cutoff`` is an amplitude ratio, not a frequency: the envelope starts where it is
    ``cutoff`` of its peak, and the spectrum is treated as ending where it falls to that ratio.
    """
    wavelength: object
    frequency_sigma_hz: object
    cutoff: float = 1e-3


@spec
class InitialField:
    """Initial Yee fields: E in V/m at t=0 and H in A/m half a step earlier, optionally scaled.

    Each of ``E`` and ``H`` is a ``(3, nz, ny, nx)`` array, a per-component tuple (``None``
    for a zero component), or a callable of ``(x, y, z)``. ``amplitude`` scales both and may be
    traced.
    """
    E: object = None
    H: object = None
    amplitude: object = 1.


@spec(positional=('name',), renamed={'delay': 'delay_seconds'})
class PlaneWave:
    """A plane wave launched from one plane across the propagation axis.

    ``fdtd.PlaneWave('light', plane=fdtd.Plane(z=0.6), direction='-z', theta=0.2,
    pulse=fdtd.GaussianPulse(wavelength=0.8, frequency_sigma_hz=2e13))``. ``theta`` and ``phi``
    are radians; ``polarization`` is ``'s'`` or ``'p'``; ``amplitude`` is the electric field in
    V/m; ``phase`` is radians and ``delay_seconds`` delays the pulse. The plane spans the whole
    transverse cell, whose faces across the wave must be :class:`Periodic`, :class:`Bloch` or
    :class:`InvariantAxis` (:meth:`Bloch.from_source` derives them).
    """
    name: str
    plane: object
    pulse: GaussianPulse
    direction: str = '-z'
    theta: float = 0.
    phi: float = 0.
    polarization: str = 's'
    amplitude: object = 1.
    phase: object = 0.
    delay_seconds: object = 0.


@spec(positional=('name',), renamed={'delay': 'delay_seconds'})
class CurrentDipole:
    """A point current source: an electric current moment in A*m, or a magnetic one in V*m.

    ``fdtd.CurrentDipole('d', center=(0., 0., 0.), component='Ey', current_moment=1e-9,
    pulse=fdtd.GaussianPulse(...))``. The waveform drives the current directly; it is not an
    electric dipole moment p(t) in C*m (that is :class:`wavebit.rcwa.ElectricDipole`), whose
    derivative would drive the Maxwell equations instead. ``kind='magnetic'`` drives an
    equivalent magnetic current on an ``H`` component. ``phase`` is radians and
    ``delay_seconds`` delays the pulse.
    """
    name: str
    center: tuple
    pulse: GaussianPulse
    component: str = 'Ey'
    kind: str = 'electric'
    current_moment: object = 1.
    phase: object = 0.
    delay_seconds: object = 0.
    stamp: str = 'bspline'

    def __post_init__(self):
        object.__setattr__(self, 'center', tuples(self.center))


@spec(positional=('name',), renamed={'delay': 'delay_seconds'})
class TFSF:
    """A total-field/scattered-field source: a plane wave injected inside an axis-aligned box.

    ``fdtd.TFSF('t', box=fdtd.SnappedBox(size=(0.4, 0.4, 0.4)), direction='-z',
    polarization='x', pulse=...)``. Inside the box the field is total (incident plus
    scattered); outside it is scattered only. ``polarization`` names the electric field's axis
    (``'x'``, ``'y'`` or ``'z'``), unlike :class:`PlaneWave`'s ``'s'``/``'p'``. ``amplitude`` is
    the electric field in V/m; ``phase`` is radians and ``delay_seconds`` delays the pulse.
    """
    name: str
    box: object
    pulse: GaussianPulse
    direction: str = '-z'
    polarization: str = 'x'
    amplitude: object = 1.
    phase: object = 0.
    delay_seconds: object = 0.


@spec(positional=('name',), renamed={'delay': 'delay_seconds'})
class ModeSource:
    """A waveguide mode launched one way from a plane across the guide.

    ``fdtd.ModeSource('src', plane=fdtd.Plane(z=-2.), direction='+z', mode=0,
    pulse=fdtd.GaussianPulse(wavelength=1.55, frequency_sigma_hz=2e13))``. ``mode`` indexes the
    modes :func:`solve_modes` finds on ``plane`` at the pulse's centre wavelength (0 is the
    largest index); the plane, its window and its refusals are the solver's. ``direction`` is
    ``'+'`` or ``'-'`` the plane's normal axis. ``amplitude`` is in square-root watts (per metre
    of each invariant axis): a continuous wave of this amplitude carries ``amplitude**2`` watts.
    ``phase`` is radians and ``delay_seconds`` delays the pulse.

    The plane is a one-sided total-field/scattered-field sheet whose incident field is the mode:
    at the centre wavelength it launches exactly that mode, exactly one way; across the band it
    launches what a :class:`ModeMonitor` with ``source=`` this source's name reports as
    ``injected``, and leaks what it reports as ``leakage``, both computed exactly for the grid.
    """
    name: str
    plane: object
    pulse: GaussianPulse
    direction: str = '+z'
    mode: int = 0
    amplitude: object = 1.
    phase: object = 0.
    delay_seconds: object = 0.

    def __post_init__(self):
        import numpy as np
        from wavebit.errors import ValidationError
        if (isinstance(self.mode, bool) or not isinstance(self.mode, (int, np.integer))
                or self.mode < 0):
            raise ValidationError('ModeSource(mode=...) is a mode index, an integer >= 0',
                                  code='source.mode', path=f'sources.{self.name}.mode',
                                  actual=self.mode)
        object.__setattr__(self, 'mode', int(self.mode))


@spec(positional=('name',), renamed={'delay': 'delay_seconds'})
class GaussianBeam:
    """A focused Gaussian beam launched one way from a plane.

    ``fdtd.GaussianBeam('b', plane=fdtd.Plane(z=-4.), direction='+z', waist=1.5,
    focus=(0., 0., 0.), polarization='y', pulse=fdtd.GaussianPulse(wavelength=1., ...))``. On the
    focal plane the E component along ``polarization`` (a transverse axis) is
    ``exp(-r**2 / waist**2)`` about ``focus`` (``waist`` is the 1/e amplitude radius, as for
    ``rcwa.GaussianBeam``); the other transverse component is zero there. The beam is the
    propagating part of that plane's angular spectrum, built from the grid's own plane waves in
    the homogeneous medium that fills the source plane, so ``focus`` is where it would focus if
    that medium filled space. ``amplitude`` is in square-root watts (per metre of each invariant
    axis): a continuous wave of this amplitude carries ``amplitude**2`` watts. ``phase`` is in
    radians and ``delay_seconds`` delays the pulse.

    The plane is a one-sided total-field/scattered-field sheet. At the pulse's centre wavelength
    it launches exactly the beam, one way; elsewhere in the band a little goes backward. The
    faces across the plane are ``Open`` (or ``InvariantAxis``), and the cross-section must hold
    the beam. ``waist`` and ``focus`` are concrete; ``amplitude``, ``phase`` and
    ``delay_seconds`` may be traced.
    """
    name: str
    plane: object
    pulse: GaussianPulse
    waist: object
    focus: tuple = (0., 0., 0.)
    polarization: str = 'x'
    direction: str = '+z'
    amplitude: object = 1.
    phase: object = 0.
    delay_seconds: object = 0.

    def __post_init__(self):
        if isinstance(self.focus, list):
            object.__setattr__(self, 'focus', tuple(self.focus))
