"""The monitors a run records, and the container that requests them.

Every monitor is named, and its result is read back as ``result[name]``. The four spectral
monitors keep the raw pulse Fourier transform; see the reference for each result's units and
normalization.
"""
from wavebit._common.backend import eager_under_torch_compile
from wavebit._common.declarations import spec
from . import _legacy_monitors
from ._spelling import Spoken
from ._structure import tuples

__all__ = ['TimeSeriesMonitor', 'FieldMonitor', 'FluxMonitor', 'DiffractionMonitor',
           'FarFieldMonitor', 'Outputs']


@eager_under_torch_compile
class TimeSeriesMonitor(Spoken, _legacy_monitors.Record):
    """A monitor that records one field component at one grid point, every time step.

    ``fdtd.TimeSeriesMonitor('probe', at=(x, y, z), component='Ey')`` is read back as
    ``result['probe']``, a :class:`wavebit.TimeSeries` with its time axis in ``times_seconds`` and
    values in V/m (A/m for an H component). ``at`` is a static point in ``Problem.units`` inside the physical domain; the
    sample is the nearest Yee point of that component, not an interpolation. ``component`` is
    one of ``'Ex'``, ``'Ey'``, ``'Ez'``, ``'Hx'``, ``'Hy'``, ``'Hz'``.
    """


@spec(positional=('name',))
class FieldMonitor:
    """A monitor that records Fourier-transformed E and H on a plane or in a box, at chosen wavelengths.

    ``fdtd.FieldMonitor('f', plane=fdtd.Plane(z=0.2), wavelengths=(0.8,), components=('Ex', 'Hy'))``
    is read back as ``result['f']``, a :class:`wavebit.RawFieldSpectrum`: the raw pulse
    transform of each component on its own Yee points, in V*s/m and A*s/m. Give exactly one of
    ``plane`` (an :class:`fdtd.Plane`, optionally with transverse spans) or ``box`` (an
    :class:`fdtd.SnappedBox`, whose volume is recorded). ``wavelengths`` are vacuum wavelengths
    in ``Problem.units``.
    """
    name: str
    wavelengths: tuple
    plane: object = None
    box: object = None
    components: tuple = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')

    def __post_init__(self):
        for name in ('wavelengths', 'components'):
            object.__setattr__(self, name, tuples(getattr(self, name)))


@spec(positional=('name',))
class FluxMonitor:
    """A monitor that records the spectral energy crossing a plane, or leaving a closed box.

    ``fdtd.FluxMonitor('t', plane=fdtd.Plane(z=-0.3), wavelengths=(0.75, 0.8))`` is read back as
    ``result['t']``, an :class:`fdtd.SpectralFlux`. Its ``spectral_energy`` is the two-sided pulse
    spectral energy in J/Hz (per metre or per square metre with invariant axes), along the
    plane's positive normal or out of the box; it is not a monochromatic power. Divide by the
    same quantity from a reference run to get a transmission. Give exactly one of ``plane`` or
    ``box`` (an :class:`fdtd.SnappedBox`); a box also reports each face in ``faces``.
    """
    name: str
    wavelengths: tuple
    plane: object = None
    box: object = None

    def __post_init__(self):
        object.__setattr__(self, 'wavelengths', tuples(self.wavelengths))


#: The two waves a diffraction monitor can record, by their role relative to the source.
WAVES = ('reflected', 'transmitted')


@spec(positional=('name',), renamed={'side': 'wave'})
class DiffractionMonitor:
    """A monitor that records the diffraction orders leaving a periodic cell through one plane.

    ``fdtd.DiffractionMonitor('t', plane=fdtd.Plane(z=-0.3), wavelengths=(0.8,),
    source='illumination', wave='transmitted')`` is read back as ``result['t']``, a
    :class:`wavebit.DiffractionOrders` normalized to the named :class:`fdtd.PlaneWave` (the only
    source in the problem). ``wave`` is ``'reflected'`` (leaving on the incidence side) or
    ``'transmitted'`` (leaving on the far side). The plane spans the whole transverse cell.
    ``orders`` is the half-width window; ``reference_plane`` and ``incident_plane`` refer the
    amplitudes to other planes along the propagation axis. ``raw=True`` returns the
    un-normalized :class:`wavebit.RawDiffractionSpectrum` instead.
    """
    name: str
    plane: object
    wavelengths: tuple
    source: str
    wave: str
    orders: object = None
    reference_plane: object = None
    incident_plane: object = None
    raw: bool = False

    def __post_init__(self):
        for name in ('wavelengths', 'orders'):
            object.__setattr__(self, name, tuples(getattr(self, name)))
        if self.wave not in WAVES:
            hint = ({'reflect': 'reflected', 'transmit': 'transmitted'}.get(self.wave)
                    if isinstance(self.wave, str) else None)
            raise ValueError(
                f'DiffractionMonitor(wave={self.wave!r}): {WAVES[0]!r} (leaving on the incidence '
                f'side) or {WAVES[1]!r} (leaving on the far side).'
                + (f' The earlier spelling {self.wave!r} is now {hint!r}.' if hint else ''))


@spec(positional=('name',))
class FarFieldMonitor:
    """A monitor that records the far-field radiation of an open scene from the fields on a closed box.

    ``fdtd.FarFieldMonitor('ff', box=fdtd.SnappedBox(size=(0.6, 0.6, 0.6)), wavelengths=(0.8,))``
    is read back as ``result['ff']``, an :class:`fdtd.FarFieldSpectrum` whose
    ``intensity(theta, phi)`` is the two-sided pulse spectral energy per solid angle (angles in
    radians). Every face must be ``Open``, the box must sit in a uniform background, and no part
    of the scene may lie outside it.
    ``propagation='continuum'`` (the default) uses the continuum Green function;
    ``'yee'`` uses the grid's own dispersion.
    """
    name: str
    wavelengths: tuple
    box: object
    propagation: str = 'continuum'

    def __post_init__(self):
        object.__setattr__(self, 'wavelengths', tuples(self.wavelengths))


@spec
class Outputs:
    """What a run returns: named monitors, and optionally the final E and H fields.

    ``fdtd.Outputs(monitors=(fdtd.FluxMonitor('t', ...),), final_fields=True)``. Monitor names
    must be unique; each is the key its result is read back by.
    """
    monitors: tuple = ()
    final_fields: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'monitors', tuple(self.monitors))
