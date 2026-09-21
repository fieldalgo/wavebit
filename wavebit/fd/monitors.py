"""What a run records: a point time series (stage 0) and the frequency-domain monitors (stage 3).

``Record(name, at=(x, y, z), component='Ey')`` reads one component at the Yee point of that
component nearest ``at``, once per step. It reads at the time level the component lives on:
an E component after the E update of step ``n`` (time ``(n + 1) dt``), an H component after
the H update (time ``(n + 1/2) dt``). The series carries those times, so a probe on ``Hz``
is not mistaken for a sample at the E times.

**Stage 3 adds four frequency-domain monitors**, every one of which lowers to the kernel's
one frequency-domain observer, ``DFTRegion`` (``wavebit/fd/_kernel/observe.py``): a running
DFT of named components over a region at a static list of frequencies. Everything a monitor
returns beyond those accumulators -- a flux, an order's amplitude, a normalisation -- is
post-processing in the backend's array module (``wavebit/fd/_post.py``), so it differentiates.

    ws = (0.5, 0.55, 0.6)                                        # wavelengths, the user's unit
    fd.Flux('out', box=fd.Box(size=(0.5,)*3), wavelengths=ws)    # closed box: net outward flux
    fd.Flux('t', plane=fd.Plane(z=-1.0), wavelengths=ws)         # a plane: flux along +axis
    fd.Field('f', plane=fd.Plane(z=0.2), components=('Ex', 'Ey'), wavelengths=ws)
    fd.Diffraction('r', z=1.2, side='reflect', wavelengths=ws)   # orders in a periodic cell
    fd.FarField('ff', box=fd.Box(size=(0.5,)*3), wavelengths=ws) # near-to-far, open scenes

``wavelengths`` are vacuum wavelengths in the scene's length unit (``f = 1/wavelength``, ``c =
1``); they are static, because the accumulator's shape and the phase each step multiplies by
are fixed by them.

Observers only read. They never touch the fields, which is what lets the kernel step apply
them last and lets the JAX scan carry their accumulators as ordinary state.
"""
import math

from ._spec import spec, static_float, static_int

__all__ = ['Record', 'Flux', 'Field', 'Diffraction', 'FarField', 'COMPONENTS', 'MONITOR_TYPES',
           'ModeMonitor', 'Decay']

#: The six components a monitor may name, in the array's component order per field.
COMPONENTS = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')


def _monitor_name(where, name):
    if not isinstance(name, str) or not name:
        raise ValueError(f'{where}(name=...): the name is the key the result is read back '
                         f'by, res[name]; it must be a non-empty string.')
    if '::' in name:
        raise ValueError(f'{where}({name!r}): "::" is reserved -- a box monitor names its '
                         f'six faces name::x- .. name::z+ internally.')


@spec(positional=('name',))
class Record:
    """A point time series of one component, at the nearest Yee point of that component."""

    name: str
    at: tuple
    component: str = 'Ey'

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('Record(name=...): the name is the key the result is read back '
                             'by, res[name]; it must be a non-empty string.')
        if not (isinstance(self.at, (tuple, list)) and len(self.at) == 3):
            raise ValueError(f'Record(at={self.at!r}): at is a point (x, y, z) in the user\'s '
                             f'length unit, three numbers.')
        # The point picks a cell index, so it is static like the grid it indexes.
        object.__setattr__(self, 'at', tuple(static_float('Record', 'at', v) for v in self.at))
        if self.component not in COMPONENTS:
            raise ValueError(f'Record(component={self.component!r}): the components are '
                             f'{COMPONENTS}.')

    @property
    def field(self):
        """``'E'`` or ``'H'``."""
        return self.component[0]

    @property
    def index(self):
        """0, 1, 2 for x, y, z."""
        return 'xyz'.index(self.component[1])


def _wavelengths(where, value):
    """The static wavelength list, as a tuple of positive finite floats."""
    if value is None:
        raise ValueError(f'{where}(wavelengths=...): a frequency-domain monitor needs the '
                         f'wavelengths it accumulates at -- a tuple of vacuum wavelengths in the '
                         f"scene's length unit, e.g. wavelengths=(0.5, 0.55, 0.6).")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = (value,)
    try:
        items = list(value)
    except TypeError:
        raise ValueError(f'{where}(wavelengths={value!r}): a tuple of vacuum wavelengths.') \
            from None
    if not items:
        raise ValueError(f'{where}(wavelengths=()): at least one wavelength.')
    out = tuple(static_float(where, 'wavelengths', v) for v in items)
    for w in out:
        if not (w > 0 and math.isfinite(w)):
            raise ValueError(f'{where}(wavelengths={value!r}): every wavelength is a positive '
                             f'length; got {w!r}.')
    if len(set(out)) != len(out):
        raise ValueError(f'{where}(wavelengths={value!r}): a wavelength is repeated; each is '
                         f'one row of the result.')
    return out


def _region(where, plane, box):
    from .geometry import Box, Plane
    if (plane is None) == (box is None):
        raise ValueError(f'{where}: give plane=fd.Plane(...) or box=fd.Box(...), exactly one.')
    if plane is not None and not isinstance(plane, Plane):
        raise TypeError(f'{where}(plane=...): an fd.Plane(z=...); got {type(plane).__name__}.')
    if box is not None and not isinstance(box, Box):
        raise TypeError(f'{where}(box=...): an fd.Box(size=..., center=...); got '
                        f'{type(box).__name__}.')


@spec(positional=('name',))
class Flux:
    """The Poynting flux through a plane (along its +normal) or out of a closed box.

    ``fd.Flux('t', plane=fd.Plane(z=-1.0), wavelengths=ws)`` or ``fd.Flux('out',
    box=fd.Box(size=..., center=...), wavelengths=ws)``. ``res[name].flux`` is ``(nf,)``,
    ``0.5 Re sum (E x conj(H)) . n dA`` of the DFT phasors, with ``n`` the plane's ``+``
    normal or the box's outward one; a box also returns each face's share in
    ``res[name].faces``. The collocation -- which H is paired with which E -- is derived in
    ``wavebit/fd/_post.py``. The phasors carry the DFT's own normalisation
    (``sum_n f(t_n) exp(i w t_n) dt``), so a flux is meaningful divided by a source's spectral
    normalisation from the same run (``res.sources[name].intensity(...)``, ``.power(...)``).

    A closed box must enclose or exclude every structure's entire support, with one cell
    of clearance for the H half-plane stencil. Planning refuses a box that cuts that
    support: it cannot be interpreted as the structure's whole absorption or scattering.
    The support test is conservative for curved bodies. ``Field(box=...)`` may sample
    the material interior; it records a volume of fields rather than closed-surface power.
    """

    name: str
    plane: object = None
    box: object = None
    wavelengths: tuple = None

    def __post_init__(self):
        _monitor_name('Flux', self.name)
        _region('Flux', self.plane, self.box)
        object.__setattr__(self, 'wavelengths', _wavelengths('Flux', self.wavelengths))


@spec(positional=('name',))
class Field:
    """DFT phasors of the named components on a plane or in a box.

    ``fd.Field('f', plane=fd.Plane(z=0.2), components=('Ex', 'Ey'), wavelengths=ws)``. Each
    component is read on **its own** Yee points nearest the plane (a plane of E_x nodes, a
    half plane of E_z) -- nothing is interpolated -- and ``res[name].fields[c]`` is ``(nf,
    nz', ny', nx')`` with ``res[name].coords[c]`` the ``(x, y, z)`` coordinates of those
    points. A box monitors the volume; keep it small, its accumulator is ``nf`` times the
    region per component.
    """

    name: str
    plane: object = None
    box: object = None
    components: tuple = COMPONENTS
    wavelengths: tuple = None

    def __post_init__(self):
        _monitor_name('Field', self.name)
        _region('Field', self.plane, self.box)
        comps = (self.components,) if isinstance(self.components, str) else tuple(self.components)
        if not comps or any(c not in COMPONENTS for c in comps):
            raise ValueError(f'Field(components={self.components!r}): a tuple drawn from '
                             f'{COMPONENTS}.')
        if len(set(comps)) != len(comps):
            raise ValueError(f'Field(components={self.components!r}): a component is repeated.')
        object.__setattr__(self, 'components', comps)
        object.__setattr__(self, 'wavelengths', _wavelengths('Field', self.wavelengths))


#: What a Diffraction monitor measures, and which way that goes relative to the incidence.
SIDES = ('reflect', 'transmit')


@spec(positional=('name',))
class Diffraction:
    """The diffraction orders through a plane of a periodic or Bloch cell.

    ``fd.Diffraction('r', z=1.2, side='reflect', wavelengths=ws)``. The plane is the node
    plane nearest the position, across the axis named (which must be the scene's
    ``PlaneWave`` propagation axis, and the cell periodic or Bloch across it). ``side`` picks
    the waves travelling away from the structure on the incidence side (``'reflect'``) or on
    the far side (``'transmit'``); the plane may sit anywhere in a uniform medium, even in the
    total-field region, because the two directions are separated from E and H (the split and
    its derivation are in ``wavebit/fd/_post.py``).

    ``res[name].orders`` is a :class:`wavebit._core.result.Result` of ``(nf, n_v, n_u)`` arrays
    -- ``R``/``T`` and their s/p parts, ``r_s``/``r_p``/``t_s``/``t_p`` in each order's own
    ``s_hat = P_hat x k_hat_par`` frame with the p sign convention of ``wavebit._core.result`` --
    with ``order_window`` and ``.order(m, n)``. Each amplitude is de-embedded from the monitor
    plane to ``reference_plane`` (default: the monitor's own) with **that order's Yee kz**,
    and divided by the PlaneWave's incident amplitude referred to ``incident_plane`` (default:
    ``reference_plane``) -- read from the PlaneWave's own column, never from a closed form.
    Transversally every phase is referred to the cell's low corner (``lateral_origin``), the
    frame ``wavebit.rcwa`` quotes its orders in. To
    compare with ``wavebit.rcwa``, which refers ``r`` to the top interface and ``t`` to the
    bottom one against an incident wave at the top: ``reference_plane=incident_plane=top`` for
    a reflect monitor, ``reference_plane=bottom, incident_plane=top`` for a transmit one.

    ``orders`` is the half width ``M`` or ``(M_u, M_v)`` of the window (default: every order
    propagating at any requested wavelength, in the medium of the plane). ``polarizations=
    'both'`` runs the scene a second time with the other polarisation and fills ``s_inc`` and
    ``p_inc`` beside ``given``; the result's ``runs`` and ``res.diagnostics['runs']`` say so.
    """

    name: str
    x: object = None
    y: object = None
    z: object = None
    side: str = 'reflect'
    wavelengths: tuple = None
    orders: object = None
    reference_plane: object = None
    incident_plane: object = None
    polarizations: str = 'given'

    def __post_init__(self):
        _monitor_name('Diffraction', self.name)
        named = [a for a in 'xyz' if getattr(self, a) is not None]
        if len(named) != 1:
            raise ValueError(f'Diffraction(...): name the plane by its one position along the '
                             f'propagation axis, as Diffraction(name, z=1.2, ...); got '
                             f'{named or "none"}.')
        axis = named[0]
        object.__setattr__(self, axis, static_float('Diffraction', axis, getattr(self, axis)))
        if self.side not in SIDES:
            raise ValueError(f'Diffraction(side={self.side!r}): {SIDES[0]!r} (away from the '
                             f'structure on the incidence side) or {SIDES[1]!r} (on the far '
                             f'side).')
        object.__setattr__(self, 'wavelengths', _wavelengths('Diffraction', self.wavelengths))
        if self.orders is not None:
            pair = self.orders if isinstance(self.orders, (tuple, list)) else (self.orders,) * 2
            if len(pair) != 2:
                raise ValueError(f'Diffraction(orders={self.orders!r}): a half width M, or the '
                                 f'pair (M_u, M_v).')
            pair = tuple(static_int('Diffraction', 'orders', v) for v in pair)
            if any(v < 0 for v in pair):
                raise ValueError(f'Diffraction(orders={self.orders!r}): half widths are >= 0.')
            object.__setattr__(self, 'orders', pair)
        for key in ('reference_plane', 'incident_plane'):
            v = getattr(self, key)
            if v is not None:
                object.__setattr__(self, key, static_float('Diffraction', key, v))
        if self.polarizations not in ('given', 'both'):
            raise ValueError(f"Diffraction(polarizations={self.polarizations!r}): 'given' (the "
                             f"PlaneWave's own polarisation) or 'both' (a second run with the "
                             f"other one, filling s_inc and p_inc).")

    @property
    def axis(self):
        return 'xyz'.index(next(a for a in 'xyz' if getattr(self, a) is not None))

    @property
    def position(self):
        return getattr(self, 'xyz'[self.axis])


@spec(positional=('name',))
class FarField:
    """The far field of an open scene: the near-to-far transform of a closed box's surface.

    ``fd.FarField('ff', box=fd.Box(size=(0.5,)*3), wavelengths=ws)``. The box's six faces are
    read exactly as a ``Flux`` box reads them (the tangential E on each face's node plane, the
    tangential H on the two half planes either side), and after the run the equivalence
    principle turns them into the field at infinity (Taflove & Hagness, *Computational
    Electrodynamics*, 3rd ed., ch. 8): ``res[name].intensity(theta, phi)`` is the radiant
    intensity per unit solid angle, ``(nf, ...)`` over the broadcast shape of the angles (in
    degrees; ``theta`` from ``+z``, ``phi`` from ``+x`` toward ``+y``), ``res[name].fields(theta,
    phi)`` the ``(E_theta, E_phi)`` of ``r exp(-i k r) E``, and ``res[name].flux`` the same box's
    net outward flux, which the intensity integrated over the sphere equals. Everything is
    post-processing in the backend's arrays, so it differentiates. ``_post.py`` derives the
    currents, the collocation and the signs.

    **Open scenes only**: every face absorbing, the box in a uniform background with nothing
    of the scene outside it (the free-space Green's function is the whole model), and no
    ``TFSF`` surface crossing it. A periodic or Bloch cell is refused, naming ``fd.Diffraction``:
    there the orders *are* the far field.
    """

    name: str
    box: object = None
    wavelengths: tuple = None

    def __post_init__(self):
        from .geometry import Box
        _monitor_name('FarField', self.name)
        if not isinstance(self.box, Box):
            raise TypeError(f'FarField(box=...): the closed surface is an fd.Box(size=..., '
                            f'center=...); got {type(self.box).__name__}. (A plane is not a '
                            f'closed surface, and the transform needs one.)')
        object.__setattr__(self, 'wavelengths', _wavelengths('FarField', self.wavelengths))

    @property
    def plane(self):
        return None


MONITOR_TYPES = (Record, Flux, Field, Diffraction, FarField)


def ModeMonitor(*args, **kwargs):
    """Refused by name: mode overlaps are stage 7."""
    raise NotImplementedError('fd.ModeMonitor (mode overlaps) belongs to stage 7, with '
                              'fd.ModeSource. Stage 3 has fd.Record, fd.Flux, fd.Field, '
                              'fd.FarField and fd.Diffraction.')


def Decay(*args, **kwargs):
    """Refused by name: stopping on decay is stage 7."""
    raise NotImplementedError('fd.Decay (stopping a run when the field has decayed) belongs to '
                              'stage 7. Stage 3 runs for Run(steps=), Run(periods=) or '
                              'Run(time=), and measures the decay after the run: '
                              "res.diagnostics['decay'], and fd.UnsettledFieldWarning when a "
                              'frequency-domain monitor was stopped too early.')
