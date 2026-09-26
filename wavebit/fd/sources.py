"""What drives a run: an initial field, the stage-2 sources, and the protocol every term obeys.

Stage 0 has no sources. The excitation is ``InitialField``: the state the leapfrog starts
from, E at ``t = 0`` and H at ``t = -dt/2``. It is not a term at all -- nothing is added into
the fields during the run -- which is exactly why it is the stage-0 excitation: it exercises
the kernel's slots with an empty term list, so the first gates test the curl, the faces and
the masks and nothing else. Sources arrive in stage 2 as terms (below).

**The Term protocol** is fixed here because the kernel step is written against it now.
A term adds into a slab of the curl that feeds the next update, in one of two phase-tagged
slots, and may carry state of its own (``mem``) that it advances once per step:

    before_H(cE, E, mem, n, params, rt) -> cE     # cE feeds  H = ah*H - bh*cE
    before_E(cH, H, mem, n, params, rt) -> cH     # cH feeds  E = ae*E + be*cH
    advance(mem, E, H, n, params, rt)   -> mem
    init_mem(plan, rt)                  -> mem

``rt`` is the runtime whose ``slab_add`` a term must use for the write (the only primitive
whose cost was measured, ``studies/fdtd_slab_add_cost.py``), and ``mem`` is the term's own
entry, not the whole memory. Terms within one slot must commute to round-off: the
permutation gate applies two of them in either order. On Torch the write is in place, so a
term must not read a view of the array it is about to write into -- the order gate's mutant
does exactly that and must raise. A term is a frozen dataclass of static values (it is part
of the plan and of its skeleton hash); anything traced belongs in ``params``.

**Stage 2 adds the sources** (below the protocol): a :class:`Band` every source's waveform is
built from, a point :class:`Dipole`, a :class:`PlaneWave` for a periodic cell and a
:class:`TFSF` box for an isolated object. What each one is in the kernel -- the slot, the
support, the sign and the half step at which it reads its waveform -- is derived in
``wavebit/fd/_kernel/sources.py``; the refusals that need the grid are in
``wavebit/fd/_lower/sources.py``. This module holds what a user may declare, and refuses
what can be refused without a grid.
"""
import math
from typing import Protocol, runtime_checkable

from ._spec import is_jax_typed, spec, static_float

__all__ = ['InitialField', 'Term', 'Band', 'Dipole', 'PlaneWave', 'TFSF', 'SOURCE_TYPES',
           'GaussianBeam', 'ModeSource']


@runtime_checkable
class Term(Protocol):
    """The phase-tagged term the kernel step calls; see the module docstring."""

    name: str

    def init_mem(self, plan, rt): ...

    def before_H(self, cE, E, mem, n, params, rt): ...

    def before_E(self, cH, H, mem, n, params, rt): ...

    def advance(self, mem, E, H, n, params, rt): ...


def _check_field(name, value):
    if value is None:
        return
    if callable(value):
        return
    shape = getattr(value, 'shape', None)
    if shape is not None:
        if len(shape) != 4 or shape[0] != 3:
            raise ValueError(
                f'InitialField({name}=array): the array is [component, z, y, x] with three '
                f'components, so its shape is (3, nz, ny, nx); got {tuple(shape)}.')
        return
    if isinstance(value, (tuple, list)):
        if len(value) != 3:
            raise ValueError(
                f'InitialField({name}=(...)): a tuple gives one entry per component, (x, y, z), '
                f'so it has three entries; got {len(value)}.')
        for c, v in zip('xyz', value):
            if not (v is None or callable(v) or hasattr(v, 'shape') or isinstance(v, (int, float, complex))):
                raise TypeError(
                    f'InitialField({name}=...): the {c} entry is {type(v).__name__}; each entry '
                    f'is a callable fn(x, y, z), an array, a number, or None for zero.')
        return
    raise TypeError(
        f'InitialField({name}={type(value).__name__}): give an array (3, nz, ny, nx), a '
        f'callable fn(x, y, z) applied to every component, or a tuple of three per-component '
        f'entries (callable, array, number or None).')


@spec
class InitialField:
    """The state the leapfrog starts from: E at ``t = 0``, H at ``t = -dt/2``.

    Each of ``E`` and ``H`` is one of

    * an array ``(3, nz, ny, nx)`` in the ``[component, z, y, x]`` layout;
    * a tuple ``(fx, fy, fz)`` of per-component entries, each a callable ``fn(x, y, z)``, an
      array ``(nz, ny, nx)``, a number, or ``None`` for zero;
    * a single callable ``fn(x, y, z)``, applied to every component.

    A callable receives the three coordinate grids of **that component's Yee points**,
    each ``(nz, ny, nx)``, in the user's length unit -- so ``lambda x, y, z: np.cos(k*x)``
    samples ``Ex`` half a cell further along x than ``Ey``, as the staggering requires.
    Nothing is interpolated: the caller is stating a field on the grid, not in the continuum.

    ``amplitude`` scales both fields and is a traced parameter, so a gradient with respect
    to the excitation's size flows through it without rebuilding the pattern (the AD gate
    differentiates final energy with respect to it).
    """

    E: object = None
    H: object = None
    amplitude: object = 1.0

    def __post_init__(self):
        if self.E is None and self.H is None:
            raise ValueError('InitialField(): neither E nor H given, so the run would start '
                             'from nothing and stay there. Give at least one.')
        _check_field('E', self.E)
        _check_field('H', self.H)


# ---------------------------------------------------------------------------------------------
# Stage 2: the band, and the three sources
# ---------------------------------------------------------------------------------------------

def _is_complex(v):
    if isinstance(v, complex):
        return True
    dtype = getattr(v, 'dtype', None)
    return dtype is not None and 'complex' in str(dtype)


def concrete_or_none(v):
    """``float(v)`` when readable now, ``None`` for a traced scalar (the materials rule)."""
    from .materials import concrete
    return concrete(v)


def _leaf(where, name, v, *, positive=False, complex_ok=False):
    """A traced-or-concrete scalar leaf, validated as far as its value can be read.

    The waveform's numbers -- amplitude, phase, delay, the band's centre and width, a
    dipole's position -- are leaves: a gradient flows through them, so a JAX tracer or a
    Torch tensor that asked for one must pass. Everything else is read and checked here.
    """
    if isinstance(v, (bool, str)) or v is None:
        raise ValueError(f'{where}({name}={v!r}): a number is needed, not a '
                         f'{type(v).__name__}.')
    if _is_complex(v) and not complex_ok:
        raise ValueError(f'{where}({name}={v!r}): a real number is needed; only the '
                         f'amplitude may be complex, and only on a complex (Bloch) run.')
    shape = getattr(v, 'shape', None)
    if shape is not None and tuple(shape) != ():
        raise ValueError(f'{where}({name}=...): one scalar; got shape {tuple(shape)}.')
    if _is_complex(v):
        return v
    c = concrete_or_none(v)
    if c is None:
        if not (is_jax_typed(v) or getattr(v, 'requires_grad', False)):
            raise ValueError(f'{where}({name}={v!r}): a real number is needed (or a traced '
                             f'scalar of jax or torch).')
        return v
    if not math.isfinite(c):
        raise ValueError(f'{where}({name}={v!r}): a finite number is needed.')
    if positive and not c > 0:
        raise ValueError(f'{where}({name}={v!r}): a positive number is needed.')
    return v


def _cutoff_sigmas(cutoff):
    """``k`` with ``exp(-k^2/2) = cutoff``: how many envelope sigmas reach the cutoff."""
    return math.sqrt(2.0 * math.log(1.0 / cutoff))


@spec
class Band:
    """A Gaussian-envelope sine carrier: the waveform every stage-2 source is built from.

    ``fd.Band(wavelength=0.55, df=0.3, cutoff=1e-3)``. In this package's units (``c = 1``)
    the centre frequency is ``f0 = 1/wavelength``; ``df`` is the **spectral standard
    deviation of the amplitude spectrum, in frequency units** (a width in wavelength would be
    asymmetric about the centre). The time signal of a source with amplitude ``A``, phase
    ``phi`` and delay ``d`` is

        w(t) = A exp(-s^2 / (2 tau^2)) sin(2 pi f0 s + phi),   s = t - t0 - d
        tau  = 1 / (2 pi df),   t0 = k tau,   exp(-k^2/2) = cutoff

    so the amplitude spectrum is ``exp(-(f - f0)^2 / (2 df^2))`` about ``+f0`` (and its mirror
    about ``-f0``), the envelope starts at ``cutoff`` of its peak at ``t = 0`` (with no delay),
    and the spectrum falls to ``cutoff`` at ``f0 +- k df``. ``shortest_wavelength = 1/(f0 +
    k df)`` is where the resolution check reads the mesh. A band that would reach zero
    frequency (``f0 <= k df``) is refused: its lower half is not a carrier any more.

    ``wavelength`` and ``df`` may be traced (a gradient with respect to the band centre is
    part of stage 2's AD gate); ``cutoff`` is static. Whatever needs a concrete centre --
    ``Run(periods=)``, a PML wavelength inferred from the sources, the cells-per-wavelength
    check -- says so when it cannot read one.
    """

    wavelength: object
    df: object
    cutoff: float = 1e-3

    def __post_init__(self):
        object.__setattr__(self, 'wavelength',
                           _leaf('Band', 'wavelength', self.wavelength, positive=True))
        object.__setattr__(self, 'df', _leaf('Band', 'df', self.df, positive=True))
        cutoff = static_float('Band', 'cutoff', self.cutoff)
        if not 0.0 < cutoff < 1.0:
            raise ValueError(f'Band(cutoff={self.cutoff!r}): the level the envelope starts '
                             f'and the spectrum ends at, a fraction in (0, 1); 1e-3 is the '
                             f'default.')
        object.__setattr__(self, 'cutoff', cutoff)
        f0, df = self.concrete_f0, concrete_or_none(self.df)
        if f0 is not None and df is not None and f0 <= self.sigmas * df:
            raise ValueError(
                f'Band(wavelength={self.wavelength!r}, df={self.df!r}): the spectrum reaches '
                f'zero frequency -- f0 = {f0:g} is not above {self.sigmas:.3f} * df = '
                f'{self.sigmas * df:g}, where it falls to the cutoff {cutoff:g}. A pulse '
                f'that wide is not a carrier; narrow df below {f0 / self.sigmas:g}.')

    @property
    def sigmas(self):
        """``k``: the envelope sigmas at which the envelope and the spectrum reach ``cutoff``."""
        return _cutoff_sigmas(self.cutoff)

    @property
    def concrete_f0(self):
        w = concrete_or_none(self.wavelength)
        return None if w is None else 1.0 / w

    @property
    def concrete_wavelength(self):
        return concrete_or_none(self.wavelength)

    @property
    def shortest_wavelength(self):
        """``1/(f0 + k df)``, where the spectrum falls to ``cutoff``; ``None`` if traced."""
        f0, df = self.concrete_f0, concrete_or_none(self.df)
        if f0 is None or df is None:
            return None
        return 1.0 / (f0 + self.sigmas * df)

    @property
    def longest_wavelength(self):
        """``1/(f0 - k df)``; ``None`` if traced."""
        f0, df = self.concrete_f0, concrete_or_none(self.df)
        if f0 is None or df is None:
            return None
        return 1.0 / (f0 - self.sigmas * df)

    def waveform(self, t, amplitude=1.0, phase=0.0, delay=0.0):
        """The closed form ``w(t)`` in NumPy, for a caller's cross-check (see the class)."""
        import numpy as np
        f0 = 1.0 / float(self.wavelength)
        tau = 1.0 / (2.0 * math.pi * float(self.df))
        s = np.asarray(t, dtype=float) - self.sigmas * tau - delay
        return amplitude * np.exp(-0.5 * (s / tau) ** 2) * np.sin(2 * math.pi * f0 * s + phase)

    def spectrum(self, f):
        """``|W(f)| / max|W|`` of the envelope about ``+f0`` (the image about ``-f0`` left out)."""
        import numpy as np
        f0, df = 1.0 / float(self.wavelength), float(self.df)
        return np.exp(-0.5 * ((np.asarray(f, dtype=float) - f0) / df) ** 2)


def _name(where, name):
    if not isinstance(name, str) or not name:
        raise ValueError(f'{where}(name=...): the name is the key the result is read back '
                         f'by, res.sources[name]; it must be a non-empty string.')


def _band(where, band):
    if not isinstance(band, Band):
        raise TypeError(f'{where}(band=...): an fd.Band(wavelength=..., df=...) is required; '
                        f'got {type(band).__name__}. Every source reads its waveform from '
                        f'one, and there is no default band.')


def _waveform_leaves(where, obj):
    object.__setattr__(obj, 'amplitude', _leaf(where, 'amplitude', obj.amplitude,
                                               complex_ok=True))
    object.__setattr__(obj, 'phase', _leaf(where, 'phase', obj.phase))
    object.__setattr__(obj, 'delay', _leaf(where, 'delay', obj.delay))


#: The stamps a :class:`Dipole` may spread its current with. See :class:`Dipole`.
STAMPS = ('bspline', 'tent')

#: The stamp a ``Dipole`` takes by default; ``wavebit/fd/_kernel/sources.py`` has the probe
#: that chose it.
DEFAULT_STAMP = 'bspline'


@spec(positional=('name',))
class Dipole:
    """A point current, electric or magnetic, spread over a smooth compact stamp.

    ``fd.Dipole('d', center=(x, y, z), component='Ey', band=band)``, or ``kind='magnetic'``
    with ``component='Hy'``. The current density is ``w(t) S(r) / dx^3`` along the named
    component, where the stamp ``S`` sums to exactly 1 over the component's Yee points, so the
    dipole moment ``sum J dV`` is ``w(t)`` -- the ``I l`` of the continuum formula -- whatever
    the stamp and wherever ``center`` sits in its cell.

    ``stamp``: ``'bspline'`` (the default: the cubic B-spline, half-width 2 cells) or
    ``'tent'`` (the hat of half-width 2 cells, normalised). Both touch at least 3 points per
    axis (4 off a node) and both reproduce linear functions, so the moment *and* the centroid
    are exact; the default was chosen by a measurement (``wavebit/fd/_kernel/sources.py``).
    ``stamp='trilinear'`` -- the 2x2x2 cloud-in-cell weights -- is refused by name: it is the
    one-cell hat, it touches two points per axis, and its derivative with respect to the
    position jumps at every node.

    ``range=((x0, x1), (y0, y1), (z0, z1))`` declares how far ``center`` may move under a
    gradient. The stamp's support -- which cells it writes -- is static, so a position that is
    a traced leaf needs a static box it stays inside; the support is the union of the stamps
    over that box. Without ``range`` the centre must be concrete; with it, a concrete centre
    outside it is refused.
    """

    name: str
    center: tuple = None
    component: str = 'Ey'
    band: object = None
    kind: str = 'electric'
    amplitude: object = 1.0
    phase: object = 0.0
    delay: object = 0.0
    range: object = None
    stamp: str = DEFAULT_STAMP

    def __post_init__(self):
        _name('Dipole', self.name)
        _band('Dipole', self.band)
        if self.kind not in ('electric', 'magnetic'):
            raise ValueError(f"Dipole(kind={self.kind!r}): 'electric' (a current J, "
                             f"component 'Ex'/'Ey'/'Ez') or 'magnetic' (a current M, "
                             f"component 'Hx'/'Hy'/'Hz').")
        field = 'E' if self.kind == 'electric' else 'H'
        if not (isinstance(self.component, str) and len(self.component) == 2
                and self.component[0] in 'EH' and self.component[1] in 'xyz'):
            raise ValueError(f'Dipole(component={self.component!r}): one of '
                             f"'{field}x', '{field}y', '{field}z' for kind={self.kind!r}.")
        if self.component[0] != field:
            other = 'magnetic' if field == 'E' else 'electric'
            raise ValueError(
                f"Dipole(component={self.component!r}, kind={self.kind!r}): a {self.kind} "
                f"dipole drives {field}; say component='{field}{self.component[1]}', or "
                f"kind='{other}' to drive {self.component}.")
        if self.stamp == 'trilinear':
            raise ValueError(
                "Dipole(stamp='trilinear'): the 2x2x2 cloud-in-cell stamp is refused. Its "
                "weights are the one-cell hat: two points per axis where the design asks for "
                "a smooth stamp of at least three, and a derivative with respect to center "
                "that jumps at every node, so a gradient in position would be piecewise "
                "constant. Say stamp='bspline' (the default) or stamp='tent'.")
        if self.stamp not in STAMPS:
            raise ValueError(f'Dipole(stamp={self.stamp!r}): one of {STAMPS}.')
        if not (isinstance(self.center, (tuple, list)) and len(self.center) == 3):
            raise ValueError(f'Dipole(center={self.center!r}): a point (x, y, z), three '
                             f'numbers.')
        center = tuple(_leaf('Dipole', 'center', v) for v in self.center)
        object.__setattr__(self, 'center', center)
        _waveform_leaves('Dipole', self)
        traced = [concrete_or_none(v) is None for v in center]
        if self.range is None:
            if any(traced):
                raise ValueError(
                    'Dipole(center=<traced>): a position that is a gradient leaf needs '
                    'range=((x0, x1), (y0, y1), (z0, z1)), the static box it may move in -- '
                    'the stamp writes a static set of cells, and without a box nothing says '
                    'which.')
            return
        rng = self.range
        if not (isinstance(rng, (tuple, list)) and len(rng) == 3
                and all(isinstance(r, (tuple, list)) and len(r) == 2 for r in rng)):
            raise ValueError(f'Dipole(range={rng!r}): three (lo, hi) pairs, one per axis, '
                             f'((x0, x1), (y0, y1), (z0, z1)).')
        pairs = tuple((static_float('Dipole', 'range', a), static_float('Dipole', 'range', b))
                      for a, b in rng)
        for axis, (a, b) in zip('xyz', pairs):
            if not a <= b:
                raise ValueError(f'Dipole(range=...): the {axis} pair ({a}, {b}) has lo > hi.')
        object.__setattr__(self, 'range', pairs)
        for axis, v, (a, b) in zip('xyz', center, pairs):
            c = concrete_or_none(v)
            if c is not None and not a <= c <= b:
                raise ValueError(f'Dipole(center=...): {axis} = {c:g} lies outside the '
                                 f'declared range ({a:g}, {b:g}). The range is what fixes the '
                                 f'cells the stamp writes; a centre outside it would write '
                                 f'cells the plan never reserved.')

    @property
    def field(self):
        return self.component[0]

    @property
    def index(self):
        return 'xyz'.index(self.component[1])

    @property
    def span(self):
        """``((lo, hi),) * 3``: the range, or the concrete centre as a zero-width box."""
        if self.range is not None:
            return self.range
        return tuple((float(c), float(c)) for c in self.center)


_DIRECTIONS = ('+x', '-x', '+y', '-y', '+z', '-z')


def _direction(where, direction, oblique_hint):
    if isinstance(direction, str) and direction in _DIRECTIONS:
        return direction
    if isinstance(direction, str) and direction in ('x', 'y', 'z'):
        raise ValueError(f"{where}(direction={direction!r}): say which way along the axis, "
                         f"'+{direction}' or '-{direction}'.")
    if isinstance(direction, (tuple, list)):
        raise ValueError(f'{where}(direction={direction!r}): a direction is one of '
                         f'{_DIRECTIONS}. {oblique_hint}')
    raise ValueError(f'{where}(direction={direction!r}): one of {_DIRECTIONS}.')


@spec(positional=('name',))
class PlaneWave:
    """A one-directional Huygens sheet in a periodic cell.

    ``fd.PlaneWave('pw', z=1.0, direction='-z', theta=20, phi=0, polarization='s',
    band=band)``. The sheet is the node plane nearest ``z`` (or ``x=``/``y=``, the axis the
    direction names); it launches a plane wave toward ``direction`` and nothing the other
    way. ``theta`` is the polar angle from the propagation axis in degrees and ``phi`` the
    azimuth in the transverse plane, from the first transverse axis of the cyclic order (x for
    z, y for x, z for y). ``'s'`` has E perpendicular to the plane of incidence, ``'p'`` in
    it; :meth:`vectors` returns ``(k, e, h)`` with ``h = k x e``.

    The cell's transverse faces must be ``Periodic`` (a zero transverse wavevector) or
    ``Bloch`` carrying the wave's transverse wavevector at the band centre --
    :meth:`Bloch.from_source` writes them. ``amplitude`` scales the tangential part of ``e``,
    which is what the sheet imposes; the reference series the run returns
    (``res.sources[name]``) is the incident field itself, so nothing downstream has to believe
    a closed form.
    """

    name: str
    x: object = None
    y: object = None
    z: object = None
    direction: str = '-z'
    theta: float = 0.0
    phi: float = 0.0
    polarization: str = 's'
    band: object = None
    amplitude: object = 1.0
    phase: object = 0.0
    delay: object = 0.0

    def __post_init__(self):
        _name('PlaneWave', self.name)
        _band('PlaneWave', self.band)
        direction = _direction('PlaneWave', self.direction,
                               'An oblique wave is theta=/phi= about an axis direction.')
        axis = direction[1]
        named = [a for a in 'xyz' if getattr(self, a) is not None]
        if named != [axis]:
            raise ValueError(f'PlaneWave(direction={direction!r}): the sheet is a plane '
                             f'across {axis}, so give its position as {axis}=... and only '
                             f'that; got {named or "none"}.')
        object.__setattr__(self, axis, static_float('PlaneWave', axis, getattr(self, axis)))
        theta = static_float('PlaneWave', 'theta', self.theta)
        phi = static_float('PlaneWave', 'phi', self.phi)
        if not 0.0 <= theta < 90.0:
            raise ValueError(f'PlaneWave(theta={self.theta!r}): the polar angle from the '
                             f'propagation axis, in degrees, in [0, 90).')
        object.__setattr__(self, 'theta', theta)
        object.__setattr__(self, 'phi', phi)
        if self.polarization not in ('s', 'p'):
            raise ValueError(f"PlaneWave(polarization={self.polarization!r}): 's' (E out of "
                             f"the plane of incidence) or 'p' (E in it).")
        _waveform_leaves('PlaneWave', self)

    @property
    def axis(self):
        return 'xyz'.index(self.direction[1])

    @property
    def sign(self):
        return 1 if self.direction[0] == '+' else -1

    @property
    def position(self):
        return getattr(self, self.direction[1])

    def vectors(self):
        """``(k, e, h)`` as unit ``(x, y, z)`` triples: ``e`` s or p, ``h = k x e``.

        With ``P`` the propagation axis and ``(u, v)`` the transverse pair in cyclic order,
        ``k = sin(theta) cos(phi) u + sin(theta) sin(phi) v +- cos(theta) P``,
        ``s = -sin(phi) u + cos(phi) v`` and ``p = s x k``, so ``e x h = k`` for both
        (``h`` is ``Z0 H``, the stored field).
        """
        P = self.axis
        u, v = (P + 1) % 3, (P + 2) % 3
        t, f = math.radians(self.theta), math.radians(self.phi)
        k = [0.0, 0.0, 0.0]
        k[u] = math.sin(t) * math.cos(f)
        k[v] = math.sin(t) * math.sin(f)
        k[P] = self.sign * math.cos(t)
        s = [0.0, 0.0, 0.0]
        s[u], s[v] = -math.sin(f), math.cos(f)
        e = s if self.polarization == 's' else _cross(s, k)
        h = _cross(k, e)

        def clean(w):
            return tuple(0.0 if abs(c) < 1e-15 else c for c in w)
        return clean(k), clean(e), clean(h)


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


#: The oblique-TFSF measurement, quoted by the refusal (``studies/fdtd_tfsf_oblique_leak.py``).
OBLIQUE_TFSF_MESSAGE = (
    'TFSF({what}): an oblique TFSF box is refused in v1. Its incident field would come from a '
    '1-D auxiliary grid with the 3-D points projected onto the direction, which is not an '
    'exact discrete solution off an axis. Measured (studies/fdtd_tfsf_oblique_leak.py, an '
    'empty box, peak scattered / peak incident): the plain aux grid leaks 3-8e-2 at 10 cells '
    'per wavelength at 15-60 degrees, and the best variant found (matched dispersion, cubic '
    'interpolation, the grid\'s own polarisation vectors) still leaks 2.2e-3 / 4.2e-4 / '
    '1.0e-4 / 2.6e-5 at 10 / 20 / 40 / 80 cells per wavelength -- second order, reaching 1e-4 '
    'only at 40. (45 degrees in a coordinate plane and the body diagonal are exact, but only '
    'because they are special directions of the grid.) Along an axis the aux grid IS the 3-D '
    'grid and an empty box leaks at round-off. For oblique incidence use fd.PlaneWave with '
    'fd.Bloch.from_source(...) in a periodic cell, or give TFSF an axis direction such as '
    "direction='-z'.")


@spec(positional=('name',))
class TFSF:
    """A total-field/scattered-field box: the incident plane wave exists only inside ``box``.

    ``fd.TFSF('t', box=fd.Region(size=(...), center=(...)), direction='-z', polarization='x',
    band=band)``. The box's faces snap to the nearest node planes; inside them the stored
    field is the total field, outside it the scattered field alone. The incident field comes
    from a 1-D auxiliary grid along the propagation axis with the **same** ``dx`` and ``dt``
    and the same background along that axis, uniform or layered (every structure that is a
    slab across the propagation axis is part of it), terminated by the scene's own absorbing
    faces on that axis -- which is why those two faces must be ``PML`` or ``Absorber``. On an
    axis the aux grid is the 3-D grid restricted to a line, so an empty box leaks at
    round-off.

    ``direction`` is an axis direction and nothing else; ``theta`` exists only to be refused
    when it is not zero, with the measurement that decided it. ``polarization`` names the axis
    E is along (``'x'`` or ``'y'`` for a z direction); ``h = k x e`` is the H partner.
    """

    name: str
    box: object = None
    direction: str = '-z'
    polarization: str = 'x'
    band: object = None
    amplitude: object = 1.0
    phase: object = 0.0
    delay: object = 0.0
    theta: float = 0.0

    def __post_init__(self):
        from .geometry import Region
        _name('TFSF', self.name)
        _band('TFSF', self.band)
        theta = static_float('TFSF', 'theta', self.theta)
        if theta != 0.0:
            raise ValueError(OBLIQUE_TFSF_MESSAGE.format(what=f'theta={self.theta!r}'))
        object.__setattr__(self, 'theta', 0.0)
        direction = _direction('TFSF', self.direction, OBLIQUE_TFSF_MESSAGE.format(
            what=f'direction={self.direction!r}'))
        from .geometry import Box
        if isinstance(self.box, Box):
            raise TypeError('TFSF(box=fd.Box(...)): fd.Box is the smoothed body a structure '
                            'fills; a TFSF surface picks node planes, and that is the '
                            'cell-snapped fd.Region(size=(...), center=(...)).')
        if not isinstance(self.box, Region):
            raise TypeError(f'TFSF(box=...): an fd.Region(size=(...), center=(...)); got '
                            f'{type(self.box).__name__}.')
        if self.polarization not in ('x', 'y', 'z') or self.polarization == direction[1]:
            others = [a for a in 'xyz' if a != direction[1]]
            raise ValueError(f'TFSF(polarization={self.polarization!r}): the axis E is along, '
                             f'transverse to direction={direction!r} -- {others[0]!r} or '
                             f'{others[1]!r}.')
        _waveform_leaves('TFSF', self)

    @property
    def axis(self):
        return 'xyz'.index(self.direction[1])

    @property
    def sign(self):
        return 1 if self.direction[0] == '+' else -1

    @property
    def pol(self):
        return 'xyz'.index(self.polarization)


SOURCE_TYPES = (Dipole, PlaneWave, TFSF)


def GaussianBeam(*args, **kwargs):
    """Refused by name: a Gaussian beam source is stage 7."""
    raise NotImplementedError('fd.GaussianBeam belongs to stage 7 (beams and mode sources). '
                              'Stage 2 has fd.Dipole, fd.PlaneWave and fd.TFSF.')


def ModeSource(*args, **kwargs):
    """Refused by name: a waveguide mode source is stage 7."""
    raise NotImplementedError('fd.ModeSource belongs to stage 7 (beams and mode sources). '
                              'Stage 2 has fd.Dipole, fd.PlaneWave and fd.TFSF.')
