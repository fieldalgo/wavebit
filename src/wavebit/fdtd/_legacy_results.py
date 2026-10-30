"""What a run returns: the recorded series by name, the final fields, the plan and the report.

``res['probe']`` is a ``Series`` with ``t`` (NumPy, the sample times -- E times for an E
component, H times for an H one) and ``values`` (an array **of the backend that ran**, so a
JAX run hands back a JAX array and a Torch run a tensor, with any gradient graph intact).
``res.final.E`` and ``res.final.H`` are the fields after the last step, ``(3, nz, ny, nx)``
on the same backend. ``res.plan`` and ``res.report`` describe the actual execution. With
explicit ``Run(decay=Decay(...))``, their step count can be shorter than the requested hard
cap; all time histories and DFT endpoint corrections use that actual prefix. The requested
cap/skeleton and stop reason remain in ``res.diagnostics['stopping']``. An early-result plan
cannot be reused as the plan of the original longer capped scene.

``res.sources[name]`` (stage 2) is each source's own reference series, a
:class:`SourceSeries` computed by the engine in the same run: for a ``TFSF`` box and a
``PlaneWave`` the incident field of its column where the total field begins (``incident``,
``(steps, 3)`` at the E times ``t``: the tangential E on the entry node plane ``plane`` and the
normal E on the half cell just inside it; ``incident_H`` at the H times ``t_H``: every H
component at the index of that half cell, the tangential ones on the half plane
``plane_H``), each at its own Yee point of cell ``(0, 0)`` across -- exactly where the main
grid's total field equals it in a uniform medium; for a
``Dipole`` the drive ``w(t)`` (the dipole moment, at its own times ``t``), the power it
delivers to the field ``work = -sum J.E dV`` at the same times, and the stamp-weighted field
``field = sum S F`` at ``t_field``. What stage 3 turns into a normalisation.

**Stage 3** adds the frequency-domain results and the sources' spectral normalisation, all of
them arrays of the backend that ran (differentiable), computed after the run from the DFT
accumulators and the reference series (``src/wavebit/fdtd/_post.py`` derives each):

    res['t'].flux                         # Flux: (nf,) along the plane's +normal / out of a box
    res['f'].fields['Ex']                 # Field: (nf, ...) phasors on the component's points
    res['r'].orders                       # Diffraction: a wavebit Result, (nf, n_v, n_u) arrays
    res['r'].theta_at                     # (nf,) the incidence angle at each wavelength, degrees
    res['ff'].intensity(theta, phi)       # FarField: (nf, ...) radiant intensity, degrees in
    res['ff'].flux                        # FarField: (nf,) the net outward flux of its box
    res.sources['pw'].incident_amplitude(ws)   # (nf,) complex, at the entry plane (or plane=)
    res.sources['t'].intensity(ws)        # (nf,) incident flux per unit area
    res.sources['d'].power(ws)            # (nf,) power the dipole delivered
    res.diagnostics                       # decay per monitor, the threshold, runs

The spectral readings of a source are **methods** taking the wavelengths, not attributes:
``incident`` already names the stage-2 time series (a golden pins it), and a source does not
know which wavelengths a caller wants. ``wavelengths=None`` means the scene's
frequency-domain monitors' wavelengths when they all share one list.
"""
from wavebit._common.backend import eager_under_torch_compile as _eager_under_torch_compile
import dataclasses
from collections.abc import Mapping

from wavebit.fdtd._public_names import PUBLIC_NAMES

__all__ = ['Results', 'Series', 'Final', 'SourceSeries', 'FluxResult', 'FieldResult',
           'FarFieldResult', 'DiffractionResult', 'DiffractionOrders']


@_eager_under_torch_compile
@dataclasses.dataclass(frozen=True)
class Series:
    """A recorded time series: ``t`` (steps,) NumPy, ``values`` (steps,) on the backend."""

    name: str
    component: str
    t: object
    values: object


@_eager_under_torch_compile
@dataclasses.dataclass(frozen=True)
class Final:
    """The fields after the last step: ``E`` at ``steps*dt``, ``H`` half a step earlier."""

    E: object
    H: object


@_eager_under_torch_compile
@dataclasses.dataclass(frozen=True)
class SourceSeries:
    """A source's reference series; which fields are set depends on ``kind`` (see the module).

    ``plane`` / ``plane_H`` are the coordinates along the propagation axis of the node plane
    ``incident`` is read at and of the half plane ``incident_H``'s tangential components sit on.
    """

    name: str
    kind: str
    t: object
    incident: object = None
    t_H: object = None
    incident_H: object = None
    plane: float = None
    plane_H: float = None
    drive: object = None
    work: object = None
    t_field: object = None
    field: object = None
    context: object = dataclasses.field(default=None, repr=False, compare=False)

    def _wavelengths(self, wavelengths):
        if wavelengths is not None:
            from wavebit.fdtd._legacy_monitors import _wavelengths
            return _wavelengths(f'res.sources[{self.name!r}]', wavelengths)
        found = getattr(self.context, 'wavelengths', None)
        if found is None:
            raise ValueError(
                f"res.sources[{self.name!r}]: say which wavelengths -- the scene's "
                f"frequency-domain monitors do not share one list to default to.")
        return found

    def _needs(self, what, kinds):
        if not any(self.kind.startswith(k) for k in kinds):
            # The kinds are the engine's source kind strings ('Dipole(E)'); the message
            # names each by its public class.
            this = self.kind.partition('(')[0]
            raise TypeError(f'res.sources[{self.name!r}].{what}(...) is for a '
                            f'{" or ".join(PUBLIC_NAMES.get(k, k) for k in kinds)} source; '
                            f'this one is a {PUBLIC_NAMES.get(this, this)}.')
        if self.context is None:
            raise ValueError(f'res.sources[{self.name!r}] carries no run context to transform.')

    def power(self, wavelengths=None):
        """``(nf,)``: the power a Dipole delivered to the field at each wavelength."""
        from wavebit.fdtd._assemble import source_power
        self._needs('power', ('Dipole',))
        return source_power(self, self._wavelengths(wavelengths))

    def intensity(self, wavelengths=None):
        """``(nf,)``: a TFSF or PlaneWave's incident flux per unit area along its direction."""
        from wavebit.fdtd._assemble import source_intensity
        self._needs('intensity', ('TFSF', 'PlaneWave'))
        return source_intensity(self, self._wavelengths(wavelengths))

    def incident_amplitude(self, wavelengths=None, plane=None):
        """``(nf,)`` complex: the incident amplitude along the polarisation, at ``plane``.

        ``plane`` is a position along the propagation axis (default: the entry plane, where the
        total field begins); the amplitude is carried there with the Yee kz.
        """
        from wavebit.fdtd._assemble import incident_amplitude
        self._needs('incident_amplitude', ('TFSF', 'PlaneWave'))
        return incident_amplitude(self, self._wavelengths(wavelengths), plane=plane)


@_eager_under_torch_compile
@dataclasses.dataclass(frozen=True)
class FluxResult:
    """A ``Flux`` monitor: ``flux`` ``(nf,)``, and ``faces`` ``{face: (nf,)}`` for a box."""

    name: str
    wavelengths: tuple
    flux: object
    faces: object = None
    accumulators: object = None   # a plane's corrected DFT phasors (Eu, Ev, Hu, Hv)
    layout: object = None         # and where they sit (_prepare/monitors.py::PlaneLayout)


@_eager_under_torch_compile
@dataclasses.dataclass(frozen=True)
class FieldResult:
    """A ``Field`` monitor: ``fields[c]`` ``(nf, nz', ny', nx')`` and ``coords[c]`` ``(x, y, z)``."""

    name: str
    wavelengths: tuple
    fields: dict
    coords: dict


@_eager_under_torch_compile
class FarFieldResult:
    """A ``FarField`` monitor: the transform of its box, evaluated at the angles you ask for.

    ``intensity(theta, phi)`` is the radiant intensity per unit solid angle ``(nf, ...)`` over
    the broadcast shape of the angles, in degrees (``theta`` from ``+z``, ``phi`` from ``+x``
    toward ``+y``); ``fields(theta, phi)`` is ``(E_theta, E_phi)`` of ``r exp(-i k r) E``; ``flux``
    ``(nf,)`` is the net outward flux of the same box, and ``faces`` its six shares.
    ``propagation='continuum'`` retains the original continuum approximation;
    its angular integral can differ from discrete box power. ``'yee'`` uses
    the lattice dispersion, reciprocal staggered fields and Green curvature.
    Then the phase is ``k_star dot r``, angles specify the outgoing ray, and
    ``vector`` retains all three Cartesian components. ``intensity`` uses
    the full vector and discrete flux weight. See docs/validation/fdtd/fd-lattice-radiation.md.
    Recorded fields remain differentiable; Yee background and angles must be
    concrete host inputs.
    """

    def __init__(self, name, wavelengths, rt, accs, faces, grid, eps, mu, flux, face_flux,
                 *, dt=None, propagation='continuum'):
        self.name = name
        self.wavelengths = wavelengths
        self.flux = flux
        self.faces = face_flux
        self.propagation = propagation
        self._dt = dt
        self._args = (rt, accs, faces, grid, tuple(1.0 / w for w in wavelengths), eps, mu)

    def _transform(self, theta, phi):
        from wavebit.fdtd._post import far_field
        rt, accs, faces, grid, freqs, eps, mu = self._args
        if self.propagation == 'yee':
            from wavebit.fdtd._radiation import modes
            from wavebit.fdtd._radiation import transform
            prepared = modes(grid.dx, self._dt, freqs, eps, mu, theta, phi)
            return transform(rt, accs, faces, grid, prepared)
        return far_field(rt, accs, faces, grid, freqs, eps, mu, theta, phi)

    def fields(self, theta, phi=0.0):
        """``(E_theta, E_phi)``, each ``(nf, ...)``: angular projections of the outgoing coefficient."""
        return self._transform(theta, phi)[:2]

    def vector(self, theta, phi=0.0):
        """``(nf, ..., 3)`` complex Cartesian amplitude for ``propagation='yee'``.

        The electric field is transverse to the discrete wavevector, which
        need not follow the observer ray. Keep its radial component when
        computing power; :meth:`intensity` includes it automatically.
        """
        if self.propagation != 'yee':
            raise ValueError("FarFieldVectorRefused: vector() requires propagation='yee'")
        return self._transform(theta, phi)[3]

    def intensity(self, theta, phi=0.0):
        """``(nf, ...)``: radiant intensity per ray solid angle for the chosen propagation model."""
        return self._transform(theta, phi)[2]

    def __repr__(self):
        return f'FarFieldResult({self.name!r}, wavelengths={self.wavelengths})'


_ORDER_CLASS = []


def _order_result():
    """The Result subclass, built once on first use (``wavebit.observations.diffraction`` is imported late)."""
    if _ORDER_CLASS:
        return _ORDER_CLASS[0]
    from wavebit.observations.diffraction import OrderView
    from wavebit.observations.diffraction import Result
    from wavebit.observations.diffraction import ResultSub

    class DiffractionOrders(Result):
        """A :class:`wavebit.observations.diffraction.Result` whose arrays carry a leading wavelength axis.

        Every array is ``(nf, n_v, n_u)``; :meth:`order` returns an ``OrderView`` of ``(nf,)``
        values, and :meth:`at` the plain ``(n_v, n_u)`` Result of one wavelength -- the shape a
        ``wavebit.rcwa`` solve at that wavelength returns, for a side-by-side comparison.
        """

        def order(self, m, n=0):
            sub = self.incident if self.incident is not None else (
                self.s_inc if self.s_inc is not None else self.p_inc)
            (cx, fx), (cy, fy) = self.order_window
            if not (cx - fx <= m <= cx + fx):
                raise IndexError(f'order m={m} outside the retained set [{cx - fx}, {cx + fx}]')
            if not (cy - fy <= n <= cy + fy):
                raise IndexError(f'order n={n} outside the retained set [{cy - fy}, {cy + fy}]')

            def pick(a):
                return None if a is None else a[..., n - cy + fy, m - cx + fx]
            def pick_vector(a):
                return None if a is None else a[..., n - cy + fy, m - cx + fx, :]
            return OrderView(m, n, pick(sub.R), pick(sub.T), pick(sub.r_s), pick(sub.r_p),
                             pick(sub.t_s), pick(sub.t_p),
                             pick_vector(sub.r_xyz), pick_vector(sub.t_xyz))

        def at(self, index):
            """The ``Result`` of wavelength ``index`` alone: arrays ``(n_v, n_u)``."""
            def cut(sub):
                if sub is None:
                    return None
                return ResultSub(*[None if getattr(sub, f) is None else getattr(sub, f)[index]
                                   for f in ResultSub._FIELDS])
            return Result(incident=cut(self.incident), s_inc=cut(self.s_inc), p_inc=cut(self.p_inc),
                          order_window=self.order_window)

    _ORDER_CLASS.append(DiffractionOrders)
    return DiffractionOrders


def DiffractionOrders(*args, **kwargs):
    """Build the Result subclass a Diffraction monitor returns (see ``_order_result``)."""
    return _order_result()(*args, **kwargs)


@_eager_under_torch_compile
class DiffractionResult:
    """A ``Diffraction`` monitor: ``orders`` (a Result), ``theta_at``, the planes, ``runs``.

    ``orders.incident`` is the run's own incidence; with ``polarizations='both'`` ``orders.s_inc``
    and ``orders.p_inc`` are filled from two runs and ``runs`` is 2. ``kz_convention`` is
    ``'yee'``: every order was de-embedded with its discrete kz. ``theta_at`` is in degrees,
    the unit ``fd.PlaneWave(theta=)`` takes; ``wavebit.rcwa`` takes radians.

    ``r_xyz`` / ``t_xyz`` carry the outgoing electric vector in global (x, y, z),
    transverse to the Yee wavevector. A single monitor does not establish a complete
    incident port or both sides of a finite stack: ``power_entering`` and ``absorption``
    stay None, and the unmeasured side's arrays stay None.
    """

    kz_convention = 'yee'

    def __init__(self, monitor, incident, extras, s_inc=None, p_inc=None, runs=1):
        cls = _order_result()
        self.name = monitor.name
        self.side = monitor.side
        self.wavelengths = monitor.wavelengths
        self.orders = cls(incident=incident, s_inc=s_inc, p_inc=p_inc,
                          order_window=extras['order_window'])
        self.order_window = self.orders.order_window
        self.theta_at = extras['theta_at']
        self.reference_plane = extras['reference_plane']
        self.incident_plane = extras['incident_plane']
        self.plane = extras['plane']
        self.kz = extras['kz']
        self.polarization = extras['polarization']
        #: The transverse point every normalized order amplitude is referred to: the cell
        #: centre, matching ``wavebit.rcwa``. Raw Fourier extraction uses the low corner.
        self.lateral_origin = extras['lateral_origin']
        self.raw = {key: extras[key] for key in ('raw_s','raw_p','raw_flux','cell_area',
                    'epsilon','mu','k_parallel','tangent_axes','outward_normal') if key in extras}
        self.runs = runs

    def with_both(self, other_sub):
        """This result with the other polarisation's run added as ``s_inc``/``p_inc``."""
        mine = self.orders.incident
        s_inc, p_inc = (mine, other_sub) if self.polarization == 's' else (other_sub, mine)
        out = DiffractionResult.__new__(DiffractionResult)
        out.__dict__.update(self.__dict__)
        out.orders = type(self.orders)(incident=mine, s_inc=s_inc, p_inc=p_inc,
                                       order_window=self.orders.order_window)
        out.runs = 2
        return out

    def __repr__(self):
        return (f'DiffractionResult({self.name!r}, side={self.side!r}, '
                f'wavelengths={self.wavelengths}, order_window={self.order_window}, '
                f'runs={self.runs})')


@_eager_under_torch_compile
class Results(Mapping):
    """See the module docstring."""

    def __init__(self, series, final, plan, report, backend, sources=None, diagnostics=None):
        self.sources = dict(sources or {})
        #: Stage 3: ``{'decay': {monitor: ratio}, 'records': ..., 'sources': ..., 'threshold':
        #: ..., 'runs': 1 or 2}`` -- see ``src/wavebit/fdtd/_assemble.py::decay_report``.
        self.diagnostics = dict(diagnostics or {})
        self._series = dict(series)
        self.final = final
        self.plan = plan
        self.report = report
        self.backend = backend

    def __getitem__(self, name):
        if name not in self._series:
            raise KeyError(f'no monitor named {name!r}; recorded: {sorted(self._series)}')
        return self._series[name]

    def __iter__(self):
        return iter(self._series)

    def __len__(self):
        return len(self._series)

    def __repr__(self):
        return (f'Results(backend={self.backend!r}, monitors={sorted(self._series)}, '
                f'cells={self.plan.cells}, steps={self.plan.steps})')
