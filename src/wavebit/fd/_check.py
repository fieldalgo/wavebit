"""``check(scene) -> Report``: what the run would be, before it is run.

Cells per axis and in total, ``dt`` and the duration, the dtype, the logical main-state
array payload (E/H, CPML histories and material memory), and the warnings a reader should
see before spending the time: a size that was snapped up to whole cells, a Bloch face
that made the run complex, a box under two cells thick. ``check`` prints the report and
returns it; ``run`` builds the same report silently and attaches it to the result.

``bytes_per_state`` includes every declared pole's P, previous and delta arrays, plus
a hybrid film's auxiliary array, over its compact support, including reserved inactive
slots. It is not a total memory forecast:
source histories and auxiliary source columns, observer and energy-diagnostic arrays,
runtime scratch, temporaries, checkpoints and allocator/Python overhead are excluded.
Parameters and DFT accumulators have separate report entries.

Stage 3 adds one line per frequency-domain monitor: the bytes of its DFT accumulators (complex
per point per frequency, plus the per-step region energy the decay diagnostic reads; NumPy
holds one scratch of the same size beside each, which is not counted), and says
so when a ``Diffraction(polarizations='both')`` makes the run two solves. A scene without such
a monitor has no monitor line.
"""
import dataclasses

import numpy as np

from ._kernel.poles import PoleTerm
from ._lower import plan as _plan
from ._lower.checks import notes as _notes, shown
from ._lower.coefficients import params_bytes

__all__ = ['Report', 'check', 'report']


@dataclasses.dataclass(frozen=True)
class Report:
    cells: tuple
    requested_size: tuple
    size: tuple
    dx: float
    dt: float
    steps: int
    duration: float
    dtype: str
    #: Main-grid E/H, CPML and material-array payload; excludes source/observer/diagnostic
    #: storage and runtime overhead. This is not the byte count of the whole state pytree.
    bytes_per_state: int
    warnings: tuple
    skeleton: str
    bytes_per_params: int = 0
    absorbers: tuple = ()
    spans: tuple = ()
    monitors: tuple = ()         # ((name, kind, bytes), ...) of the DFT monitors (stage 3)
    solves: int = 1
    #: ((term name, discretisation, active rows, row width, bytes), ...) -- one row per
    #: structure that asked for a material discretisation other than the default. A scene
    #: where every structure took the default has none and prints exactly what it printed
    #: before, which is the point: the selector shows up in the report only where it was used.
    discretisations: tuple = ()
    #: ((x lo, x hi), (y lo, y hi), (z lo, z hi)) of the grid, so a structure row can say
    #: whether its got differs from its asked by the box's face or by a snap (C5).
    box: tuple = ()
    #: One ``checks.BodyRow`` per smoothed structure, printed as its block (C5), and the bytes
    #: of each constitutive term's cross coefficients (params) and flux memory (state).
    bodies: tuple = ()
    term_bytes: tuple = ()
    #: False where the params could not be built on NumPy (a traced material): the params
    #: line then says so instead of printing 0.00 MB, which a threshold would read as small.
    params_known: bool = True
    #: ((term name, stars moved, stars with a cross coefficient, largest t), ...): the
    #: star lift of each diagonal-rule constitutive term (``stars.lift``).
    lifts: tuple = ()
    junctions: tuple = ()
    #: (requested cap steps, amplitude threshold, requested window, window steps), or None.
    stopping: object = None

    def __str__(self):
        lines = [
            f'cells      {self.cells[0]} x {self.cells[1]} x {self.cells[2]} = '
            f'{self.cells[0] * self.cells[1] * self.cells[2]}',
            f'size       {self.size} (requested {self.requested_size})',
            f'dx         {self.dx:g}',
            f'dt         {self.dt:g}   steps {self.steps}   duration {self.duration:g}',
            f'dtype      {self.dtype}',
            f'state      {self.bytes_per_state / 1e6:.2f} MB (E, H, CPML and material memory)',
            (f'params     {self.bytes_per_params / 1e6:.2f} MB (coefficients, profiles)'
             if self.params_known else
             'params     unknown (a traced value: the params could not be built on NumPy)'),
        ]
        if self.stopping is not None:
            cap, threshold, window, count = self.stopping
            lines.append(f'stopping   Decay amplitude <= {threshold:g}, window {window:g} '
                         f'({count} steps), hard cap {cap} steps; state-tail heuristic')
        for name, kind, cells in self.absorbers:
            lines.append(f'absorber   {name}  {kind}, {cells} cells')
        for axis, lo, hi, slo, shi in self.spans:
            note = self._span_note(axis, lo, hi, slo, shi)
            lines.append(f'structure  {axis}=({lo:g}, {hi:g}) -> ({slo:g}, {shi:g}){note}')
        for row in self.bodies:
            lines.extend(self._body_lines(row))
        for name, kind, nbytes in self.monitors:
            lines.append(f'monitor    {name}  {kind}, {nbytes / 1e6:.2f} MB (DFT accumulators)')
        for name, rule, rows, width, nbytes in self.discretisations:
            lines.append(f'material   {name}  discretisation={rule}, {rows} active rows x '
                         f'{width}, {nbytes / 1e6:.2f} MB (prepared on the host)')
        if self.solves != 1:
            lines.append(f'solves     {self.solves} (a Diffraction monitor asked for both '
                         f'polarisations: the scene runs once per polarisation)')
        lines.append(f'skeleton   {self.skeleton[:12]}')
        for w in self.warnings:
            lines.append(f'warning    {w}')
        return '\n'.join(lines)

    def _body_lines(self, row):
        """The block of one smoothed structure: what the grid holds of it, and what it costs."""
        (i0, i1), (j0, j1), (k0, k1) = row.support
        out = [f'body       structures[{row.index}] {row.describe}, smoothed',
               f'  support  x {i0}..{i1}  y {j0}..{j1}  z {k0}..{k1}  '
               f'({(i1 - i0) * (j1 - j0) * (k1 - k0)} cells)']
        if row.across is None:
            out.append('  across   a layer (uniform along two axes): no feature size to resolve')
        else:
            out.append(f'  across   {row.across[0]:.2f} cells at its narrowest '
                       f'({row.across[1]:g})')
        names = ' '.join(row.held)
        if row.asked is None:
            held = ', '.join(f'{name} {v:.6g}' for name, v in row.held.items())
            out.append(f'  volume   held {held} (the body answers no volume())')
        else:
            # Each component against the body's volume inside its own voxels: a node
            # component's reach half a cell past a minus face and stops half a cell short of a
            # plus one, so against the box a correct body leaving through one face read 2.9%
            # off, and one across a mirror 12% (the C5 review).
            worst = max((row.held[n] / row.cover[n] - 1.0 for n in row.held if row.cover[n]),
                        key=abs, default=0.0)
            out.append(f'  volume   {row.asked:.6g} asked, {row.inside:.6g} in the box; held on '
                       f'{names} to {worst:+.1e} of what their voxels cover, at worst')
        if row.outside:
            share = '' if row.share is None else f'{100 * row.share:.1f}% '
            out.append(f'  outside  {share}past {", ".join(row.outside)}')
        out.append(f'  cut      {" / ".join(str(v) for v in row.cut.values())} voxels '
                   f'({names})')
        for standing, face, axis, side, plane, rule, count, rim, beside in self.junctions:
            if standing == row.index:
                letter = "xyz"[axis]
                sign = "+" if side == 1 else "-"
                out.append(f"  junction on structures[{face}] at {letter} = {plane:g} "
                           f"(its {letter}{sign} face): {rule}, {count} voxels "
                           f"({rim} on the rim), {beside} beside the base")
        sizes = dict((name, (p, s)) for name, p, s in self.term_bytes)
        for name in row.terms:
            p, s = sizes.get(name, (None, None))
            cost = ('bytes unknown' if p is None else
                    f'{p / 1e6:.2f} MB params + {s / 1e6:.2f} MB state')
            others = row.shared.get(name, ())
            shared = ('' if not others else ', shared with ' + ', '.join(
                f'structures[{j}]' for j in others))
            out.append(f'  cross    {name}: {cost}{shared}')
            lifted = next((row for row in self.lifts if row[0] == name), None)
            if lifted is not None and lifted[1]:
                out.append(f'  lift     {name}: {lifted[1]} of {lifted[2]} stars moved toward '
                           f'the node tensor, the largest {lifted[3]:.3g} of the way{shared}')
        for name in row.stencil:
            out.append(f'  cross    {name}: the prepared stencil operator (its material row)')
        if row.normals == 'none' and not row.stencil:
            out.append('  cross    none: its cross coefficients are zero (an unturned box)')
        if row.normals == 'exact':
            out.append('  normals  exact moment (the sphere)')
        elif row.normals != 'none':
            out.append(f'  normals  nearest-surface proxy, max |n_c^2 - w_c| {row.normals:.6g}')
        for face, plane in row.mirrors:
            out.append(f'  mirror   {face[0]} = {shown(plane, self.dx):g} ({face}): the scene is '
                       f'this and its image')
        return out

    def _span_note(self, label, lo, hi, slo, shi):
        """``<- clipped`` where the box's face cut the span, ``<- snapped`` where cells moved it.

        Compared to a billionth of a cell, not exactly: the snapped ``-0.2`` of a box that
        asked ``-0.2`` comes back ``-0.20000000000000004`` from the cell arithmetic in a 0.6
        box at ``dx = 0.05`` (``-0.19999999999999996`` in a 1.0 one), and the exact comparison
        printed ``<- snapped`` on an axis where nothing moved (C5 audit). The row's axis is
        read from its label only where the label names one the way every shipped shape does
        -- ``x`` or ending ``.x`` -- and an external shape's other label is told apart from a
        clip by its move alone, as ``<- snapped`` (it raised on a label not ending in an axis
        letter, the C5 review).
        """
        tol = 1e-9 * self.dx

        def same(u, v):
            return abs(u[0] - v[0]) <= tol and abs(u[1] - v[1]) <= tol
        if same((lo, hi), (slo, shi)):
            return ''
        clipped = (lo, hi)
        axis = label if label in ('x', 'y', 'z') else (
            label[-1] if label[-2:] in ('.x', '.y', '.z') else None)
        if self.box and axis is not None:
            wall = self.box['xyz'.index(axis)]
            clipped = (max(lo, wall[0]), min(hi, wall[1]))
        notes = [word for word, moved in (('clipped', not same((lo, hi), clipped)),
                                          ('snapped', not same(clipped, (slo, shi)))) if moved]
        return '   <- ' + ' and '.join(notes)


def report(scene, p=None, params=None):
    """The report of ``scene`` (under plan ``p``, built if not given), without printing.

    ``params`` is the lowered parameters when the caller already has them (``run`` does), so
    that the byte count is of the arrays the run really built, on the backend it built them
    on. Without them ``check`` lowers on NumPy, which a scene with a *traced* material value
    cannot do -- there the count is reported as unknown rather than guessed.

    A scene the *lowering* refuses is refused here too, with the same message: ``check``
    promises what the run would be, and a complete, healthy-looking report for a scene
    ``fd.run`` turns away is the opposite of that. Only what NumPy cannot hold is caught.
    """
    p = _plan(scene) if p is None else p
    grid = p.grid
    warnings = []
    requested = scene.domain.size
    if any(abs(s - r) > 1e-12 * max(abs(r), 1.0) for s, r in zip(grid.size, requested)):
        warnings.append(f'size snapped up to whole cells: {requested} -> {grid.size}; the '
                        f'centre stays at {scene.domain.center}')
    if p.dtype == 'complex128':
        warnings.append('a Bloch face makes the run complex: twice the bytes and the flops of '
                        'a real run')
    axis1d = [name[0] for name, rule in p.face_rules if rule.kind == 'axis1d' and name[1] == '-']
    if axis1d:
        warnings.append(f'Axis1D on {", ".join(axis1d)}: one cell, derivative i k exactly '
                        f'(always complex)')
    for axis, n in zip('xyz', grid.n):
        if n < 4 and axis not in axis1d:
            warnings.append(f'{n} cell(s) along {axis}: a box this thin is all boundary')
    bodies, lifted = [], {}
    for message, _ in ((m, c) for c, m in _notes(scene, p, rows=bodies, lifted=lifted)):
        warnings.append(message)
    itemsize = np.dtype(p.dtype).itemsize
    # The params' bytes are counted by lowering on NumPy -- the coefficient arrays and the
    # CPML profiles are the same shapes on every backend, so this costs no device memory and
    # never imports JAX -- and with the initial field left out, both because evaluating it
    # would make `check` as expensive as a step and because it *becomes* the state at n = 0
    # and is already counted there.
    psi = sum(4 * face.cells * (p.cells // p.grid.n['xyz'.index(face.name[0])])
              for face in p.cpml) * itemsize
    flux_memory = sum(3 * np.prod(t.region.shape) for t in p.constitutive) * itemsize
    # PoleTerm reserves three histories; a hybrid film declares a fourth auxiliary
    # array per slot. Count every declared slot, including inactive ones, statically:
    # initializing full buffers here would defeat a pre-run report.
    polarization_memory = sum(3 * getattr(t, 'memory_arrays_per_pole', 3)
                              * t.n_poles * np.prod(t.region.shape)
                              for t in p.terms if isinstance(t, PoleTerm)) * itemsize
    absorbers = tuple((name, 'PML' if (name, cells) in p.pml_cells else 'Absorber', cells)
                      for name, cells in p.absorbing_faces)
    params_known = True
    if params is None:
        from ._lower import lower as _lower
        try:
            params = _lower(scene.replace(initial=None), p, 'numpy')
        except (TypeError, RuntimeError) as err:  # a traced leaf that NumPy cannot hold
            # Narrow on purpose. A refusal is a ValueError, and `check` promising "what the
            # run would be" must not print a complete, healthy-looking report for a scene
            # `run` then refuses outright -- the refusal's own text is what names the fix.
            # This catch is for the one thing it was written for: `np.asarray` on a tracer.
            params, params_known = {}, False
            warnings.append(f'the params could not be built on NumPy for the byte count: '
                            f'{type(err).__name__}: {err}; run the scene to see it')
    cross = params.get('constitutive', {}) if isinstance(params, dict) else {}
    # Counted from shape and dtype, as params_bytes counts: under jax.jit(fd.run) the arrays
    # are tracers, and np.asarray on one raised (the C5 review).
    term_bytes = tuple((t.name, params_bytes({'term': cross[t.name]}) if t.name in cross
                        else None, int(3 * np.prod(t.region.shape) * itemsize))
                       for t in p.constitutive)
    monitors, solves = _monitor_lines(scene, p)
    from ._lower.junction import report_rows as junction_rows
    stopping = None
    if p.decay is not None:
        from ._decay import window_steps
        from ._lower.plan import run_steps
        stopping = (run_steps(scene, p.dt), p.decay.threshold, p.decay.window,
                    window_steps(p.decay.window, p.dt))
    return Report(cells=grid.n, requested_size=requested, size=grid.size, dx=p.dx, dt=p.dt,
                  stopping=stopping,
                  discretisations=_discretisation_lines(p), junctions=junction_rows(p),
                  steps=p.steps, duration=p.steps * p.dt, dtype=p.dtype,
                  bytes_per_state=(2 * 3 * p.cells * itemsize + psi + flux_memory
                                   + polarization_memory),
                  warnings=tuple(warnings),
                  skeleton=p.skeleton, bytes_per_params=params_bytes(params),
                  absorbers=absorbers, spans=p.spans, monitors=monitors, solves=solves,
                  box=tuple((grid.low[a], grid.high[a]) for a in range(3)),
                  bodies=tuple(bodies), term_bytes=term_bytes, params_known=params_known,
                  lifts=tuple((name, out.moved, out.stars, out.largest)
                              for name, out in lifted.items()))


def _discretisation_lines(p):
    """One row per prepared material operator; empty for every scene that asked for none.

    Read off the term itself -- the arrays it carries *are* the prepared operator -- so this
    costs nothing and cannot disagree with what the run will use.
    """
    from ._kernel.stencil import StencilTerm
    return tuple((t.name, t.discretisation, len(t.index), t.index.shape[1],
                  t.index.nbytes + t.value.nbytes + t.output.nbytes + t.diagonal.nbytes)
                 for t in p.terms if isinstance(t, StencilTerm))


def _monitor_lines(scene, p):
    """``((name, kind, bytes), ...)`` per frequency-domain monitor, and the solve count."""
    out = []
    for m in getattr(scene, 'monitors', ()):
        regions = [o for o in p.observers if getattr(o, 'monitor', None) == m.name]
        if regions:
            out.append((m.name, type(m).__name__, sum(o.nbytes(p) for o in regions)))
    solves = 2 if any(getattr(m, 'polarizations', 'given') == 'both'
                      for m in getattr(scene, 'monitors', ())) else 1
    return tuple(out), solves


def check(scene):
    """Print and return the :class:`Report` of ``scene``."""
    r = report(scene)
    print(r)
    return r
