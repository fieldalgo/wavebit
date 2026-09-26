"""``check(scene) -> Report``: what the run would be, before it is run.

Cells per axis and in total, ``dt`` and the duration, the dtype, the bytes one state
occupies (E and H; the curl scratch on NumPy doubles it), and the warnings a reader should
see before spending the time: a size that was snapped up to whole cells, a Bloch face
that made the run complex, a box under two cells thick. ``check`` prints the report and
returns it; ``run`` builds the same report silently and attaches it to the result.

Stage 3 adds one line per frequency-domain monitor: the bytes of its DFT accumulators (complex
per point per frequency, plus the per-step region energy the decay diagnostic reads; NumPy
holds one scratch of the same size beside each, which is not counted), and says
so when a ``Diffraction(polarizations='both')`` makes the run two solves. A scene without such
a monitor prints exactly what it printed before.
"""
import dataclasses

import numpy as np

from ._lower import plan as _plan
from ._lower.checks import notes as _notes
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

    def __str__(self):
        lines = [
            f'cells      {self.cells[0]} x {self.cells[1]} x {self.cells[2]} = '
            f'{self.cells[0] * self.cells[1] * self.cells[2]}',
            f'size       {self.size} (requested {self.requested_size})',
            f'dx         {self.dx:g}',
            f'dt         {self.dt:g}   steps {self.steps}   duration {self.duration:g}',
            f'dtype      {self.dtype}',
            f'state      {self.bytes_per_state / 1e6:.2f} MB (E and H)',
            f'params     {self.bytes_per_params / 1e6:.2f} MB (coefficients, profiles)',
        ]
        for name, kind, cells in self.absorbers:
            lines.append(f'absorber   {name}  {kind}, {cells} cells')
        for axis, lo, hi, slo, shi in self.spans:
            note = self._span_note(axis, lo, hi, slo, shi)
            lines.append(f'structure  {axis}=({lo:g}, {hi:g}) -> ({slo:g}, {shi:g}){note}')
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

    def _span_note(self, label, lo, hi, slo, shi):
        """``<- clipped`` where the box's face cut the span, ``<- snapped`` where cells moved it.

        Compared to a billionth of a cell, not exactly: the snapped ``-0.2`` of a box that
        asked ``-0.2`` comes back ``-0.20000000000000004`` from the cell arithmetic, and the
        exact comparison printed ``<- snapped`` on an axis where nothing moved (C5 audit).
        The row's axis is its label's last letter (``x``, ``box.x``, ``sphere.x``).
        """
        tol = 1e-9 * self.dx

        def same(u, v):
            return abs(u[0] - v[0]) <= tol and abs(u[1] - v[1]) <= tol
        if same((lo, hi), (slo, shi)):
            return ''
        clipped = (lo, hi)
        if self.box:
            wall = self.box['xyz'.index(label[-1])]
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
    for message, _ in ((m, c) for c, m in _notes(scene, p)):
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
    absorbers = tuple((name, 'PML' if (name, cells) in p.pml_cells else 'Absorber', cells)
                      for name, cells in p.absorbing_faces)
    if params is None:
        from ._lower import lower as _lower
        try:
            params = _lower(scene.replace(initial=None), p, 'numpy')
        except (TypeError, RuntimeError) as err:  # a traced leaf that NumPy cannot hold
            # Narrow on purpose. A refusal is a ValueError, and `check` promising "what the
            # run would be" must not print a complete, healthy-looking report for a scene
            # `run` then refuses outright -- the refusal's own text is what names the fix.
            # This catch is for the one thing it was written for: `np.asarray` on a tracer.
            params = {}
            warnings.append(f'the params could not be built on NumPy for the byte count: '
                            f'{type(err).__name__}: {err}; run the scene to see it')
    monitors, solves = _monitor_lines(scene, p)
    return Report(cells=grid.n, requested_size=requested, size=grid.size, dx=p.dx, dt=p.dt,
                  discretisations=_discretisation_lines(p),
                  steps=p.steps, duration=p.steps * p.dt, dtype=p.dtype,
                  bytes_per_state=2 * 3 * p.cells * itemsize + psi + flux_memory,
                  warnings=tuple(warnings),
                  skeleton=p.skeleton, bytes_per_params=params_bytes(params),
                  absorbers=absorbers, spans=p.spans, monitors=monitors, solves=solves,
                  box=tuple((grid.low[a], grid.high[a]) for a in range(3)))


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
