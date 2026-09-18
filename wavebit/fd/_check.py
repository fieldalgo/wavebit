"""``check(scene) -> Report``: what the run would be, before it is run.

Cells per axis and in total, ``dt`` and the duration, the dtype, the bytes one state
occupies (E and H; the curl scratch on NumPy doubles it), and the warnings a reader should
see before spending the time: a size that was snapped up to whole cells, a Bloch face
that made the run complex, a box under two cells thick. ``check`` prints the report and
returns it; ``run`` builds the same report silently and attaches it to the result.
"""
import dataclasses

import numpy as np

from ._lower import plan as _plan

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

    def __str__(self):
        lines = [
            f'cells      {self.cells[0]} x {self.cells[1]} x {self.cells[2]} = '
            f'{self.cells[0] * self.cells[1] * self.cells[2]}',
            f'size       {self.size} (requested {self.requested_size})',
            f'dx         {self.dx:g}',
            f'dt         {self.dt:g}   steps {self.steps}   duration {self.duration:g}',
            f'dtype      {self.dtype}',
            f'state      {self.bytes_per_state / 1e6:.2f} MB (E and H)',
            f'skeleton   {self.skeleton[:12]}',
        ]
        for w in self.warnings:
            lines.append(f'warning    {w}')
        return '\n'.join(lines)


def report(scene, p=None):
    """The report of ``scene`` (under plan ``p``, built if not given), without printing."""
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
    for axis, n in zip('xyz', grid.n):
        if n < 4:
            warnings.append(f'{n} cell(s) along {axis}: a box this thin is all boundary')
    itemsize = np.dtype(p.dtype).itemsize
    return Report(cells=grid.n, requested_size=requested, size=grid.size, dx=p.dx, dt=p.dt,
                  steps=p.steps, duration=p.steps * p.dt, dtype=p.dtype,
                  bytes_per_state=2 * 3 * p.cells * itemsize, warnings=tuple(warnings),
                  skeleton=p.skeleton)


def check(scene):
    """Print and return the :class:`Report` of ``scene``."""
    r = report(scene)
    print(r)
    return r
