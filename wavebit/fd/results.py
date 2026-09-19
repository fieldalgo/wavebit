"""What a run returns: the recorded series by name, the final fields, the plan and the report.

``res['probe']`` is a ``Series`` with ``t`` (NumPy, the sample times -- E times for an E
component, H times for an H one) and ``values`` (an array **of the backend that ran**, so a
JAX run hands back a JAX array and a Torch run a tensor, with any gradient graph intact).
``res.final.E`` and ``res.final.H`` are the fields after the last step, ``(3, nz, ny, nx)``
on the same backend. ``res.plan`` and ``res.report`` are what the run was built from.

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
"""
import dataclasses
from collections.abc import Mapping

__all__ = ['Results', 'Series', 'Final', 'SourceSeries']


@dataclasses.dataclass(frozen=True)
class Series:
    """A recorded time series: ``t`` (steps,) NumPy, ``values`` (steps,) on the backend."""

    name: str
    component: str
    t: object
    values: object


@dataclasses.dataclass(frozen=True)
class Final:
    """The fields after the last step: ``E`` at ``steps*dt``, ``H`` half a step earlier."""

    E: object
    H: object


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


class Results(Mapping):
    """See the module docstring."""

    def __init__(self, series, final, plan, report, backend, sources=None):
        self.sources = dict(sources or {})
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
