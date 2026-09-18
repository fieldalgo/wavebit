"""What a run returns: the recorded series by name, the final fields, the plan and the report.

``res['probe']`` is a ``Series`` with ``t`` (NumPy, the sample times -- E times for an E
component, H times for an H one) and ``values`` (an array **of the backend that ran**, so a
JAX run hands back a JAX array and a Torch run a tensor, with any gradient graph intact).
``res.final.E`` and ``res.final.H`` are the fields after the last step, ``(3, nz, ny, nx)``
on the same backend. ``res.plan`` and ``res.report`` are what the run was built from.
"""
import dataclasses
from collections.abc import Mapping

__all__ = ['Results', 'Series', 'Final']


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


class Results(Mapping):
    """See the module docstring."""

    def __init__(self, series, final, plan, report, backend):
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
