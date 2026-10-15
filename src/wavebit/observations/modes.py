"""Concrete ringdown analysis with SI timestamps and frequencies."""
from dataclasses import field
from wavebit._common.backend import eager_under_torch_compile
from wavebit._common.declarations import spec
from wavebit._common.execution import frozen_mapping


@spec
class ModeFit:
    frequency_hz: object
    decay_rate_per_second: object
    quality_factor: object
    amplitude: object
    time_origin_seconds: float
    times_seconds: object
    fitted: object
    diagnostics: object = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self,'diagnostics',frozen_mapping(self.diagnostics))


@eager_under_torch_compile
def fit_modes(times_seconds,signal,*,modes=1,start_seconds=None,stop_seconds=None,pencil=None,rcond=1e-10):
    """Fit sum(a exp((-decay - 2j*pi*f)*(t-time_origin))) to a concrete trace.

    Times must be uniform SI seconds. Positive decay means damping. A real
    cosine requires two conjugate modes. Source-free window and model-order
    convergence remain the caller's checks; this NumPy fit does not retain AD.
    """
    from wavebit.fdtd._periodic.probes import fit_fdtd_modes
    raw=fit_fdtd_modes(times_seconds,signal,modes=modes,start=start_seconds,stop=stop_seconds,
                       pencil=pencil,rcond=rcond)
    return ModeFit(frequency_hz=raw['frequency'],decay_rate_per_second=raw['decay_rate'],
        quality_factor=raw['quality_factor'],amplitude=raw['amplitude'],
        time_origin_seconds=raw['time_origin'],times_seconds=raw['t'],fitted=raw['fitted'],
        diagnostics={k:raw[k] for k in ('relative_residual','singular_values','basis_condition','modes','pencil','discrete_pole')})


__all__=['ModeFit','fit_modes']
