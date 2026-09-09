"""Study optical readouts over the shared conforming NVF cell implementation."""

from wavebit._core.conforming.continuous_flux import (
    ContinuousFluxGapModel as _Cell,
    polynomial_moments as polynomial_moments,
    flux_integrals as flux_integrals,
    project_flux_strip as project_flux_strip,
    project_flux_polynomial as project_flux_polynomial
)
from studies.nvf_gap_optics import GapOptics


class ContinuousFluxGapModel(GapOptics, _Cell):
    """The shared cell with the original fixed-fixture optical study methods."""

if __name__ == '__main__':
    from studies.nvf_harmonic_coordinates import main
    main(model_class=ContinuousFluxGapModel, kind='continuous_flux', gap_enrichment=True)
