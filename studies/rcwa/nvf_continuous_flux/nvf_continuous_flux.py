"""Study optical readouts over the shared conforming NVF cell implementation."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))

from wavebit.rcwa._engine.conforming.continuous_flux import ContinuousFluxGapModel as _Cell, polynomial_moments as polynomial_moments, flux_integrals as flux_integrals, project_flux_strip as project_flux_strip, project_flux_polynomial as project_flux_polynomial
from studies.rcwa.nvf_gap_optics.nvf_gap_optics import GapOptics


class ContinuousFluxGapModel(GapOptics, _Cell):
    """The shared cell with the original fixed-fixture optical study methods."""

if __name__ == '__main__':
    from studies.rcwa.nvf_harmonic_coordinates.nvf_harmonic_coordinates import main
    main(model_class=ContinuousFluxGapModel, kind='continuous_flux', gap_enrichment=True)
