"""Study optical readouts over the shared conforming NVF cell implementation."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))

from wavebit.rcwa._engine.conforming.polynomial_body import PolynomialBodyGapModel as _Cell, body_space as body_space, project_body_polynomial as project_body_polynomial
from studies.rcwa.nvf_gap_optics.nvf_gap_optics import GapOptics


class PolynomialBodyGapModel(GapOptics, _Cell):
    """The shared cell with the original fixed-fixture optical study methods."""

if __name__ == '__main__':
    from studies.rcwa.nvf_harmonic_coordinates.nvf_harmonic_coordinates import main
    main(model_class=PolynomialBodyGapModel, kind='polynomial_body_gap', gap_enrichment=True,
         mesh_grading=True, body_enrichment=True)
