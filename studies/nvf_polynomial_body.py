"""Study optical readouts over the shared conforming NVF cell implementation."""

from wavebit._core.conforming.polynomial_body import (
    PolynomialBodyGapModel as _Cell,
    body_space as body_space,
    project_body_polynomial as project_body_polynomial
)
from studies.nvf_gap_optics import GapOptics


class PolynomialBodyGapModel(GapOptics, _Cell):
    """The shared cell with the original fixed-fixture optical study methods."""

if __name__ == '__main__':
    from studies.nvf_harmonic_coordinates import main
    main(model_class=PolynomialBodyGapModel, kind='polynomial_body_gap', gap_enrichment=True,
         mesh_grading=True, body_enrichment=True)
