"""Study optical readouts over the shared conforming NVF cell implementation."""

from wavebit._core.conforming.material_grading import (
    MaterialGradedGapModel as _Cell,
    graded_fractions as graded_fractions
)
from studies.nvf_gap_optics import GapOptics


class MaterialGradedGapModel(GapOptics, _Cell):
    """The shared cell with the original fixed-fixture optical study methods."""

if __name__ == '__main__':
    from studies.nvf_harmonic_coordinates import main
    main(model_class=MaterialGradedGapModel, kind='material_graded_gap', gap_enrichment=True, mesh_grading=True)
