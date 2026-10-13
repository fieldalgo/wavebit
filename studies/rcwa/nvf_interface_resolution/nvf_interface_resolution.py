"""Study optical readouts over the shared conforming NVF cell implementation."""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[3]))

from wavebit.rcwa._engine.conforming.material_grading import MaterialGradedGapModel as _Cell, graded_fractions as graded_fractions
from studies.rcwa.nvf_gap_optics.nvf_gap_optics import GapOptics


class MaterialGradedGapModel(GapOptics, _Cell):
    """The shared cell with the original fixed-fixture optical study methods."""

if __name__ == '__main__':
    from studies.rcwa.nvf_harmonic_coordinates.nvf_harmonic_coordinates import main
    main(model_class=MaterialGradedGapModel, kind='material_graded_gap', gap_enrichment=True, mesh_grading=True)
