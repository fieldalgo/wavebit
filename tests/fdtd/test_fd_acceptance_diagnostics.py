"""A missing growth ratio must not erase available quiet-window evidence."""
import numpy as np
import pytest
from tests.fixtures.fdtd.acceptance_diagnostics import diagnostic_basis


def evidence():
    t = np.arange(0., 13., .25)
    e = np.where(t < 3., 1., 1e-12)
    return dict(stable=True, sources_end=2., interior_growth=None,
                interior_energy=(t, e))


def test_complete_quiet_trace_is_evidence_even_without_a_growth_ratio():
    assert diagnostic_basis(evidence(), 13., 1., False) == 'quiet_floor'


@pytest.mark.parametrize('defect', ['missing', 'short', 'no_signal', 'nonfinite', 'above_floor',
                                   'warning', 'still_driving', 'unstable', 'missing_samples'])
def test_missing_or_bad_evidence_cannot_use_the_quiet_floor_exception(defect):
    d = evidence()
    if defect == 'missing': d['interior_energy'] = None
    if defect == 'short': d['sources_end'] = 10.
    if defect == 'no_signal': d['interior_energy'][1][:] = 0.
    if defect == 'nonfinite': d['interior_energy'][1][-1] = np.nan
    if defect == 'above_floor': d['interior_energy'][1][-1] = 1e-6
    if defect == 'still_driving': d['sources_end'] = 13.
    if defect == 'unstable': d['stable'] = False
    if defect == 'missing_samples': d['interior_energy'] = ([0., 12.], [1., 1e-12])
    assert diagnostic_basis(d, 13., 1., defect == 'warning') is None


def test_finite_reported_ratio_is_kept_but_a_growth_warning_blocks_it():
    d = evidence(); d['interior_growth'] = .4
    assert diagnostic_basis(d, 13., 1., False) == 'reported_growth'
    assert diagnostic_basis(d, 13., 1., True) is None
