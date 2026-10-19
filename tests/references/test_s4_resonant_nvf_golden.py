"""The resonant-NVF contact fixture against S4, read from numbers S4 froze.

S4 solved each case in its own interpreter through
``interface/s4_resonant_nvf.py``; this file only reads the archive, so it needs
no S4 and runs everywhere. The tolerances are those the in-process comparison
used before S4 moved out of the Wavebit process.
"""

import json
import sys
import types

import numpy as np
import pytest

from studies.rcwa.nvf_resonant_spectrum.resonant_nvf_probe import make_solver, solve_case
from tests.references.interface import s4_resonant_nvf as bridge
from tools.repository_paths import repository_path

GOLDEN = repository_path('tests/references/golden_s4_resonant_nvf.npz')


@pytest.fixture(scope='module')
def golden():
    with np.load(GOLDEN, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def test_archive_holds_exactly_the_bridge_cases(golden):
    cases = json.loads(str(golden['__cases']))
    assert set(cases) == set(bridge.CASES)
    for name, (order, shape, method) in bridge.CASES.items():
        assert cases[name]['order'] == order and cases[name]['method'] == method
        assert cases[name]['shape'] == {k: v if k == 'type' else float(v) for k, v in shape.items()}
        assert golden[name].shape == (2,) and np.isfinite(golden[name]).all()


@pytest.mark.parametrize('method', sorted(bridge.BASES))
def test_s4_bridge_matches_the_oblique_uniform_slab(golden, method):
    result = make_solver(1, 32, shape=bridge.uniform_slab()).solve()
    np.testing.assert_allclose(golden[f'uniform_{method}'], [result.T.sum(), result.R.sum()],
                               atol=3e-13, rtol=0)


def test_s4_rotation_units_preserve_the_same_rectangle(golden):
    np.testing.assert_allclose(golden['rotation_fixture'], golden['rotation_equivalent'], atol=3e-12, rtol=0)


@pytest.mark.parametrize('order', [2, 4])
@pytest.mark.parametrize('label', sorted(bridge.SHIFTS))
def test_same_direct_rule_matches_s4_on_the_patterned_contact_fixture(golden, order, label):
    shape = bridge.shifted_fixture(bridge.SHIFTS[label])
    np.testing.assert_allclose(solve_case(order, 64, 'wavebit-direct', shape),
                               golden[f'contact_{label}_order{order}'], atol=3e-12, rtol=0)


@pytest.mark.parametrize('incoming', [np.nan, 0.])
def test_invalid_external_flux_is_rejected_as_a_reference(monkeypatch, incoming):
    """The bridge's own guard, driven by a stand-in: no S4 is loaded here."""
    class Simulation:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

        def GetBasisSet(self):
            return [(m, n) for m in range(-1, 2) for n in range(-1, 2)]

        def GetPowerFlux(self, **kwargs):
            return complex(incoming), 0j

    monkeypatch.setitem(sys.modules, 'S4', types.SimpleNamespace(New=lambda **kwargs: Simulation()))
    with pytest.raises(FloatingPointError, match='not a reference'):
        bridge.s4_solve(1)
