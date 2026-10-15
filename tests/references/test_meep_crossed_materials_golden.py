"""Independent magnetic and electric-pole crossed references from black-box Meep.

The two grids measure finite-discretization sensitivity. RCWA is an additional internal
cross-check, not the source of the golden. No Meep dependency is needed to run this file.
"""

from wavebit.fdtd._periodic.api import fdtd as _legacy_wavebit_fdtd

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path
import hashlib
import json
import sys

import numpy as np
import pytest


HERE = repository_path('tests/references')
sys.path.insert(0, str(HERE))
from tests.references._meep_crossed_material_cases import CASES, epsilon  # noqa: E402
from tests.references._meep_crossed_cases import margin  # noqa: E402

GOLDEN = HERE / 'golden_meep_crossed_materials.npz'


def scene(case):
    return dict(wavelength=case['wavelength'], period=list(case['period']),
                thickness=list(case['thickness']), structure=case['index']**2,
                top=case['n_top']**2, bot=case['n_bot']**2, mu=case['mu'],
                theta=np.deg2rad(case['theta_deg']), phi=np.deg2rad(case['phi_deg']),
                psi=.4, truncation_order=dict(x=1, y=1))


def extract(result, case):
    u, v = list(result.orders_u), list(result.orders_v)
    ix = np.ix_([v.index(n) for n in case['orders_v']], [u.index(m) for m in case['orders_u']])
    return {f'{leg}.{key}': np.asarray(getattr(getattr(result, f'{leg}_inc'), key))[ix]
            for leg in ('s', 'p') for key in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T')}


from tests.references._legacy_conventions import AMPLITUDES, centre_phase


def frozen(data, name, resolution):
    from tests.references._meep_crossed_metal_case import CASE as METAL
    case = METAL if name == 'metal' else CASES[name]
    return {f'{leg}.{key}': data[f'{name}__{resolution}__{leg}__{key}']
                            * (centre_phase(case) if key in AMPLITUDES else 1.)
            for leg in ('s', 'p') for key in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T')}


def gap(a, b, case, power=False):
    errors = []
    for key in a:
        component = key.split('.')[1]
        if (component in ('R', 'T')) != power:
            continue
        n = case['n_top'] if component in ('r_s', 'r_p', 'R') else case['n_bot']
        mask = np.array([[margin(case, m, l, n) > .05 for m in case['orders_u']]
                         for l in case['orders_v']])
        errors.extend(np.abs(a[key][mask] - b[key][mask]))
    assert len(errors) >= 8
    return float(np.max(errors))


@pytest.fixture(scope='module')
def golden():
    with np.load(GOLDEN) as data:
        assert bool(data['complete'])
        yield data


@pytest.fixture(scope='module', params=list(CASES))
def comparison(request):
    name = request.param
    case = CASES[name]
    kw = scene(case)
    results = {}
    for res in (20, 40):
        result, diag = _legacy_wavebit_fdtd(**kw, dispersion=case['dispersion'], poles=case['poles'],
                                   resolution=res, n_periods=160, n_pml=12, pad_scale=1.,
                                   courant=.99, subpixel=True, grading=None,
                                   return_diagnostics=True)
        results[f'fdtd{res}'] = extract(result, case)
        results[f'residual{res}'] = diag['residual']
    kw['structure'] = epsilon(case, case['wavelength'])
    for order in (7, 11):
        kw['truncation_order'] = dict(x=order, y=order)
        results[f'rcwa{order}'] = extract(_legacy_wavebit_rcwa(backend='numpy', **kw).solve(), case)
    # A material omission must be distinguishable from the oracle's grid discrepancy.
    if name == 'magnetic':
        kw['mu'] = np.ones_like(case['mu'])
    else:
        kw['structure'] = case['index']**2
    results['omitted'] = extract(_legacy_wavebit_rcwa(backend='numpy', **kw).solve(), case)
    return name, case, results


@pytest.mark.slow
def test_material_amplitudes_match_independent_reference(comparison, golden):
    name, case, got = comparison
    fine = frozen(golden, name, 40)
    error = gap(got['fdtd40'], fine, case)
    print(name, 'fdtd/meep', error, 'fdtd/ref', gap(got['fdtd40'], got['rcwa11'], case),
          'meep/ref', gap(fine, got['rcwa11'], case), flush=True)
    assert error < .05
    assert gap(got['fdtd40'], fine, case, power=True) < .04


@pytest.mark.slow
def test_both_spatial_refinements_reduce_reference_error(comparison, golden):
    name, case, got = comparison
    assert gap(got['rcwa7'], got['rcwa11'], case) < .005
    for solver in ('meep', 'fdtd'):
        coarse = frozen(golden, name, 20) if solver == 'meep' else got['fdtd20']
        fine = frozen(golden, name, 40) if solver == 'meep' else got['fdtd40']
        a, b = gap(coarse, got['rcwa11'], case), gap(fine, got['rcwa11'], case)
        print(name, solver, 'coarse/fine', a, b, flush=True)
        assert b < a


@pytest.mark.slow
def test_material_omission_is_resolved(comparison, golden):
    name, case, got = comparison
    fine = frozen(golden, name, 40)
    good = gap(got['rcwa11'], fine, case)
    omitted = gap(got['omitted'], fine, case)
    print(name, 'omitted/good', omitted, good, flush=True)
    assert omitted > max(.05, 3*good)


@pytest.mark.parametrize('name', list(CASES))
def test_incident_controls_are_clean(name, golden):
    for resolution in (20, 40):
        for leg in ('s', 'p'):
            assert golden[f'{name}__{resolution}__{leg}__leak'] < 1e-9
            assert golden[f'{name}__{resolution}__{leg}__purity'] < .01


def test_frozen_runtime_controls_belong_to_this_reference():
    report = json.loads((HERE / 'golden_meep_crossed_materials_runtime.json').read_text())
    assert report['complete']
    assert report['reference_sha256'] == hashlib.sha256(GOLDEN.read_bytes()).hexdigest()
    assert report['resolution'] == 40 and report['until_after_sources'] == 240.
    assert set(report['cases']) == {f'{name}.{leg}' for name in CASES for leg in ('s', 'p')}
    for control in report['cases'].values():
        assert 0 <= control['errors']['amplitude'] < .001
        assert control['leak'] < 1e-9 and control['purity'] < .01
