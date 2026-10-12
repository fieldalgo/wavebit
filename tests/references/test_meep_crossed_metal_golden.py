"""A lossy negative-epsilon 3D Drude pattern against an independent Meep reference."""
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

import wavebit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _meep_crossed_metal_case import CASE  # noqa: E402
from _meep_crossed_material_cases import epsilon  # noqa: E402
from _meep_crossed_cases import margin  # noqa: E402
from test_meep_crossed_materials_golden import scene, extract, frozen, gap  # noqa: E402


@pytest.fixture(scope='module')
def golden():
    combined = {}
    for filename in ('golden_meep_crossed_metal.npz', 'golden_meep_crossed_metal_fine.npz'):
        with np.load(HERE / filename) as data:
            assert bool(data['complete'])
            combined.update({key: data[key] for key in data.files})
    return combined


@pytest.fixture(scope='module')
def computed():
    result, diagnostics = wavebit.fdtd(
        **scene(CASE), dispersion=CASE['dispersion'], poles=CASE['poles'],
        resolution=60., n_periods=240, n_pml=12, pad_scale=1., courant=.99,
        subpixel=True, grading=None, return_diagnostics=True)
    assert diagnostics['residual'] < 1e-3
    return extract(result, CASE)


@pytest.mark.slow
def test_complex_metal_amplitudes_match_meep(computed, golden):
    error = gap(computed, frozen(golden, 'metal', 120), CASE)
    print('METAL_AMPLITUDE_ERROR', error, flush=True)
    assert error < .06


@pytest.mark.parametrize('leg', ['s', 'p'])
@pytest.mark.slow
def test_metal_absorption_matches_meep(computed, golden, leg):
    measured = 1 - computed[f'{leg}.R'].sum() - computed[f'{leg}.T'].sum()
    want = float(1 - golden[f'metal__120__{leg}__R'].sum()
                 - golden[f'metal__120__{leg}__T'].sum())
    print('METAL_ABSORPTION', leg, measured, want, flush=True)
    # Removing the pole gives a lossless vacuum layer on a lossless substrate:
    # its exact absorption is zero, far outside this comparison bound.
    assert .1 < measured < 1.
    assert abs(measured - want) < .05
    for key in ('R', 'T'):
        np.testing.assert_allclose(computed[f'{leg}.{key}'],
                                   golden[f'metal__120__{leg}__{key}'], atol=.05, rtol=0.)


def test_external_source_and_runtime_controls(golden):
    assert golden['coarse_reference_sha256'].item() == hashlib.sha256(
        (HERE / 'golden_meep_crossed_metal.npz').read_bytes()).hexdigest()
    for leg in ('s', 'p'):
        for tag in ('30', '60', '60_time240', '120'):
            assert golden[f'metal__{tag}__{leg}__leak'] < 1e-9
            assert golden[f'metal__{tag}__{leg}__purity'] < .01
        assert golden[f'metal__60_time240__{leg}__runtime_change'] < .001


def test_declared_material_regime_and_radiating_order_window():
    for wavelength in (.75, .9, 1.05):
        active = epsilon(CASE, wavelength)[CASE['dispersion'] == 1]
        assert np.all(active.real < 0) and np.all(active.imag > 0)
    for n in (CASE['n_top'], CASE['n_bot']):
        propagating = [(m, l) for m in CASE['orders_u'] for l in CASE['orders_v']
                       if margin(CASE, m, l, n) > .05]
        assert propagating == [(0, 0)]


def test_finer_external_grid_reduces_reference_error(golden):
    # Coarse Meep grids failed the original amplitude/absorption comparison. Keep
    # those arrays and require the added tier to move toward an internal reference,
    # rather than changing a tolerance or replacing the discrepant data.
    kw = scene(CASE)
    kw['structure'] = epsilon(CASE, CASE['wavelength'])
    kw['truncation_order'] = dict(x=11, y=11)
    ref = extract(wavebit.rcwa(backend='numpy', **kw).solve(), CASE)
    errors = [gap(frozen(golden, 'metal', resolution), ref, CASE) for resolution in (30, 60, 120)]
    print('METAL_MEEP_REFINEMENT', errors, flush=True)
    assert errors[2] < errors[1] < errors[0]
