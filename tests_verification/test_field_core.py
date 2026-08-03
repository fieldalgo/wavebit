"""Near-field reconstruction (fieldwave._core.fields) against the frozen legacy output.

Field reconstruction had no coverage at all before the rewrite reached it, so its reference
values were captured from the legacy per-backend implementation just before that was deleted
(gen_goldens_field.py -> golden_legacy_field.npz). These are the first real assertions the
subsystem has ever had.
"""
import pathlib

import numpy as np
import pytest

import fieldwave

GOLDEN = np.load(pathlib.Path(__file__).parent / 'golden_legacy_field.npz')

# Must stay identical to gen_goldens_field.py.
CASES = {
    'te_1d':    dict(pol=0, phi=None, ucell=[[[3., 3., 3., 1., 1., 1.]]], thickness=[0.15],
                     n_top=2., n_bot=1., theta=0.2, fto=[4, 0], period=[0.77], wavelength=0.777),
    'tm_1d':    dict(pol=1, phi=None, ucell=[[[3., 3., 3., 1., 1., 1.]]], thickness=[0.15],
                     n_top=2., n_bot=1., theta=0.2, fto=[4, 0], period=[0.77], wavelength=0.777),
    '1d_2layer': dict(pol=0, phi=None, ucell=[[[3., 3., 1., 1., 3., 1.]], [[2., 2., 2., 1., 1., 1.]]],
                      thickness=[0.15, 0.22], n_top=1., n_bot=1., theta=0.1, fto=[3, 0],
                      period=[0.7], wavelength=0.9),
    'conical':  dict(psi=0.4, phi=0.3, ucell=[[[3., 3., 3., 1., 1., 1.]]], thickness=[0.15],
                     n_top=2., n_bot=1., theta=0.2, fto=[4, 0], period=[0.77], wavelength=0.777),
    '2d':       dict(psi=0.9, phi=0.4, ucell=np.random.default_rng(0).uniform(1, 4, (1, 5, 5)),
                     thickness=[0.2], n_top=1., n_bot=1.5, theta=0.3, fto=[2, 2],
                     period=[1., 1.], wavelength=0.9),
    '2d_2layer': dict(psi=1.1, phi=0.5, ucell=np.random.default_rng(1).uniform(1, 4, (2, 4, 4)),
                      thickness=[0.15, 0.22], n_top=1.2, n_bot=1., theta=0.2, fto=[2, 2],
                      period=[0.9, 0.9], wavelength=0.7),
}
VARIANTS = (('default', (True, False, False)), ('all', (True, True, True)))


def _solved(name, backend=0):
    kw = dict(CASES[name])
    kw['ucell'] = np.asarray(kw['ucell'], dtype=float)
    mee = fieldwave.call_mee(backend=backend, **kw)
    mee.conv_solve()
    return mee


@pytest.mark.parametrize('tag,sfi', VARIANTS, ids=[v[0] for v in VARIANTS])
@pytest.mark.parametrize('name', list(CASES))
def test_field_matches_legacy_golden(name, tag, sfi):
    mee = _solved(name)
    assert mee.grating_type_assigned == GOLDEN[f'{name}__grating_type']
    got = np.asarray(mee.calculate_field(res_x=5, res_y=4, res_z=6, set_field_input=sfi))
    want = GOLDEN[f'{name}__{tag}']
    assert got.shape == want.shape, (got.shape, want.shape)
    assert np.allclose(want, got, atol=1e-12, rtol=0), np.max(np.abs(want - got))


@pytest.mark.parametrize('backend', [1, 2], ids=['jax', 'torch'])
@pytest.mark.parametrize('name', ['te_1d', 'conical', '2d'])
def test_field_cross_backend(name, backend):
    """One implementation, so jax and torch reconstruct the same field as numpy."""
    pytest.importorskip('jax' if backend == 1 else 'torch')
    mee = _solved(name, backend=backend)
    got = np.asarray(mee.calculate_field(res_x=5, res_y=4, res_z=6))
    want = GOLDEN[f'{name}__default']
    assert got.shape == want.shape, (got.shape, want.shape)
    assert np.allclose(want, got, atol=1e-10, rtol=0), np.max(np.abs(want - got))
