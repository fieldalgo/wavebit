"""The same answer ladder for ordinary incidence and every source input."""

from wavebit.rcwa._engine.sources import ElectricDipole as _legacy_wavebit_ElectricDipole

from wavebit.rcwa._engine.sources import GaussianBeam as _legacy_wavebit_GaussianBeam

from wavebit.rcwa._engine.sources import OrderExcitation as _legacy_wavebit_OrderExcitation

from wavebit.rcwa._engine.sources import PlaneWave as _legacy_wavebit_PlaneWave

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa
from dataclasses import fields
import weakref

import numpy as np
import pytest

import wavebit

RUNGS = ('efficiency', 'amplitude', 'field')
KINDS = ('ordinary', 'order_map', 'plane', 'orders', 'two_sided', 'sectors', 'gaussian', 'dipole')


def host(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def solver_for(kind, backend='numpy', **updates):
    options = dict(backend=backend, wavelength=1.3, period=(2.7, 2.3),
                   thickness=[.4], truncation_order=(1, 0),
                   structure=np.array([[[4.+.2j]]]), top=1., bot=2.25)
    if kind == 'ordinary':
        options.update(theta=.17, phi=.2, polarization=(.6, .8j))
    elif kind == 'order_map':
        options.update(theta=.17, phi=.2, polarization=(.6, .8j), excitation={0:1., 1:.3j})
    elif kind == 'plane':
        options['excitation'] = _legacy_wavebit_PlaneWave(theta=.17, phi=.2, polarization=(.6, .8j))
    elif kind == 'orders':
        options['excitation'] = _legacy_wavebit_OrderExcitation(kx=.2, orders={0:1., 1:.3j})
    elif kind in ('two_sided', 'sectors'):
        options.update(excitation=[_legacy_wavebit_PlaneWave(kx=.2),
            _legacy_wavebit_PlaneWave(side='bottom', kx=.2 if kind == 'two_sided' else .7, amplitude=.4j)],
            power_region='infinite')
        # Include incident/reflected interference in lossy magnetic half-spaces.
        options.update(top=wavebit.Material(1.+.1j, 1.2), bot=wavebit.Material(2.25+.15j, 1.1))
    elif kind == 'gaussian':
        options.update(excitation=_legacy_wavebit_GaussianBeam(waist=5., sampling=(3, 1)),
                       power_region=((-4., 4.), (-3., 3.)))
    else:
        options.update(excitation=_legacy_wavebit_ElectricDipole(position=(.1, 0., .17),
                       moment=(0., 1e-29, .2e-29j)), length_unit=1e-6)
    return _legacy_wavebit_rcwa(**(options | updates))


def power_values(result):
    if hasattr(result, 'power'):
        # Output fractions are intentionally absent for an internal source.
        return [getattr(result.power, f.name) for f in fields(result.power)
                if f.name not in ('region', 'has_internal_sources')] + [result.output_top, result.output_bottom]
    return [result.R, result.T, result.power_entering, result.absorption]


@pytest.mark.parametrize('backend', ['numpy', 'jax', 'torch'])
@pytest.mark.parametrize('kind', KINDS)
def test_answer_levels_keep_the_same_power_and_expose_the_same_field_functions(kind, backend):
    solver = solver_for(kind, backend)
    # Exercise solve-time level changes on one solver, including dropping state.
    results = {rung: solver.solve(answers=rung) for rung in RUNGS}
    eff, amp, field = (results[rung] for rung in RUNGS)
    for result in (amp, field):
        for lower, higher in zip(power_values(eff), power_values(result)):
            if lower is None:
                assert higher is None
            else:
                np.testing.assert_allclose(host(lower), host(higher), atol=2e-12, rtol=2e-12)
    with pytest.raises(RuntimeError, match="answers='amplitude'"):
        eff.boundary_fields('top')
    for result in (eff, amp):
        with pytest.raises(RuntimeError, match="answers='field'"):
            result.reconstruct_field([0.], [0.], [.1])
    for side in ('top', 'bottom'):
        for part in ('incoming', 'outgoing', 'total'):
            np.testing.assert_allclose(host(amp.boundary_fields(side, [.05], [.03], part)),
                                       host(field.boundary_fields(side, [.05], [.03], part)), atol=2e-12)
    assert field.reconstruct_field([0.], [0.], [.1]).shape == (1, 1, 1, 6)
    assert eff._field_state is None
    assert not amp._field_state.data['groups']
    solver.solve(answers='efficiency')
    # Retained results keep their own access even after lowering the solver level.
    assert field.reconstruct_field([0.], [0.], [.1]).shape == (1, 1, 1, 6)
    assert amp.boundary_fields('top').shape == (1, 1, 6)
    if kind not in ('ordinary', 'order_map'):
        assert eff._spectra is None and not eff._groups
        before = power_values(eff)
        with pytest.raises(RuntimeError, match="answers='amplitude'"):
            eff.boundary_power('cell')
        with pytest.raises(RuntimeError, match='result.power'):
            eff.boundary_power()
        assert all(a is b for a, b in zip(before[:5], power_values(eff)[:5]))
        box = ((-.7, .8), (-.6, .9))
        revised = amp.boundary_power(box)
        direct = solver.solve(answers='efficiency', power_region=box).power
        for name in ('incoming_top', 'incoming_bottom', 'outgoing_top', 'outgoing_bottom', 'net_into_stack'):
            np.testing.assert_allclose(host(getattr(revised, name)), host(getattr(direct, name)), atol=2e-12)


def test_default_source_answer_preserves_boundary_fields_and_region_integration():
    solver = solver_for('plane')
    result = solver.solve()
    assert solver._scene.answers == 'amplitude'
    assert result.boundary_fields('top').shape == (1, 1, 6)
    assert result.boundary_power('cell').output_top == result.output_top
    with pytest.raises(RuntimeError, match="answers='field'"):
        result.reconstruct_field()


def test_efficiency_does_not_keep_hidden_spectra_or_build_a_field_snapshot(monkeypatch):
    import wavebit.rcwa._engine.illumination as module
    references = []
    spectrum_init = module._Spectrum.__init__
    def capture(self, *args, **kwargs):
        spectrum_init(self, *args, **kwargs)
        references.extend([weakref.ref(self), weakref.ref(self.field)])
    def unexpected(*args, **kwargs):
        pytest.fail('An efficiency solve must not build a field snapshot')
    monkeypatch.setattr(module._Spectrum, '__init__', capture)
    monkeypatch.setattr(module, 'source_state', unexpected)
    solver = solver_for('sectors', answers='efficiency')
    result = solver.solve()
    assert np.isfinite(result.output_top)
    assert references and all(ref() is None for ref in references)
    assert solver._result is result and result._field_state is None
