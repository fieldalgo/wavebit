"""Calibrated incident-basis tails agree with separately truncated scene solves."""
import numpy as np
import pytest
import wavebit
from wavebit._core.scene_fdtd import _scattering_change_bound


KEYS = ('r_s', 'r_p', 't_s', 't_p')


def settings(**changes):
    return dict(dict(structure=np.array([[[2., 3.], [1.2, 1.8]]]), thickness=[.213],
                     top=1., bot=1., period=[.4, .4], wavelength=[.97, 1.03], theta=.2, phi=.6,
                     polarization=(1., .6j), backend=0, resolution=10, n_periods=25,
                     n_pml=4, truncation_order=0, align_layers=True, return_diagnostics=True), **changes)


@pytest.mark.parametrize('side,grading,stride', [('top', None, 1), ('bottom', True, 2)])
def test_each_calibrated_basis_matches_an_independent_prefix(side, grading, stride):
    kw = settings(incidence_side=side, grading=grading, dft_subsample=stride,
                  record_field=41, record_components=['ex'], record_bounds={'z': (-.05, .25)})
    full, d = wavebit.fdtd(**kw, convergence_window=80)
    prefix, earlier = wavebit.fdtd(**dict(kw, n_periods=(d['steps']-80-.25)*d['dt']))
    assert earlier['steps'] == d['steps']-80
    c = d['convergence']
    for basis in ('s_inc', 'p_inc'):
        for key in KEYS:
            expected = np.stack([getattr(getattr(a, basis), key)-getattr(getattr(b, basis), key)
                                 for a, b in zip(full, prefix)])
            np.testing.assert_allclose(c[basis]['amplitude_change'][key], expected, atol=3e-14, rtol=0)
        assert c[basis]['calibration_valid'].shape == (2,)
        assert c[basis]['calibration_valid'].all()
    for key in KEYS:
        # The supplied Jones vector is normalized to unit incident electric field.
        combination = (c['s_inc']['amplitude_change'][key]+.6j*c['p_inc']['amplitude_change'][key])/np.sqrt(1.36)
        np.testing.assert_allclose(c['amplitude_change'][key], combination, atol=1e-15)


def test_scattering_bound_is_attained_by_a_unit_jones_input():
    kw = settings(n_periods=22)
    _, d = wavebit.fdtd(**kw, convergence_window=120)
    c = d['convergence']
    # Choose the output row with the largest bound, then the incident Jones
    # vector that makes its two independently measured tail contributions add.
    key = max(KEYS, key=lambda name: np.max(c['scattering_change_bound'][name]))
    index = np.unravel_index(np.argmax(c['scattering_change_bound'][key]), c['scattering_change_bound'][key].shape)
    row = np.array([c[b]['amplitude_change'][key][index] for b in ('s_inc', 'p_inc')])
    magnitude = np.linalg.norm(row)
    assert magnitude > 1e-8
    jones = np.conj(row)/magnitude
    selected, selected_d = wavebit.fdtd(**dict(kw, polarization=tuple(jones)), convergence_window=120)
    prefix, _ = wavebit.fdtd(**dict(kw, polarization=tuple(jones), n_periods=(d['steps']-120-.25)*d['dt']))
    measured = abs(getattr(selected[index[0]].given, key)[index[1:]]
                   - getattr(prefix[index[0]].given, key)[index[1:]])
    np.testing.assert_allclose(measured, c['scattering_change_bound'][key][index], atol=3e-14)
    for basis in ('s_inc', 'p_inc'):
        for component in KEYS:
            np.testing.assert_array_equal(c[basis]['amplitude_change'][component],
                                          selected_d['convergence'][basis]['amplitude_change'][component])
    # The bound is independent of the supplied polarization and diagnoses the
    # operator itself. Setting tolerance around its largest value flips the flag.
    for factor, expected in ((.9, False), (1.1, True)):
        _, flagged = wavebit.fdtd(**kw, convergence_window=120, convergence_tol=factor*magnitude)
        assert bool(flagged['convergence']['scattering_settled']) is expected


def test_unexcited_prefix_refuses_every_basis_and_operator_bound():
    kw = settings()
    _, plain = wavebit.fdtd(**kw)
    _, d = wavebit.fdtd(**kw, convergence_window=plain['steps']-1)
    c = d['convergence']
    assert not c['scattering_settled']
    for basis in ('s_inc', 'p_inc'):
        assert not c[basis]['calibration_valid'].any()
        assert not c[basis]['amplitudes_settled']
        assert all(np.isnan(value).all() for value in c[basis]['amplitude_change'].values())
    assert all(np.isnan(value).all() for value in c['scattering_change_bound'].values())


@pytest.mark.parametrize('backend', [1, 2])
def test_zero_scattering_bound_has_finite_zero_subgradient(backend):
    # Include an exactly zero row next to a nonzero row. A naive sqrt(sum(abs^2))
    # has an undefined reverse derivative at zero, even with a zero cotangent.
    def objective(value, xp):
        columns = xp.stack((xp.stack((value, 1j*value)), xp.stack((.3+.4j+0*value, .1-.2j+0*value))))
        return _scattering_change_bound(xp, columns).sum()
    if backend == 1:
        jax = pytest.importorskip('jax')
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        value, gradient = jax.jit(jax.value_and_grad(lambda x: objective(x, jnp)))(0.)
    else:
        torch = pytest.importorskip('torch')
        x = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        value = objective(x, torch)
        gradient, = torch.autograd.grad(value, x)
        value, gradient = value.detach(), gradient.detach()
    assert float(value) == pytest.approx(np.sqrt(.3), abs=1e-15)
    assert float(gradient) == 0.
