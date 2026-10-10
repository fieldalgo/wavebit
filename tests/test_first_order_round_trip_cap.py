"""The first-order split's thickness cap: every forward/backward round trip within
``eps**-0.5``, swapped in balanced pairs inside the invariant blocks of the layer operator.

The first-order split had no cap. At a complex ``theta`` (or in a gain layer) a propagating mode's
label can grow across a layer, and the reflection state carries the round trip ``exp(rt)``; past
``rt ~ 37`` both ETM and SMM were O(1) wrong, and past ``rt ~ 700`` they raised
``FloatingPointError``. :func:`~wavebit._core.solve_2d_first_order._cap_round_trips` swaps labels in
pairs, and each part of how it picks them is load-bearing; the controls below put each part back
and must break a case:

* no cap at all (``k0_d=None``): every thick case fails;
* pairs ranked over the whole layer instead of within each invariant block: a uniform layer's
  orders are swapped against each other, a pivot on a reflection that is exactly zero, and SMM
  returned 4e2 in orders that must be dark at 5 um, up to 1e17 at 20 um;
* ranked by the growth of each *label* instead of ``Im(lam)``: a z-coupled tensor shifts every
  ``lam`` of an order by a common amount that cancels in every round trip, so a label can grow
  while its round trip does not and the reverse; at 200 um the rule then left round trips of
  e**75 standing at 0.5+0.3j, e**173 at the 0.2+0.6j the control uses;
* each slice of a layer cut in two deciding on its own thickness instead of the run's
  (:func:`~wavebit._core.conventions.run_k0_d`): the thick slice swaps, the thin one does not,
  and the plane between them -- which reflects nothing -- is a pivot on that zero. SMM has no
  gap medium on this path, so its interface matrix held one eigenvector twice: 0.8 off the uncut
  layer at 20 um + 1.5 um.
"""
import numpy as np
import pytest

import wavebit
import wavebit._core.solve_2d_first_order as first_order

E_GYRO = np.array([[3.1, .23+.11j, .31-.07j], [.23-.11j, 4.2, -.19+.09j],
                   [.31+.07j, -.19-.09j, 2.8]], complex)
M_GYRO = np.array([[1.3, .04j, .08], [-.04j, 1.1, .03j], [.08, -.03j, 1.2]], complex)
S0 = np.array([[[0, 1, 0, 1, 1, 1]]]) * 24 + 1
NUDGE = 1 + 1e-14


def uniform(d, alg, to=(2, 2), theta=0.5+0.3j):
    return wavebit.rcwa(backend='numpy', wavelength=730., period=(2000., 2000.), thickness=[d],
                        truncation_order=to, theta=theta, phi=.19, psi=0.7, factorization='li',
                        top=1., bot=2.25, layer_algorithm=alg,
                        structure=[dict(background=wavebit.Material(E_GYRO, M_GYRO), shapes=[])])


def amplitudes(solver):
    r = solver.solve(auto_detune=False)
    return np.stack([np.asarray(getattr(getattr(r, i), k)) for i in ('s_inc', 'p_inc')
                     for k in ('r_s', 'r_p', 't_s', 't_p')])


def grating(theta, alg, d, structure=S0, formulation='first_order'):
    s = wavebit.rcwa(backend='numpy', top=1.0, bot=25.0, theta=theta, truncation_order=[10],
                     wavelength=900e-9, period=[2000e-9], structure=structure, thickness=[d],
                     complex_dtype=0, fourier_coefficients='integrated', factorization='li',
                     layer_algorithm=alg, psi=np.pi/2, phi=0.3)
    if formulation:
        s.formulation = formulation
    r = s.solve()
    return np.concatenate([np.ravel(np.asarray(getattr(getattr(r, i), k)))
                           for i in ('s_inc', 'p_inc') for k in ('r_s', 'r_p', 't_s', 't_p')])


def relative(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.fixture
def control(monkeypatch):
    """Swap the cap for a broken variant for the duration of one test."""
    def install(name):
        original = first_order._cap_round_trips
        if name == 'none':
            monkeypatch.setattr(first_order, '_cap_round_trips',
                                lambda xp, M, lam, psi, fwd, k0_d: fwd)
        elif name == 'global':
            monkeypatch.setattr(first_order, '_mode_blocks',
                                lambda xp, M, psi: xp.zeros(psi.shape[1], dtype=xp.int64))
        elif name == 'label_growth':
            def by_label(xp, M, lam, psi, fwd, k0_d):
                # the rule with each label's own growth as its rank and contestedness
                h = xp.imag(lam)
                g = xp.where(fwd, -h, h)
                shifted = xp.where(g > 0, lam, xp.real(lam) + 1j*xp.where(fwd, 1e3, -1e3))
                return original(xp, M, shifted, psi, fwd, k0_d)
            monkeypatch.setattr(first_order, '_cap_round_trips', by_label)
    return install


def specular_and_dark(solver):
    """(the specular column of every amplitude, the largest amplitude in any other order)."""
    a = amplitudes(solver)
    a = a.reshape(a.shape[0], -1)
    centre = a.shape[1] // 2
    return a[:, centre], np.max(np.abs(np.delete(a, centre, axis=1)), initial=0.0)


@pytest.mark.parametrize('alg', ['ETM', 'SMM'])
@pytest.mark.parametrize('d', [5e3, 2e4, 2e5])
def test_uniform_tensor_layer_keeps_dark_orders_dark(alg, d):
    # A uniform layer between uniform half-spaces reflects each order into itself, so only the
    # specular order may carry light, and it must equal the single-order solve.
    specular, dark = specular_and_dark(uniform(d, alg))
    single, _ = specular_and_dark(uniform(d, alg, (0, 0)))
    assert dark < 1e-11 * np.max(np.abs(single))
    assert relative(specular, single) < 1e-11


@pytest.mark.parametrize('variant', ['none', 'global'])
def test_uniform_controls_break(control, variant):
    control(variant)
    d = {'none': 5e4, 'global': 5e3}[variant]
    try:
        specular, dark = specular_and_dark(uniform(d, 'SMM'))
    except FloatingPointError:
        return
    single, _ = specular_and_dark(uniform(d, 'SMM', (0, 0)))
    assert max(dark / np.max(np.abs(single)), relative(specular, single)) > 1e-6


def test_label_growth_control_breaks(control):
    # 0.2+0.6j at 200 um: ranking by label growth leaves round trips the cap should have removed
    control('label_growth')
    try:
        specular, dark = specular_and_dark(uniform(2e5, 'ETM', theta=0.2+0.6j))
    except FloatingPointError:
        return
    single, _ = specular_and_dark(uniform(2e5, 'ETM', (0, 0), theta=0.2+0.6j))
    assert max(dark / np.max(np.abs(single)), relative(specular, single)) > 1e-6


@pytest.mark.parametrize('alg', ['ETM', 'SMM'])
@pytest.mark.parametrize('theta,d,tol', [(0.5+0.3j, 50e-6, 1e-10), (0.2+0.6j, 50e-6, 1e-9),
                                         (1.0+0.1j, 200e-6, 1e-9)])
def test_isotropic_grating_forced_first_order_matches_second_order_smm(alg, theta, d, tol):
    reference = grating(theta, 'SMM', d, formulation=None)
    assert relative(grating(theta, alg, d), reference) < tol


@pytest.mark.parametrize('alg', ['ETM', 'SMM'])
def test_gain_layer_at_a_real_angle(alg):
    gain = np.where(S0 > 1, 25-0.5j, 1.0+0j)
    reference = grating(0.5, 'SMM', 200e-6, gain, formulation=None)
    assert relative(grating(0.5, alg, 200e-6, gain), reference) < 1e-12


def coherent_door(d, structure):
    """``R`` and ``T`` of :func:`grating`'s s-polarized scene through ``excitation=``."""
    s = wavebit.rcwa(top=1.0, bot=25.0, truncation_order=(10, 0), wavelength=900e-9,
                     period=(2000e-9, 2000e-9), structure=structure, thickness=[d],
                     fourier_coefficients='integrated', factorization='li',
                     excitation={'top': wavebit.PlaneWave(theta=0.5, phi=0.3,
                                                          polarization=(1., 0.))})
    r = s.solve()
    return np.array([float(np.real(r.output_top)), float(np.real(r.output_bottom))])


@pytest.mark.parametrize('d', [50e-6, 200e-6])
def test_the_coherent_source_door_is_capped_too(d, control):
    """``excitation=`` runs its own first-order cascade (``illumination._cascade``), and it takes
    the cap.

    It passed none until 2026-10-10, so the gain grating above was 1.1e-2 off there at 50 um and
    came back as ``R = 5861`` (for 3.659) at 200 um while every ordinary solve was right; capped
    on its run's thickness, as the ordinary first-order solve is, it is at 2e-15 and 7e-15 in ``R``.
    """
    gain = np.where(S0 > 1, 25-0.5j, 1.0+0j)
    s = wavebit.rcwa(backend='numpy', top=1.0, bot=25.0, theta=0.5, phi=0.3, polarization=(1., 0.),
                     truncation_order=(10, 0), wavelength=900e-9, period=(2000e-9, 2000e-9),
                     structure=gain, thickness=[d], fourier_coefficients='integrated',
                     factorization='li', layer_algorithm='SMM')
    r = s.solve(auto_detune=False)
    reference = np.array([float(np.sum(r.R)), float(np.sum(r.T))])
    assert np.all(np.abs(coherent_door(d, gain) - reference) < 1e-10 * reference)
    control('none')
    assert np.any(np.abs(coherent_door(d, gain) - reference) > 1e-3 * reference)


def test_grating_control_breaks(control):
    control('none')
    reference = grating(0.5+0.3j, 'SMM', 50e-6, formulation=None)
    assert relative(grating(0.5+0.3j, 'ETM', 50e-6), reference) > 1e-3


def test_swaps_are_balanced_within_blocks_and_ignore_a_common_shift():
    rng = np.random.default_rng(7)
    blocks = [rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)) for _ in range(5)]
    M = np.zeros((20, 20), complex)
    for b, A in enumerate(blocks):
        idx = [b, 5 + b, 10 + b, 15 + b]           # interleaved, as the order blocks are
        M[np.ix_(idx, idx)] = A
    lam, psi = np.linalg.eig(M)
    fwd = np.zeros(20, bool)
    for b in range(5):
        members = np.where(np.argmax(np.abs(psi), axis=0) % 5 == b)[0]
        fwd[members[np.argsort(lam[members].real)[2:]]] = True
    for k0_d in (1.0, 30.0, 300.0, 3e4):
        out = np.asarray(first_order._cap_round_trips(np, M, lam, psi, fwd, k0_d))
        block = np.argmax(np.abs(psi), axis=0) % 5
        for b in range(5):
            assert out[block == b].sum() == fwd[block == b].sum()
        h = lam.imag * k0_d
        for b in range(5):
            f, k = out & (block == b), ~out & (block == b)
            cap = -0.5 * np.log(np.finfo(float).eps) + 1e-9
            assert not f.any() or np.max(h[k]) - np.min(h[f]) <= cap
        # A shift of every Im(lam) by a constant changes no round trip, so it changes no label.
        shifted = np.asarray(first_order._cap_round_trips(np, M + 0.37j * np.eye(20), lam + 0.37j,
                                                          psi, fwd, k0_d))
        assert np.array_equal(shifted, out)


def split_uniform(ds, alg, theta=0.5+0.3j):
    """The gyrotropic layer as adjacent slices of one material, thicknesses ``ds``."""
    return wavebit.rcwa(backend='numpy', wavelength=730., period=(2000., 2000.), thickness=list(ds),
                        truncation_order=(2, 2), theta=theta, phi=.19, psi=0.7,
                        factorization='li', top=1., bot=2.25, layer_algorithm=alg,
                        structure=[dict(background=wavebit.Material(E_GYRO, M_GYRO), shapes=[])
                                   for _ in ds])


@pytest.mark.parametrize('alg', ['ETM', 'SMM'])
@pytest.mark.parametrize('theta', [0.5+0.3j, 0.2+0.6j])
def test_a_split_layer_is_the_uncut_layer(monkeypatch, alg, theta):
    """Slices of one material cap as one layer: 20 um + 1.5 um answers as 21.5 um does.

    The control lets each slice decide on its own thickness; SMM then went 0.8 (0.5+0.3j) and
    0.2 (0.2+0.6j) off the uncut layer, ETM 1.4e-10 at 0.5+0.3j -- a hundred times this
    test's bound, where with the runs both sit at 1e-15.
    """
    reference = amplitudes(split_uniform([21.5e3], 'SMM', theta))
    assert relative(amplitudes(split_uniform([20e3, 1.5e3], alg, theta)), reference) < 1e-12
    monkeypatch.setattr(first_order, 'layer_runs', lambda xp, slots, rules=None:
                        [False] * (len(slots) - 1))
    try:
        broken = relative(amplitudes(split_uniform([20e3, 1.5e3], alg, theta)), reference)
    except (FloatingPointError, np.linalg.LinAlgError):
        return
    assert broken > 1e-10, broken


def test_a_layer_written_twice_to_rounding_is_one_run(monkeypatch):
    """Two slices of one gyrotropic tensor, the second scaled by ``1 + 1e-15``, under a grating.

    Not bit-equal, so a bit-equality test kept them apart: the 3.5 um slice swapped labels, the
    1.5 um one did not, and SMM -- no gap medium on this path -- went 0.41 off the uncut layer at
    the plane between them. Joined to ``sqrt(eps)`` they decide together (4e-14). The control
    puts bit equality back.
    """
    grating = dict(background=1., shapes=[dict(type='rectangle', lx=800., ly=900., material=6.25)])

    def solver(materials, thickness, alg):
        return wavebit.rcwa(backend='numpy', wavelength=730., period=(2000., 2000.), thickness=thickness,
                            truncation_order=(2, 2), theta=0.5+0.3j, phi=.19, psi=0.7,
                            factorization='li', top=1., bot=2.25, layer_algorithm=alg,
                            structure=[grating] + [dict(background=m, shapes=[])
                                                   for m in materials])

    uncut = amplitudes(solver([wavebit.Material(E_GYRO, M_GYRO)], [300., 5000.], 'SMM'))
    split = ([wavebit.Material(E_GYRO, M_GYRO), wavebit.Material(E_GYRO * (1 + 1e-15), M_GYRO)],
             [300., 3500., 1500.])
    for alg in ('ETM', 'SMM'):
        assert relative(amplitudes(solver(*split, alg)), uncut) < 1e-11

    from wavebit._core import conventions

    def bit_equal(xp, a, b):
        return all(x is y or (x is not None and y is not None and x.shape == y.shape
                              and bool(xp.all(x == y))) for x, y in zip(a, b))
    monkeypatch.setattr(conventions, '_same_layer', bit_equal)
    assert relative(amplitudes(solver(*split, 'SMM')), uncut) > 1e-3


def test_a_slot_zero_in_one_spelling_still_joins_the_run(monkeypatch):
    """The same tilted uniaxial tensor, the thin slice rotated by ``2*pi``: one material.

    The rotation leaves rounding noise (6e-17) in slots that are exactly zero in the other
    spelling. Scaled by each slot's own size that noise is a full difference, the two slices
    decided apart, and SMM went 22 off the uncut 81.5 um slab at ``theta = 0.5+0.05j``. Scaled
    by the pair's largest entry they join. The control puts the per-slot scale back.
    """
    R = np.array([[np.cos(.6), 0, np.sin(.6)], [0, 1, 0], [-np.sin(.6), 0, np.cos(.6)]])
    tilted = wavebit.Material((R @ np.diag([2.4, 2.4, 3.2]) @ R.T).astype(complex))

    def solve(materials, thickness, alg):
        s = wavebit.rcwa(backend='numpy', wavelength=730., period=(2000., 2000.), thickness=thickness,
                         truncation_order=(2, 2), theta=0.5+0.05j, phi=.19, psi=0.7,
                         factorization='li', top=1., bot=2.25, layer_algorithm=alg,
                         structure=[dict(background=m, shapes=[]) for m in materials])
        s.formulation = 'first_order'
        return amplitudes(s)

    uncut = solve([tilted], [81500.], 'SMM')
    split = ([tilted, tilted.rotated(2 * np.pi, 'z')], [80000., 1500.])
    for alg in ('ETM', 'SMM'):
        assert relative(solve(*split, alg), uncut) < 1e-12

    from wavebit._core import conventions

    def per_slot(xp, a, b):
        tol = float(np.sqrt(np.finfo(float).eps))
        return all(x is y or (x is not None and y is not None and x.shape == y.shape and bool(
            xp.max(xp.abs(x - y)) <= tol * max(float(xp.max(xp.abs(x))), float(xp.max(xp.abs(y))))))
            for x, y in zip(a, b))
    monkeypatch.setattr(conventions, '_same_layer', per_slot)
    assert relative(solve(*split, 'SMM'), uncut) > 1e-2


def test_traced_thickness_matches_eager():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    frozen = uniform(1e3, 'SMM', (1, 1)).freeze_geometry()
    def loss(b, d):
        r = wavebit.rcwa(backend=b, wavelength=730., period=(2000., 2000.), thickness=[d],
                         truncation_order=(1, 1), theta=0.5+0.3j, phi=.19, psi=0.7,
                         factorization='li', top=1., bot=2.25, layer_algorithm='SMM',
                         structure=[dict(background=wavebit.Material(E_GYRO, M_GYRO), shapes=[])]
                         ).solve(geometry=frozen, auto_detune=False)
        xp = jnp if b == 'jax' else np
        return xp.sum(xp.abs(r.s_inc.r_s) ** 2)
    compiled = jax.jit(lambda d: loss('jax', d))
    for d in (3e3, 8e3, 1.2e4, 5e4):                 # none, some, more and all labels swapped
        assert float(compiled(jnp.asarray(d))) == pytest.approx(loss('numpy', d), rel=1e-12)


def test_unsettled_coupling_graph_is_one_block():
    # A chain through the coordinates in scrambled order does not settle in the fixed rounds; the
    # blocks then fall back to one, never to a wrong partition.
    n = 64
    order = np.random.default_rng(3).permutation(n)
    M = np.eye(n, dtype=complex)
    M[order[:-1], order[1:]] = 1.0
    psi = np.eye(n, dtype=complex)
    assert np.array_equal(np.asarray(first_order._mode_blocks(np, M, psi)), np.zeros(n, dtype=int))
    blocks = np.kron(np.eye(4), np.ones((n // 4, n // 4)))     # four dense blocks settle at once
    found = np.asarray(first_order._mode_blocks(np, blocks.astype(complex), psi))
    assert len(set(found)) == 4
