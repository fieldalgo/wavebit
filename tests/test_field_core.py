"""Near-field reconstruction (wavebit._core.fields) against the frozen legacy output.

Field reconstruction had no coverage at all before the rewrite reached it, so its reference
values were captured from the legacy per-backend implementation just before that was deleted
(gen_goldens_field.py -> golden_legacy_field.npz). These are the first real assertions the
subsystem has ever had.
"""
import pathlib

import numpy as np
import pytest

import wavebit

GOLDEN = np.load(pathlib.Path(__file__).parent / 'data' / 'golden_legacy_field.npz')

# Must stay identical to gen_goldens_field.py.
CASES = {
    'te_1d':    dict(tetm='te', phi=None, ucell=[[[3., 3., 3., 1., 1., 1.]]], thickness=[0.15],
                     n_top=2., n_bot=1., theta=0.2, fto=[4, 0], period=[0.77], wavelength=0.777),
    'tm_1d':    dict(tetm='tm', phi=None, ucell=[[[3., 3., 3., 1., 1., 1.]]], thickness=[0.15],
                     n_top=2., n_bot=1., theta=0.2, fto=[4, 0], period=[0.77], wavelength=0.777),
    '1d_2layer': dict(tetm='te', phi=None, ucell=[[[3., 3., 1., 1., 3., 1.]], [[2., 2., 2., 1., 1., 1.]]],
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
    solver = wavebit.rcwa(backend=backend, **kw)
    solver.conv_solve()
    return solver


# The one deliberate departure from the frozen values, applied to the GOLDEN rather than to
# the output so the file keeps meaning what it meant. Slot-for-slot, per layout:
#
#   3 components, TE  (Ey, Hx, Hz)      ->  E_t and H_z stand, H_x flips
#   3 components, TM  (Hy, Ex, Ez)      ->  E_t stands, H_y and E_z flip
#   6 components      (Ex,Ey,Ez,Hx,Hy,Hz) -> E_z, H_x, H_y flip
#
# See wavebit/_core/fields.py's module docstring: the legacy implementation returned those
# three in the frame whose +z points *out* of the stack -- contradicting docs/conventions.md,
# docs/formulation.md's own sanity anchor, and the pair _vacuum_modes writes, all three of
# which have said `(Ey, Hx) = (1, -kz)` for a +z wave the whole time.
_LEGACY_PARITY = {3: {0: (1, -1, 1), 1: (-1, 1, -1)}, 6: (1, 1, -1, -1, -1, 1)}


def legacy_to_declared(golden, tetm_index=None):
    """The frozen legacy cell, carried into the frame the library declares."""
    n = golden.shape[-1]
    par = _LEGACY_PARITY[6] if n == 6 else _LEGACY_PARITY[3][tetm_index]
    return golden * np.asarray(par)


def _tetm_index(name):
    return {'te': 0, 'tm': 1}.get(CASES[name].get('tetm'))


@pytest.mark.parametrize('tag,sfi', VARIANTS, ids=[v[0] for v in VARIANTS])
@pytest.mark.parametrize('name', list(CASES))
def test_field_matches_legacy_golden(name, tag, sfi):
    """Still the legacy field — up to the one sign change made on purpose on 2026-08-14.

    The golden is not regenerated for that change and must not be: the legacy per-backend
    implementation it was captured from was deleted as soon as the rewrite reached this
    subsystem, so re-running the generator would re-freeze *this* solver's output and quietly
    turn the only legacy-parity evidence in the repo into a change detector. (The generator
    now refuses to run, for that reason.) Transforming the golden instead keeps the claim
    intact and puts the departure where a reader can see it.
    """
    solver = _solved(name)
    assert solver.grating_type_assigned == GOLDEN[f'{name}__grating_type']
    got = np.asarray(solver.calculate_field(res_x=5, res_y=4, res_z=6, set_field_input=sfi))
    want = legacy_to_declared(GOLDEN[f'{name}__{tag}'], _tetm_index(name))
    assert got.shape == want.shape, (got.shape, want.shape)
    assert np.allclose(want, got, atol=1e-12, rtol=0), np.max(np.abs(want - got))


@pytest.mark.parametrize('name', list(CASES))
def test_the_legacy_parity_is_necessary(name):
    """The transform above must not be quietly simplified away as a no-op.

    Comparing against the raw golden has to *fail*, and fail on exactly the components the
    parity touches — otherwise this file would still be green with the sign fix reverted, which
    is the state the repo was in before it was found.
    """
    got = np.asarray(_solved(name).calculate_field(res_x=5, res_y=4, res_z=6))
    raw = GOLDEN[f'{name}__default']
    par = np.asarray(_LEGACY_PARITY[6] if raw.shape[-1] == 6
                     else _LEGACY_PARITY[3][_tetm_index(name)])
    assert not np.allclose(raw, got, atol=1e-12, rtol=0), 'the parity would be a no-op here'
    for i, sign in enumerate(par):
        scale = np.max(np.abs(raw[..., i]))
        if scale < 1e-9:
            continue                       # identically zero in this case, so it says nothing
        gap = np.max(np.abs(raw[..., i] - got[..., i])) / scale
        assert (gap > 1e-3) == (sign < 0), f'component {i}: sign {sign}, raw gap {gap:.2e}'


# ------------------------------------------------------- the frame the components live in

FRAME_WL = 620.0
FRAME_N = 1.5
FRAME_D = [90.0, 260.0, 90.0]


def maxwell_residual(F, kx, ky, eps, d, res_z, layer, wavelength=FRAME_WL):
    """Max curl-equation residual over one layer's interior, per equation.

    On Fourier coefficients, in the frame where ``+z`` runs *with* the cell's ``z`` index:
    ``d/dx -> i k0 kx``, ``d/dy -> i k0 ky``, ``d/dz`` by a central difference.
    ``linspace(0, res_z, res_z)`` is inclusive, so the step is ``d / (res_z - 1)``.

    ``eps`` is the full 3x3 tensor, so ``tests/test_field_4ff.py`` can share this on a
    z-coupled layer; a scalar is promoted. ``F`` is ``(n_z, 6)``, which is what a single
    retained order gives at ``res_x = res_y = 1`` — the field is then one plane wave and the
    sample at the origin *is* its Fourier coefficient.
    """
    eps = np.asarray(eps, dtype=complex)
    if eps.ndim == 0:
        eps = eps * np.eye(3, dtype=complex)
    k0 = 2 * np.pi / wavelength
    h = d / (res_z - 1)
    s = np.asarray(F[layer * res_z:(layer + 1) * res_z], dtype=complex)
    Ex, Ey, Ez, Hx, Hy, Hz = (s[:, i] for i in range(6))
    D = [sum(eps[r, c] * (Ex, Ey, Ez)[c] for c in range(3)) for r in range(3)]
    m = slice(1, -1)

    def dz(v):
        return (v[2:] - v[:-2]) / (2 * h)

    residuals = [
        1j * k0 * ky * Ez[m] - dz(Ey) - 1j * k0 * Hx[m],          # (curl E)_x = i k0 Hx
        dz(Ex) - 1j * k0 * kx * Ez[m] - 1j * k0 * Hy[m],          # (curl E)_y = i k0 Hy
        1j * k0 * (kx * Ey[m] - ky * Ex[m]) - 1j * k0 * Hz[m],    # (curl E)_z = i k0 Hz
        1j * k0 * ky * Hz[m] - dz(Hy) + 1j * k0 * D[0][m],        # (curl H)_x = -i k0 Dx
        dz(Hx) - 1j * k0 * kx * Hz[m] + 1j * k0 * D[1][m],        # (curl H)_y = -i k0 Dy
        1j * k0 * (kx * Hy[m] - ky * Hx[m]) + 1j * k0 * D[2][m],  # (curl H)_z = -i k0 Dz
    ]
    scale = k0 * np.max(np.abs(s))
    return np.array([np.max(np.abs(r)) / scale for r in residuals])


def _as_six(v, path):
    """Promote the two 3-component 1D layouts to the full (Ex,Ey,Ez,Hx,Hy,Hz)."""
    z = np.zeros_like(v[:, 0])
    if path == 'te_slab':          # (Ey, Hx, Hz)
        return np.stack([z, v[:, 0], z, v[:, 1], z, v[:, 2]], axis=-1)
    if path == 'tm_slab':          # (Hy, Ex, Ez)
        return np.stack([v[:, 1], z, v[:, 2], z, v[:, 0], z], axis=-1)
    return v


FRAME_PATHS = {
    'te_slab': dict(tetm='te', phi=None, fto=(0, 0), period=[400.0]),
    'tm_slab': dict(tetm='tm', phi=None, fto=(0, 0), period=[400.0]),
    'conical': dict(psi=0.7, phi=0.0, fto=(0, 0), period=[400.0]),
    '2d': dict(psi=0.7, phi=np.deg2rad(41.0), fto=(0, 0), period=(400.0, 380.0)),
}


def _frame_cell(path, res_z, formulation=None):
    one = np.array([[[1.0]], [[FRAME_N]], [[1.0]]])
    two = np.array([[[1.0, 1.0], [1.0, 1.0]],
                    [[FRAME_N, FRAME_N], [FRAME_N, FRAME_N]],
                    [[1.0, 1.0], [1.0, 1.0]]])
    kw = dict(backend=0, n_top=1.0, n_bot=1.0, wavelength=FRAME_WL, thickness=FRAME_D,
              theta=np.deg2rad(33.0), ucell=two if path == '2d' else one, **FRAME_PATHS[path])
    solver = wavebit.rcwa(**kw)
    if formulation:
        solver.formulation = formulation
    solver.conv_solve()
    raw = np.asarray(solver.calculate_field(res_x=1, res_y=1, res_z=res_z))
    kx, ky = solver.get_kx_ky_vector(FRAME_WL)
    if raw.ndim == 5:
        raw = raw[0]
    return (_as_six(raw[:, 0, 0, :], path),
            complex(np.asarray(kx).reshape(-1)[0]),
            0.0 if ky is None else complex(np.asarray(ky).reshape(-1)[0]))


@pytest.mark.parametrize('path,formulation', [('te_slab', None), ('tm_slab', None),
                                              ('conical', None), ('2d', '2ff'), ('2d', '4ff')])
def test_every_path_returns_a_maxwell_field_in_the_declared_frame(path, formulation):
    """The check that was missing, and that a whole convention slipped through the gap in.

    ``docs/conventions.md`` declares ``+z`` down, into the stack. Every path must therefore
    return six components that satisfy the curl equations **with that ``+z``** — and until
    2026-08-14 none of them did: ``E_z``, ``H_x`` and ``H_y`` came back in the opposite frame,
    which is a perfectly good field that answers a different question.

    The frame was never in doubt anywhere else. ``docs/formulation.md`` §1 anchors its
    ``(M1)/(M2)`` on a vacuum wave travelling ``+z`` and writes the pair ``(Ey, Hx) =
    (1, -kz)``; ``tests/test_anisotropic_berreman.py::_vacuum_modes`` writes that same pair;
    ``solve_2d_4ff`` derives in it. Only the reconstruction disagreed.

    Nothing caught it for a reason worth recording. ``docs/time-convention-audit.md`` §8 did
    look at field reconstruction and passed it, on two checks — does ``|E|`` decay going down
    an absorbing layer, and does a uniform slab's interior field match the closed form in
    amplitude and phase. Both are about ``E_t``, which is the one part the parity leaves
    alone. Cross-backend and cross-path checks were blind for the same reason: they compare
    the reconstruction with itself.

    So this asserts the *rate* as well as the size. The field is analytic inside a layer, so
    halving the step must quarter the residual; a wrong frame does not converge at all, it
    sits at O(1). Size alone would be a tolerance argument.
    """
    worst = []
    for res_z in (32, 64, 128):
        F, kx, ky = _frame_cell(path, res_z, formulation)
        r = maxwell_residual(F, kx, ky, FRAME_N ** 2, FRAME_D[1], res_z, layer=1)
        worst.append(np.max(r))

    assert worst[0] < 1e-2, worst
    assert worst[-1] < 1e-3, worst
    for coarse, fine in zip(worst, worst[1:]):
        assert 3.4 < coarse / fine < 4.6, f'rate {coarse / fine:.2f}, residuals {worst}'


@pytest.mark.parametrize('path,formulation', [('te_slab', None), ('tm_slab', None),
                                              ('conical', None), ('2d', '2ff'), ('2d', '4ff')])
def test_the_declared_frame_is_the_one_that_converges(path, formulation):
    """The control: the frame the library used to ship in must now *fail* the check above.

    Without this, undoing the fix would leave the test above green if it were ever loosened —
    and "the residual is small" was never the claim. The other frame is not slightly worse,
    it is O(1) and flat in the step.
    """
    F, kx, ky = _frame_cell(path, 64, formulation)
    F_old = np.asarray(F, dtype=complex).copy()
    F_old[:, [2, 3, 4]] *= -1                      # back into the pre-2026-08-14 frame
    r = maxwell_residual(F_old, kx, ky, FRAME_N ** 2, FRAME_D[1], 64, layer=1)
    assert np.max(r) > 0.5, np.max(r)


@pytest.mark.parametrize('backend', [1, 2], ids=['jax', 'torch'])
@pytest.mark.parametrize('name', ['te_1d', 'conical', '2d'])
def test_field_cross_backend(name, backend):
    """One implementation, so jax and torch reconstruct the same field as numpy."""
    pytest.importorskip('jax' if backend == 1 else 'torch')
    solver = _solved(name, backend=backend)
    got = np.asarray(solver.calculate_field(res_x=5, res_y=4, res_z=6))
    want = legacy_to_declared(GOLDEN[f'{name}__default'], _tetm_index(name))
    assert got.shape == want.shape, (got.shape, want.shape)
    assert np.allclose(want, got, atol=1e-10, rtol=0), np.max(np.abs(want - got))
