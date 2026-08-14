"""Near-field reconstruction (wavebit._core.fields): the frame it returns, and drift.

Two jobs, and they are different claims — keeping them apart is the whole lesson of this
file's history.

* **Is it right?** Answered by physics and by an outside solver, never by our own past output.
  :func:`test_every_path_returns_a_maxwell_field_in_the_declared_frame` below checks the curl
  equations *and their convergence rate* on every path;
  ``tests/references/test_s4_field_golden.py`` compares against S4 entry for entry at 1e-15 on
  uniform stacks; ``tests/test_field_4ff.py`` adds the cross-formulation twin, interface
  continuity and the flux-to-efficiency tie.
* **Has it moved?** Answered by ``data/golden_field.npz``, frozen from this solver over cases
  the checks above cannot reach — patterned layers, many orders, several layers, every
  ``set_field_input`` variant, the inverse-Fourier kernel and the array layout.

That second file used to be called ``golden_legacy_field.npz``, was captured from the
project's own earlier implementation, and was described by its generator as "the first oracle
it has ever had". It was not an oracle. What it froze was a wrong sign convention — every path
returned ``E_z``, ``H_x`` and ``H_y`` in the frame whose ``+z`` points out of the stack, which
``docs/conventions.md``, ``docs/formulation.md``'s sanity anchor and
``test_anisotropic_berreman.py::_vacuum_modes`` had all contradicted the whole time — and it
kept the suite green for as long as it existed, because a self-captured golden agrees with
whatever produced it. It was retired on 2026-08-14 and re-frozen, once, after the fix; the
re-freeze was verified to equal the old file times exactly that parity and nothing else.

The cases are **imported** from the generator rather than restated here. A copy that has to
"stay identical" to another file is the same failure mode one level down.
"""
import pathlib
import sys

import numpy as np
import pytest

import wavebit

sys.path.insert(0, str(pathlib.Path(__file__).parent / 'data'))
from gen_goldens_field import CASES, RES, VARIANTS, solved  # noqa: E402

GOLDEN = np.load(pathlib.Path(__file__).parent / 'data' / 'golden_field.npz')


@pytest.mark.parametrize('tag,sfi', VARIANTS, ids=[v[0] for v in VARIANTS])
@pytest.mark.parametrize('name', list(CASES))
def test_the_reconstruction_has_not_drifted(name, tag, sfi):
    """The frozen cells, unchanged — a change detector, and only that.

    It cannot say the reconstruction is correct; it says nobody moved it by accident. The
    file it replaced made the stronger-sounding claim and was wrong, so the assertion here is
    deliberately narrow and the correctness claims live in the tests named in the module
    docstring.

    ``grating_type`` and ``formulation`` are checked alongside the numbers: a case that
    silently reroutes to a different path would otherwise be compared against the right values
    through the wrong code.
    """
    solver = solved(name)
    assert solver.grating_type_assigned == GOLDEN[f'{name}__grating_type']
    # Both sides forced to plain str: a str-Enum compared against a numpy string array
    # is True whatever the array holds, which made this assertion vacuous once.
    assert solver.formulation_assigned.value == str(GOLDEN[f'{name}__formulation'])
    got = np.asarray(solver.calculate_field(set_field_input=sfi, **RES))
    want = GOLDEN[f'{name}__{tag}']
    assert got.shape == want.shape, (got.shape, want.shape)
    assert np.allclose(want, got, atol=1e-12, rtol=0), np.max(np.abs(want - got))


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
@pytest.mark.parametrize('name', ['te_1d', 'conical', '2d', '2d_4ff'])
def test_field_cross_backend(name, backend):
    """One implementation, so jax and torch reconstruct the same field as numpy.

    Worth being clear about what this does *not* say: all three backends running one source
    agree by construction, so a convention error is invisible here — it was, for as long as
    the old golden existed. This is a port check.
    """
    pytest.importorskip('jax' if backend == 1 else 'torch')
    got = np.asarray(solved(name, backend=backend).calculate_field(**RES))
    want = GOLDEN[f'{name}__default']
    assert got.shape == want.shape, (got.shape, want.shape)
    assert np.allclose(want, got, atol=1e-10, rtol=0), np.max(np.abs(want - got))
