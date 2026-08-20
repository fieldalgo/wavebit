"""The in-cell field against S4 — the external oracle field reconstruction never had.

Every other reference test in this directory checks efficiencies or amplitudes. This one
checks the field itself, which until 2026-08-14 was pinned only by a golden captured from this
project's own earlier implementation. That golden froze a wrong sign convention and kept the
suite green for as long as it existed; see ``gen_goldens_s4_field.py`` for what replaced it and
why S4 stops at uniform stacks.

The comparison is **exact, not converged**: a uniform stack truncates on neither side, so
agreement is at 1e-15 rather than at some tolerance chosen to fit. All five reconstruction
paths are run against the same S4 numbers — pure TE, pure TM, conical, 2D, and the first-order
(4ff) route — because they are five different assemblies of the same physics and the oracle
costs nothing extra per path.
"""
import pathlib

import numpy as np
import pytest

import wavebit

GOLDEN = np.load(pathlib.Path(__file__).parent / 'golden_s4_field.npz')

WL = 620.0
PERIOD = (400.0, 380.0)
RES_Z = 7

# S4's `pAmplitude=1` is exactly `-1` times this library's `psi=0`, on every component at every
# angle (spread ~1e-15, measured). A global sign on the incident amplitude, not a frame parity
# -- a parity would move E_z, H_x, H_y and leave E_t alone, and this moves all six together.
POL_SIGN = {'s': 1.0, 'p': -1.0}

# Which reconstruction each path exercises, and what it can express. The pure 1D layouts have
# no azimuth and one polarization each, so they only claim the cases they can actually reach.
PATHS = ('te', 'tm', 'conical', '2d', '4ff')


def _interior(cell):
    """Drop the two face samples of every layer, leaving what the golden stores.

    ``calculate_field`` samples ``linspace(0, d, res_z)`` inclusive, so index 0 and index
    ``res_z-1`` sit exactly on the layer boundaries -- where ``E_z`` is discontinuous and two
    solvers may report different one-sided limits and both be right.
    """
    n_layers = cell.shape[0] // RES_Z
    keep = np.concatenate([np.arange(k * RES_Z + 1, k * RES_Z + RES_Z - 1)
                           for k in range(n_layers)])
    return cell[keep]


def _as_six(v, path):
    """Promote the two 3-component 1D layouts to ``(Ex, Ey, Ez, Hx, Hy, Hz)``."""
    z = np.zeros_like(v[:, 0])
    if path == 'te':                       # (Ey, Hx, Hz)
        return np.stack([z, v[:, 0], z, v[:, 1], z, v[:, 2]], axis=-1)
    if path == 'tm':                       # (Hy, Ex, Ez)
        return np.stack([v[:, 1], z, v[:, 2], z, v[:, 0], z], axis=-1)
    return v


def _reconstruct(path, stack, theta, phi, pol):
    n = np.array([x[1] for x in stack]).reshape(-1, 1, 1)
    d = [float(x[0]) for x in stack]
    kw = dict(backend=0, n_top=1.0, n_bot=1.0, wavelength=WL, thickness=d,
              theta=np.deg2rad(theta), answers='field')
    if path in ('te', 'tm'):
        kw.update(tetm=path, phi=None, fto=(0, 0), period=[PERIOD[0]], ucell=n)
    elif path == 'conical':
        kw.update(psi=np.pi / 2 if pol == 's' else 0.0, phi=np.deg2rad(phi),
                  fto=(0, 0), period=[PERIOD[0]], ucell=n)
    else:
        kw.update(psi=np.pi / 2 if pol == 's' else 0.0, phi=np.deg2rad(phi),
                  fto=(0, 0), period=PERIOD,
                  ucell=np.repeat(np.repeat(n, 2, axis=1), 2, axis=2))
    solver = wavebit.rcwa(**kw)
    if path == '4ff':
        solver.formulation = '4ff'
    solver.conv_solve()
    cell = np.asarray(solver.calculate_field(res_x=1, res_y=1, res_z=RES_Z))
    if cell.ndim == 5:
        cell = cell[0]
    return _as_six(_interior(cell[:, 0, 0, :]), path)


def _cases():
    for key in GOLDEN.files:
        if key.endswith('__z') or key.endswith('__stack'):
            continue
        stack_name, spec = key.split('__')
        theta, phi, pol = spec.split('_')
        yield key, stack_name, float(theta), float(phi), pol


CASES = list(_cases())


@pytest.mark.parametrize('path', PATHS)
@pytest.mark.parametrize('key,stack_name,theta,phi,pol',
                         CASES, ids=[c[0] for c in CASES])
def test_the_reconstructed_field_matches_s4(path, key, stack_name, theta, phi, pol):
    """Entry for entry, every component, at 1e-13 of the cell's own scale.

    A uniform stack is where the sign and frame conventions live and where neither solver
    truncates, so this is the check that would have caught the 2026-08-14 defect on the day it
    was introduced — and the one that now stops it coming back through any of the five paths.
    """
    if path in ('te', 'tm'):
        # no azimuth, and one polarization each: `te` is s, `tm` is p
        if phi != 0.0 or POL_SIGN[pol] != POL_SIGN['s' if path == 'te' else 'p']:
            pytest.skip(f'{path} cannot express phi={phi:g}, pol={pol}')

    stack = GOLDEN[f'{stack_name}__stack']
    want = POL_SIGN[pol] * GOLDEN[key]
    got = _reconstruct(path, stack, theta, phi, pol)

    assert got.shape == want.shape, (got.shape, want.shape)
    scale = np.max(np.abs(want))
    assert np.max(np.abs(want - got)) < 1e-13 * scale, np.max(np.abs(want - got)) / scale


def test_every_component_is_exercised_somewhere():
    """The comparison above must not be passing on components that are identically zero.

    Six components, and a normal-incidence case has two of them at zero in every path — so
    without this the suite could lose ``E_z`` entirely and still look thorough.
    """
    seen = np.zeros(6, dtype=bool)
    for key, stack_name, theta, phi, pol in CASES:
        arr = GOLDEN[key]
        seen |= np.max(np.abs(arr), axis=0) > 1e-6 * np.max(np.abs(arr))
    assert seen.all(), f'never exercised: {[i for i in range(6) if not seen[i]]}'


def test_the_polarization_sign_is_a_global_one():
    """``POL_SIGN`` is an amplitude convention, and this states the evidence for that.

    If S4's p wave differed from ours by a *frame* parity instead, ``E_z``, ``H_x``, ``H_y``
    would flip and ``E_x``, ``E_y``, ``H_z`` would not. Then a single global sign could not
    reconcile them, and this test is what tells the two apart — the distinction that the whole
    2026-08-14 fix turned on.
    """
    key, stack_name, theta, phi, pol = next(
        c for c in CASES if c[4] == 'p' and c[2] != 0.0 and c[3] != 0.0)
    stack = GOLDEN[f'{stack_name}__stack']
    got = _reconstruct('2d', stack, theta, phi, pol)
    raw = GOLDEN[key]

    ratio = got[np.abs(raw) > 1e-9 * np.max(np.abs(raw))] / raw[np.abs(raw) > 1e-9 * np.max(np.abs(raw))]
    assert np.max(np.abs(ratio - np.mean(ratio))) < 1e-12, 'not a single constant'
    assert abs(np.mean(ratio) + 1.0) < 1e-12, np.mean(ratio)
