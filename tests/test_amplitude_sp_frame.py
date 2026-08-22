"""The per-order s/p frame the complex amplitudes are quoted in.

``R`` and ``T`` are ``|r|**2``, so nothing built out of them can see the sign of an
amplitude, and until this file every check in the suite was built out of them or out of
quantities a per-order sign cancels in. That blind spot shipped one: :func:`order_azimuth`
computed each order's frame as ``atan(ky/kx)``, and a ratio has already discarded
``sign(kx)``, so every order with ``Re(kx) < 0`` was quoted in ``-s_hat``, ``-p_hat``. A
half-turn negates ``cos`` and ``sin`` together, which makes it a diagonal ``+-1`` similarity
on the stack algebra: it cancels out of every efficiency, out of energy conservation, out of
the reconstructed field, and out of every external golden this project has. It survives in
exactly one place, the sign of the reported ``r_s`` / ``r_p`` / ``t_s`` / ``t_p``, which is
what a caller doing phase-sensitive work reads.

Three independent statements, because no one of them is conclusive alone:

1. **the transposed twin** -- a physical identity between two solves, which is what first
   made the sign visible;
2. **the reconstructed field** -- an outside opinion on what the amplitudes must be, since
   the field is pinned against S4 / fmmax / RETICOLO and the flip cancels inside it;
3. **continuity** -- a convention may be surprising, but it may not be discontinuous in a
   continuous input, and this rules out "it is just a different frame".

Kept out of ``test_order_azimuth.py`` on purpose: that file is about the *singular points*
of the azimuth, and these are about the frame everywhere else.
"""
import numpy as np
import pytest

import wavebit

# Chiral on purpose -- no mirror line in x or in y, so no order can come out right by a
# symmetry the cell happens to have. Two arms of different index hanging off one bar.
_UC = np.ones((1, 16, 16))
_UC[0, 2:7, 3:12] = 12.1104          # permittivity: 3.48**2
_UC[0, 6:13, 9:13] = 2.10
_UC[0, 11:14, 2:6] = 1.60

_WL, _THETA, _PSI = 250.0, np.deg2rad(25.0), 0.7
_PX, _PY, _FX, _FY = 500.0, 800.0, 5, 3
# wl/Px = 0.5 and wl/Py = 0.3125 at wl=250, so 18 orders propagate and the retained grid
# straddles kx = 0 -- both halves of the plane are populated, which is the premise.
_THICK = [120.0]


def _solve(structure, period, truncation_order, phi, psi, theta=_THETA, backend=0, **kw):
    return wavebit.rcwa(backend=backend, top=1.0, bot=1.0, thickness=_THICK,
                        wavelength=_WL, theta=theta, phi=phi, psi=psi, factorization='nvf',
                        structure=structure, period=list(period), truncation_order=list(truncation_order),
                        **kw).solve().given


def _order_kxy(phi, theta=_THETA, px=_PX, py=_PY, fx=_FX, fy=_FY):
    """The in-plane wavevector of every retained order, on the ``(n_orders_y, n_orders_x)`` grid."""
    kx = np.sin(theta) * np.cos(phi) + np.arange(-fx, fx + 1) * _WL / px
    ky = np.sin(theta) * np.sin(phi) + np.arange(-fy, fy + 1) * _WL / py
    return np.meshgrid(kx, ky)


def test_the_premise_both_halves_of_the_kx_plane_are_populated():
    """Guard: without orders on each side of ``kx = 0`` this file asserts nothing.

    ``==`` on the count rather than ``>``, so that retuning the case has to come back here
    and restate what it now covers instead of silently thinning the coverage to one side.
    """
    KX, _ = _order_kxy(np.pi / 2)
    assert (KX < 0).sum() == 35 and (KX > 0).sum() == 42, (KX < 0).sum()


# ------------------------------------------------------------------ 1. the transposed twin

def test_a_grating_and_its_transpose_report_the_same_amplitudes():
    """Transposing the cell and the incidence is the same physical problem, amplitudes too.

    The recipe is a reflection of the whole scene about the plane ``x = y``: transpose the
    cell, swap the period pair, swap the ``truncation_order`` pair, send ``phi -> pi/2 - phi``, and
    conjugate the polarization (``a_s -> -a_s``), because a reflection reverses handedness.
    Everything physical must survive it, and the amplitudes are physical once a frame is
    fixed -- the reflection maps order ``(m, n)``'s frame onto order ``(n, m)``'s, so the
    two solves must agree entry for entry with no per-order correction at all.

    That last clause is the whole test. The efficiencies transferred at 2e-15 throughout the
    defect's life and so did ``|r_s|``; what did not was ``r_s`` itself, by 4.4e-1 -- a sign
    on exactly the orders with ``sign(kx) != sign(ky)``. Those are the orders the reflection
    moves *across* ``kx = 0``, which is why a transpose is the cheap way to see a half-plane
    error and a rotation or a translation is not.

    Asserted per order rather than on a norm: a sign on one weak order is the failure mode,
    and a norm over 77 of them would average it away.
    """
    a = _solve(_UC, (_PX, _PY), (_FX, _FY), np.pi / 2, _PSI)
    b = _solve(np.transpose(_UC, (0, 2, 1)), (_PY, _PX), (_FY, _FX), 0.0, -_PSI)

    for name in ('R', 'R_s', 'R_p', 'T_s', 'T_p'):
        x = np.asarray(getattr(a, name), dtype=np.complex128)
        y = np.asarray(getattr(b, name), dtype=np.complex128).T
        assert np.abs(x - y).max() < 1e-13, f'{name} does not transfer'

    for name in ('r_s', 'r_p', 't_s', 't_p'):
        x = np.asarray(getattr(a, name), dtype=np.complex128)
        y = np.asarray(getattr(b, name), dtype=np.complex128).T
        if name.endswith('_s'):
            y = -y                                  # the a_s -> -a_s of the reflection
        assert np.abs(x).max() > 0.1, f'{name} came back empty, so this asserts nothing'
        worst = np.abs(x - y).max()
        assert worst < 1e-13, (
            f'{name} disagrees by {worst:.3e} while |{name}| transfers at '
            f'{np.abs(np.abs(x) - np.abs(y)).max():.3e} -- a per-order sign, not a magnitude')


# ------------------------------------------------------------------ 2. the field says so

def test_the_amplitudes_are_the_field_projected_on_zhat_cross_k():
    """The reported ``r_s`` is the top face's field in the ``s_hat = z_hat x k_hat`` frame.

    This is the outside opinion, and it works because the defect *cannot* reach the field:
    reversing an order's frame reverses its basis vectors and its coefficients together, so
    the reconstruction -- which is pinned entry for entry against S4 and fmmax -- is right
    either way. That makes it a valid arbiter for the amplitudes rather than a restatement
    of them.

    Transverse ``E`` is continuous across the top interface, so its Fourier coefficient at
    order ``(m, n)`` is the superstrate's: the incident wave, which lives only at ``(0, 0)``,
    plus the reflected amplitude at its own phase reference. Every other order is therefore
    the reflected field outright, and no phase bookkeeping is needed.

    The scale comes out ``1`` rather than fitted, which is a second assertion for free: it
    says ``r_s`` *is* that field amplitude against a unit incident one, the contract
    ``_core/result.py`` states.
    """
    solver = wavebit.rcwa(backend=0, top=1.0, bot=1.0, thickness=_THICK, wavelength=_WL,
                          theta=_THETA, phi=np.pi / 2, psi=_PSI, factorization='nvf',
                          structure=_UC, period=[_PX, _PY], truncation_order=[_FX, _FY], answers='field')
    res = solver.solve().given

    # More sample points than retained orders, so the decomposition is overdetermined and a
    # missing order would show up as a residual instead of being absorbed.
    xs = np.linspace(0.0, _PX, 4 * _FX + 3, endpoint=False)
    ys = np.linspace(0.0, _PY, 4 * _FY + 3, endpoint=False)
    cell = np.asarray(solver.reconstruct_field(sample_x=xs, sample_y=ys, sample_z=np.array([0.0])),
                      dtype=np.complex128)
    cell = cell.reshape((-1,) + cell.shape[-3:])[0]           # (ny, nx, 6) at z = 0

    KX, KY = _order_kxy(np.pi / 2)
    k0 = 2 * np.pi / _WL
    basis = (np.exp(1j * k0 * np.outer(ys, KY[:, 0]))[:, None, :, None]
             * np.exp(1j * k0 * np.outer(xs, KX[0]))[None, :, None, :])
    basis = basis.reshape(ys.size * xs.size, KX.size)
    cx, cy = (np.linalg.lstsq(basis, cell[:, :, i].reshape(-1), rcond=None)[0].reshape(KX.shape)
              for i in (0, 1))
    assert np.abs(basis @ cx.reshape(-1) - cell[:, :, 0].reshape(-1)).max() < 1e-12, (
        'the top-face field is not a sum of the retained orders, so the projection below '
        'is reading something other than the reflected amplitudes')

    varphi = np.arctan2(KY, KX)                               # s_hat = z_hat x k_hat_par
    E_s = np.cos(varphi) * cy - np.sin(varphi) * cx
    r_s = np.asarray(res.r_s, dtype=np.complex128)

    # (0, 0) also carries the incident wave; the weak orders carry no sign worth reading.
    live = np.abs(r_s) > 1e-3
    live[_FY, _FX] = False
    assert (live & (KX < 0)).sum() >= 10, 'too few kx<0 orders to test'

    assert np.abs(E_s[live] - r_s[live]).max() < 1e-12, (
        'the reported amplitudes are not this field in the z_hat x k_hat frame; '
        f'the {(live & (KX < 0)).sum()} kx<0 orders are the ones to look at '
        f'(|E_s + r_s| there = {np.abs(E_s[live & (KX < 0)] + r_s[live & (KX < 0)]).max():.3e})')


# ------------------------------------------------------------------ 3. continuity

def test_an_orders_amplitude_is_continuous_when_its_own_kx_crosses_zero():
    """Nothing physical happens at ``kx = 0``, so nothing reported may jump there.

    At ``phi = 0`` order ``m = -1`` has ``kx = sin(theta) - wl/Px``, which crosses zero at
    ``theta = 30 deg`` exactly. The order stays propagating and well away from a Rayleigh
    point -- ``R`` and ``|r_s|`` cross with no feature at all -- so a sign change in ``r_s``
    there cannot be physics, and cannot be a convention either. Conventions are allowed to
    be surprising; they are not allowed to be discontinuous in a continuous input.

    ``n = 1``, not ``n = 0``: the ``ky == 0`` row has ``atan(0/kx) == 0`` on both sides of
    the crossing, so the defect is invisible along it and a test sited there would have
    passed throughout. That row is a different question -- a frame reversed identically on
    both sides, which only the two tests above can see.

    Calibrated against itself rather than against a chosen tolerance: ``r_s`` varies with
    ``theta`` for real physical reasons, so the assertion is that the step *across* the
    crossing is the same size as an equal step that does not cross it. There is no number to
    tune, and it stays honest if the case is ever retuned. With the defect present the
    crossing step was 2\\|r\\| against a control step of ~1e-3 -- three orders clear.
    """
    at = (_FY + 1, _FX - 1)                         # order (m, n) = (-1, +1)
    # 30 deg exactly is where kx of order -1 changes sign; the four points are equally
    # spaced, so exactly one of the three gaps straddles it.
    steps = [_solve(_UC, (_PX, _PY), (_FX, _FY), 0.0, _PSI, theta=np.deg2rad(deg))
             for deg in (29.997, 29.999, 30.001, 30.003)]
    assert float(np.real(np.asarray(steps[0].R)[at])) > 1e-3, 'the order carries no power'

    for name in ('R', 'r_s', 'r_p'):
        v = [complex(np.asarray(getattr(s, name), dtype=np.complex128)[at]) for s in steps]
        crossing = abs(v[2] - v[1])
        control = max(abs(v[1] - v[0]), abs(v[3] - v[2]))
        assert crossing < 5 * control, (
            f'{name} jumped across kx = 0: step {crossing:.3e} against a control step of '
            f'{control:.3e} ({v[1]:.8f} -> {v[2]:.8f})'
            + ('  (a sign)' if abs(v[1] + v[2]) < 0.1 * abs(v[1]) else ''))


@pytest.mark.parametrize('backend, name', [(1, 'jax'), (2, 'torch')])
def test_the_traced_backends_quote_the_same_frame(backend, name):
    """The sign the fix puts back is a boolean cast to a complex dtype, which is the one line
    here the three array namespaces could disagree about.

    NumPy is covered by the test above; this is the same claim on the two backends where an
    ``astype`` of a comparison is not obviously the same operation, and where a wrong answer
    would reach anyone differentiating through a 2D solve.
    """
    pytest.importorskip({1: 'jax', 2: 'torch'}[backend])
    if backend == 1:
        import wavebit.on_jax  # noqa: F401  -- lifts x64 before any array is built

    a = _solve(_UC, (_PX, _PY), (_FX, _FY), np.pi / 2, _PSI, backend=backend)
    b = _solve(np.transpose(_UC, (0, 2, 1)), (_PY, _PX), (_FY, _FX), 0.0, -_PSI,
               backend=backend)
    for attr in ('r_s', 'r_p', 't_s', 't_p'):
        x = np.asarray(getattr(a, attr), dtype=np.complex128)
        y = np.asarray(getattr(b, attr), dtype=np.complex128).T
        if attr.endswith('_s'):
            y = -y
        assert np.abs(x).max() > 0.1, f'{name}: {attr} came back empty'
        assert np.abs(x - y).max() < 1e-12, f'{name}: {attr}'


@pytest.mark.parametrize('layer_algorithm', ['ETM', 'SMM'])
def test_both_stack_algorithms_quote_the_same_frame(layer_algorithm):
    """``scattering.py`` calls the azimuth twice more, on paths the tests above do not take.

    The two stack algorithms are separate code that must land on one answer; comparing them
    on amplitudes rather than efficiencies is what makes this a frame check rather than a
    duplicate of ``test_stability.py``.
    """
    a = _solve(_UC, (_PX, _PY), (_FX, _FY), np.pi / 2, _PSI)
    b = _solve(_UC, (_PX, _PY), (_FX, _FY), np.pi / 2, _PSI, layer_algorithm=layer_algorithm)
    for name in ('r_s', 'r_p', 't_s', 't_p'):
        x = np.asarray(getattr(a, name), dtype=np.complex128)
        y = np.asarray(getattr(b, name), dtype=np.complex128)
        assert np.abs(x - y).max() < 1e-11, f'{layer_algorithm}: {name}'
