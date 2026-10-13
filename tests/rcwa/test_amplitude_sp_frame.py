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

Section 4 is the pure 1D ``tetm`` path, which joined this frame on 2026-09-27: against the
conical path at ``phi = 0`` amplitude for amplitude and field for field, and the one place the
frame is *meant* to step -- an order's own ``kx = 0`` on the ``ky == 0`` row.
"""

from wavebit.compat import legacy as _legacy_wavebit
import numpy as np
import pytest


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


def _solve(structure, period, truncation_order, phi, psi, theta=_THETA, backend='numpy', **kw):
    return _legacy_wavebit.rcwa(backend=backend, top=1.0, bot=1.0, thickness=_THICK,
                        wavelength=_WL, theta=theta, phi=phi, psi=psi, factorization='nvf',
                        structure=structure, period=list(period), truncation_order=list(truncation_order),
                        **kw).solve().incident


def _order_kxy(phi, theta=_THETA, px=_PX, py=_PY, fx=_FX, fy=_FY, n=1.0):
    """The in-plane wavevector of every retained order, on the ``(n_orders_v, n_orders_u)`` grid.

    ``n`` is the incidence medium's index -- ``theta`` is measured there.
    """
    kx = n * np.sin(theta) * np.cos(phi) + np.arange(-fx, fx + 1) * _WL / px
    ky = n * np.sin(theta) * np.sin(phi) + np.arange(-fy, fy + 1) * _WL / py
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

# A 1D grating lit off its plane, for the conical path: asymmetric in x so no order is saved
# by a mirror, and phi != 0 so every order has ky != 0 -- its kx = 0 crossings then sit away
# from the origin, where the frame is defined on both sides of them.
_UC1D = np.ones((1, 1, 32))
_UC1D[0, 0, 3:14] = 12.1104
_UC1D[0, 0, 17:22] = 2.10
_FX1D, _PHI1D = 7, 0.4
# Glass below, vacuum above: the two faces open into different media, so the p frame's
# cos(theta) is a different number on each, and lit from below the incidence medium is not 1.
_N_BOT = 1.5


def _host(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def _field_scene(grating, side, backend='numpy', theta=_THETA):
    """A solve with the field kept, its result, and its orders' in-plane wavevectors.

    ``'te'`` / ``'tm'`` are the pure 1D path on the conical scene's grating, at the only
    azimuth that path has -- so every order sits on the ``ky == 0`` row, where its frame is
    ``+-y``, and the incident pair is ``(1, 0)`` or ``(0, 1)``.
    """
    if grating == '2d':
        phi, fx, fy, kw = np.pi / 2, _FX, _FY, dict(structure=_UC, factorization='nvf', psi=_PSI)
    elif grating == 'conical':
        phi, fx, fy, kw = _PHI1D, _FX1D, 0, dict(structure=_UC1D, psi=_PSI)
    else:
        phi, fx, fy, kw = 0.0, _FX1D, 0, dict(structure=_UC1D, tetm=grating)
    solver = _legacy_wavebit.rcwa(backend=backend, top=1.0, bot=_N_BOT ** 2, thickness=_THICK,
                          wavelength=_WL, theta=theta, phi=phi, incidence_side=side,
                          period=[_PX, _PY], truncation_order=[fx, fy], answers='field', **kw)
    return solver, solver.solve(), _order_kxy(phi, theta=theta, fx=fx, fy=fy,
                                              n=1.0 if side == 'top' else _N_BOT)


def _face_orders(result, z, KX, KY):
    """``(Ex, Ey)`` of every retained order on the plane at height ``z``, and the fit residual.

    More sample points than retained orders, so the decomposition is overdetermined and a
    missing order would show up as a residual instead of being absorbed. `xs` / `ys` are public
    coordinates, measured from the centre of the cell, and are handed to `sample_x` /
    `sample_y` and to the plane-wave basis as they are: the amplitudes are Fourier
    coefficients about the centre, so no half-period appears anywhere. (Until 2026-09-27 they
    were about the corner, and this sampled from the corner to match.)
    """
    fy, fx = (s // 2 for s in KX.shape)
    xs = np.linspace(-0.5 * _PX, 0.5 * _PX, 4 * fx + 3, endpoint=False)
    ys = np.linspace(-0.5 * _PY, 0.5 * _PY, 4 * fy + 3, endpoint=False)
    cell = result.reconstruct_field(sample_x=xs, sample_y=ys, sample_z=np.array([z]))
    cell = _host(cell).astype(np.complex128).reshape(ys.size, xs.size, 6)
    k0 = 2 * np.pi / _WL
    basis = (np.exp(1j * k0 * np.outer(ys, KY[:, 0]))[:, None, :, None]
             * np.exp(1j * k0 * np.outer(xs, KX[0]))[None, :, None, :])
    basis = basis.reshape(ys.size * xs.size, KX.size)
    ex, ey = (np.linalg.lstsq(basis, cell[:, :, i].reshape(-1), rcond=None)[0].reshape(KX.shape)
              for i in (0, 1))
    return ex, ey, np.abs(basis @ ex.reshape(-1) - cell[:, :, 0].reshape(-1)).max()


def _sp_components(ex, ey, KX, KY, n, reflected):
    """An order's transverse ``E`` read in its s/p frame, as the two amplitudes.

    ``s_hat = z_hat x k_hat_par``. ``p_hat``'s in-plane part lies along ``k_hat_par`` with
    length ``cos(theta_m) = kz / (n k0)`` in the medium the order travels in: **``+`` for a
    reflected order, ``-`` for a transmitted one and for the incident wave**, whichever face is
    lit -- continued onto the decaying branch for an evanescent order. That is what makes
    normal incidence ``E = (-a_p, a_s)`` on both sides (``docs/reference/common/conventions.md``). In the public
    frame it is ``p_hat = k_hat x s_hat`` lit from below and ``s_hat x k_hat`` lit from above:
    the in-plane rule is the one held fixed, so the handedness of ``(s_hat, p_hat, k_hat)``
    follows the lit side.
    """
    varphi = np.arctan2(KY, KX)
    c, s = np.cos(varphi), np.sin(varphi)
    cos_m = np.sqrt(n ** 2 - KX ** 2 - KY ** 2 + 0j) / n
    return c * ey - s * ex, (c * ex + s * ey) / (cos_m if reflected else -cos_m)


def _signs_by_half_plane(E, a, live, KX):
    """Which sign relates ``E`` to ``a`` on each half of the kx plane -- for a failure message.

    The first report of the 2026-09-24 failure read a half-turn into a moved sample plane,
    because this message used to print only ``|E + a|`` on ``kx < 0``. Say what was measured.
    """
    out = []
    for name, half in (('kx>0', KX > 0), ('kx<0', KX < 0)):
        m = live & half
        if not m.any():
            continue
        if np.abs(E - a)[m].max() < 1e-9:
            out.append(f'{name}: agrees')
        elif np.abs(E + a)[m].max() < 1e-9:
            out.append(f'{name}: agrees up to a sign -- a half-turn of the frame')
        else:
            out.append(f'{name}: neither sign')
    return '; '.join(out)


def _incident_pair(grating):
    """The incident ``(a_s, a_p)`` each scene is lit with, in the incident order's own frame."""
    return {'te': (1.0, 0.0), 'tm': (0.0, 1.0)}.get(grating, (np.sin(_PSI), np.cos(_PSI)))


def _assert_amplitudes_are_the_field(result, pair, KX, KY, side, tol, min_negative_kx):
    res = result.incident
    top = sum(_THICK)                 # z = 0 is the bottom interface, z = sum(thickness) the top
    lit, far = (top, 0.0) if side == 'top' else (0.0, top)
    n_lit, n_far = (1.0, _N_BOT) if side == 'top' else (_N_BOT, 1.0)
    centre = tuple(s // 2 for s in KX.shape)

    for channel, z, n in (('r', lit, n_lit), ('t', far, n_far)):
        ex, ey, residual = _face_orders(result, z, KX, KY)
        assert residual < tol, (
            f'the field at z = {z} is not a sum of the retained orders, so the projection '
            f'below is reading something other than the {channel} amplitudes')
        E_s, E_p = _sp_components(ex, ey, KX, KY, n, reflected=channel == 'r')
        a_s, a_p = (_host(getattr(res, f'{channel}_{pol}')).astype(np.complex128)
                    for pol in ('s', 'p'))

        # On the lit face order (0, 0) also carries the incident wave, `pair` in the incident
        # order's own frame, whose p_hat leans the opposite way from a reflected order's.
        # Take it off, and what is left is reflection like every other order.
        if channel == 'r':
            E_s[centre] -= pair[0]
            E_p[centre] += pair[1]

        # The plane first, because nothing else identifies it: every face of the stack is a
        # sum of the retained orders, so the residual above passes on any of them. One
        # channel alone -- s, or p where the incidence has no s -- so that an error in the
        # other frame is reported as one below and not blamed on the plane.
        E_c, a_c = (E_s, a_s) if pair[0] != 0 else (E_p, a_p)
        miss = abs(E_c[centre] - a_c[centre])
        assert miss < tol, (
            f'z = {z} does not read as the {"lit" if channel == "r" else "far"} face of a '
            f'{side}-lit stack: order (0, 0) there is not {"incident + " if channel == "r" else ""}'
            f'{channel}_{"s" if pair[0] != 0 else "p"} (off by {miss:.3e}). z = 0 has been the bottom interface and '
            'z = sum(thickness) the top one since ee31979 (docs/reference/common/conventions.md). Settle the '
            'plane before reading a frame into the orders -- a moved plane reads as a sign.')

        for pol, E, a in (('s', E_s, a_s), ('p', E_p, a_p)):
            if not np.any(a):
                # A pure 1D solve carries one polarization; the other must be empty in the
                # field as well as in the report.
                assert np.abs(E).max() < tol, f'{channel}_{pol} is reported zero and is not'
                continue
            live = np.abs(a) > 1e-3        # the weak orders carry no sign worth reading
            assert (live & (KX < 0)).sum() >= min_negative_kx, f'too few kx<0 orders in {channel}_{pol}'
            worst = np.abs(E[live] - a[live]).max()
            assert worst < tol, (
                f'{side}-lit {channel}_{pol} is not the field in its documented frame, off by '
                f'{worst:.3e} ({_signs_by_half_plane(E, a, live, KX)})')


@pytest.mark.parametrize('theta', [_THETA, -_THETA], ids=['+25deg', '-25deg'])
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('grating', ['2d', 'conical', 'te', 'tm'])
def test_the_amplitudes_are_the_field_projected_on_zhat_cross_k(grating, side, theta):
    """Every reported amplitude is its face's field in the ``s_hat = z_hat x k_hat`` frame.

    This is the outside opinion, and it works because the defect *cannot* reach the field:
    reversing an order's frame reverses its basis vectors and its coefficients together, so
    the reconstruction -- which is pinned entry for entry against S4 and fmmax -- is right
    either way. That makes it a valid arbiter for the amplitudes rather than a restatement
    of them.

    Transverse ``E`` is continuous across an interface, so its Fourier coefficient at order
    ``(m, n)`` on the lit face is the incidence medium's: the incident wave, which lives only
    at ``(0, 0)``, plus the reflected amplitude at its own phase reference. Every other order
    is therefore the reflected field outright, and no phase bookkeeping is needed; the far
    face is the transmitted field outright, ``(0, 0)`` included.

    The scale comes out ``1`` rather than fitted, which is a second assertion for free: it
    says ``r_s`` *is* that field amplitude against a unit incident one, the contract
    ``src/wavebit/observations/diffraction.py`` states.

    Both lit sides, all four amplitudes, and the conical path beside the 2D one: until
    2026-09-27 this read ``r_s`` alone, top-lit, 2D, so a change to the p frame or to the
    backside could not fail it. It also sampled ``z = 0`` as "the top face" -- true when it
    was written, and not since ee31979 put ``z = 0`` at the bottom interface, after which it
    decomposed the transmitted field and reported a frame error the amplitudes did not have
    (they were byte-identical across that commit). Hence the plane check inside, which names
    that failure before any per-order comparison can impersonate a sign.

    The pure 1D path (``'te'`` / ``'tm'``) since 2026-09-27, when it moved into this frame from
    the Cartesian ``E_y`` / ``H_y``: this is its outside arbiter, for the frame and for TM's unit
    E input together -- lit from below the incidence medium is glass, so a unit-H input would
    come back scaled by 1.5 rather than by 1. Both angle signs, because at ``-25 deg`` the
    incident order itself sits at ``kx < 0`` and is subtracted in its own, turned-over frame.
    """
    _, result, (KX, KY) = _field_scene(grating, side, theta=theta)
    _assert_amplitudes_are_the_field(result, _incident_pair(grating), KX, KY, side, tol=1e-12,
                                     min_negative_kx=10 if grating == '2d' else 4)


@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_the_traced_backends_reconstruct_in_the_same_frame(backend, side):
    """The field arbiter above, on the two traced backends.

    The twin test below compares these backends' amplitudes with themselves, which a
    backend-specific error in the amplitudes *and* the field together would pass; this one
    holds them to the field.
    """
    pytest.importorskip(backend)
    if backend == 'jax':
        _legacy_wavebit.enable_x64()
    _, result, (KX, KY) = _field_scene('2d', side, backend=backend)
    _assert_amplitudes_are_the_field(result, _incident_pair('2d'), KX, KY, side, tol=1e-11,
                                     min_negative_kx=10)


# ------------------------------------------------------------------ 3. continuity

def test_an_orders_amplitude_is_continuous_when_its_own_kx_crosses_zero():
    """Nothing physical happens at ``kx = 0`` off the ``ky == 0`` row, so nothing may jump there.

    At ``phi = 0`` order ``(m, n) = (-1, +1)`` has ``kx = sin(theta) - wl/Px``, which crosses
    zero at ``theta = 30 deg`` exactly, with ``ky != 0``. The order stays propagating and well
    away from a Rayleigh point -- ``R`` and ``|r_s|`` cross with no feature at all -- and its
    ``k_par`` never vanishes, so ``s_hat = z_hat x k_hat_par`` turns smoothly through ``+-x``:
    a sign change in ``r_s`` there cannot be physics, and cannot be the frame either. That
    is what made the 2026-08-16 half-turn a defect rather than a convention.

    ``n = 1``, not ``n = 0``, and the difference is the frame's own singular point. On the
    ``ky == 0`` row ``k_par`` passes *through* zero, ``z_hat x k_hat_par`` has no limit there,
    and ``s_hat`` is ``+y`` on one side and ``-y`` on the other: an order's ``r_s`` changes sign
    at its own ``kx = 0`` by design, on every path, the pure 1D one included since 2026-09-27.
    Section 4 pins that step as a step; the Cartesian per-order vector ``r_xyz`` is the
    quantity that stays continuous through it.

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


@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_the_traced_backends_quote_the_same_frame(backend):
    """The sign the fix puts back is a boolean cast to a complex dtype, which is the one line
    here the three array namespaces could disagree about.

    NumPy is covered by the test above; this is the same claim on the two backends where an
    ``astype`` of a comparison is not obviously the same operation, and where a wrong answer
    would reach anyone differentiating through a 2D solve.
    """
    pytest.importorskip(backend)
    if backend == 'jax':
        _legacy_wavebit.enable_x64()

    a = _solve(_UC, (_PX, _PY), (_FX, _FY), np.pi / 2, _PSI, backend=backend)
    b = _solve(np.transpose(_UC, (0, 2, 1)), (_PY, _PX), (_FY, _FX), 0.0, -_PSI,
               backend=backend)
    for attr in ('r_s', 'r_p', 't_s', 't_p'):
        x = np.asarray(getattr(a, attr), dtype=np.complex128)
        y = np.asarray(getattr(b, attr), dtype=np.complex128).T
        if attr.endswith('_s'):
            y = -y
        assert np.abs(x).max() > 0.1, f'{backend}: {attr} came back empty'
        assert np.abs(x - y).max() < 1e-12, f'{backend}: {attr}'


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


# ------------------------------------------------------------------ 4. the pure 1D path

# The scene of the 2026-09-27 investigation: an absorbing, x-asymmetric first layer on a second
# one, so orders m and -m differ and nothing is saved by a mirror. 13 orders, from
# kx/k0 = -1.95 to +2.55 at theta = 0.3; six of them sit at kx < 0 when the incident one does
# not. Rows are refractive indices, squared into permittivity; bottom layer first.
_ROW_TOP = np.array([2.2 + 0.3j] * 5 + [1] * 3 + [3] * 2 + [1] * 6) ** 2
_ROW_BOT = np.array([3] * 7 + [1.5] * 3 + [1] * 6, dtype=complex) ** 2
_1D = dict(structure=np.stack([_ROW_BOT, _ROW_TOP])[:, None, :], thickness=[0.15, 0.22],
           period=[1.6], wavelength=0.6, top=1.0, bot=1.5 ** 2, truncation_order=[6, 0])
_PAIR = {'te': (1, 0), 'tm': (0, 1)}
_PSI_OF = {'te': np.pi / 2, 'tm': 0.0}
_EVERY = ('r_s', 'r_p', 't_s', 't_p', 'R', 'R_s', 'R_p', 'T', 'T_s', 'T_p', 'power_entering')


def _pure_and_coupled(tetm, side, theta, algorithm='ETM', backend='numpy', spelling='psi'):
    """The same light three ways: ``tetm``, and ``phi = 0`` with ``psi`` or with the pair."""
    kw = dict(_1D, backend=backend, theta=theta, incidence_side=side,
              layer_algorithm=algorithm, answers='field')
    pure = _legacy_wavebit.rcwa(tetm=tetm, **kw)
    incidence = {'psi': _PSI_OF[tetm]} if spelling == 'psi' else {'polarization': _PAIR[tetm]}
    coupled = _legacy_wavebit.rcwa(phi=0.0, **incidence, **kw)
    return pure, pure.solve(), coupled, coupled.solve()


def _kx(solver):
    return np.real(_host(solver.order_kxy()[0]).astype(np.complex128)).ravel()


@pytest.mark.parametrize('spelling', ['psi', 'pair'])
@pytest.mark.parametrize('algorithm', ['ETM', 'SMM'])
@pytest.mark.parametrize('theta', [0.0, 0.3, -0.3])
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_the_pure_1d_path_is_the_conical_path_at_zero_azimuth(tetm, side, theta, algorithm,
                                                               spelling):
    """``tetm='te'`` is ``polarization=(1, 0)`` at ``phi = 0``, amplitude for amplitude.

    One frame for every path since 2026-09-27: each order in its own ``s_hat = z_hat x
    k_hat_par``, which on the ``ky == 0`` row is ``sign(Re kx) * y_hat``. Until then the pure
    path quoted its Cartesian primary on every order, so the two disagreed by
    ``sigma_m * sigma_inc`` -- a sign on every order whose ``kx`` is on the other side of zero
    from the incident one's -- while every efficiency agreed to 3e-15 and hid it.

    The premise is asserted, or the comparison could pass on orders that happen to vanish: at
    least four ``kx < 0`` orders carry a real amplitude, and at ``theta = -0.3`` the incident
    order itself sits at ``kx < 0``, so the incident wave is ``-y`` and not ``+y``.
    """
    pure, a, _, b = _pure_and_coupled(tetm, side, theta, algorithm, spelling=spelling)
    kx = _kx(pure)
    main = _host(getattr(a.incident, 'r_s' if tetm == 'te' else 'r_p')).ravel()
    assert ((kx < 0) & (np.abs(main) > 1e-2)).sum() >= 4, 'too few kx < 0 orders carry a sign'
    if theta < 0:
        assert kx[kx.size // 2] < 0, 'the incident order was meant to sit at kx < 0'
    for name in _EVERY:
        x = _host(getattr(a.incident, name)).astype(np.complex128)
        y = _host(getattr(b.incident, name)).astype(np.complex128)
        worst = np.abs(x - y).max()
        assert worst < 1e-12, (
            f'{name}: {worst:.2e}' + ('' if np.ndim(x) == 0 else
                                      f' ({_signs_by_half_plane(x, y, np.abs(y) > 1e-3, kx.reshape(x.shape))})'))


@pytest.mark.parametrize('theta', [0.3, -0.3])
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('tetm', ['te', 'tm'])
@pytest.mark.parametrize('backend', ['jax', 'torch'])
def test_the_traced_backends_put_the_pure_1d_path_in_the_same_frame(backend, tetm, side,
                                                                     theta):
    """The sign is a real ``where`` multiplied into complex arrays, on every backend."""
    pytest.importorskip(backend)
    if backend == 'jax':
        _legacy_wavebit.enable_x64()
    _, a, _, b = _pure_and_coupled(tetm, side, theta, backend=backend)
    for field in _EVERY:
        x = _host(getattr(a.incident, field)).astype(np.complex128)
        y = _host(getattr(b.incident, field)).astype(np.complex128)
        assert np.abs(x - y).max() < 1e-12, (backend, field)


@pytest.mark.parametrize('algorithm', ['ETM', 'SMM'])
@pytest.mark.parametrize('theta', [0.3, -0.3])
@pytest.mark.parametrize('side', ['top', 'bottom'])
@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_the_pure_1d_field_is_the_conical_field(tetm, side, theta, algorithm):
    """One incident wave, so one field: all six components, exactly, both sides.

    Until 2026-09-27 the ratio was ``sigma_inc * h`` -- the pure path was lit by ``+y`` whatever
    the angle, and its TM input was a unit ``H_y`` (``h = n/mu`` of the incidence medium, 1.5
    lit from the glass below). Now both are the incident order's own unit-E wave.
    """
    _, a, _, b = _pure_and_coupled(tetm, side, theta, algorithm)
    x = _host(a.reconstruct_field(sample_x=12, sample_y=1, sample_z=4))
    y = _host(b.reconstruct_field(sample_x=12, sample_y=1, sample_z=4))
    gap = np.abs(x - y).max() / np.abs(y).max()
    assert gap < 1e-12, f'off by {gap:.2e}; best scalar {np.vdot(y, x) / np.vdot(y, y):.6f}'


def test_an_order_at_exactly_kx_zero_takes_the_origin_frame():
    """``kx == 0.0`` on an order other than the incident one is ``+y``, as the origin rule gives.

    A direct ``kx = 2*pi/P`` puts order ``-1`` at exactly zero (asserted, not assumed: it holds
    only if the division rounds to it). ``order_azimuth`` answers that order from ``phi``, which
    is 0 on this row, so ``s_hat = +y``, and the pure path's ``order_frame_sign`` has to agree.
    With a direct ``kx`` the incident column is the lit one, so this reaches the elementwise
    branch of the column rather than the scalar ``sigma_inc`` one.

    NumPy only, on purpose: the same input rounds order ``-1`` to ``kx = -5.55e-17`` on Torch,
    which then takes the ``-y`` frame on both paths. An order at an exact zero gets its sign
    from rounding (``docs/reference/common/conventions.md``), so what holds across backends is this relation
    between the paths, not the sign itself.
    """
    kw = dict(_1D, kx=2 * np.pi / _1D['period'][0], answers='amplitude')
    pure = _legacy_wavebit.rcwa(tetm='te', **kw)
    a = pure.solve()
    coupled = _legacy_wavebit.rcwa(polarization=(1, 0), **kw)     # a direct kx has no phi to give
    b = coupled.solve()
    assert coupled.grating_type == 1, 'the pair was meant to take the conical path'
    kx = _kx(pure)
    assert kx[kx.size // 2 - 1] == 0.0, kx[kx.size // 2 - 1]
    assert abs(complex(_host(a.incident.r_s).ravel()[kx.size // 2 - 1])) > 1e-2
    for name in _EVERY:
        x = _host(getattr(a.incident, name)).astype(np.complex128)
        y = _host(getattr(b.incident, name)).astype(np.complex128)
        assert np.abs(x - y).max() < 1e-12, name


@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_a_complex_angle_keys_the_frame_on_the_real_part(tetm):
    """``theta = pi/6 + 0.2j``: every ``kx`` is complex, and ``Re(kx)`` decides the frame.

    ``sign(kx)`` on a complex array is ``kx/|kx|``, not ``+-1``, so a spelling that reached for
    it would put a phase on every order rather than a sign. The conical path keys its half-turn
    on ``Re(kx)``; the pure one has to land on the same numbers. The incidence medium and the
    angle are those of the RETICOLO complex-``kx`` lossless case.
    """
    kw = dict(structure=np.array([[[4.0, 4.0, 1.0, 1.0, 1.0, 1.0]]]), thickness=[0.3],
              period=[0.7], wavelength=0.9, top=4.0, bot=2.25, truncation_order=[5, 0],
              theta=np.pi / 6 + 0.2j, answers='amplitude')
    pure = _legacy_wavebit.rcwa(tetm=tetm, **kw)
    a = pure.solve()
    b = _legacy_wavebit.rcwa(polarization=_PAIR[tetm], phi=0.0, **kw).solve()
    kx = _host(pure.order_kxy()[0]).astype(np.complex128).ravel()
    assert np.all(np.imag(kx) != 0) and (np.real(kx) < 0).sum() >= 3, kx
    for name in ('r_s', 'r_p', 't_s', 't_p', 'R', 'T'):
        x = _host(getattr(a.incident, name)).astype(np.complex128)
        y = _host(getattr(b.incident, name)).astype(np.complex128)
        assert np.abs(x - y).max() < 1e-12, name


@pytest.mark.parametrize('tetm', ['te', 'tm'])
def test_on_the_ky_zero_row_an_amplitude_turns_over_where_its_own_kx_crosses_zero(tetm):
    """The price of one frame: ``s_hat = z_hat x k_hat_par`` has no limit at ``k_par = 0``.

    On the ``ky == 0`` row ``s_hat`` is ``+y`` on one side of ``kx = 0`` and ``-y`` on the other,
    so an order's ``r_s`` (TE) or ``r_p`` (TM) changes sign exactly where its own ``kx``
    crosses zero, while ``R`` sails through. That is the frame's singular point, not physics,
    and it is what the pure 1D path took on with the frame on 2026-09-27; before, it quoted
    ``E_y`` / ``H_y`` and was continuous here. Pinned both ways -- a sign across the crossing
    and a smooth control either side -- so a revert to the Cartesian frame fails, and so does
    a real jump.

    Order ``-1`` has ``kx = sin(theta) - 0.375``, zero at ``theta = asin(0.375)``.
    """
    theta_c = float(np.arcsin(0.375))
    steps = [_legacy_wavebit.rcwa(tetm=tetm, theta=theta_c + d, backend='numpy', answers='amplitude',
                          **_1D).solve().incident for d in (-3e-5, -1e-5, 1e-5, 3e-5)]
    at = (0, _1D['truncation_order'][0] - 1)
    name = 'r_s' if tetm == 'te' else 'r_p'
    v = [complex(_host(getattr(s, name))[at]) for s in steps]
    r = [float(np.real(_host(s.R)[at])) for s in steps]
    control = max(abs(v[1] - v[0]), abs(v[3] - v[2]))
    assert abs(v[0]) > 1e-2 and control < 1e-2, (v, control)
    assert abs(v[2] + v[1]) < 5 * control, f'{name} did not turn over: {v[1]} -> {v[2]}'
    assert abs(v[2] - v[1]) > 1.9 * abs(v[1]), f'{name} did not turn over: {v[1]} -> {v[2]}'
    assert abs(r[2] - r[1]) < 5 * max(abs(r[1] - r[0]), abs(r[3] - r[2])), r
