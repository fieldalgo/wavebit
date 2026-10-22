"""Direct scalar interfaces for differentiable, homogeneous nonmagnetic stacks.

No material is removed, including at zero contrast. Rationalized Fresnel
coefficients preserve both the zero reflection and its independent tangent.
The same recursion supplies both ports and the two-face field amplitudes.
"""
from wavebit._common.conventions import boundary_kz, order_azimuth
from wavebit.rcwa._engine.enums import Answers, answers_of
from wavebit.rcwa._engine.scattering import (
    _block, _diag, _field_records, _is_backside, _prepare,
    _sp_scattering_result, order_grid_kxy,
)


def _refine_root(xp, e, gx, gy, q):
    """Newton residual with compensated products, retaining boundary_kz's branch.

    A long slab amplifies one ulp of the normal root into a resolved phase error.
    Evaluate e-kx²-ky²-q² without discarding the product residuals first.
    """
    split = 2**27 + 1 if xp.finfo(xp.real(q).dtype).bits == 64 else 2**12 + 1
    def product(a, b):
        ca, cb = split*a, split*b
        ah, bh = ca-(ca-a), cb-(cb-b)
        al, bl = a-ah, b-bh
        v = a*b
        return v, ((ah*bh-v)+ah*bl+al*bh)+al*bl
    def add(hi, lo, a, b):
        v = hi+a
        part = v-hi
        error = (hi-(v-part))+(a-part)
        return v, lo+b+error
    re, im = xp.real(e)+xp.zeros_like(xp.real(q)), xp.imag(e)+xp.zeros_like(xp.real(q))
    er, ei = xp.zeros_like(re), xp.zeros_like(im)
    for v in (xp.reshape(gx, (-1,)), xp.reshape(gy, (-1,)), q):
        a, b = xp.real(v), xp.imag(v)
        hi, lo = product(a, a)
        re, er = add(re, er, -hi, -lo)
        hi, lo = product(b, b)
        re, er = add(re, er, hi, lo)
        hi, lo = product(a, b)
        im, ei = add(im, ei, -2*hi, -2*lo)
    residual = (re+er)+1j*(im+ei)
    return q + residual/(2*xp.where(q == 0, xp.ones_like(q), q))


def _interfaces(xp, eps, roots, k2):
    pairs = []
    for ea, eb, qa, qb in zip(eps, eps[1:], roots, roots[1:]):
        rs = (ea - eb) / (qa + qb)**2
        # The coupled p port reverses the magnetic reflection coefficient.
        rp = (ea - eb) * (ea*eb - k2*(ea + eb)) / (eb*qa + ea*qb)**2
        t = xp.concat((2*qa/(qa + qb), 2*eb*qa/(eb*qa + ea*qb)))
        pairs.append((xp.concat((rs, rp)), t))
    return pairs


def _walk(xp, pairs, roots, thickness, k0):
    """Reflectance and transmission from each interface, without growing exponentials."""
    r, t = pairs[-1]
    states = []
    for (ri, ti), q, d in zip(pairs[-2::-1], roots[-2:0:-1], xp.flip(thickness, axis=0)):
        exponent = 1j*xp.concat((q, q))*(k0*d)
        grows = xp.real(exponent) > 0
        u = xp.exp(xp.where(grows, -exponent, exponent))
        u2 = u*u
        den = xp.where(grows, u2 + ri*r, 1 + ri*u2*r)
        # f: forward amplitude at entry; b: backward amplitude at exit;
        # through: forward amplitude at exit. Each multiplies incidence at this interface.
        f = ti*xp.where(grows, u2, xp.ones_like(u2))/den
        through = ti*u/den
        b = through*r
        states.append((f, b, through))
        r = xp.where(grows, ri*u2 + r, ri + u2*r)/den
        t = through*t
    return r, t, states[::-1]


def solve(wavelength, kx, ky, n_start, n_end, incidence, thickness, eps,
          anchor, *, phi, incidence_side, answers, lit=None):
    """Kernel-frame bottom-to-top stack, in the existing coupled s/p port basis."""
    xp, C, kx, ky = _prepare(anchor, kx, ky)
    gx, gy = order_grid_kxy(xp, kx, ky)
    k2 = xp.reshape(gx*gx + gy*gy, (-1,))
    roots = [boundary_kz(xp, e, k2) for e in eps]
    roots = [_refine_root(xp, e, gx, gy, q) for e, q in zip(eps, roots)]
    k0 = 2*3.141592653589793/wavelength
    pairs = _interfaces(xp, eps, roots, k2)
    r, t, states = _walk(xp, pairs, roots, thickness, k0)
    rr, tt, reverse = _walk(xp, _interfaces(xp, eps[::-1], roots[::-1], k2),
                             roots[::-1], xp.flip(thickness, axis=0), k0)
    S = tuple(_diag(xp, a) for a in (r, tt, t, rr))
    backside = _is_backside(incidence_side)
    out, column = _sp_scattering_result(
        xp, C, S, kx, ky, roots[0], roots[-1], n_start, n_end,
        incidence, backside, answers, 1., 1., eps[0], eps[-1], lit)
    if not answers_of(answers).covers(Answers.FIELD):
        return out
    if backside:
        states, roots, eps, thickness = reverse, roots[::-1], eps[::-1], xp.flip(thickness, axis=0)
    angle = xp.reshape(order_azimuth(xp, gy, gx, phi), (-1,))
    c, s = xp.cos(angle), xp.sin(angle)
    modes, amplitudes = [], []
    count = k2.shape[0]
    I = xp.astype(xp.eye(count), C)
    incoming = column
    for e, q, d, (f, b, through) in zip(eps[1:-1], roots[1:-1], thickness, states):
        # Invert the Cartesian-to-s/p rotation used by sp_modes_2d.
        W = _block(xp, [[_diag(xp, -s), _diag(xp, -1j*c*q/e)],
                        [_diag(xp, c), _diag(xp, -1j*s*q/e)]])
        V = _block(xp, [[_diag(xp, -1j*c*q), _diag(xp, -s)],
                        [_diag(xp, -1j*s*q), _diag(xp, c)]])
        modes.append([I/e, W, V, -1j*xp.concat((q, q)), d])
        amplitudes.append((f[:, None]*incoming, b[:, None]*incoming))
        incoming = through[:, None]*incoming
    def arrange(a):
        return xp.permute_dims(a, (1, 0))[:, :, None]
    out.update(layer_info_list=_field_records(modes, amplitudes, [None]*len(modes), arrange),
               T1=arrange(column))
    return out
