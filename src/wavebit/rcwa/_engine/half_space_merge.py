"""Layers of a half-space's own medium, merged into that half-space.

A layer whose material is the half-space's beside it is not an obstacle: the plane between them
reflects nothing, and the light in that layer is the half-space's own plane waves. Solved as a
layer it is not harmless, though. At a complex ``k_par`` a lit wave can grow across it, and the
growth turns the last bit of rounding into an O(1) change of the answer -- a *knife edge*
(``docs/theory/rcwa/connecting-algorithms.md``, section 10). The theory value there is the stack
with that layer deleted and the reference planes moved back to the faces, so that is what this
module computes:

* the kernels solve the remaining stack (:class:`MergePlan` names which layers went);
* :func:`face_exponents` carries each amplitude from the remaining stack's faces to the full
  stack's: an incident wave that reaches the remaining stack ``d_in`` deeper, a reflected order
  that climbs back ``d_in``, a transmitted order that runs ``d_out`` further;
* :func:`shift_result` and :func:`shift_scattering` apply them to the amplitudes, efficiencies and
  scattering matrix, and :func:`zone_coefficients` writes the field inside a merged layer as the
  half-space's plane waves.

``kz`` is each half-space's own root, the one the kernels use for that medium
(:func:`~wavebit._common.conventions.boundary_kz` with
:func:`~wavebit._common.conventions.half_space_medium`), so a double negative takes its mirror
root here too.

Which layers merge is a material question, decided once per solve from the scene's own values
(:func:`material_match`), never from a convolution matrix -- a uniform raster's matrices carry
1e-16 of Fourier round-off, and ``solve_conv`` takes matrices that need not be the scene's. The
decision changes the number of layers, which is the shape of the program, so it is taken only on
concrete values that carry no derivative, or on a public adapter's proof that scalar region
materials share the same live input. An independently differentiated layer is not the
half-space, whatever its value is at the point of evaluation; a shared input retains its
derivative through the half-space and the reference-plane shift.
"""
from typing import NamedTuple

import numpy as np

from wavebit._common.backend import array_namespace


class MergePlan(NamedTuple):
    """The public layers a solve merged into its half-spaces.

    Layers ``[0, bottom)`` are the bottom half-space and ``[n - top, n)`` the top one; the
    kernels solve ``[bottom, n - top)``, which may be empty. ``snapped`` lists the public indices
    that merged by ``half_space_merge_tolerance`` rather than by equality: their material was
    within the tolerance of the half-space's and was taken to be it.
    """
    bottom: int
    top: int
    snapped: tuple = ()

    def kept(self, n_layers):
        """The slice of the public stack the kernels solve."""
        return slice(self.bottom, n_layers - self.top)


def _lifted(value, like):
    """A half-space scalar in a layer value's shape: a scalar, ``(3,)`` or ``(..., 3, 3)``."""
    if like.shape[-2:] == (3, 3):
        return value * np.eye(3)
    return np.full(like.shape, value)


def material_match(values, medium, tolerance, dtype=np.complex128):
    """Whether every value of a layer is the half-space's: ``'equal'``, ``'near'`` or ``None``.

    ``values`` is a sequence of ``(epsilon, mu)`` host arrays, one per region of the layer (a
    raster is one region of many pixels; ``mu`` is ``None`` where the layer names none, which is
    ``1``). ``medium`` is the half-space's scalar ``(epsilon, mu)``. Tensors are compared with
    ``medium * I``, so their off-diagonals must vanish.

    ``'equal'`` is bit for bit at the solve's own precision ``dtype``, both sides cast to it:
    the numbers the kernels would see, so an integer ``9`` is ``9.0``, and a float32 ``2.2`` is
    the half-space's ``2.2`` in a single-precision solve and not in a double one.
    ``'near'`` is within ``tolerance`` on one scale for the pair, ``max|delta| <= tolerance *
    max|entry|`` over both media, permittivity and permeability together -- the rule
    ``conventions._same_layer`` uses, and for its reason: a per-entry scale lets a zero
    off-diagonal against rotation round-off decide the verdict.
    """
    eps_h, mu_h = medium
    deviation, scale, equal = 0.0, max(abs(complex(eps_h)), abs(complex(mu_h))), True
    for eps, mu in values:
        for value, h in ((eps, eps_h), (1.0 if mu is None else mu, mu_h)):
            value = np.asarray(value)
            if not np.all(np.isfinite(value)):
                return None                     # not a material, and not the half-space's
            target = _lifted(complex(h), value)
            equal = equal and bool(np.all(value.astype(dtype) == target.astype(dtype)))
            if value.size:
                deviation = max(deviation, float(np.max(np.abs(value - target))))
                scale = max(scale, float(np.max(np.abs(value))))
    if equal:
        return 'equal'
    if tolerance > 0 and deviation <= tolerance * scale:
        return 'near'
    return None


def face_exponents(xp, k0, kz_in, kz_out, specular, d_in, d_out):
    """The exponents of the phases that carry the remaining stack's answer to the full stack's.

    ``kz_in`` / ``kz_out`` are the lit and far half-spaces' roots per order (flattened as the
    amplitudes are), ``d_in`` / ``d_out`` the thickness merged on each side. Returns, each the
    argument of an ``exp``:

    * ``incident`` -- ``i k0 kz_in[spec] d_in``, the incident wave's phase at the remaining
      stack's lit face, which multiplies every field the remaining stack describes;
    * ``reflected`` / ``transmitted`` -- per order, ``incident`` plus the order's own way back
      (``i k0 kz_in[m] d_in``) or onward (``i k0 kz_out[m] d_out``);
    * ``lit`` / ``far`` -- the per-order legs alone, which a scattering matrix needs.

    Exponents rather than exponentials, so that every factor is one ``exp`` of a sum: a growing
    and a decaying leg never meet as ``inf * 0``, and an entry that is exactly zero can be
    masked *before* its ``exp`` is formed (:func:`scaled`).
    """
    lead = 1j * k0 * kz_in[specular] * d_in
    return dict(incident=lead, reflected=lead + 1j * k0 * kz_in * d_in,
                transmitted=lead + 1j * k0 * kz_out * d_out,
                lit=1j * k0 * kz_in * d_in, far=1j * k0 * kz_out * d_out)


def scaled(value, exponent, power=False):
    """``value * exp(exponent)`` -- or ``value * |exp(exponent)|**2`` with ``power`` -- with an
    exactly-zero ``value`` kept zero however large the factor would be.

    An unlit, decoupled order is exactly zero, and its factor may overflow: ``0 * inf`` would be
    NaN where the theory value is zero. The exponent is masked before the ``exp``, not the
    factor after it, so a reverse-mode derivative never multiplies a zero cotangent by
    ``exp(x) = inf`` either.
    """
    xp = array_namespace(value)
    exponent = xp.reshape(exponent, value.shape) if exponent.ndim else exponent
    if power:
        exponent = 2 * xp.real(exponent)
    zero = value == 0
    with np.errstate(over='ignore', invalid='ignore'):
        factor = xp.exp(xp.where(zero, xp.zeros_like(exponent), exponent))
        return xp.where(zero, value, value * factor)


def shift_result(result, exponents):
    """Amplitudes and efficiencies of every incidence, moved to the full stack's faces.

    Each efficiency is its amplitude's squared modulus over a fixed incident power, so it scales
    by ``|factor|**2``; the kernel's own per-order totals ``R`` / ``T`` are scaled rather than
    re-summed, so the efficiency and amplitude rungs keep one set of bits. ``power_entering`` is
    the caller's (see :func:`entering_power`).
    """
    reflected, transmitted = exponents['reflected'], exponents['transmitted']
    shifted = {}
    for key, sub in result.items():
        if key not in ('incident', 's_inc', 'p_inc') or sub is None:
            shifted[key] = sub
            continue
        sub = dict(sub)
        for name, exponent in (('r_s', reflected), ('r_p', reflected),
                               ('t_s', transmitted), ('t_p', transmitted)):
            if sub.get(name) is not None:
                sub[name] = scaled(sub[name], exponent)
        for name, exponent in (('R', reflected), ('R_s', reflected), ('R_p', reflected),
                               ('T', transmitted), ('T_s', transmitted), ('T_p', transmitted)):
            if sub.get(name) is not None:
                sub[name] = scaled(sub[name], exponent, power=True)
        shifted[key] = sub
    return shifted


def shift_scattering(scattering, exponents):
    """The scattering matrix of the full stack from the remaining stack's.

    Rows are outgoing and columns incoming amplitudes, port 1 the lit side
    (``S11' = Dl S11 Dl``, ``S12' = Dl S12 Df``, ``S21' = Df S21 Dl``, ``S22' = Df S22 Df``),
    with ``Dl`` / ``Df`` the lit and far legs per order, repeated over the s and p halves of a
    coupled layout; each entry's factor is one exponential of its row's and its column's legs.
    The ports themselves -- media, wavevectors, channels -- do not change.
    """
    if scattering is None:
        return None
    lit, far = exponents['lit'], exponents['far']
    xp = array_namespace(scattering.S11)
    if scattering.S11.shape[-1] == 2 * lit.shape[0]:
        lit, far = xp.concat([lit, lit]), xp.concat([far, far])
    return scattering._replace(
        S11=scaled(scattering.S11, lit[:, None] + lit[None, :]),
        S12=scaled(scattering.S12, lit[:, None] + far[None, :]),
        S21=scaled(scattering.S21, far[:, None] + lit[None, :]),
        S22=scaled(scattering.S22, far[:, None] + far[None, :]))


def entering_power(incoming, reflected):
    """The net power the lit face passes into the stack over the incident power.

    ``incoming`` / ``reflected`` are one incidence's six-component spectra on the lit face
    (``(Ex, Ey, Ez, Z0 Hx, Z0 Hy, Z0 Hz)`` per order, ``ordinary_spectra``). Each order's normal
    Poynting flux is ``Re(Ex conj(Hy) - Ey conj(Hx))``, interference between the incident and
    reflected waves included; at a real ``k_par`` distinct orders carry no cross flux over a
    cell. The common factor and the direction cancel in the ratio. Needed only when a lossy lit
    medium was merged at a real ``k_par``: the kernels' value is the remaining stack's lit face,
    and between the two faces the merged medium absorbs.
    """
    xp = array_namespace(incoming)

    def flux(c):
        return xp.sum(xp.real(c[:, 0] * xp.conj(c[:, 4]) - c[:, 1] * xp.conj(c[:, 3])))
    return flux(incoming + reflected) / flux(incoming)


def zone_coefficients(xp, z, owner, edges, merged, data, name, n_layers):
    """Each order's six components at heights ``z`` inside the merged layers, else zero.

    ``merged`` is ``(bottom, top, lit side)``, the field state's static plan; ``data`` its
    ``'merge'`` entry: ``k0``, the half-spaces' roots ``kz_bottom`` / ``kz_top``, the incident
    exponent (:func:`face_exponents`), and per incidence the full stack's incoming spectrum
    (``incoming``) and the remaining stack's own reflected and transmitted spectra
    (``inner``), referenced at its own faces and lit at unit amplitude there; the incident
    exponent joins their own in one ``exp``. On the lit side
    the field is the incident wave from the full stack's face plus the reflected waves from the
    remaining stack's face; on the far side, the transmitted waves from the remaining stack's
    face. Each family is written from the face it is referenced at, so a wave that grows across
    the merged layer is never formed beyond double range only to be divided back. A wave
    ``exp(i k0 kz_z (z - z_ref))`` has ``kz_z`` the ``z`` component of its wavevector: up
    (``+kz``) for a wave entering through the bottom face or leaving through the top one, down
    (``-kz``) otherwise. Heights outside a zone are clipped into it before the exponential and
    masked after, as :func:`~wavebit.rcwa._engine.field_result._local_height` does for layers, so
    an unselected sample cannot overflow or poison a gradient, and a selected one keeps its own
    tangent.
    """
    bottom, top, lit_side = merged
    k0 = data['k0']
    total = edges[-1]
    lit_bottom = lit_side == 'bottom'
    incoming = data['incoming'][name]
    reflected, transmitted = data['inner'][name]
    lead = data['incident']

    def wave(spectrum, kz_z, z_ref, low, high, mask, offset=None):
        # A selected height keeps its own tangent, a face sample included (min/max would split
        # it with the bound); an unselected one is clipped into the zone.
        local = xp.where(mask, z, xp.minimum(xp.maximum(z, low), high)) - z_ref
        exponent = 1j * k0 * kz_z[None, :] * local[:, None]
        if offset is not None:
            exponent = exponent + offset
        # Orders this family does not carry, and samples outside the zone, take exp(0): masked
        # before the exp, so neither an overflow nor its derivative reaches the answer.
        live = mask[:, None] & xp.any(spectrum != 0, axis=-1)[None, :]
        with np.errstate(over='ignore', invalid='ignore'):
            phase = xp.exp(xp.where(live, exponent, xp.zeros_like(exponent)))
            value = spectrum[None, :, :] * phase[:, :, None]
        return xp.where(live[:, :, None], value, xp.zeros_like(value))

    zones = []
    if bottom:
        zones.append(('bottom', owner < bottom, edges[0], edges[bottom]))
    if top:
        zones.append(('top', owner >= n_layers - top, edges[n_layers - top], total))
    result = None
    for side, mask, low, high in zones:
        kz = data['kz_' + side]
        up = kz if side == 'bottom' else -kz       # into the stack through this side's face
        inner = high if side == 'bottom' else low  # the remaining stack's face
        outer = low if side == 'bottom' else high  # the full stack's face
        if (side == 'bottom') == lit_bottom:
            part = (wave(incoming, up, outer, low, high, mask)
                    + wave(reflected, -up, inner, low, high, mask, lead))
        else:
            part = wave(transmitted, -up, inner, low, high, mask, lead)
        result = part if result is None else result + part
    return result
