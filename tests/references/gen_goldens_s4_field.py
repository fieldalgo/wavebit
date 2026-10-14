"""Freeze S4's in-cell **field** into ``golden_s4_field.npz`` — the first external field oracle.

Run once, in an environment with the S4 extension built; ``test_s4_field_golden.py`` then runs
anywhere.

    conda run -n fieldwave python tests/references/gen_goldens_s4_field.py

**Why this file exists.** Every other golden in this directory checks efficiencies or
amplitudes. Field reconstruction had no external check at all — its only reference was
``golden_legacy_field.npz``, captured from this project's own earlier implementation, which is
a change detector rather than an oracle: it can say "we have not changed", never "we are
right". It duly froze a wrong sign convention (``E_z``, ``H_x``, ``H_y`` in the frame whose
``+z`` points *out* of the stack, on samples that ran *into* it -- the declared frame was then
``+z`` into the stack; since 2026-09-21 it is ``+z`` out of it with samples running bottom to
top) and kept the suite green for as long as it existed. That
golden is gone; this one replaces it, and ``tests/fixtures/legacy_rcwa/golden_field.npz`` covers the breadth
S4 cannot reach.

**Where S4 reaches, measured 2026-08-14.** On a *uniform* stack the two solvers agree entry
for entry at ~1e-15, on every field component, every reconstruction path (pure TE, pure TM,
conical, 2D, and the first-order route), at 0/24/47 degrees and both azimuths. There is no
truncation on either side, so this is an exact comparison rather than a converged one.

On a **patterned** layer it is not an oracle. Measured on a 1D binary grating: the continuous
components land at 1e-3 to 1e-2 and ``E_z`` at up to 7.6e-1, because the two codes truncate
and factorize differently and ``E_z`` is discontinuous at the material boundary. That is a
convergence study, not a frozen tolerance, and it is deliberately not attempted here.

**Two conventions this had to establish, both measured rather than assumed.**

* **S4's p amplitude points the other way.** For ``pAmplitude=1`` the whole field is exactly
  ``-1`` times what this library returns for ``psi=0`` — every component, every angle, every
  azimuth, spread ~1e-15. It is a global sign on the incident amplitude, *not* a frame parity:
  a parity would move ``E_z``, ``H_x``, ``H_y`` and leave ``E_t`` alone, and this moves all of
  them together. ``sAmplitude=1`` needs no sign at all.
* **Sample the layer interior, never the interfaces.** ``E_z`` is discontinuous across a layer
  boundary, so the two codes are free to report different one-sided limits there and both be
  right. Sampling the interfaces made ``E_z`` look like the one component that disagreed —
  ratios scattered over 3 orders while everything else sat at 1e-15 — which reads exactly like
  a bug in the component this project had just finished fixing. It was the sample points.
  ``reconstruct_field`` puts a sample on both faces of every layer (``linspace`` is inclusive),
  so the interior is indices ``1 .. sample_z-2``.

**Real permittivity only** — this S4 build rejects a complex ``Epsilon``, so the cases below
are lossless. Absorption is covered against the Airy closed form in ``tests/rcwa/test_lossy_media.py``
and by the ``|E|``-decay check in ``docs/validation/rcwa/time-convention-audit.md`` section 8.

**The stacks are stated in permittivity, and the ``** 2`` that used to sit at ``SetMaterial`` is
gone rather than moved.** S4's ``Epsilon`` always wanted a permittivity; what changed on
2026-08-20 is that wavebit does too, so the conversion this generator was performing has no
remaining side to be on. One consequence is worth knowing before rerunning this: the
``__stack`` arrays in the **committed** golden were written before that flip and hold a
refractive index in their second column. ``test_s4_field_golden`` reads them through
``refractive_index_from_eps`` and says so; rerun this and they become permittivities, at which point
that conversion has to come out. Nothing else in the file moves — the physics is identical and the
field arrays are bit-for-bit the same, which is the only reason this is a safe thing to leave
half-migrated.
"""

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path
import sys

import numpy as np

HERE = repository_path('tests/references')
OUT = HERE / 'golden_s4_field.npz'

WL = 620.0
PERIOD = (400.0, 380.0)
RES_Z = 7                       # per layer, inclusive of both faces; interior is 1..RES_Z-2

# (thickness, relative permittivity) top to bottom -- S4's order, with its z running down from
# the first layer; the test reverses the list into the public bottom-to-top order and its cell
# back into this depth frame. Both start and end in vacuum so the
# incident and transmitted sides are unambiguous, and both put a high-index layer in the middle
# so the interior samples sit where the field is actually structured.
#
# 2.25 and 4.41 are the index-1.5 and index-2.1 layers this list has always described; the
# scene is unchanged and the committed golden still applies to it.
STACKS = {
    'two_media': [(90.0, 1.0), (260.0, 2.25), (90.0, 1.0)],
    'three_media': [(90.0, 1.0), (260.0, 2.25), (140.0, 4.41), (90.0, 1.0)],
}
THETAS = (0.0, 24.0, 47.0)
PHIS = (0.0, 35.0)
POLS = ('s', 'p')


def z_samples(stack):
    """Interior sample points of every layer, as depths from the first layer's front face.

    That is S4's ``z`` and the order ``reconstruct_field`` returned samples in until
    2026-09-21; it now returns them bottom to top, and the test reverses its cell to match.

    Reads only the thicknesses, so it is unaffected by which quantity the second column holds
    — which is what lets ``test_s4_field_golden`` call it on the committed golden's ``__stack``
    as well as on ``STACKS``.
    """
    zs, off = [], 0.0
    for d, _ in stack:
        zs += [off + j * d / (RES_Z - 1) for j in range(1, RES_Z - 1)]
        off += d
    return np.array(zs)


def s4_field(stack, theta, phi, pol, zs):
    import S4
    S = S4.New(Lattice=((PERIOD[0], 0), (0, PERIOD[1])), NumBasis=1)
    for i, (d, eps) in enumerate(stack):
        S.SetMaterial(Name=f'm{i}', Epsilon=float(eps))
        S.AddLayer(Name=f'L{i}', Thickness=float(d), Material=f'm{i}')
    S.SetFrequency(1.0 / WL)
    S.SetExcitationPlanewave(IncidenceAngles=(float(theta), float(phi)),
                             sAmplitude=1.0 if pol == 's' else 0.0,
                             pAmplitude=0.0 if pol == 's' else 1.0, Order=0)
    out = np.empty((len(zs), 6), dtype=complex)
    for i, z in enumerate(zs):
        E, H = S.GetFields(0.0, 0.0, float(z))
        out[i] = [*E, *H]
    return out


def main():
    out = {}
    for stack_name, stack in STACKS.items():
        zs = z_samples(stack)
        out[f'{stack_name}__z'] = zs
        out[f'{stack_name}__stack'] = np.array(stack, dtype=float)
        for theta in THETAS:
            for phi in PHIS:
                for pol in POLS:
                    key = f'{stack_name}__{theta:g}_{phi:g}_{pol}'
                    out[key] = s4_field(stack, theta, phi, pol, zs)
    np.savez_compressed(str(OUT), **out)
    print(f'wrote {len(out)} arrays to {OUT.name}')


if __name__ == '__main__':
    try:
        import S4  # noqa: F401
    except ImportError:
        sys.exit('needs the S4 extension built in this environment')
    main()
