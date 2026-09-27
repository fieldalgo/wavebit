"""Freeze this solver's field cells into ``golden_field.npz`` — a change detector, not an oracle.

    conda run -n fieldwave python tests/data/gen_goldens_field.py --rebaseline

**Read this before regenerating.** These numbers come from this codebase, so they cannot say
the reconstruction is *right* — only that it has not moved. That distinction is not academic
here. The file this replaced, ``golden_legacy_field.npz``, was captured from the project's own
earlier implementation and described in its own generator as "the first oracle it has ever
had". It was not an oracle, and what it actually froze was a **wrong sign convention**: every
path returned ``E_z``, ``H_x`` and ``H_y`` in the frame whose ``+z`` points out of the stack
while sampling ``z`` down into it, contradicting ``docs/conventions.md`` (then ``+z`` down,
into the stack), ``docs/formulation.md``'s own sanity anchor and the pair
``test_anisotropic_berreman.py::_vacuum_modes`` writes. The suite stayed green the whole time,
because a self-captured golden agrees with whatever produced it.

**The archive keeps that day's frame.** ``CASES`` list their layers top to bottom and
``golden_field.npz`` stores cells with ``z`` measured down from the top, as frozen on
2026-08-14. The public API has run bottom to top with ``+z`` up since 2026-09-21, so
:func:`solved` reverses the layer lists on the way in, :func:`build` mirrors each cell back
(reversal along ``z`` plus the ``E_z``/``H_x``/``H_y`` parity) before it would store it, and
``tests/test_field_core.py`` restates the archive the other way (``upward_reference``) at the
comparison. The turn re-froze nothing.

So this file is deliberately scoped to the job it can do:

* **What proves the reconstruction correct** is elsewhere, and now exists.
  ``tests/references/test_s4_field_golden.py`` compares against S4 — a different code, in C++,
  from a different lineage — entry for entry at 1e-15 on uniform stacks, where the conventions
  live and neither solver truncates.
  ``tests/test_field_core.py::test_every_path_returns_a_maxwell_field_in_the_declared_frame``
  checks the curl equations and their convergence rate, which is theory rather than a
  reference. ``tests/test_field_4ff.py`` adds the cross-formulation twin, interface continuity
  and the flux-to-efficiency tie.
* **What this file does** is catch drift across breadth those cannot reach: patterned layers,
  many orders, multiple layers, every ``incidences`` variant, the inverse-Fourier kernel
  and the ``(sample_z, sample_y, sample_x, comp)`` layout — cheaply, on every push.

**Regenerating is not how a failing test gets fixed.** It needs ``--rebaseline`` and prints
what moves, so re-freezing is a decision someone made and a reviewer can see, not a reflex. If
the output changes on purpose, the reason belongs in the commit message.

**The permittivity flip (2026-08-20) left these numbers in place, and the small residue it did
leave is worth naming so nobody re-freezes for it.** The layers are exact: ``1d_2layer``, whose
half-spaces are both vacuum, reproduces every array **bit for bit** with its raster rewritten
from ``3`` to ``9``, which is what says the flip itself moves nothing. Six of the seven cases
*do* move, by 1.0e-14 to 2.4e-13, and all six are the ones with a non-vacuum half-space: a
half-space material now reaches the solve as ``index_from_eps(eps)``, which is complex by
construction (``2+0j`` where ``2.0`` used to sit), and complex arithmetic reassociates the same
expression. That is round-off from a dtype, not a change of answer, and it sits an order inside
the ``atol=1e-12`` both this file's own diff and ``tests/test_field_core.py`` compare at.
"""
import pathlib
import sys

import numpy as np

import wavebit

OUT = pathlib.Path(__file__).resolve().parent / 'golden_field.npz'

def _rand_eps(shape, seed):
    """A random **permittivity** raster: the same draw as before, squared.

    These cases were written as refractive-index rasters drawn from ``uniform(1, 4)``, and the
    solver squared them on the way in. It no longer does (2026-08-20), so the squaring moves
    here and the numbers reaching the solve are bit-for-bit the ones the golden was frozen
    against — verified by regenerating and diffing, which is what ``main`` below prints.

    Squaring the *draw* rather than widening the range is deliberate: ``uniform(1, 16)`` would
    be a different set of permittivities from the same seed, and the golden would move for a
    reason that has nothing to do with the flip.
    """
    return np.random.default_rng(seed).uniform(1, 4, shape) ** 2


# Chosen to hit every reconstruction path and both 1D layouts, at more than one layer and more
# than one order. `formulation` is the only key not passed to the constructor.
#
# **Permittivity, and the scene is the one the golden holds.** Every literal is the square of
# the index this list carried before 2026-08-20 -- 9 was 3, 4 was 2, `top=4.` was `n_top=2.`,
# `bot=2.25` was `n_bot=1.5`. `1.9599999999999997` is `1.4 ** 2` as a double and not a rounding
# of it: this file's goldens are compared at atol=1e-12, but the habit of writing the tidy
# decimal is what breaks the bit-exact assertions elsewhere in the suite.
CASES = {
    'te_1d':     dict(tetm='te', phi=None, structure=[[[9., 9., 9., 1., 1., 1.]]],
                      thickness=[0.15], top=4., bot=1., theta=0.2, truncation_order=[4, 0], period=[0.77],
                      wavelength=0.777),
    'tm_1d':     dict(tetm='tm', phi=None, structure=[[[9., 9., 9., 1., 1., 1.]]],
                      thickness=[0.15], top=4., bot=1., theta=0.2, truncation_order=[4, 0], period=[0.77],
                      wavelength=0.777),
    '1d_2layer': dict(tetm='te', phi=None,
                      structure=[[[9., 9., 1., 1., 9., 1.]], [[4., 4., 4., 1., 1., 1.]]],
                      thickness=[0.15, 0.22], top=1., bot=1., theta=0.1, truncation_order=[3, 0],
                      period=[0.7], wavelength=0.9),
    'conical':   dict(psi=0.4, phi=0.3, structure=[[[9., 9., 9., 1., 1., 1.]]],
                      thickness=[0.15], top=4., bot=1., theta=0.2, truncation_order=[4, 0], period=[0.77],
                      wavelength=0.777),
    '2d':        dict(psi=0.9, phi=0.4, structure=_rand_eps((1, 5, 5), 0),
                      thickness=[0.2], top=1., bot=2.25, theta=0.3, truncation_order=[2, 2],
                      period=[1., 1.], wavelength=0.9),
    '2d_2layer': dict(psi=1.1, phi=0.5, structure=_rand_eps((2, 4, 4), 1),
                      thickness=[0.15, 0.22], top=1.44, bot=1., theta=0.2, truncation_order=[2, 2],
                      period=[0.9, 0.9], wavelength=0.7),
    # The first-order route, forced onto a problem the second-order one can also express. Its
    # own physics is covered by tests/test_field_4ff.py; what this adds is the public path and
    # the same drift net the others get.
    '2d_4ff':    dict(psi=0.7, phi=0.35, structure=_rand_eps((2, 4, 4), 2),
                      thickness=[0.18, 0.13], top=1., bot=1.9599999999999997, theta=0.25,
                      truncation_order=[2, 1], period=[0.85, 0.8], wavelength=0.66, formulation='4ff'),
}
VARIANTS = (('default', ('incident',)), ('all', ('incident', 's', 'p')))

#: The cases whose solve carries three incidences. A ``tetm`` declaration on a one-row raster
#: routes to the pure 1D TE/TM path, which carries exactly one -- the declaration is what takes
#: the other degree of freedom away -- so ``('incident', 's', 'p')`` is refused there rather than
#: answered (:func:`wavebit._core.fields.refuse_extra_incidences`, 2026-08-20).
COUPLED = tuple(name for name, kw in CASES.items() if 'tetm' not in kw)

#: The (case, tag, incidences) triples this file freezes, and the ones the parity test reads.
#: ``golden_field.npz`` still holds ``te_1d__all`` / ``tm_1d__all`` / ``1d_2layer__all`` from
#: before the refusal existed, and they are **bit-identical to their ``__default`` twins** --
#: which is the argument being ignored, frozen. `test_field_core.py` asserts exactly that
#: rather than dropping them, so the file stays byte-identical: re-running this generator to
#: delete three redundant keys would re-freeze the other 25 as well, and absorbing whatever
#: drift they have accumulated is the one thing a drift detector must not do.
CASE_VARIANTS = tuple((name, tag, which) for name in CASES for tag, which in VARIANTS
                      if tag == 'default' or name in COUPLED)
RES = dict(sample_x=5, sample_y=4, sample_z=6)


def solved(name, backend=0):
    kw = dict(CASES[name])
    formulation = kw.pop('formulation', None)
    kw['structure'] = np.asarray(kw['structure'], dtype=float)[::-1]
    kw['thickness'] = kw['thickness'][::-1]
    solver = wavebit.rcwa(backend=backend, answers='field', **kw)
    if formulation:
        solver.formulation = formulation
    solver.solve()
    return solver


def build():
    out = {}
    for name in CASES:
        solver = solved(name)
        for case, tag, which in CASE_VARIANTS:
            if case != name:
                continue
            cell = np.asarray(solver.reconstruct_field(incidences=which, **RES))
            # Preserve this archive's original downward frame, never recapture a new baseline.
            parity = ((1, 1, -1, -1, -1, 1) if cell.shape[-1] == 6 else
                      ((1, -1, 1) if solver.tetm_index == 0 else (-1, 1, -1)))
            out[f'{name}__{tag}'] = np.flip(cell, axis=1 if cell.ndim == 5 else 0) * parity
        out[f'{name}__grating_type'] = np.array(solver.grating_type)
        # `.value`, not the enum: numpy reads a str-Enum's data ('2ff') for the
        # itemsize but its __str__ ('Formulation.TWO_FIELD') for the content, storing
        # 'For'. Worse, `enum == that_array` is True anyway, so the assertion that
        # reads it back looks like it checks something and does not.
        out[f'{name}__formulation'] = np.array(solver.formulation_assigned.value)
    return out


def main():
    fresh = build()
    if OUT.exists():
        old = np.load(OUT)
        moved = []
        for k, v in fresh.items():
            if k not in old.files:
                moved.append(f'  + {k} (new)')
            elif old[k].shape != v.shape:
                moved.append(f'  ~ {k}: shape {old[k].shape} -> {v.shape}')
            elif not np.issubdtype(v.dtype, np.number):
                # the grating type and formulation labels: compare as written, not numerically
                if str(old[k]) != str(v):
                    moved.append(f'  ~ {k}: {old[k]} -> {v}')
            elif not np.allclose(np.asarray(old[k], dtype=complex),
                                 np.asarray(v, dtype=complex), rtol=0, atol=1e-12):
                gap = np.max(np.abs(np.asarray(old[k], dtype=complex)
                                    - np.asarray(v, dtype=complex)))
                moved.append(f'  ~ {k}: max change {gap:.3e}')
        for k in old.files:
            if k not in fresh:
                moved.append(f'  - {k} (dropped)')
        print('\n'.join(moved) if moved else '  (nothing moves)')
        print(f'{len(moved)} of {len(fresh)} arrays change.')
    np.savez_compressed(str(OUT), **fresh)
    print(f'wrote {len(fresh)} arrays to {OUT.name}')


if __name__ == '__main__':
    if '--rebaseline' not in sys.argv:
        sys.exit(
            'refusing to write without --rebaseline. These numbers come from this codebase, '
            'so regenerating them cannot fix a wrong answer -- it can only hide one. Read the '
            'module docstring, then pass --rebaseline if the change is deliberate.')
    main()
