"""Freeze this solver's field cells into ``golden_field.npz`` — a change detector, not an oracle.

    conda run -n fieldwave python tests/data/gen_goldens_field.py --rebaseline

**Read this before regenerating.** These numbers come from this codebase, so they cannot say
the reconstruction is *right* — only that it has not moved. That distinction is not academic
here. The file this replaced, ``golden_legacy_field.npz``, was captured from the project's own
earlier implementation and described in its own generator as "the first oracle it has ever
had". It was not an oracle, and what it actually froze was a **wrong sign convention**: every
path returned ``E_z``, ``H_x`` and ``H_y`` in the frame whose ``+z`` points out of the stack,
contradicting ``docs/conventions.md``, ``docs/formulation.md``'s own sanity anchor and the pair
``test_anisotropic_berreman.py::_vacuum_modes`` writes. The suite stayed green the whole time,
because a self-captured golden agrees with whatever produced it.

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
  many orders, multiple layers, every ``set_field_input`` variant, the inverse-Fourier kernel
  and the ``(res_z, res_y, res_x, comp)`` layout — cheaply, on every push.

**Regenerating is not how a failing test gets fixed.** It needs ``--rebaseline`` and prints
what moves, so re-freezing is a decision someone made and a reviewer can see, not a reflex. If
the output changes on purpose, the reason belongs in the commit message.
"""
import pathlib
import sys

import numpy as np

import wavebit

OUT = pathlib.Path(__file__).resolve().parent / 'golden_field.npz'

# Chosen to hit every reconstruction path and both 1D layouts, at more than one layer and more
# than one order. `formulation` is the only key not passed to the constructor.
CASES = {
    'te_1d':     dict(tetm='te', phi=None, ucell=[[[3., 3., 3., 1., 1., 1.]]], thickness=[0.15],
                      n_top=2., n_bot=1., theta=0.2, fto=[4, 0], period=[0.77],
                      wavelength=0.777),
    'tm_1d':     dict(tetm='tm', phi=None, ucell=[[[3., 3., 3., 1., 1., 1.]]], thickness=[0.15],
                      n_top=2., n_bot=1., theta=0.2, fto=[4, 0], period=[0.77],
                      wavelength=0.777),
    '1d_2layer': dict(tetm='te', phi=None,
                      ucell=[[[3., 3., 1., 1., 3., 1.]], [[2., 2., 2., 1., 1., 1.]]],
                      thickness=[0.15, 0.22], n_top=1., n_bot=1., theta=0.1, fto=[3, 0],
                      period=[0.7], wavelength=0.9),
    'conical':   dict(psi=0.4, phi=0.3, ucell=[[[3., 3., 3., 1., 1., 1.]]], thickness=[0.15],
                      n_top=2., n_bot=1., theta=0.2, fto=[4, 0], period=[0.77],
                      wavelength=0.777),
    '2d':        dict(psi=0.9, phi=0.4, ucell=np.random.default_rng(0).uniform(1, 4, (1, 5, 5)),
                      thickness=[0.2], n_top=1., n_bot=1.5, theta=0.3, fto=[2, 2],
                      period=[1., 1.], wavelength=0.9),
    '2d_2layer': dict(psi=1.1, phi=0.5, ucell=np.random.default_rng(1).uniform(1, 4, (2, 4, 4)),
                      thickness=[0.15, 0.22], n_top=1.2, n_bot=1., theta=0.2, fto=[2, 2],
                      period=[0.9, 0.9], wavelength=0.7),
    # The first-order route, forced onto a problem the second-order one can also express. Its
    # own physics is covered by tests/test_field_4ff.py; what this adds is the public path and
    # the same drift net the others get.
    '2d_4ff':    dict(psi=0.7, phi=0.35, ucell=np.random.default_rng(2).uniform(1, 4, (2, 4, 4)),
                      thickness=[0.18, 0.13], n_top=1., n_bot=1.4, theta=0.25, fto=[2, 1],
                      period=[0.85, 0.8], wavelength=0.66, formulation='4ff'),
}
VARIANTS = (('default', (True, False, False)), ('all', (True, True, True)))
RES = dict(res_x=5, res_y=4, res_z=6)


def solved(name, backend=0):
    kw = dict(CASES[name])
    formulation = kw.pop('formulation', None)
    kw['ucell'] = np.asarray(kw['ucell'], dtype=float)
    solver = wavebit.rcwa(backend=backend, **kw)
    if formulation:
        solver.formulation = formulation
    solver.conv_solve()
    return solver


def build():
    out = {}
    for name in CASES:
        solver = solved(name)
        for tag, sfi in VARIANTS:
            out[f'{name}__{tag}'] = np.asarray(
                solver.calculate_field(set_field_input=sfi, **RES))
        out[f'{name}__grating_type'] = np.array(solver.grating_type_assigned)
        # `.value`, not the enum: numpy reads a str-Enum's data ('2ff') for the
        # itemsize but its __str__ ('Formulation.SECOND') for the content, storing
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
