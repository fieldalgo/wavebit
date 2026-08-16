"""Re-freeze the ``2d_nvf`` case, captured from THIS solver rather than from the legacy one.

    python tests/data/gen_goldens_nvf.py

**This is a self-captured golden and says so in its name.** It agrees with whatever produced
it, so it can report that an answer moved and nothing else -- it is a tripwire, not an oracle.
``gen_goldens.py`` beside it refuses to run for exactly that reason: re-freezing the *legacy*
file against the current implementation would turn the repo's only legacy-parity evidence into
a change detector, silently, with every test still green. So ``2d_nvf`` moved out here instead
and ``golden_legacy.npz`` stays byte-identical, its other twelve cases still legacy-derived.

That split makes the file names honest about something that was already true: since 2026-08-07
``2d_nvf`` pinned continuity against this implementation, not against the one it replaced. It is
also the route ``gen_goldens.py``'s own note prescribes, and the one
``tests/test_field_core.py`` took on 2026-08-14 when it retired ``golden_legacy_field.npz`` and
re-froze once as ``golden_field.npz``.

**Why it was re-frozen (2026-08-16).** The sampled Fourier transform used to omit the
half-pixel cell-centre phase, so the normal field sat half a pixel off the permittivity it
multiplies -- see :func:`wavebit._core.factorization._half_pixel_phase`. Per-order efficiency
moves up to 0.30 transmitted and 0.09 reflected, and energy is conserved to 1.000000000000
either way. Large because this case is the half-pixel term's worst case and not in spite of it:
5x5 puts half a pixel at a tenth of the cell, and ``fto=2`` is nowhere near converged.

**A random 5x5 raster cannot arbitrate that**, and it is not being asked to. NVF's premise is
that a staircase stands for a smooth boundary and a random raster has no boundary behind it --
``test_golden_legacy_parity.py``'s header says so about this very case. The evidence is
elsewhere, where an exact limit exists: the sampled transform's own quadrature. Sampling a known
continuous field at ``(j+0.5)/N`` and comparing against its true coefficients, the phase makes
the error fall as ``1/N**2`` -- the midpoint rule's own order -- against ``1/N`` without it,
measured 1.0e-4/1.5e-5/3.1e-6/7.7e-7/1.9e-7 at N = 32/64/128/256/512 versus
1.5e-2/7.5e-3/3.7e-3/1.9e-3/9.4e-4. That is a comparison against the transform's exact limit
rather than against a rival formulation, so unlike this file it can say which answer is right.

The same argument shape as the 2026-08-07 re-freeze, which moved this case further (0.684 to
0.352) and justified it on smooth shapes via ``docs/nvf-upsampling.md``.
"""
import pathlib
import sys

import numpy as np

# The checkout this file lives in, ahead of everything else. Run as a script, `sys.path[0]` is
# `tests/data/` rather than the repo root, so `import wavebit` either fails or -- inside a
# worktree, with the package importable from elsewhere -- silently captures a *different*
# checkout's solver into this checkout's golden.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import wavebit                                                          # noqa: E402

NAME = '2d_nvf'
CASE = dict(psi=0.9, phi=0.4, ucell=np.random.default_rng(3).uniform(1.0, 4.0, size=(1, 5, 5)),
            thickness=[0.2], n_top=1., n_bot=1., theta=0.2, fto=[2, 2], period=[1., 1.],
            wavelength=0.9, factorization='nvf')

#: Efficiency keys, stored under the legacy spelling the comparison already speaks.
EFFICIENCIES = (('de_ri', 'R'), ('de_ri_s', 'R_s'), ('de_ri_p', 'R_p'),
                ('de_ti', 'T'), ('de_ti_s', 'T_s'), ('de_ti_p', 'T_p'))


def amplitude_entries(n_top, n_bot):
    """Legacy amplitude keys, and the factor the comparison applies to them.

    ``test_golden_legacy_parity._amp_map`` reads a golden entry and multiplies it to reach
    today's attribute, so what is stored here is today's value *divided* by that factor -- the
    number the legacy solver would have written for today's physics. Storing it that way keeps
    the schema identical to ``golden_legacy.npz``, so the comparison needs no branch for this
    case and cannot drift away from how the other twelve are read.
    """
    return (('R_s', 'r_s', 1.0), ('R_p', 'r_p', 1.0 / (-1j * n_top)),
            ('T_s', 't_s', 1.0), ('T_p', 't_p', 1.0 / (-1j * n_bot)))


def main():
    kw = dict(CASE)
    kw['ucell'] = np.asarray(kw['ucell'], dtype=float)
    solver = wavebit.rcwa(backend=0, **kw)
    result = solver.conv_solve()

    out = {f'{NAME}__grating_type': np.array(solver.grating_type_assigned)}
    for sub in ('res', 'res_te_inc', 'res_tm_inc'):
        s = getattr(result, sub)
        if s is None:
            continue
        for key, attr in EFFICIENCIES:
            out[f'{NAME}__{sub}__{key}'] = np.asarray(getattr(s, attr))
        for key, attr, factor in amplitude_entries(CASE['n_top'], CASE['n_bot']):
            out[f'{NAME}__{sub}__{key}'] = np.asarray(getattr(s, attr)) / factor

    out[f'{NAME}__T1'] = np.asarray(solver.T1)
    for i, li in enumerate(solver.layer_info_list):
        for j, nm in enumerate(('epz_i', 'W', 'V', 'q', 'd', 'A_i', 'B')):
            out[f'{NAME}__layer{i}__{nm}'] = np.asarray(li[j])

    energy = float(np.sum(np.asarray(result.res.R)) + np.sum(np.asarray(result.res.T)))
    if abs(energy - 1.0) > 1e-11:
        raise SystemExit(f'R + T = {energy:.12f} on a lossless stack - do not freeze this')

    path = pathlib.Path(__file__).parent / 'golden_nvf.npz'
    np.savez_compressed(str(path), **out)
    print(f'wrote {len(out)} arrays for {NAME}; R + T = {energy:.12f}')


if __name__ == '__main__':
    main()
