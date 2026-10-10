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
5x5 puts half a pixel at a tenth of the cell, and ``truncation_order=2`` is nowhere near converged.

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

**Why it was re-frozen again (2026-08-17), and why this one is a stronger case than either.**
The normal field was not periodic. A unit cell is a torus, and the ``'distance'`` construction
read it as a flat array in three places -- the interface mask compared only inside the array,
the distance transform measured to the in-cell interface rather than the periodically nearest
one, and ``np.gradient`` is one-sided on the first and last row and column. Transmitted
efficiency moves 0.714 to 0.643 here; energy is conserved to 1.000000000000 either way.

The two earlier re-freezes had to argue their new values were better from *somewhere else* --
smooth shapes for the 2026-08-07 lift, midpoint-rule quadrature order for the half-pixel phase
-- precisely because a random 5x5 raster cannot arbitrate a normal field. **This one does not
need an outside argument, because the case arbitrates itself.** Rolling a raster by whole
pixels is an exact lattice translation: the same structure, named differently, so every
efficiency must be unchanged. On this very case, rolled two pixels in x, the old field moved
``max|dR|`` by 5.7e-02 and the new one by 2.3e-15. An implementation that fails an exact
invariant on the golden's own geometry is wrong on that geometry, whatever any other case says.

That invariant is now under test in its own right (``tests/test_nvf_lattice_translation.py``),
so this file is no longer the only thing standing between the fix and a silent regression --
which is the condition under which a self-captured golden is doing its proper job.

**Why the permittivity flip (2026-08-20) did *not* re-freeze it, which was not obvious in
advance.** NVF reads the raster twice over -- once as the permittivity it factorizes and once
as the field whose gradient is the interface normal -- so squaring the raster is a change to
the normal field's input, not only to the layer's material. It turns out to change nothing:
regenerated with the raster stated as ``eps`` instead of ``n``, all 39 arrays come back **bit
for bit** identical to the committed file. The reason is that squaring is monotone and the
normal is a *direction*: ``grad(u**2) = 2u grad(u)`` points where ``grad(u)`` points, and the
construction normalises. Measured on this case rather than argued in general -- a construction
that thresholded the raster somewhere would not have this property, and the check is one
command.
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
# The same random raster this case has always used, squared: it was drawn as a refractive index
# from `uniform(1, 4)` and the solver squared it on the way in, which it stopped doing on
# 2026-08-20. Squaring the draw keeps the *permittivities* identical, where re-drawing from
# `uniform(1, 16)` would have produced a different structure from the same seed.
CASE = dict(psi=0.9, phi=0.4,
            structure=np.random.default_rng(3).uniform(1.0, 4.0, size=(1, 5, 5)) ** 2,
            thickness=[0.2], top=1., bot=1., theta=0.2, truncation_order=[2, 2], period=[1., 1.],
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

    The two arguments are **refractive indices**, not the permittivities the case now states:
    the factor is the Fresnel ``H -> E`` conversion ``-i*n``, which is an amplitude ratio and
    has an index in it however the scene is spelled. Both half-spaces here are vacuum, so the
    two readings coincide numerically -- which is exactly why the caller converts explicitly
    rather than relying on that.
    """
    return (('R_s', 'r_s', 1.0), ('R_p', 'r_p', 1.0 / (-1j * n_top)),
            ('T_s', 't_s', 1.0), ('T_p', 't_p', 1.0 / (-1j * n_bot)))


def frame_sign(solver):
    """The per-order ``+-1`` that :func:`test_golden_legacy_parity._frame_sign` applies.

    **The amplitudes here are stored in the LEGACY s/p frame, deliberately.** The legacy solver
    built each order's frame as ``atan(ky/kx)``, which cannot see ``sign(kx)``, so every order
    with ``Re(kx) < 0`` was quoted in ``-s_hat``, ``-p_hat``; that was fixed on 2026-08-17 in
    :func:`~wavebit._core.conventions.order_azimuth`. The comparison treats the difference as a
    convention conversion -- one more factor beside the ``H -> E`` ones in ``_amp_map`` -- and
    applies it to all thirteen cases uniformly, this one included.

    So this generator has to divide it back out, exactly as it already divides out the ``H -> E``
    factors, or the freeze lands in the new frame, the comparison flips it again, and the
    ``Re(kx) < 0`` orders come back negated. Keeping the freeze frame-neutral is what lets
    ``_frame_sign`` stay one uniform rule rather than gaining a branch for the one case whose
    generator happens to run.

    Must stay identical to ``_frame_sign``; the two are checked against each other by
    ``test_results_match_legacy_golden[2d_nvf]`` every run.
    """
    if solver.grating_type == 0:
        return 1.0
    kx = np.asarray(solver.order_kxy()[0], dtype=np.complex128)
    return np.where(np.real(np.reshape(kx, (1, -1))) < 0, -1.0, 1.0)


def main():
    kw = dict(CASE)
    kw['structure'] = np.asarray(kw['structure'], dtype=float)
    # `answers='field'` because this generator pins the per-layer state as well as the
    # result surface; without it `solver.T1` is None and the goldens regenerate empty.
    solver = wavebit.rcwa(backend='numpy', answers='field', **kw)
    result = solver.solve()

    out = {f'{NAME}__grating_type': np.array(solver.grating_type)}
    sign = frame_sign(solver)
    # The golden's key component is the legacy attribute name; today's is beside it.
    # See SUBS in tests/test_golden_legacy_parity.py for why the file keeps the old
    # vocabulary rather than being rewritten.
    for sub, attr_name in (('res', 'incident'), ('res_te_inc', 's_inc'), ('res_tm_inc', 'p_inc')):
        s = getattr(result, attr_name)
        if s is None:
            continue
        for key, attr in EFFICIENCIES:
            out[f'{NAME}__{sub}__{key}'] = np.asarray(getattr(s, attr))
        for key, attr, factor in amplitude_entries(wavebit.index_from_eps(CASE['top']),
                                                   wavebit.index_from_eps(CASE['bot'])):
            out[f'{NAME}__{sub}__{key}'] = np.asarray(getattr(s, attr)) / factor / sign

    out[f'{NAME}__T1'] = np.asarray(solver.T1)
    for i, li in enumerate(solver.layer_info_list):
        for j, nm in enumerate(('epz_i', 'W', 'V', 'q', 'd', 'A_i', 'B')):
            out[f'{NAME}__layer{i}__{nm}'] = np.asarray(li[j])

    energy = float(np.sum(np.asarray(result.incident.R)) + np.sum(np.asarray(result.incident.T)))
    if abs(energy - 1.0) > 1e-11:
        raise SystemExit(f'R + T = {energy:.12f} on a lossless stack - do not freeze this')

    path = pathlib.Path(__file__).parent / 'golden_nvf.npz'
    np.savez_compressed(str(path), **out)
    print(f'wrote {len(out)} arrays for {NAME}; R + T = {energy:.12f}')


if __name__ == '__main__':
    main()
