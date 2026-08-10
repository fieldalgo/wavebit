# Coming from meent — the one result that changes

Written: 2026-08.

Targets:
- [`../README.md`](../README.md#coming-from-meent) — the rename table this page continues
- `fieldwave/_core/conventions.py` — where the axis convention is stated once
- [`../studies/`](../studies) — the Reticolo comparison that settles which value is right

The rename table is in the README. This page covers the single case where FieldWave returns a
*different number* for the same input, so you can tell whether it touches you before you
conclude something regressed.

## TL;DR

meent's CFS path read `ucell` row `0` as the last y; every other path read it as the first.
FieldWave reads it as `y = 0` everywhere. If you are at oblique azimuth on `integrated`, with a
structure that is not y-symmetric, your number changes — and the new one is the correct one.

## The row-0 disagreement

meent's CFS path read `ucell` row `0` as the **last** y, while its other paths read it as the
first. The same array therefore described two different structures depending on meent's
`fourier_type` — a y-flip of one another.

At `phi = 0` the two structures are optically equivalent and the disagreement is invisible.
Away from it, it is not: meent's CFS was ~1e-2 off Reticolo at oblique azimuth, while its other
paths were not.

FieldWave reads row `0` as `y = 0` on every path — raster, block and smooth, and for the
reconstructed field cell too (see *Axes follow the array* in the README's Conventions). With
that settled, CFS agrees with Reticolo to ~1e-13 across the twelve angle and polarization
combinations the suite checks.

## Whether it touches you

All three conditions have to hold:

1. `phi != 0` — at normal azimuth nothing changes, and
2. you are on `fourier_coefficients='integrated'` (meent's `fourier_type=1`), and
3. your structure is not symmetric in y — a y-flip has to be a different structure.

If all three hold, expect a difference against the meent number. **The FieldWave value is the
correct one**, and Reticolo is the reference that says so; the old agreement at `phi = 0` was
the two conventions cancelling, not either of them being right.

If you need to reproduce an old meent result exactly for comparison, flip the `ucell` rows
(`ucell[:, ::-1, :]`) — that reconstructs the structure meent's CFS path was actually solving.
