# Complete retained contact elimination

Written: 2026-09-16

Targets:

- `wavebit/_core/conforming/pair_contact_retained.py`
- `wavebit/_core/conforming/pair_complete_solve.py`
- `wavebit/_core/conforming/pair_contact_precision.py`
- `tests/test_nvf_complete_solve.py`
- `tests/test_nvf_contact_geometric_frame.py`

A high-degree contact cell can have thousands of common potentials and far
fewer retained Fourier fields. A complete geometric coordinate map requires
dense common-matrix products even though the retained constraints are small.
Eliminating through the original physical stiffness avoids those products.
It preserves every polynomial, transmission, disappearing and seam potential,
the same directly integrated equations, and every physical acceptance bound.

## The finite equations

Let `K` be the full unprojected common stiffness, `Q` its retained Fourier
moments, `R` its material-weighted moments, and `A` the free retained material
matrix. `S` is the corresponding reciprocal source. For each retained input,
`uc`, `x` and `H` satisfy

```text
K*uc + S*x = Q.H*H
x + Q*uc = I
H = A*x + R*uc.
```

Solve `K*V = Q.H` and `K*W = S` for all columns. Substitution gives

```text
[ I-Q*W       Q*V ] [ x ] = [ I ]
[ -A+R*W    I-R*V ] [ H ]   [ 0 ],

uc = V*H - W*x.
```

For `N` scalar Fourier modes this block has dimension `4*N`. The automatic
complete path uses it when that dimension is smaller than the common
potential count and no common frame was already supplied. Otherwise the
complete geometric-frame path remains available. This decision uses matrix
dimensions only; it does not select physical fields or change material data.

Every disappearing potential is recovered from the same full normal block.
The analytic common variation includes the opening measure, moving Fourier
moments, common gradients, all live material directions and normal coupling.
It reuses the retained factorization for the differentiated right-hand side.
The fixed reference field bank stays fixed during a material variation.

## Complete pivots and physical verification

The retained block may be poorly conditioned in its raw coordinates. Its
leading floating-point word can even be singular while the full high/low
matrix is nonsingular. A complex partial-pivot factorization therefore keeps
both words during every elimination and triangular substitution. Pivoting
permutes equations. An exactly zero pivot raises; no small direction is
removed and no diagonal shift is introduced.

The independent verifier receives the full high/low physical field,
fixed-potential geometry variation and all potential coefficients. It
reconstructs pointwise fields and integrates energy, derivative energy and
retained means without receiving the candidate tensor or its JVP. The weak,
retained, flux, energy, reciprocity and material checks keep their original
bounds. See [the energy-coordinate note](nvf-contact-energy-coordinates.md).

Metadata reports `closed_elimination='physical_retained'` for this path and
`contact_solve_size` for its matrix dimension. `condition` describes that
matrix. A condition reported by the geometric path describes a different
coordinate system; neither value alone is a physical error bound.

## Independent controls and measured limits

Independent 100-digit complex Gaussian elimination checks all columns of
four Hilbert-type systems of sizes 3, 6, 10 and 14. The respective measured
relative component errors were `1.60e-30`, `8.78e-27`, `3.10e-21` and
`8.08e-15`, against bounds declared before the runs. The forward error grows
with conditioning even though every matrix entry is retained. Separate
controls resolve a nonsingular low-word perturbation of a singular leading
matrix, reuse the factorization, preserve inputs and reject exact singularity.

A 324-by-324 algebra control built from preserved order-4 sextic matrices had
condition `1.98e13`. Complete pivoting and four-step residual refinement
agreed over the full solution within `5.68e-18`; the complete scaled residual
was `4.87e-34`. This control deliberately used a rounded free input to define
a reproducible algebra problem. It is not a physical closure qualification.
The complete factor and solve took 2.44 seconds, versus 0.21 seconds for
residual refinement in this small block. The intended cost reduction comes
from removing large common coordinate products, not from a faster small solve.

Separate complete physical controls retain all 20 real transmission directions
at quintic order `(1,0)` and sextic order zero. They compare retained elimination, the integrated geometric frame and an
independently sampled QR frame under joint material and opening directions.
Each full physical field is independently verified, with original `3e-10` / `3e-9` reflection bounds and full
response comparisons at `3e-12` / `3e-11`. These are finite-equation checks.
They do not establish joint polynomial, mesh, branch or Fourier convergence,
or a universal optical stopping rule.
