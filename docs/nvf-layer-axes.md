# Conforming NVF with different layer tiling axes

Written: 2026-09-09

Targets:
- `wavebit/_core/conforming/__init__.py`
- `tests/test_conforming_layer_axes.py`

`ConformingNVF` accepts nonoverlapping rectangle sequences along either lattice
axis. Each layer chooses its own axis, so horizontal and vertical material
sequences can share one optical stack. Equivalent rectangle descriptions
obtained by adding a quarter turn and exchanging lengths give the same
matrices. The [manual](guide/conforming-nvf.md) includes a mixed-stack example.

## Frame and Fourier mapping

Write the right-handed orthogonal lattice as
`A = R diag(Lx, Ly)`, where R is a rotation. The existing cell solver
parameterizes material partitions along its first axis. A layer tiled
along the second lattice axis uses the right-handed frame

```text
J = [[0, -1], [1, 0]]
A_cell = A J = (a2, -a1)
R_cell = R J
(Lx_cell, Ly_cell) = (Ly, Lx)
(cx_cell, cy_cell) = (cy, -cx)
aspect_cell = Ly/Lx
```

The center coordinates in this expression are local physical coordinates;
the implementation applies the same exchange after separate period
normalization. Rectangle extents are exchanged, and the perpendicular gap
is measured against Lx. The cell solve therefore uses the reciprocal of the
original aspect ratio, including the physical energy metric described in
the [rectangular-cell derivation](nvf-rectangular-cells.md).

For each original Fourier order pair `(m, n)`, the cell pair is `(n, -m)`.
The adapter applies this mapping to every retained-order difference before
the energy closure. It rotates both the field and its energy by R_cell,
so all five returned slots use the original laboratory components and
original solver order layout. Layer scattering needs no per-layer order
permutation, and rectangular retained-order sets remain supported.

The cell kernel's centered origin still requires the half-cell translation
used by the public shape API. The center of `(a2, -a1)` differs from that of
`(a1, a2)` by `-a1`, an integer lattice translation. It adds no Fourier
phase for integer orders. The scalar material convolution is checked
against an independent analytic integral in the original lattice; this
detects order signs and origin errors that total power could conceal.

At allowed rectangle angles, squared direction cosines exchange `lx` and
`ly` into the lattice frame. The live alignment guard accepts multiples
of pi/2 within roundoff and refuses oblique edges. This normalization has
no absolute-value cusp or orientation branch within a JAX trace.
The original lengths are also checked for strict positivity. Checking only
transformed extents can accept a zero or tiny negative length because the
sine or cosine of a quarter turn is not exactly zero in floating point.
The original-length guard remains active in traced replay.

## Planning and replay

Planning tries the first lattice axis, then the second. An immutable plan
stores the input-index ordering and chosen axis for each layer. Uniform
layers use the first axis. Geometry, material values, periods and common
rotation remain live during replay.

A single rectangle often permits both frames. The first-axis preference
preserves the previous choice for such inputs. Replay keeps its choice
even if another frame later becomes valid: two finite trial spaces need
not give identical approximations. In particular, a full-first-period
stripe requires the second-axis frame and retains it if its width later
shrinks. A first-axis plan that becomes invalid must be explicitly
replanned; the builder does not switch frames inside differentiation.

## Controls and scope

The added controls in `tests/test_conforming_layer_axes.py` cover:

- Independent analytic material convolutions at five retained-order sets,
  including unequal counts and zero-order axes.
- Equivalent quarter-turn rectangle representations in both tiling frames.
- A horizontal-contact, uniform, vertical-contact stack against independent
  inverse-rule matrices and actual NumPy, JAX and Torch engine readouts.
- Every diffraction power for the same physical scene expressed in
  `(a1, a2)` and `(a2, -a1)`, with order labels mapped between the two.
- Eleven live geometry, material, period and pose derivatives under JIT,
  compared with two centered finite-difference steps.
- The first-period contact derivative at exact contact and at one-ULP
  roundoff, with two one-sided step windows.
- Frozen-axis replay, invalid plans and NaN output for invalid traced input.
- Refusal of nonpositive original lengths despite rotation roundoff, both
  during planning and traced replay.

Commands and results are recorded in
`studies/results/nvf_layer_axes_validation.log`. Existing contact,
Hermitian, residual and derivative criteria are unchanged. These checks
qualify the finite construction and integration, not optical convergence.
The selected suites pass 107 public construction checks, 62 engine and pose
regressions, and 31 documentation contract checks. Both manual examples,
the strict Sphinx build and lint also pass.

Within each layer, rectangles must avoid overlaps, share
their perpendicular extent and center, and span less than one period along
the tiling axis. Full perpendicular-period contact is supported. L-shaped
unions, unequal perpendicular extents, oblique edges and sheared cells
remain unsupported. The
[common-loss construction](nvf-common-loss.md) accepts a shared passive
factor independently for each layer; the later
[complex-material construction](nvf-complex-material.md) also permits
independent loss for each shape and background. General shape NVF is available
through the existing solver API; this explicit builder does not substitute
another construction for unsupported input.

The subsequent [background-interval extension](nvf-separated-rectangles.md)
allows rectangles to be separated along the chosen axis. Intervals may open
or close under the same plan; the axis and input-index ordering stay fixed.

The [lattice-rebasing extension](nvf-lattice-rebasing.md) accepts skew basis
vectors that describe the same orthogonal point lattice. It preserves
physical rectangles, the supplied cell-center convention and the exact
original Fourier set. Genuinely nonorthogonal point lattices still need a
different cell construction.
