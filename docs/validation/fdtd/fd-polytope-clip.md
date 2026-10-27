# Two smoothed surfaces in one voxel: the polytope clip (stage 4, C10)

Written: 2026-10-27

Targets: `src/wavebit/fdtd/_prepare/checks.py`, `src/wavebit/fdtd/_prepare/coefficients.py`,
`src/wavebit/fdtd/_prepare/constitutive.py`, `src/wavebit/fdtd/_prepare/junction.py`,
`src/wavebit/fdtd/_shapes/_cut.py`, `src/wavebit/fdtd/_shapes/_bodies.py`,
`studies/fdtd/subpixel_media_census/fdtd_subpixel_media_census.py`.

## Status

This page is written in two parts. The first part, **Registered acceptance**, was committed
before any C10 code was written and before any number below was computed. It is not edited
after the measurements; results are recorded in later sections that cite it.

## Registered acceptance (2026-10-27, before any C10 measurement)

### What C10 is to answer

Door 5 of `checks.check_smoothed` refuses two smoothed bodies whose surfaces pass through one
voxel, because the two-phase rule `coefficients.mix` composes wrongly there (73% on two half
spaces of one material, up to 122% on one sphere declared twice). Study S7
(`studies/fdtd/subpixel_media_census/`) found such voxels rare: none for single bodies, 1-4%
of cut voxels for a pillar flush on a substrate spelled as a second body, and for a dimer of
two different spheres at a gap of 0.05 diameters (at 20 cells per unit only).

C10 replaces the refusal by an answer **only where every quantity the answer needs is computed
exactly or by a rule already used for one body**, and keeps the refusal, by name, everywhere
else. The planned construction:

1. **Per-medium fills.** For the pair `A` (listed first) and `B` (listed later, so it wins
   where they overlap), the voxel holds `B` with fill `F_B`, `A` with fill `F_A - S` and the
   ground (what the voxel held before `A` was painted) with fill `1 - F_A - F_B + S`, where
   `F_A`, `F_B` are the bodies' existing exact fills and `S` is the volume they share in the
   voxel. `S` is admitted only where it is exact:
   * `S = 0` where the bodies are proved disjoint (per voxel by the existing exact test of
     their supporting half spaces, `_bodies._no_common_volume`, for convex bodies; or globally,
     for a sphere whose centre is at least its radius from the partner's exact signed distance);
   * `S` = the exact two-plane cut (`_cut.two_plane`, already validated against an independent
     ConvexHull oracle) where both bodies are proved flat in the voxel;
   * `S` = the partner's exact volume over an axis-aligned sub-box, where one member is flat
     in the voxel with an axis-aligned normal (an unturned Box face), so that the voxel on that
     member's side of its face is itself an axis-aligned box.
2. **Interface moment.** `w_c` is the combined second moment of the interfaces: each member's
   own area and moment (exact for polygonal bodies, the tangent plane's for curved ones, as for
   one body), with the part of `A`'s surface buried inside `B` removed.
3. **Rule.** With `<1/eps> = sum_m f_m/eps_m` and `<eps> = sum_m f_m eps_m` over the media
   present, each component reads `1/eps_c = w_c <1/eps> + (1 - w_c)/<eps>`, and the common-node
   tensor is `1/<eps> I + (<1/eps> - 1/<eps>) P` with `P` the area-weighted sum of the members'
   nearest-surface `n n^T` (the existing node proxy). For parallel planar interfaces this is
   the exact laminate (normal component harmonic, tangential arithmetic over all layers). With
   one medium absent it is the existing two-phase rule.
4. **Kept refused by name**: flush joints (two coincident back-to-back faces; that is the
   junction's question, `Structure(junction=)`), any voxel whose shared volume is not in one of
   the exact cases above, overlapping bodies of one material (a union, whose CSG spelling is not
   public), a third structure reaching a written voxel other than a whole earlier ground,
   magnetic, conductive, dispersive or anisotropic media, `discretisation='stencil'`, and
   traced geometry (`range=`) on a member.

### Gate 1: geometry and rule, unit tests (a new test module `test_fd_clip.py` under `tests/fdtd/`)

* **G1a, polyhedral fills.** For pairs of unturned and turned Boxes and convex Prisms in each
  admitted case (disjoint, both flat, one axis-aligned flat member), every medium fill of every
  written voxel against an independent oracle: the voxel and each body written as half spaces,
  intersected by `scipy.spatial.HalfspaceIntersection` and measured by `ConvexHull`, sharing
  no line with `_cut` or `_exact`. **Registered: absolute error at most 2e-12 of a voxel.**
  (The two-plane cut measured 9.8e-13 against this kind of oracle; 2e-12 rounds that up.)
* **G1b, fills with a sphere.** A sphere disjoint from a partner, touching a substrate, and
  sunk into a substrate: every medium fill against adaptive quadrature in the style of
  `tests/fdtd/test_fd_shapes.py`, `scipy.integrate.quad` along the substrate normal of exactly
  clipped cross-section areas, broken at the cross-sections' kinks. **Registered: at most 1e-10
  of a voxel.**
* **G1c, polyhedral interface moments.** The exposed interface area and its three per-component
  moments against the polytope oracle's facets (a member's facets inside the voxel, less the
  facets of the shared polytope on that member's plane). **Registered: at most 2e-9 of `dx^2`**
  (the two-plane cut's measured worst, 1.3e-9 on nearly anti-parallel faces, rounded up).
* **G1d, rule.** (i) Three parallel planar layers in one voxel: each lowered component of `eps`
  against the laminate closed form computed from the layer thicknesses alone, **relative 1e-12**.
  (ii) With one medium's fill exactly zero, the clip rule's diagonal equals `coefficients.mix`,
  and its node tensor, cross term and arithmetic mean equal the single-body expressions in
  `constitutive.lower`, **bit for bit** (`numpy.array_equal`).
* **G1e, refusals.** Each case kept refused in "What C10 is to answer" fires, by name.
* **Mutations.** Every new test is shown to fail under a deliberate edit of the product code,
  which is then restored exactly; an import error does not count.

### Gate 2: convergence (new study `studies/fdtd/subpixel_polytope_clip/`)

* **Arm L, exact reference, clip active at every grid.** A planar stack along z of two smoothed,
  unturned Boxes spanning the transverse cell: `eps` 6 on `z in [-0.30, 0]`, a vacuum gap of
  0.003, `eps` 2.25 on `z in [0.003, 0.203]`, in vacuum. The gap is thinner than every grid
  used, so its two faces share voxels at every grid. Two polarisations: normal incidence (the
  tangential, arithmetic part of the rule) and p polarisation at 30 degrees in the x-z plane
  (the normal, harmonic part), the latter with `Axis1D` transverse faces. Reflectance of the
  zeroth order at 21 wavelengths in `[0.8, 1.2]` against the continuum transfer-matrix
  reflectance of the same four-interface stack. Grids 20, 28, 40, 56, 80, 112 and 160 cells per
  unit length. Error at a wavelength: `|R_fdtd - R_exact|`; band error: the root mean square
  over the 21 wavelengths. **Pass: for each polarisation, the least-squares log-log order of the
  band error over all seven grids is at least 1.8.** Reported beside it: the fit's scatter
  (`exp(max |log residual|) - 1`), the per-wavelength orders, the per-grid errors, and the
  count of clip-written voxels at each grid. Control arm L0: the same stack with a gap of 0.2
  (no shared voxel) is run on the same grids and reported the same way; it is an instrument
  check and not part of the pass.
* **Arm D, the S7 dimer.** S7's geometry: spheres of radius 0.25 centred at
  `(-0.2625, 0, 0.021)` and `(0.2625, 0, 0.021)`, a gap of 0.05 diameters. S7 named two metals;
  **a metal on a curved smoothed body is refused (door 1; C8 smooths planar films only), so
  the materials are two lossless dielectrics, `eps` 12 and 2.25, in vacuum**, a substitution
  recorded here and not a choice made after measurement. A TFSF plane wave along +z, polarised
  along the dimer axis (x), scattering cross-section from a closed flux box (the instrument of
  `studies/fdtd/subpixel_mie_sphere/`) at 11 wavelengths in `[1.0, 2.0]`. Grids 20, 24, 28, 32,
  40 and 56 cells per unit; reference: 80 cells per unit. Per wavelength and grid: relative
  error against the reference; band error: mean over wavelengths. Reported: fitted order and
  scatter over the six grids, per-wavelength orders, the per-grid table, and the clip-written
  voxel count at each grid. **Pass: (a) every grid plans without refusal and runs with the
  stability diagnostic finite; (b) at every grid where the clip writes, the band error lies
  within a factor of 2 of the power law fitted through the grids where it does not write,
  evaluated at that grid (the clip-active grids are not outliers).** If (b) fails, voxels
  answered between two curved surfaces are refused by name and the failure is recorded.
* **Arm P, the S7 flush pillar.** A cylinder flush on a smoothed Box substrate is the
  junction's (`junction='split'`), not C10's. **Pass: it plans with no clip record and its
  lowered coefficients are unchanged** (Gate 3).

### Gate 3: every existing two-media scene is unchanged

* The four frozen stage goldens reproduce in check mode, nothing written
  (`tests/references/interface/freeze_fd_stage0.py` to `freeze_fd_stage3.py`). `tests/references/fd_golden_stage*.npz`
  are not regenerated; **if any moves, the work stops and is reported.**
* Thirteen existing two-media smoothed scenes (sphere, magnetic sphere, unturned and turned Box,
  Cylinder, hexagonal Prism, core-shell pair, a dimer far apart, cylinder and Box pillars under
  `split` and `staircase`, a stencil sphere, a high-contrast sphere): plan skeletons and every
  lowered parameter array **byte-identical** before and after C10, from a dump taken on
  unmodified `main` (183baf37) before any C10 edit.

### Gate 4: backends and derivatives

* Lowered coefficients (`be`, constitutive cross terms, lifted `be`) of clip scenes on NumPy,
  JAX (eager and jit) and Torch: **maximum relative difference at most 1e-12** (the existing
  coefficient-lowering budget). Fields after 40 steps of a clip scene with a dipole source:
  **at most 1e-11 relative**.
* Derivatives of a field-energy objective with respect to each clip medium's `eps` (first,
  second, ground), JAX and Torch automatic differentiation against a central difference with
  relative step 1e-4: **relative difference at most 1e-6**, and JAX against Torch **at most
  1e-10**. Traced geometry on a clip member: refused by name.

Agreement across three backends is consistency, not physical correctness; Gate 2 is the
physical gate.
