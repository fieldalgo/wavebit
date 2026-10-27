# Stage 4 report: shapes and accuracy (C0-C12)

Written: 2026-10-27

Targets: `src/wavebit/fdtd/_shapes/__init__.py`, `src/wavebit/fdtd/_prepare/occupancy.py`,
`src/wavebit/fdtd/_prepare/coefficients.py`, `src/wavebit/fdtd/_prepare/constitutive.py`,
`src/wavebit/fdtd/_prepare/checks.py`, `src/wavebit/fdtd/_prepare/junction.py`,
`src/wavebit/fdtd/_prepare/clip.py`, `src/wavebit/fdtd/_prepare/stencil.py`,
`src/wavebit/fdtd/_prepare/shape_ranges.py`, `tests/fdtd/test_fd_shape_protocol.py`,
`tests/fdtd/test_fd_shapes.py`, `tests/fdtd/test_fd_occupancy.py`, `tests/fdtd/test_fd_bodies.py`,
`tests/fdtd/test_fd_clip.py`.

Stage 4 of the general FDTD engine (`wavebit.fdtd`) set out to give it curved and rotated
bodies with subpixel smoothing, differentiable edges, and bodies that run out through an
absorbing layer, graded against exact Mie theory (the pass mark) and a metasurface against
RCWA. It was planned as thirteen commits, C0 to C12. This page
states what each one is today and where its evidence is. **It is a status record, not a
completion claim: two registered gates are not passed (C7, and C10's dimer arm), C3's passed
only with the optional stencil, three commits are partial, and one is left for the owner.**

## Status at a glance

| | What | Status | Evidence |
| --- | --- | --- | --- |
| C0 | The shape protocol every shape answers (support, filled cells or a body) | done | the protocol in `src/wavebit/fdtd/_legacy_geometry.py`; `tests/fdtd/test_fd_shape_protocol.py` |
| C1 | The geometry kernel `_shapes/`: exact fills, the `<n n^T>` moment, CSG on the distance (internal) | done | [smoothed occupancy](fd-subpixel-occupancy.md); `tests/fdtd/test_fd_shapes.py` |
| C2 | Occupancy per Yee component, `Sphere`, the mixing rule | done | [smoothed occupancy](fd-subpixel-occupancy.md) |
| C3 | **The pass mark: a sphere against Mie** | **passed only with `discretisation='stencil'`**; the default voxel rule fails the cylinder | [Mie validation](fd-mie-validation.md) |
| C4 | `Cylinder`, `Prism`, turned `Box`, the `Box`/`Region` rename | done | [smoothed occupancy, C4](fd-subpixel-occupancy.md#c4-the-extruded-bodies-2026-09-26) |
| C5 | The refusals and what `check` prints; the star lift | done | [smoothed occupancy, C5 and the star lift](fd-subpixel-occupancy.md) |
| -- | The flush junction (`junction='split'`, default) | done | [smoothed occupancy, the junction](fd-subpixel-occupancy.md#the-junction-a-body-standing-flush-on-a-face) |
| C6 | Differentiable edges (`range=`), AD against central differences | done | [smoothed occupancy, C6 sections](fd-subpixel-occupancy.md); [differentiation guide](../../guide/fdtd/fd-differentiation.md) |
| C7 | **A metasurface against RCWA** | **not passed** (M3 order 1.64 / 1.75 in amplitude, 1.51 / 1.62 in efficiency, under 1.8) | [metasurface validation](fd-metasurface-validation.md) |
| C8 | Dispersive and lossy smoothing | **partial**: flat films and separated stacks only | the C8 film and stack pages below |
| C9 | A structure crossing the absorbing layer | **narrow routes only** | [matched absorption](fd-constitutive-absorber.md); [substrate continuation](../../guide/fdtd/substrate-gradients.md) |
| C10 | Two smoothed surfaces in one voxel (the polytope clip) | **partial**: answered where one surface is flat; arm L passes, **the dimer gate fails**, so two curved surfaces stay refused | [the polytope clip](fd-polytope-clip.md) |
| C11 | The off-diagonal constitutive slot | **not written; not closed by its rule; left to the owner** | [C11 record](fd-stage4-c11.md) |
| C12 | These documents and this report | this page | -- |

## C0-C2: the protocol, the kernel and the rule

A shape answers, per field component's own Yee point, what fraction of that component's voxel
it fills and the second moment of its surface normal there; the coefficients mix the body's
material into the cell by the diagonal of the Farjadpour/Kottke effective inverse tensor. The
geometry kernel's fills are exact closed forms (ball-box clip, clipped disc and polygon areas,
the cube-half-space cut) held to independent oracles; the CSG nodes (`Union`, `Intersection`,
`Difference`) exist inside `_shapes/` and answer every voxel with a reported bound where an
exact clip does not exist. **They are not public**, and C10 does not change that (below).

## C3: the Mie pass mark -- passed with the stencil only

The registered gate passes for `fd.Structure(..., discretisation='stencil')`: the epsilon-12
sphere converges at mean / RMS order 2.2442819 / 2.2443152 (1.48% scatter) over 40/56/80 cells
per wavelength and three placements, the TE epsilon-12 cylinder at 2.137819. **The default
voxel rule does not pass**: it passes every sphere arm (epsilon 12 at 1.8268534, -6.18% at 80
cells) but fails the TE epsilon-12 cylinder at order 1.6033652. The stencil applies to one
`Sphere` scene at a time, takes plain numbers and needs a `PlaneWave` or `TFSF` source. The
default rule's adjoint cross coupling (the off-diagonal terms at common nodes, see C11) is what
raised the glass sphere from 1.714 to 2.023.

## C4-C6: bodies, refusals, derivatives

`Cylinder`, `Prism` and turned `Box` ship with exact fills; every crease inside one body is two
media and exact. The refusals name what they refuse and carry the number the scene reads with
the door held open. The star lift makes the smoothed operator positive by construction.
Geometry derivatives (`range=`) are available for the sphere, cylinder, box and prism, under
plane-wave illumination, for several separated bodies and for a pillar on its substrate, each
held to central differences on JAX and Torch.

The flush junction -- a `Cylinder` or unturned `Box` standing on the face of an unturned smoothed
`Box` -- has its own measured rule (`split`, default; second order on the flat-film anchor,
an unresolved rim order on one pillar scene) and alternatives (`staircase`, `refuse`).

## C7: the metasurface against RCWA -- not passed

Under its registered rule the campaign of 2026-10-03 does not pass. M1 (epsilon-4 cylinder)
passes both polarisations (orders 1.90 s, 1.97 p). **M3 (epsilon-12 box, conical incidence)
fails the order criterion**: amplitude orders 1.64 (s) and 1.75 (p), efficiency 1.51 / 1.62,
under 1.8, while agreeing with RCWA in all 54 cells. M2 p fails one cell of 27 (a
cross-polarised term RCWA holds at zero by symmetry); M2 s is not judgeable at its RCWA ladder.
Registered follow-ups found that M3's low order does not need the pillar-substrate rim (a box
lifted off its glass converges at the same order), did not find it in the corners of the xz
cross-section, and left the y faces a loose suspect. The cause is not identified; the record
states it as a limitation.

## C8: dispersive and lossy smoothing -- partial

Implemented and measured, each in its own bounded scope: one flat conductive film
([conductive film](fd-conductive-film.md)), separated conductive films
([conductive stack](fd-conductive-stack.md)) and two exactly touching ones
([conductive contact](fd-conductive-contact.md)); one flat Drude or Lorentz film
([single pole](fd-single-pole-film.md)); one Debye film ([Debye film](fd-debye-film.md)), two
Debye poles ([two Debye](fd-two-debye-film.md)) and finite Debye sums
([Debye sum](fd-debye-sum-film.md)); finite Drude/Lorentz sums
([oscillator sum](fd-oscillator-sum-film.md)); mixed Debye and oscillator films
([hybrid film](fd-hybrid-film.md)); separated Debye and single-oscillator stacks
([Debye stack](fd-debye-stack.md), [oscillator stack](fd-oscillator-stack.md)). Each page
records its unresolved cells. Each scope is concrete (no traced material values), axis-aligned,
in a lossless dielectric and clear of absorbing layers. **Refused**: a conductive or dispersive
material on any curved or rotated body -- the manual's "no smoothed lossy or dispersive bodies
outside flat films" -- and a lossy medium at a junction. A smoothed curved metal -- a metal
nanoparticle, for example -- is therefore not available today.

## C9: a structure crossing the absorbing layer -- narrow routes only

Two routes exist. A coupled dielectric support may sit inside a conductive `Absorber` when every
interpolation link connects equal damping rates ([matched absorption](fd-constitutive-absorber.md));
all its physical calibration runs stayed unsettled and no convergence result is claimed. A
fixed, unturned, transversely uniform `Box` substrate may continue through an `Open()` face when
it declares `Structure.continuation` ([substrate continuation](../../guide/fdtd/substrate-gradients.md)).
**A finite or curved smoothed body crossing a PML is refused.**

## C10: two smoothed surfaces in one voxel

C10 turns part of door 5's refusal into an answer ([record](fd-polytope-clip.md)): where two
smoothed bodies' surfaces share a voxel and the volume they share there is exact -- proved
apart, both flat, or one an unturned `Box` face -- each medium's fill is exact and one
projection over both interfaces mixes the three media; it is the exact laminate for parallel
layers and the two-phase rule bit for bit with one medium absent. Its acceptance was registered
before any code, and amended three times before the measurements it changed.

* Gates 1, 3 and 4 pass: fills and interface moments against polytope and quadrature oracles,
  every existing two-media scene byte-identical and the four goldens reproduced, backend
  agreement and material derivatives against central differences.
* **Gate 2, arm L passes**: a planar stack whose gap shares voxels at every grid converges at
  order 1.866 (s) and 1.817 (p) against the transfer matrix, with 50% and 63% scatter; the
  control with no clip voxel reads 1.852 / 1.856.
* **Gate 2, arm D fails** both criteria. The S7 dimer (two dielectric spheres, gap 0.05
  diameters; S7's two metals cannot be smoothed at all) is refused by the resolution door at the
  coarse grids where its clip writes, and where it runs, its error sits 0.08 to 0.49 times the
  law fitted through the finer grids, outside the registered factor of 2. **As registered, a
  voxel two curved surfaces share stays refused.**
* The S7 flush pillar stays the junction's: no clip record, its coefficients byte-identical.

So C10 answers bodies resting on, raised just above or sunk into a `Box` face, and boxes or
prisms beside each other; it refuses dimers of curved bodies, flush joints the junction does
not cover, curved overlaps and one-material unions. **Public CSG (`fd.Union`, `fd.Difference`)
stays internal**: C10 does not make the CSG nodes exact (they still answer with bounds where
two children's surfaces meet), so nothing about them falls out of it.

## C11: the off-diagonal constitutive slot -- left to the owner

The slot was registered to be never written if studies S1 and S2 showed the off-diagonal terms
buy no order. S1 (tilted flat interface) says no; S2 (2-D Mie cylinder) shows the diagonal rule
failing the TE arm at contrast 12, and the cross terms raised the glass sphere from 1.714 to
2.023. The rule's premise is therefore not met, C11 is not closed by it, and nothing is built.
Isotropic cross coupling already exists in the engine (from C3's work). What C11 could still
mean -- anisotropic materials in smoothed cells, or another off-diagonal writing for the
high-contrast cylinder -- is unmeasured and listed in the [C11 record](fd-stage4-c11.md).

## What stage 4 leaves

* the default voxel rule's TE epsilon-12 cylinder (order 1.603 with cross terms, under 1.8);
* C7's M3 order and its unidentified cause;
* dispersive and lossy smoothing beyond flat films and stacks (curved metals);
* a finite or curved body through a PML;
* the C11 decision;
* two curved surfaces in one voxel (C10's failed dimer gate), the other C10 refusals listed in
  its record, and public CSG.
