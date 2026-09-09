# Independent FEM references for NVF interfaces

This investigation is still open. The new reference instrument is
[`studies/nvf_fem_reference.py`](../studies/nvf_fem_reference.py). It solves the
three-dimensional Maxwell problem with NGSolve H(curl) elements and imports no
Wavebit solver or factorization code. It includes the tilted contact fixture
and a separate affine two-material gap profile. **It is a reference study, not a replacement
for rectangular NVF or a converged golden answer.**

The immediate purpose is to obtain an independently converged answer before
judging changes to the NVF extension or its assembly. S4 direct rule and the
Wavebit direct-rule control agree at the same Fourier truncation, but that does
not make either one a converged oracle. See the earlier comparison in
[`nvf-contact-followup.md`](nvf-contact-followup.md#independent-tilted-fixture-comparison-2026-09-08).

## Physical problem and discretization

All lengths in the FEM instrument are micrometers: period 0.7 square, wavelength
0.9, film thickness 0.3, rectangle width 0.64, rectangle height 0.7/sqrt(5), and
angle atan2(1,2). Inclusion/background/substrate permittivities are 12.25/1/2.25.
The incident plane wave has theta=0.2, phi=0, and p polarization.

The rectangular prism and its periodic images are clipped and united in the
geometry kernel, then meshed conformingly. Its volume is checked against the
analytic value, 0.0601055072352. This preserves the approximately 13.9 nm contact
run without a raster cutoff. Both lateral periodic identifications are checked
on the **generated mesh**. The Bloch phase is factored out of the unknown; the
curl in the weak form is `curl(u) + i k_parallel cross u`.

The incident and bare-substrate Fresnel fields are analytic. The scattered-field
right-hand side is `k0^2 (epsilon - epsilon_background) E_background` in the
inclusion. Electric and magnetic PML tensors follow the same coordinate
transformation. Power is the spatially averaged Poynting flux in the physical
superstrate and substrate, excluding the PML. Static condensation eliminates
interior element unknowns and reconstructs them after the sparse solve. A full
operator residual is checked after reconstruction.

NGSolve documents the [periodic finite-element construction](https://ngsolve.org/ngsolve/docs/i-tutorials/unit-2.12-periodicity/periodicity.html)
and its [H(curl) elements](https://docu.ngsolve.org/nightly/i-tutorials/unit-2.3-hcurlhdiv/hcurlhdiv.html).
The field and PML formulation in the instrument was written independently; no
external solver implementation was copied into Wavebit.

The second boundary option, `--boundary rayleigh`, omits both PML layers. It
couples the lateral-periodic finite-element trace to outgoing Rayleigh waves.
For lateral wave vector k and outgoing vertical rate gamma, the tangential
admittance is `Gamma = gamma I + k k.T/gamma`. Every propagating order must be
included, or the instrument raises. The base sparse form uses local impedance;
a low-rank correction replaces it with this exact admittance for the retained
orders. A Woodbury solve applies the correction using one sparse factorization.
Omitted evanescent orders retain local impedance, so the Rayleigh cut and physical
padding remain independent convergence parameters. This is **not** an exact
infinite boundary operator at a finite cut.

The general finite-element/DtN approach to biperiodic Maxwell scattering is
described by [Jiang et al., 2018](https://arxiv.org/abs/1811.12449). The present
instrument does not implement their adaptive error estimator, and their error
bounds must not be assigned to these results. It also reports two flux readouts:
outgoing Rayleigh power and volume-averaged physical Poynting flux.

## Rayleigh data for a physical primitive lattice

The boundary-data helper now accepts
`rayleigh_data(cut, epsilon, lattice=A, theta=theta, phi=phi)`, where A
contains physical primitive vectors in columns, in micrometers. Its default
retains the original square lattice and incidence. The unit-index incident
half space supplies the lateral incident vector; the requested epsilon
belongs to the outgoing half space. For integer labels `(m,n)` it computes

```text
k = k_incident + 2*pi*A^(-T)*(m,n)
gamma = sqrt(k0^2*epsilon - k.k), with Im(gamma) >= 0
Gamma = gamma*I + k*k.T/gamma.
```

The propagating-order check uses a bound for each integer label separately.
If a_i is a primitive vector, then
`label_i = a_i.(k-k_incident)/(2*pi)` over the disk
`|k| < k0*sqrt(epsilon)`. Projecting that disk onto each a_i gives a finite
rectangle enclosing every propagating label, including for strongly
sheared bases. It does not assume that small physical wavevectors have
small integer labels in the supplied basis.

For the [affine comparison](nvf-affine-convergence.md),
`A=[[.7,.315],[0,.49]]`, theta=.2 and phi=0 give only `(0,0)` in the top
half space. The substrate also propagates `(-1,0)` and `(-1,-1)` under
this FEM helper's `+G` convention; the latter has
`gamma=1.526674994510` per micrometer. Cut 1 contains these propagating
orders but does not establish convergence of the omitted evanescent
boundary correction. Exactly grazing retained orders remain unsupported.

Nine mesh-free algebraic controls pass: the earlier independent slab
and square-lattice admittance checks, plus seven added cases for affine
plane-wave admittances, explicit reciprocal components, rotation
covariance, strongly sheared integer labels, default compatibility,
invalid inputs and exact grazing. One control represents the square
physical lattice with `A=[[.7,2.8],[0,.7]]`; its original near-grazing
wave has labels `(1,4)`, so cut 3 must be rejected. These checks require
no tetrahedral mesh or large FEM solve.

`solve` and `solve_rayleigh` retain the fixed square geometry, p source,
trace phases and area normalization described above. The separate
`solve_affine_rayleigh` entry point now connects the physical mesh below
to both polarization sources, affine trace phases and physical-area flux
normalization. Its discretization and outgoing-boundary uncertainties
still need qualification before assigning an affine optical reference value.

## Physical affine geometry and periodic pairs

`make_affine_profile_mesh()` constructs the affine comparison's material
volumes directly in physical space and returns `(mesh, materials)`. All
lengths are in micrometers. With `J=[[1,.45],[0,.7]]`, the reference
period .7, width .350, gap .017 and seam .031 produce two adjoining
parallelogram prisms of permittivities 4 and 9. Their reference widths
are .206 and .144; their reference height is .683 and physical film
thickness is .3. The reference rectangles are transformed before
extrusion. Ordinary physical rectangles are not reinterpreted as this
affine profile.

The cell has physical area .343 square micrometers. It includes .1 of
physical padding above and below, with permittivities 1 and 2.25. There
are no absorbing layers. `uniform=4.` replaces the entire film with a
uniform slab for analytic controls. The scattering entry point consumes
that mesh; the geometry helper itself solves no field equation.
The patterned material volumes are .02954658, .02065392 and .05269950
cubic micrometers for the first piece, second piece and background.
Each exterior region occupies .0343 cubic micrometers.

Opposite faces are selected using reference coordinates on the physical
shape and identified by translation through the corresponding physical
primitive vector. `periodic_counts(mesh, lattice=A)` checks the resulting
point-pair displacements against each complete vector or its negative.
It rejects componentwise sign flips, unexpected displacements and either
missing direction. Its default square lattice remains compatible with
both existing 3D and static 2D reference meshes.

Eight small-mesh controls check material and total volume, outer-face
areas and heights, material values on both sides of each interface,
actual periodic pairs, reduced H(curl) unknown counts, whole-lattice
rotation, the uniform slab geometry and invalid inputs. The existing
nine mesh-free algebraic controls also pass. At maximum tetrahedron
size .2, the observed mesh sizes are:

| geometry | tetrahedra | vertices | ordinary order-1 H(curl) unknowns | periodic unknowns |
| --- | ---: | ---: | ---: | ---: |
| affine profile | 436 | 138 | 1342 | 1128 |
| profile rotated by .53 radians | 448 | 140 | 1370 | 1122 |
| uniform affine slab | 414 | 134 | 1294 | 1074 |

These are geometry and finite-element-space checks, with no Maxwell
system assembled or solved. The helper requires positive padding,
thickness and gap, an interior seam, width below the reference period,
a finite real orientation-preserving transform, and h smaller than half
each physical cell width between opposite faces. Invalid inputs, failed
face matching, and incorrect material volumes or exterior areas raise.
Exact contact and general physical-rectangle inputs remain outside this
helper's scope. The new scattering entry point uses these meshes with
the source, boundary-trace and power normalization described next.

## Affine Maxwell connection and independent controls

`solve_affine_rayleigh` solves the physical affine profile for p and s
incidence in one call. It shares one mesh, sparse Maxwell factorization
and Fourier trace response between the two source vectors. A two-column
Woodbury solve applies the same retained Rayleigh boundary correction to
both fields. Interior unknowns are reconstructed after that correction,
and each polarization is checked against the full uncondensed operator.
The existing square-fixture entry points retain their previous behavior.

The incident electric field has unit magnitude for both polarizations.
Writing `t=(cos(phi),sin(phi),0)` and `s=(-sin(phi),cos(phi),0)`, its
tangential p amplitude is `cos(theta)*t` and its s amplitude is `s`.
The bare-substrate incident, reflected and transmitted electric and
magnetic fields are analytic, with `H=k cross E/k0`. Only the film
contrast `epsilon-1` drives the scattered-field source; no additional
source is applied inside the substrate.

The periodic field has its incident Bloch factor removed. For physical
lattice A and reciprocal label `(m,n)`, the trace basis is
`exp(i*G.r)/sqrt(area)`, where `G=2*pi*A^(-T)*(m,n)` and
`area=det(A)`. Its amplitude is therefore `sqrt(area)` times the
physical tangential electric amplitude. Bare-interface reflected and
transmitted waves are added to the zero-order ports before computing
outgoing powers. The modal power divisor is `area*k0*cos(theta)`;
the volume-averaged Poynting-flux divisor is `area*pad*cos(theta)`.
Both readouts and every retained diffraction-order R/T power are returned.

Four new optional tests compare the result with an independent Airy
slab formula, the exact zero-contrast interface, full/condensed solutions,
and invalid requests. The zero-contrast test includes both polarizations,
nonzero azimuth, normal incidence, and a different affine cell area.
The driven slab test checks the specular order and absent nonzero orders
as well as total R/T. It spies on the actual sparse `Inverse` call to
confirm one factorization supplies both polarizations.

The driven epsilon-4 slab at h=.18, theta=.2, phi=.37, pad=.1 and cut 1
fails the predetermined `5e-4` accuracy target at FEM degree 3: its p
transmission differs from the analytic value by `1.78042e-3`. Refining
to degree 4 on the same 438-element mesh passes that unchanged target:

| polarization | analytic T | FEM degree-4 T | absolute T error | modal/volume disagreement | lossless balance error |
| --- | ---: | ---: | ---: | ---: | ---: |
| p | .838052025533 | .838490396510 | 4.38371e-4 | 1.262e-7 | 4.276e-5 |
| s | .826108566299 | .826577853556 | 4.69287e-4 | 1.472e-7 | 4.405e-5 |

The degree-4 slab has 23,085 unknowns and takes 7.80 s with peak process
RSS 1.191 GB on the observed host. These are measurements of this control,
not an accuracy or performance guarantee for the patterned profile.
The combined selection passes 21 tests: four new scattering checks,
eight affine geometry checks and nine existing algebraic checks.
An independent review compares 64 electric/magnetic field samples with
Maxwell plane waves to within `5.56e-16` and modal admittance power with
direct Poynting power to within `2.23e-16`.

These controls verify the affine connection. The `5e-4` slab criterion
is an instrument check, not a chosen NVF convergence threshold. The
patterned profile needs separate mesh/degree, Rayleigh-cut and padding
refinements. Small linear residuals alone cannot establish optical accuracy.

## A failed periodic-boundary request was caught by the control

The first local FEM prototype selected outer faces within 1e-8. OpenCascade pads
face bounding boxes by about 1e-7 in these coordinates, so **both selections were
empty**. Calling `Identify` on those lists did not raise an error. The resulting
mesh had zero periodic pairs and answered a different boundary-value problem.
Its small algebraic residual did not reveal this error.

The current instrument checks nonempty, matched outer-face selections and checks
actual point-pair displacements in both directions after meshing. Tests also
verify that the periodic space has fewer independent unknowns than the ordinary
space. The rejected nonperiodic values are not included below.

## Measured controls and preliminary result

Here `h` is the outer maximum tetrahedron size; the patterned film requests h/2.
`p` is the H(curl) polynomial order. PML thickness and physical padding are each
0.4, with stretch 1+2i. A small energy error is not a continuum error bound.

The exact uniform-epsilon-4 slab has T=0.838052025533.

| h | p | uniform slab T | absolute error against analytic T | energy error |
| --- | --- | ---: | ---: | ---: |
| 0.14 | 2 | 0.853731655190 | 0.015679629657 | 0.003192760229 |
| 0.18 | 3 | 0.841376878322 | 0.003324852789 | 0.000209273140 |
| 0.18 | 4 | 0.838232037221 | 0.000180011688 | 0.000004092306 |

These controls were run with the corrected local prototype before packaging the
instrument. The packaged version adds a higher quadrature order for the analytic
source and explicit metadata/guards; its nonzero-source control is also exercised
by the optional test suite.

For the actual tilted contact:

| h | p | elements | unknowns before condensation | T | energy error |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.14 | 2 | 5884 | 56415 | 0.779444332792 | 0.001660836954 |
| 0.14 | 3 | 5884 | 146064 | 0.873537338349 | 0.000032048686 |
| 0.18 | 4 | 4254 | 217000 | 0.876390743842 | 0.000003790406 |
| 0.14 | 4 | 5884 | 300555 | 0.876511687509 | 0.000001813687 |

The same-p mesh change is only 0.000120944, but this sequence has **not** converged
in the absorbing boundary. At h=0.18, p=4, changing the PML stretch from 1+2i to
1+3i changes T from 0.876390744 to 0.870897410, a shift of 0.005493334. Both have
energy errors about 4e-6. Therefore **the 0.876 values are excluded as an accuracy
reference**, despite their small mesh difference, residual, and energy error.

The substrate (1,0) Rayleigh wave has gamma=1.507144577 per micrometer. With the
short 0.4-micrometer PML, its predicted round-trip attenuation in amplitude is
`exp(-2*gamma*0.4*stretch)`: 0.0896868 at stretch=2, and 0.0268592 at stretch=3.
The uniform slab does not excite this order, explaining why that control alone
could not qualify the patterned structure's PML.

## Rayleigh-boundary measurements

Physical padding is 0.4. Preliminary patterned results from the local prototype
used volume flux only; the packaged instrument retains that value as `volume_T`
and uses outgoing Rayleigh power as `T`.

| h | p | cut | elements | unknowns | volume T | volume energy error |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 0.18 | 3 | 2 | 3478 | 86316 | 0.862943303916 | 0.000277763392 |
| 0.18 | 4 | 2 | 3478 | 177625 | 0.866055904424 | 0.000010118006 |

Thus 0.866055904 is also a **preliminary value**, not a golden answer. The
h=0.14, p=4 Rayleigh refinement did not return an observable: it was terminated
after severe paging. A process-specific `vmmap` measurement reported a 23.6G
peak footprint and 15.8G swapped out. This is a resource-limited refinement,
not a numerical result and not evidence of a physical divergence.

At h=0.18, p=3, reducing the Rayleigh cut from 2 to 1 changes volume T by only
0.00000652516 (0.862943303916 to 0.862949829077). Both cuts retain every propagating
order. The cut=1 modal T is 0.863038885671, and its modal/volume disagreement is
0.0000890566. This boundary-cut check is much smaller than the current p=3 to p=4
change. It does not replace the finer-mesh check at p=4.

Several bounded linear-solver alternatives were checked before stopping this
refinement. SciPy GMRES with a projected NGSolve BDDC preconditioner failed its
600-iteration cap on the much smaller h=0.25, p=2 uniform control; positive and
indefinite preconditioner forms both failed. A SuperLU/MMD experiment without
diagonal pivoting reproduced the h=0.18, p=3 uniform Rayleigh result within
2e-14, with a full residual of 2.4e-13, but took about 50 seconds versus about
14 seconds for the original prototype's UMFPACK solve. Its patterned accuracy
and memory scaling were not qualified. The ordinary-pivot SuperLU variant was
stopped after several minutes on that small control. No alternative was
installed, and no failed/incomplete solve was converted into comparison data.
The large refinement and all linear-solver experiments have been stopped or
completed; none is left running by this investigation.

The packaged Rayleigh instrument's epsilon-4 uniform slab at h=0.18, p=3, cut=1
gives T=0.838312384096, volume T=0.837902615436, and analytic T=0.838052025533.
The two readouts disagree by 0.000409769. This observed flux error is kept
explicit instead of selecting the readout that happens to be closer.
No production NVF behavior or acceptance tolerance was changed.

## Scope of the optional reference environment

The inspected environment contains `ngsolve==6.2.2606`,
`netgen-mesher==6.2.2606` and `netgen-occt==7.8.1`. The first two packages
declare LGPL-2.1-only; OCCT has its LGPL license and additional exception.
The package declarations do not describe every bundled component.
In particular, the installed `netgen/libngla.dylib` exports
`SuiteSparse_version`, `umfpack_dl_report_status` and
`umfpack_zl_report_status`, and contains an UMFPACK banner and GPL notice.
The exact bundled UMFPACK version has not been identified. The official
[UMFPACK license](https://github.com/DrTimothyAldenDavis/SuiteSparse/blob/dev/UMFPACK/Doc/License.txt)
identifies GPL-2.0-or-later, so this environment must not be described as
containing only LGPL components or as cleared for product redistribution.

The reference use remains internal numerical validation, with independently
written study code and numerical outputs that contain no solver
implementation. The running/output provisions of the
[NGSolve license](https://github.com/NGSolve/ngsolve/blob/master/LICENSE)
and [GNU GPL version 2](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html)
distinguish that use from redistribution. No FEM binaries or external
solver source are added to the Wavebit runtime or its installation
dependencies. Bundling, linking or distributing the reference environment
requires a separate review of exact component versions and their terms.
The [OCCT license](https://github.com/Open-Cascade-SAS/OCCT/blob/V7_8_1/LICENSE_LGPL_21.txt)
and [exception](https://github.com/Open-Cascade-SAS/OCCT/blob/V7_8_1/OCCT_LGPL_EXCEPTION.txt)
are part of that review, not a substitute for it.

## Reproduction

On the current host NGSolve 6.2.2606 is already installed in
the optional NGSolve environment. Run from the
NVF worktree:

Set `NVF_FEM_PYTHON` to the Python executable with the optional NGSolve dependencies.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 "$NVF_FEM_PYTHON" studies/nvf_fem_reference.py --hs .18 .14 --orders 4 --boundary pml
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 "$NVF_FEM_PYTHON" studies/nvf_fem_reference.py --hs .18 --orders 3 4 --uniform 4 --boundary pml
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 "$NVF_FEM_PYTHON" studies/nvf_fem_reference.py --hs .18 .14 --orders 4 --boundary rayleigh --cut 2
PYTHONPATH=$PWD "$NVF_FEM_PYTHON" -m unittest discover -s tests -p test_nvf_fem_reference.py
```

The optional tests use standard-library unittest because this reference
environment does not contain pytest. The ordinary Wavebit pytest environment
skips them if NGSolve is unavailable. Tests cover actual periodic pairs, missing
and one-axis-only identifications, the clipped periodic material geometry,
analytic and nonzero-source slab controls, the zero-contrast limit, and
condensation/reconstruction equivalence for both boundary formulations. The
Rayleigh tests also reject an omitted propagating order and require the mesh to
contain no PML. All **12 original optional tests passed** in the NGSolve environment. They
validate the instrument; they are not additional NVF physics successes.
The lattice-helper extension adds seven tests and moves two existing
algebraic tests into `FEMAlgebraicTests`, which constructs no mesh. Its
nine checks pass separately; the ten unchanged mesh/solve checks were
not rerun for this boundary-data-only change. Run the small selection with
`python -m unittest tests.test_nvf_fem_reference.FEMAlgebraicTests -v`.
The subsequent geometry extension passes all eight affine geometry tests
plus those nine algebraic tests in 0.358 s:

```bash
PYTHONPATH=$PWD "$NVF_FEM_PYTHON" -m unittest tests.test_nvf_fem_affine_geometry tests.test_nvf_fem_reference.FEMAlgebraicTests -v
```

Five existing square-fixture mesh and periodic-identification tests also
pass in 0.387 s. The five existing scattering-solve tests are not rerun for
the geometry-only change. The new checks establish geometry and space
consistency, not a converged optical reference.

The full affine integration is checked separately, including both
polarizations and the driven degree-4 slab:

```bash
PYTHONPATH=$PWD "$NVF_FEM_PYTHON" -m unittest tests.test_nvf_fem_affine_scattering tests.test_nvf_fem_affine_geometry tests.test_nvf_fem_reference.FEMAlgebraicTests -v
```

This selection passes 21 tests in 9.686 s. To reproduce a physical affine
solve through the Python entry point:

```bash
PYTHONPATH=$PWD "$NVF_FEM_PYTHON" - <<'PY'
import json
import ngsolve as ng
from studies.nvf_fem_reference import solve_affine_rayleigh
ng.SetNumThreads(1)
result = solve_affine_rayleigh(h=.18, order=4, cut=2, pad=.1)
print(json.dumps(result, allow_nan=False))
PY
```

Adding `uniform=4., phi=.37` reproduces the slab geometry and incidence;
its reported control above uses `cut=1`. The default command-line entry
point continues to select the historical square fixture.
Selected results, with prototype-versus-packaged flux provenance, are retained in
[`studies/results/nvf_fem_reference.log`](../studies/results/nvf_fem_reference.log).
