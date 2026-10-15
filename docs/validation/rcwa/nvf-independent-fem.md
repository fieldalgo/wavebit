# Independent FEM references for NVF interfaces

This investigation is still open. The new reference instrument is
[`studies/rcwa/nvf_fem_reference/nvf_fem_reference.py`](../../../studies/rcwa/nvf_fem_reference/nvf_fem_reference.py). It solves the
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

## Separating the omitted Robin boundary power

The affine solver also returns `robin_complement_power` for each side and
their sum. This measures the outgoing power assigned by the local Robin
condition to the part of the scattered trace outside the retained Fourier
space. For an orthonormal trace basis and retained coefficients `ports`,
the contribution of a side with exterior permittivity epsilon is

```text
L_side = sqrt(epsilon) * (integral_side |u_sc,t|^2 - sum |ports|^2)
         / (area*cos(theta)).
energy_budget_residual = 1 - R - T - L_top - L_bottom.
```

Only physical tangential x/y components enter the norm. The bare-interface
annex is a retained constant trace, so adding it does not change the
complement. This quantity is a numerical outgoing sink, not physical
absorption by evanescent waves. Quadrature or projection errors can make
the observed difference negative; the diagnostic preserves its sign.
The remaining signed energy budget can contain FEM, quadrature, source
and linear-solve error. Neither number is an optical error bound.

Six small-mesh tests integrate independent analytic coefficient functions,
without assembling a Maxwell matrix. They check Parseval power on both
sides, removal of a retained mode, constant-annex invariance, physical
rotation and scale, signed projection overshoot, and invalid inputs.
Together with the affine scattering, geometry and algebraic controls,
27 tests pass. An independent review confirms the formula, port ordering,
normalization and use of the scattered trace before the modal add-back.

On the patterned h=.18, FEM-degree-4, cut-1, pad=.1 calculation, adding
this postprocessing leaves all previous R/T, volume readouts, residuals
and individual diffraction powers exactly identical:

| polarization | signed `1-R-T` | Robin complement power | remaining signed budget |
| --- | ---: | ---: | ---: |
| p | 3.78233e-3 | 3.80466e-3 | -2.23299e-5 |
| s | 4.51552e-3 | 4.55214e-3 | -3.66228e-5 |

The omitted Robin trace accounts for most of this cut-1 energy deficit.
The small remaining budget does not qualify the patterned transmission.

## Patterned affine boundary and degree refinement

The following calculations use the physical affine profile above, theta=.2,
phi=0, pad=.1 and the original quadrature bonus 6. Here h is the global
maximum tetrahedron size; the affine helper does not impose the h/2 film
restriction used by the historical square fixture below. This is a
three-dimensional Maxwell FEM polynomial
degree, distinct from the degree-10 two-dimensional auxiliary field used
inside the separate Wavebit comparison. No material or optical parameter
changes between the rows.

First, the Rayleigh cut increases at h=.18 and FEM degree 4 on the same
469-tetrahedron, 24,645-unknown mesh. The signed balance is retained because
the raw energy deficit changes to an excess as the boundary improves.

| cut | p T | s T | p `1-R-T` | s `1-R-T` | largest modal/volume disagreement |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | .848428442375 | .727567584962 | 3.78233e-3 | 4.51552e-3 | 3.973e-3 |
| 2 | .852230459897 | .731148729630 | 2.09300e-5 | 4.98574e-6 | 2.478e-5 |
| 3 | .852278152427 | .731186559488 | -2.01608e-5 | -3.48461e-5 | 1.296e-5 |
| 4 | .852280465907 | .731188011462 | -2.20430e-5 | -3.64442e-5 | 1.463e-5 |

The last cut increment changes p/s transmission by
`+2.31348e-6`/`+1.45197e-6`. It is one observed finite increment, not a
bound on the omitted boundary tail. In particular, all signed energy
errors cannot be described as outgoing Robin loss.

Next, at the same h=.18 and cut 4, the degree increases without changing
the mesh:

| FEM degree | unknowns | p T | s T | p absolute balance error | s absolute balance error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 24,645 | .852280465907 | .731188011462 | 2.20430e-5 | 3.64442e-5 |
| 5 | 43,854 | .852055828117 | .730483079606 | 5.85361e-7 | 9.76317e-7 |
| 6 | 71,106 | .852182720280 | .730695883465 | 5.10051e-8 | 2.68792e-8 |

The degree-4-to-5 transmission changes are
`-2.24638e-4`/`-7.04932e-4`; the degree-5-to-6 changes reverse sign,
to `+1.26892e-4`/`+2.12804e-4`. Degree 6 has algebraic residuals below
`1.17e-14` and modal/volume disagreements below `3.44e-6`, but its tiny
energy balance error does not establish transmission convergence.

At degree 5, reducing h from .18 to .14 regenerates a 555-element mesh
with 51,906 unknowns. Its p/s transmission is
`.852106590399`/`.730558488159`, changing by
`+5.07623e-5`/`+7.54086e-5`. The largest modal/volume disagreement grows
to `1.54e-5`, despite absolute energy balance errors below `9.31e-8`.
These meshes are not nested, and neither the maximum element size nor
the energy budget alone orders the error of every output.

Increasing the h=.14 calculation to degree 6 gives 84,161 unknowns and
p/s transmission `.852191204024`/`.730701980509`. The h=.18-to-.14 changes
at degree 6 are `+8.48374e-6`/`+6.09704e-6`, smaller than the measured
mesh changes at degree 5. However, the degree-5-to-6 changes at h=.14
remain `+8.46136e-5`/`+1.43492e-4`. This comparison does not establish a
tail estimate. The degree-6 signed energy remainders after accounting
for the Robin complement are only `-4.93e-12`/`-2.80e-11`, whereas the
modal/volume disagreements remain `8.67e-6`/`4.34e-6`. A nearly closed
global boundary budget does not imply equally accurate interior fields.

The h=.18 degree-4 cut ladder takes 7.68, 14.26, 24.94 and 37.86 s.
The cut-4 degree-5 and degree-6 solves take 88.10 and 200.13 s, with
peak process RSS 2.208 and 2.977 GB. The h=.14 degree-5 solve takes
123.43 s and 2.501 GB; h=.14 degree 6 takes 300.07 s and 3.585 GB.
These are sequential shared-host observations,
not controlled speed comparisons or an accuracy guarantee.

## Controlling nonpolynomial integration independently

The affine entry point accepts a nonnegative integer `quadrature_bonus`,
defaulting to 6 to preserve the earlier numerical settings. This is a
study parameter separate from the FEM degree: it sets the source bonus
to that value, the Fourier trace bonus to `2*cut+quadrature_bonus`, and
the volume-flux and Robin-norm integration order to
`2*max_order+quadrature_bonus`, where `max_order` includes any local
degree enrichment. Both the result and its settings record
the selected value. Invalid values are rejected before meshing.
The polynomial bulk and local Robin forms retain their default quadrature.

A local element-matrix check on the affine mesh compares the default
bulk/Robin quadrature with bonus 8 at degrees 3 through 6. For the tested
tetrahedron and outer triangle, the combined Bloch bulk form differs
relatively by at most `8.95e-14`; the separate mass, curl and Robin
comparisons are all below `9.92e-14`. This finds no evidence of polynomial
underintegration in those elements, including the incident-wavevector
cross terms. It is not a bound for every mesh or field.

The full patterned h=.18, FEM-degree-4, cut-4, pad=.1 solve was then
repeated with quadrature bonus 14. Its largest individual diffraction
power change is `3.48e-14`, and its largest total or volume R/T change
is `3.59e-14`. The original p/s energy excesses of
`2.20430e-5`/`3.64442e-5` and modal/volume discrepancies therefore persist
under this integration refinement. Raising numerical integration order
does not resolve the observed optical discretization variation in this
control. The repeated solve takes 46.10 s with peak process RSS 1.353 GB.

## Moving the outgoing boundaries

Increasing physical padding from .1 to .2 moves the Rayleigh planes
farther from the patterned film. It also regenerates the tetrahedral
mesh, so the resulting transmission difference is not a pure measure
of boundary truncation. At h=.18 the new mesh has 464 tetrahedra,
compared with 469 for pad=.1.

At FEM degree 5 on the new 43,038-unknown mesh, increasing the Rayleigh
cut from 2 to 4 gives:

| cut | p T | s T | p Robin complement | s Robin complement |
| ---: | ---: | ---: | ---: | ---: |
| 2 | .851943145390 | .730434752280 | 1.24444e-7 | 1.35059e-7 |
| 4 | .851943269585 | .730434871114 | 1.35534e-8 | 8.07164e-9 |

The cut change is `+1.24194e-7`/`+1.18834e-7` in p/s transmission.
By comparison, pad=.1-to-.2 at degree 5 and cut 4 changes transmission
by `-1.12559e-4`/`-4.82085e-5`. The measured same-mesh cut change is
much smaller; separating padding from remeshing still requires
discretization controls. The cut-4 signed
energy remainders are `-1.66890e-6`/`-2.67648e-6`, even though the
omitted Robin power is much smaller. The two cut calculations take
39.15 and 94.30 s, with peak process RSS 2.232 and 2.301 GB.

Increasing to degree 6 on the pad=.2 mesh gives 69,867 unknowns and
p/s transmission `.852132216048`/`.730707118806`. Relative to degree 5,
the changes remain `+1.88946e-4`/`+2.72248e-4`. At degree 6 and cut 4,
pad=.1-to-.2 changes p/s transmission by `-5.05042e-5`/`+1.12353e-5`;
the s padding increment has reversed sign from the degree-5 comparison.
The omitted Robin powers are only `4.39e-9`/`2.47e-9`, while the
remaining signed budgets are `-2.45e-8`/`-3.52e-8` and modal/volume
disagreements are `3.27e-6`/`1.37e-6`. These finite controls leave the
FEM discretization uncertainty unresolved. This solve takes 191.64 s
with peak process RSS 3.453 GB.

## Refining the physical dielectric edges

The optional `edge_h` setting requests a physical mesh-size target along
the patterned film's 20 dielectric edges. Each of the two prisms has
12 edges, with four shared at the internal interface. Assigning the
target to these actual prism edges before gluing the solids follows
the geometry under affine maps and rotations; it does not select
artificial periodic-cell edges using a coordinate box.

The edges include six vertical lines at reference positions
`x=-.175,.031,.175`, `y=+/- .3415`, and seven horizontal lines on each
film interface `z=0,.3`. Two vertical lines at x=.031 meet permittivities
4, 9 and 1. Their endpoints also meet the exterior materials. This
control provides a way to test resolution near these interfaces without
asserting that edge singularities dominate the observed optical error.

The default `edge_h=None` retains the original global-size mesh. An
explicit target must be a finite positive real value no larger than h;
it cannot accompany the uniform-slab control, which lacks these patterned
edges. Netgen treats the value as a mesh-size target, not a strict upper
bound on every generated segment: a .10 target can leave a .144-long
short CAD edge in one segment. It also does not by itself guarantee
a chosen number of subdivisions through the 11.9 nm physical normal
gap corresponding to the 17 nm reference gap.

The result now records `edge_h` and `coupling_dofs` as well as the full
unknown count. `coupling_dofs` measures the global space after static
condensation; the count is still reported if an uncondensed solve is
requested. Equal full unknown counts at different polynomial degrees
can leave different global factorization sizes, so they are not enough
to predict equal memory use.

The small-mesh controls define the 20 physical line segments independently
of the CAD edge selection. At target .08, each line receives more mesh
points and a smaller largest interval, both before and after whole-cell
rotation. Material volumes and samples, exterior areas and heights,
actual periodic pairs and the reduced H(curl) space are checked on both
refined meshes. A separate pre-assembly progress check verifies the
requested target and actual global-space count in the solve metadata.
The full affine selection now passes 29 tests in 11.693 s.

An independent review compares the old helper with both current default
forms and obtains exactly identical vertices, tetrahedron connectivity
and periodic identifications. A distinct affine map
`J=[[.9,-.25],[.3,1.1]]` at target .12 confirms that precisely the
20 physical dielectric edges carry the target after gluing, while the
28 other distinct CAD edges do not. Independent material volumes,
outer-face areas and actual primitive-vector displacements also pass.

For a matched optical comparison, h=.18, cut 4, pad=.2 and quadrature
bonus 6 stay fixed. At FEM degree 4 the global-size mesh has 464
tetrahedra and 24,145 full unknowns, with 17,185 remaining after
condensation. The target-.10 mesh has 908 tetrahedra, 46,720 full
unknowns and 33,100 after condensation. The degree-4 energy
balance errors decrease from `5.42e-5`/`8.27e-5` to
`1.21e-6`/`1.62e-6`, but this change alone cannot establish that the
transmissions are closer to their continuum values.

Increasing to degree 5 on each fixed mesh gives the following matched
comparison. The target-.10 degree-5 space has 83,448 full unknowns and
50,760 after condensation.

| FEM degree | global-size p T | target-.10 p T | global-size s T | target-.10 s T |
| ---: | ---: | ---: | ---: | ---: |
| 4 | .852385240563 | .851926548859 | .732112533661 | .730496704608 |
| 5 | .851943269585 | .852099612881 | .730434871114 | .730691646657 |

The target-.10 degree-4-to-5 increments are
`+1.73064e-4`/`+1.94942e-4`, compared with
`-4.41971e-4`/`-1.67766e-3` on the original mesh. Their magnitudes
decrease by 60.8% for p and 88.4% for s. The largest individual R/T
power changes decrease from `1.98264e-3`/`3.96889e-3` to
`2.17578e-4`/`1.96559e-4`, reductions of 89.0%/95.0%. The smaller
variation therefore also appears in the individual diffraction powers.
These are observed changes over one degree interval, with reversed
signs between the two meshes; they are not estimates or bounds for
the remaining optical error.

The local degree-5 modal/volume disagreements are
`5.65e-6`/`2.55e-6` and signed energy remainders are
`-3.26e-8`/`-8.64e-9`. Its total transmission differs from the
unrefined degree-6 result by `-3.26032e-5`/`-1.54721e-5`. Neither
agreement nor the small energy budget qualifies an independent oracle.
The local degree-4 p modal/volume disagreement actually grows from
`2.00e-5` to `2.37e-5` despite its smaller energy imbalance.

As in the earlier per-order comparison, a close total can be misleading:
the unrefined degree-4 p total differs from the finite Wavebit row by
only `+4.53e-6`, while its transmitting `(-1,0)` and `(0,0)` orders
differ by about `-1.915e-3` and `+2.101e-3`. Local sizing reduces the
magnitude of each transmitting-order difference in this comparison
while increasing the total difference. Wavebit's own Fourier tail
remains unresolved, so this is a cancellation diagnostic, not an
accuracy ranking against a converged reference.

The degree-4 default and target-.10 solves take 42.31 and 108.35 s,
with peak process RSS 1.297 and 2.919 GB. The measured local-size
calculation fits below the 37,387 global unknowns of the unrefined
degree-6 case, which took 191.64 s and 3.453 GB. These observations
describe different finite approximations; they are not an equal-accuracy
performance comparison. The target-.10 degree-5 solve takes 250.73 s
and 4.011 GB, versus 94.30 s and 2.301 GB for the original degree-5
mesh. The reduction in degree sensitivity comes with higher cost at
this fixed degree; no default mesh choice has been changed.

## Refining the physical dielectric vertices

The independent `vertex_h` setting requests a physical mesh-size target
at the patterned film's 12 distinct dielectric vertices. Each prism has
eight vertices; four are shared at their common material interface.
Targets are assigned to the actual prism vertices before gluing, so
the selected points follow the physical affine transform. Artificial
periodic-cell corners are not selected.

The value must be finite, positive and no larger than h, and cannot
accompany a uniform-slab control. It can be used with or without
`edge_h`; the two CAD attributes retain their independent requested
values, without requiring one to be smaller than the other. Netgen
combines these sizing controls when regenerating the mesh. They are
targets, not strict segment-length bounds, and a coarser vertex target
is not promised to leave the existing mesh unchanged.

`vertex_h=None` preserves the original mesh construction. Independent
comparisons against the preceding source revision find exactly the
same points, volume/boundary connectivity and periodic identifications
for default, edge-only and uniform meshes. The focused geometry
selection now passes 14 tests, including the actual 12 target points,
rotation, material volumes, outer areas/heights, periodic displacements
and input validation. At target .07 with edge target .12, the vertical
edge segment next to each physical vertex is shorter than on the
matched edge-only mesh, before and after rotation.

At h=.18, outgoing pad .2, edge target .12 and vertex target .07,
the generated mesh has 1,002 tetrahedra. Its global degree-5 space
has 92,016 full and 55,944 condensed unknowns, compared with
71,796/43,716 on the edge-only mesh. This supplies a separate spatial
refinement axis alongside the fixed-mesh degree controls. The counts
and shorter corner segments do not establish that vertex resolution
dominates the optical error or that the refined optical result is
closer to the continuum solution. Results record the requested
`vertex_h` in both the top-level metadata and settings.

The first optical intervals keep global degree 5, h=.18, edge target
.12, outgoing pad .2, Rayleigh cut 4 and quadrature bonus 6 fixed:

| vertex target | tetrahedra | full / condensed unknowns | p T | s T |
| ---: | ---: | ---: | ---: | ---: |
| none | 780 | 71,796 / 43,716 | .852038068565 | .730662922035 |
| .070 | 1,002 | 92,016 / 55,944 | .852108495556 | .730802741386 |
| .065 | 1,093 | 100,428 / 61,080 | .852094870566 | .730861555984 |

Adding target .070 changes p/s T by `+7.04270e-5`/`+1.39819e-4`;
the largest individual R/T changes are `8.80527e-5`/`2.09671e-4`.
Reducing the target from .070 to .065 changes T by
`-1.36250e-5`/`+5.88146e-5`, with individual changes up to
`5.28178e-5`/`7.30630e-5`. The p increment reverses sign and its
total again hides larger individual-order changes. These targets
regenerate different meshes, not a nested finite-element space, and
do not establish a convergence rate or remaining-error estimate.
The boundary space also changes from 1,584 to 1,848 unknowns between
the two vertex targets, even at the same Rayleigh cut. Consequently,
the observed change cannot be attributed only to corner-adjacent cells.

The .065 modal/volume discrepancies are `9.17e-6`/`5.35e-6`, larger
than `5.45e-6`/`2.55e-6` at .070. Its energy errors remain only
`2.43e-8`/`1.62e-8`, while signed remaining budgets are
`-3.77e-8`/`-1.99e-8`. The smaller geometric target therefore does
not monotonically improve the optical diagnostics. The matched
[degree comparison below](#separating-vertex-and-degree-refinement)
checks degree sensitivity on the target-.070 mesh.

The .070 and .065 solves take 231.82 and 260.27 s, with peak process
RSS 4.032 and 3.816 GB. The larger unknown count does not give a
larger observed process peak in these two runs; sparse factorization
structure and shared-host observations must be kept separate from
simple DOF counts. These timings compare different finite solutions,
not performance at an established equal accuracy.

## Retaining only boundary rows of the Fourier responses

The affine solver originally stored two complex arrays with shape
`(full_unknowns, 4*(2*cut+1)**2)`: the Fourier boundary traces and their
responses under the factorized Robin operator. A trace has support only
on the top and bottom boundary degrees of freedom. The updated solver
gets those rows from the final periodic, compressed space and stores
only them in both arrays.

Let U denote the full trace array, B the selected boundary rows, Q the
existing inverse action, D the port correction and b the two base
solutions. The smaller representation uses the same Woodbury equation:

```text
T = U[B, :]
V = (Q U)[B, :]
w = solve(I + D T* V, D T* b[B, :])
q[B, :] = T w; q[other rows, :] = 0
solution = b - Q q
```

Here `T*` denotes conjugate transpose. Applying Q to the final two
right-hand sides recovers all correction rows; the sparse factorization
is reused. Interior reconstruction follows that correction, as before.
For an uncondensed solve, this also recovers the interior correction
directly. Keeping only coupling rows of the solution would lose that
correction in the uncondensed case.

The same boundary selection evaluates physical ports and adds the
Rayleigh term to the full uncondensed residual. It does not reduce the
retained Fourier set, approximate the inverse or change the source.
The extra cost is two inverse applications, compared with 324 trace
right-hand sides at cut 4. Small independent complex block systems
check both condensation choices against a direct solve, including
interior reconstruction and the full residual.

The result records `boundary_dofs` and `boundary_array_bytes`. The
latter counts the two retained arrays, excluding sparse factorization,
dense port matrices, work vectors and temporary products. At edge
target .12, degree 5 and cut 4, the original 71,796-row arrays require
709.90 MiB together; the 1,584 boundary rows require 15.66 MiB. This
is an array-footprint reduction, not a prediction of total process RSS
or runtime, and it changes no optical convergence criterion.

A matched repeat at that setting changes individual R/T powers by at
most `3.34e-16` and total or volume R/T by at most `5.56e-16`.
The full residuals remain below `1.19e-14`. Peak process RSS changes
from 3.168 to 2.986 GB, a 5.76% reduction; measured runtime changes
from 177.08 to 169.19 s. The RSS reduction is much smaller than the
retained-array reduction because the sparse solve and other stages
also contribute to the process peak. These are two shared-host runs,
with small validation controls overlapping the repeat; they do not
establish a speedup across workloads or repeated timing uncertainty.

Separate patterned controls with nonzero azimuth compare the original
and revised solver under both condensation choices. Their largest
reported numerical differences are `1.79e-15` and `1.15e-15`. The
focused tests verify actual trace support, one factorization and the
two extra inverse applications, and driven full/condensed agreement.
Together with the existing scattering, Robin and square-fixture
controls, the selected 31 tests pass in 57.533 s.

## Recomputing the internal condensation blocks

The optional `recompute_internal=True` argument to `solve_affine_rayleigh`
retains the condensed Maxwell system while omitting the stored harmonic
extension and internal inverse matrices. It requires `condense=True` and
accepts a boolean. The default False keeps the existing storage path.
Both settings and stage metadata record the selected value.

Let I denote element-interior unknowns, E the retained coupling unknowns
and A the original local-Robin operator. For each of the two original
volume sources, the new path constructs:

```text
z_E = 0;  z_I = inverse(A_II) f_I
rhs_E = (f - A z)_E = f_E - A_EI inverse(A_II) f_I
```

The installed `ComputeInternal` operation evaluates z, and a full form
application evaluates A z. The temporary vectors are released after this
source condensation. The same single sparse factorization supplies the
two source solves, every boundary-trace response and the two final
Rayleigh corrections. A boundary trace has zero interior load, so those
inverse applications need no additional source elimination.

After the boundary correction, each field's internal values are recovered
from its corrected coupling values and its original volume source. This
preserves the full uncondensed residual check. The Robin boundary form
does not act on the zero tangential traces of the interior H(curl) bubbles;
that property is essential when reconstructing local interior blocks.

The installed NGSolve 6.2.2606 API behavior was checked against its
[versioned implementation](https://github.com/NGSolve/ngsolve/blob/v6.2.2606/comp/bilinearform.cpp).
This option calls the existing optional runtime; no external source is
copied into Wavebit. It adds four local reconstruction sweeps and two full
form applications for the two polarizations. The global inverse count
remains `rank+4`; including the existing final residual checks, there
are four full form applications in the new path. Avoiding stored blocks
does not imply a faster solve or guarantee that a larger case fits a
particular RSS limit.

The small patterned control compares the full, stored-condensed and
recomputed-condensed paths at h=.2, degree 3, cut 1, theta=.31 and
phi=.23. It retains the existing `2e-10` absolute criteria for total and
individual R/T, volume flux and signed Robin budgets, and `1e-11` for
full residuals. Instrumentation checks one factorization, `rank+4`
inverse applications, the four reconstruction calls and absent internal
matrix storage. A driven uniform slab uses the independent Airy formula
with its existing `5e-4` criterion. A mixed base-2/film-3 control checks
the same equivalence with variable entity degrees. These controls test
the alternative elimination path, not optical discretization accuracy.
All seven focused controls pass in 28.733 s with the original thresholds.
The [vertex-degree-7 retry](#resource-limited-global-degree-7) nevertheless
exceeds the same 8-GiB process RSS guard. The small equivalence controls
therefore establish the arithmetic option, not a solution to that resource
limit.

## Testing a second physical edge interval

Further edge-size controls keep h=.18, cut 4, pad=.2 and quadrature bonus
6 fixed. They compare degree 5 at targets .12 and .10, and degree 4 at
targets .10 and .08. Each target remeshes the same physical geometry;
the two intervals at different degrees are not one convergence sequence.

| edge target | degree | tetrahedra | full / condensed unknowns | p T | s T |
| ---: | ---: | ---: | ---: | ---: | ---: |
| .12 | 5 | 780 | 71,796 / 43,716 | .852038068565 | .730662922035 |
| .10 | 5 | 908 | 83,448 / 50,760 | .852099612881 | .730691646657 |
| .10 | 4 | 908 | 46,720 / 33,100 | .851926548859 | .730496704608 |
| .08 | 4 | 1,441 | 73,850 / 52,235 | .852018977392 | .730725046570 |

At degree 5, target .12-to-.10 changes p/s T by
`+6.15443e-5`/`+2.87246e-5`; the largest individual R/T changes are
`8.57057e-5`/`2.99748e-5`. The .12 row already has energy errors of
only `2.54e-8`/`1.28e-8`, yet its optical result still changes at these
larger scales. Its modal/volume disagreements are `9.99e-6`/`5.47e-6`.

At degree 4, target .10-to-.08 changes T by
`+9.24285e-5`/`+2.28342e-4`; the largest individual R/T changes are
`9.33087e-5`/`2.64493e-4`. The finer-edge row has energy errors of
`1.34e-6`/`1.87e-6`, slightly larger than at target .10, and
modal/volume disagreements of `2.22e-5`/`2.40e-5`. Thus a finer edge
target alone does not yield monotone improvement in these diagnostics.
These controls leave both degree and mesh uncertainty open and do not
establish a remaining-error estimate.

The new target-.12 degree-5 and target-.08 degree-4 solves take
177.08 and 167.12 s, with peak process RSS 3.168 and 3.416 GB.
They ran sequentially using the original full-row boundary arrays.
These shared-host observations compare different finite approximations,
not cost at equal optical accuracy.

## Raising the film degree on a fixed mesh

The optional `film_degree` and `film_degree_pad` arguments change the
H(curl) space while preserving the physical geometry and tetrahedral
mesh. The base `order` still applies outside the selected region.
An edge, face or cell receives the higher degree if its vertex-average
z lies in the closed band `[-film_degree_pad, .3+film_degree_pad]`,
with a floating-point tolerance at the endpoints. The pad is a physical
length in micrometers, independent of the outgoing-boundary `pad`.

The helper sets entity degrees in NGSolve's variable-order space before
applying lateral periodic identifications and compression. Lateral
translations preserve z, so paired edges and faces receive consistent
degrees. `film_degree=None` and an explicit degree equal to `order`
use the original global-order construction. A larger degree must be
an integer, and the band padding must be finite and nonnegative; a
positive band padding requires an explicit film degree.

The result's `space_refinement` records the base and maximum degrees,
requested band, directly selected entity counts, and full/condensed
unknown counts. A neighboring element can inherit a higher maximum
degree from a shared edge or face even if its own cell was not
selected. Thus the cell-selection count is not the number of all
elements having a high-order finite-element basis. Both the volume
flux and Robin-norm readouts use `2*max_order+quadrature_bonus`.

On the target-.12, h=.18, pad=.2 mesh, base degree 5 with film degree
6 and band padding .05 selects 765 edges, 1,129 faces and 587 cells.
The space has 104,326 full unknowns and 56,288 after condensation,
compared with 71,796/43,716 for global degree 5 and 116,690/62,090
for global degree 6 on that same mesh. The outer boundary still has
1,584 unknowns, so the retained boundary arrays remain 15.66 MiB
at Rayleigh cut 4. These sizes are resource observations, not evidence
that either approximation has resolved the optical error.

Seven space controls verify the unchanged default and equal-degree
spaces, equality with the higher global space when the band covers
the entire domain, actual entity dimensions, periodic identified DOFs,
finite-element degree metadata and invalid-input refusals. On a
transition tetrahedron and face, the default polynomial-form quadrature
agrees with bonus 6 to relative `5.77e-14` and `3.33e-14`.

An independent review on `J=[[.9,-.25],[.3,1.1]]` checks the degree-3
to degree-4 band, including boundary trace support and periodic DOFs.
Two physical integration controls check patterned full/condensed and
higher-quadrature agreement, and an independently evaluated Airy slab.
The slab retains the earlier `5e-4` absolute R/T criterion; the
patterned algebra/quadrature comparisons use `2e-10` and do not serve
as continuum-accuracy tolerances. All 40 affine geometry, algebra,
scattering, Robin, boundary-storage and local-degree tests pass in
42.154 s; this qualifies the finite instrument, not optical convergence.

The first patterned comparison fixes h=.18, edge target .12, cut 4,
outgoing pad .2 and quadrature bonus 6. Raising the film band by one
degree gives:

| space | p T | s T | p modal/volume disagreement | s modal/volume disagreement |
| --- | ---: | ---: | ---: | ---: |
| global degree 5 | .852038068565 | .730662922035 | 9.98735e-6 | 5.47086e-6 |
| degree 5, film band degree 6 | .852128153667 | .730772027375 | 6.03287e-6 | 3.61977e-6 |
| global degree 6 | .852147871193 | .730777052860 | 5.32953e-6 | 2.62282e-6 |

The p/s changes to the film-band space are `+9.00851e-5`/`+1.09105e-4`, with maximum
individual R/T changes `9.00954e-5`/`1.09111e-4`. The refined row has
energy errors of `1.51e-8`/`7.58e-9`, omitted Robin powers of
`1.48e-8`/`4.25e-9`, and signed remaining budgets of
`-3.00e-8`/`-1.18e-8`. Those smaller budgets do not bound the much
larger observed degree sensitivity. The enriched solve takes 311.79 s
with peak process RSS 4.277 GB. It is a different finite approximation,
not an equal-accuracy performance comparison with global degree 5.
There is some cancellation between transmitting orders. In particular,
the s `(-1,0)` difference from the finite Wavebit row grows from
`-1.17534e-5` to `-2.51812e-5` even though total T moves closer.
This comparison does not show improvement in every diffraction order.

Completing the same-mesh increase to global degree 6 changes total T
by another `+1.97175e-5`/`+5.02549e-6`. That increase also raises the
outer-boundary space from 1,584 to 2,184 unknowns, so it is not solely
an exterior-cell interior-basis change. Both degree-6 readouts use
quadrature order 18. The transmitting-order changes expose cancellation:

| T order | global 5 to 6, p | global 5 to 6, s | film-band 6 to global 6, p | film-band 6 to global 6, s |
| --- | ---: | ---: | ---: | ---: |
| (-1,-1) | +4.49134e-6 | +1.79042e-5 | -1.31212e-6 | +4.63624e-7 |
| (-1,0) | +1.44154e-4 | +9.87997e-6 | +5.74162e-5 | +2.33078e-5 |
| (0,0) | -3.88424e-5 | +8.63466e-5 | -3.63865e-5 | -1.87459e-5 |
| total | +1.09803e-4 | +1.14131e-4 | +1.97175e-5 | +5.02549e-6 |

The largest individual local-to-global R/T changes are
`5.74162e-5`/`2.33078e-5`, substantially larger than the corresponding
total changes. The global-degree-6 signed energy remainders are
`-1.77e-9`/`-6.10e-10`, yet modal/volume disagreements remain
`5.33e-6`/`2.62e-6`. No optical error bound or convergence rate follows.
This solve takes 340.82 s with peak process RSS 4.584 GB; its two
retained boundary arrays occupy 21.59 MiB. The cost comparison covers
different finite spaces on a shared host.

## Separating vertex and degree refinement

Completing global degree 6 on the target-.070 mesh gives a two-by-two
comparison of vertex sizing and degree. All four rows retain h=.18,
edge target .12, pad .2, Rayleigh cut 4 and quadrature bonus 6:

| vertex target | global FEM degree | p T | s T |
| --- | ---: | ---: | ---: |
| none | 5 | .852038068565 | .730662922035 |
| none | 6 | .852147871193 | .730777052860 |
| .070 | 5 | .852108495556 | .730802741386 |
| .070 | 6 | .852198590802 | .730875018549 |

Degree is assigned after mesh generation. Within each degree pair,
the recorded geometry, element counts and periodic-pair summaries match;
the historical output does not include exact mesh-topology fingerprints.
The vertex change regenerates the mesh and can alter elements away
from the selected corners. It is not a nested spatial refinement.

| change | p total T | s total T | largest individual p R/T change | largest individual s R/T change |
| --- | ---: | ---: | ---: | ---: |
| vertex target none to .070, degree 5 | +7.04270e-5 | +1.39819e-4 | 8.80527e-5 | 2.09671e-4 |
| vertex target none to .070, degree 6 | +5.07196e-5 | +9.79657e-5 | 5.90348e-5 | 1.53018e-4 |
| degree 5 to 6, without vertex target | +1.09803e-4 | +1.14131e-4 | 1.44154e-4 | 1.14144e-4 |
| degree 5 to 6, vertex target .070 | +9.00952e-5 | +7.22772e-5 | 1.15136e-4 | 7.22608e-5 |

The mixed difference, defined as the degree increment with target .070
minus the degree increment without it, is `-1.97074e-5`/`-4.18537e-5`
for p/s total T. The observed degree variation is smaller on the
vertex mesh, but both refinement axes still change individual powers
by about `1e-4`. Neither this mixed difference nor either interval
estimates the remaining error. For example, the degree-6 spatial change
in s T combines `+2.87469e-5`, `-8.37994e-5` and `+1.53018e-4` in
the three transmitting orders. The total hides that cancellation.

The new row's modal/volume disagreements are `2.81115e-6`/`1.08678e-6`,
while its energy errors are `1.36e-10`/`1.46e-9`. Omitted Robin powers
are `1.83008e-9`/`1.73054e-9` and signed remaining budgets are
`-1.69394e-9`/`-2.67070e-10`. The full linear residuals stay below
`1.05e-14`; none of these small algebraic diagnostics resolves the
larger optical changes. Against the finite degree-10 auxiliary-field
Wavebit row at `(24,32)`, total T still differs by
`-1.82120e-4`/`-1.38200e-4`. Both methods retain unresolved
discretization uncertainty, so that comparison supplies no oracle.

The new 1,002-tetrahedron space has 149,604 full and 79,464 condensed
unknowns, with 2,184 boundary unknowns. Its two retained response arrays
occupy 21.59 MiB. The solve takes 488.62 s with peak process RSS
5.795 GB; the observed peak after factorization is 4.526 GB, rising
during the boundary responses. These are shared-host measurements of
different finite approximations, not equal-accuracy speed comparisons.
All four completed results pass independent metadata, power-sum,
signed-budget and per-order difference checks; no convergence
acceptance threshold is inferred from the rectangle.

## Checking the nearer vertex interval at both degrees

The target-.065 global-degree-6 solve completes a second rectangle,
using vertex targets .070/.065 and global degrees 5/6. It retains
h=.18, edge target .12, pad .2, Rayleigh cut 4 and quadrature bonus 6.
The new space has 1,093 tetrahedra, 163,268 full and 86,758 condensed
unknowns, with 2,548 boundary unknowns and 26/67 periodic point pairs.
Its geometry summaries match the existing .065 degree-5 row; historical
logs still contain no exact topology fingerprint.

| vertex target | global FEM degree | p T | s T |
| ---: | ---: | ---: | ---: |
| .070 | 5 | .852108495556 | .730802741386 |
| .070 | 6 | .852198590802 | .730875018549 |
| .065 | 5 | .852094870566 | .730861555984 |
| .065 | 6 | .852189878745 | .730915491210 |

The .070-to-.065 spatial increments keep their signs at both degrees:
at degree 5, p/s T changes by `-1.36250e-5`/`+5.88146e-5`; at degree
6, by `-8.71206e-6`/`+4.04727e-5`. The smaller total changes still
include larger, cancelling changes in individual transmitting orders:

| T order | spatial change at degree 6, p | spatial change at degree 6, s |
| --- | ---: | ---: |
| (-1,-1) | +1.90569e-6 | +5.99615e-6 |
| (-1,0) | -3.66723e-5 | -1.52531e-5 |
| (0,0) | +2.60545e-5 | +4.97297e-5 |
| total | -8.71206e-6 | +4.04727e-5 |

At target .065, the degree-5-to-6 p/s T increments are
`+9.50082e-5`/`+5.39352e-5`, with maximum individual R/T changes
`1.31281e-4`/`5.39520e-5`. Relative to target .070, the p degree
increment grows while the s increment shrinks. The mixed difference,
defined as the .065 degree increment minus the .070 degree increment,
is `+4.91293e-6`/`-1.83419e-5`. The two axes remain coupled; neither
the smaller spatial difference nor the mixed difference supplies a
remaining-error estimate or a monotone accuracy ranking.

The new modal/volume disagreements are `4.95502e-6`/`2.64781e-6`,
larger than `2.81115e-6`/`1.08678e-6` on the .070 degree-6 mesh.
Energy errors are `3.00e-9`/`6.04e-10`; omitted Robin powers are
`4.89342e-9`/`1.19511e-9` and signed remaining budgets are
`-1.89230e-9`/`-5.90956e-10`. The full residuals stay below
`1.02e-14`. Independent checks of the full power arrays, physical
port mapping and signed identities pass, but these small algebraic
quantities do not qualify an optical reference.

Against the finite degree-10 auxiliary-field Wavebit row at `(24,32)`,
the new total-T differences are `-1.90832e-4`/`-9.77274e-5`. The p
difference grows in magnitude while the s total moves closer. However,
the s `(-1,0)` difference grows to `-1.00926e-4` and the s `(0,0)`
difference crosses to `+1.28527e-5`. Neither the spatial refinement
nor closer s total establishes improvement in every channel.

The solve takes 593.84 s with peak process RSS 7.034 GB, compared
with 488.62 s and 5.795 GB for .070 at degree 6. The retained boundary
arrays account for 25.19 MiB; the sparse factorization and other
workspaces dominate the process peak. These sequential shared-host
measurements describe different finite approximations, not equal-accuracy
performance. The higher cost does not resolve the remaining degree
sensitivity.

## Resource-limited global degree 7

Raising the target-.070 vertex mesh to global degree 7 retains 1,002
tetrahedra and periodic pair counts 25/62, but increases the full/coupling
space to 227,280/107,040 unknowns. The 2,880 outer-boundary unknowns retain
29,859,840 bytes of trace/response arrays. These counts pass the
pre-assembly space limits; they do not bound factorization memory.

With the optional boundary progress callback, factorization finishes at
335.696 s with reported process peak RSS 4,759,371,776 bytes. After the
first 32 of 324 trace inverse applications, at 521.971 s, peak RSS reaches
9,447,981,056 bytes and the caller's unchanged 8-GiB RSS guard raises.
The next trace and the optical readout are not evaluated. A separate
process-specific `vmmap` observation during boundary responses reports
physical footprint 10.8G, peak footprint 17.1G and 8.8G swapped out; these
retain the tool's units and displayed precision. Footprint, RSS and the
small retained boundary arrays have different accounting scopes.

This attempt returns no optical result and cannot extend the degree
ladder or define a stopping criterion. It demonstrates why checking RSS
only after factorization is insufficient. Even the new callback can
detect a limit only between completed native calls.

A second attempt uses `recompute_internal=True` from commit `ee47fb4` (landed as `ecda894`),
with the same physical problem, mesh settings, global degree, quadrature,
full/coupling limits and 8-GiB RSS ceiling. It confirms the same mesh and
space counts. The reporting interval is reduced from 32 to 8 completed
trace responses. Factorization finishes at 310.463 s with reported peak
RSS 4,895,817,728 bytes. At the first report, after 8 of 324 responses
and 569.202 s, peak RSS is 9,416,654,848 bytes and the guard raises again.
This attempt also exits without a boundary completion or optical result.

Omitting internal blocks has not made this case fit the limit. These
two partial executions stop at different trace counts, so their peak RSS
and elapsed times do not establish a memory saving or speed comparison.
The interval between factorization and the first trace report includes
source condensation, source inverse applications, trace assembly and
trace inverse applications. The available observations do not identify
which operation produced the peak. In particular, the 29,859,840 retained
boundary-array bytes cannot account for the full process RSS. Both raw
failures remain recorded; neither attempt supplies an optical value or
stopping criterion. The later completed
[reference-release control](#degree-7-reference-release-control) is a
separate execution.

A mesh-only check identifies a smaller global-degree-7 interval on the
existing edge-.12 mesh without the extra vertex restriction: 780
tetrahedra, periodic pairs 22/36, full/coupling counts 177,232/83,632 and
the same 2,880 boundary unknowns. Its degree-5 and degree-6 baselines
already exist. This changes the mesh branch of the study; it is not a
replacement for a value on the distinct vertex-.070 mesh branch. A separate
base-6/film-7 space on this smaller mesh has 149,287/74,187 unknowns
without an added film neighborhood, or 160,774/76,824 with a .05
neighborhood. Preflight counts alone are not solved fields or memory
guarantees. The zero-neighborhood local film-7 space remains unsolved;
the .05-neighborhood result is recorded below. None relaxes the existing
linear, power or optical controls.

## Locating memory growth before the first boundary trace

Separate diagnostic processes revisit the completed vertex-.070 degree-6
mesh, with 149,604 full, 79,464 coupling and 2,184 outer-boundary unknowns.
They keep h=.18, edge=.12, cut 4, pad=.2, quadrature bonus 6 and the
8-GiB peak-RSS guard. Both stop intentionally after the first completed
trace response. They measure operations, without repeating the complete
optical calculation or supplying new optical results.

The diagnostic samples process peak RSS and current `ps` RSS before and
after assembly, factorization, internal reconstruction, full form
application and each inverse. Because matrix multiplication is lazy,
the inverse wrapper forces evaluation into an additional full complex
vector before the after-sample. That adds 2,393,664 bytes and a subsequent
copy, while process observations add timing overhead. A smaller wrapper
control first verifies the operation counts and intentional stop.

| Observation | Stored blocks | Recomputed blocks |
| --- | ---: | ---: |
| Current RSS after assembly, bytes | 1,275,527,168 | 793,772,032 |
| Current RSS after factorization, bytes | 1,659,879,424 | 2,033,303,552 |
| Current RSS before first inverse, bytes | 1,875,410,944 | 2,049,245,184 |
| Current RSS after first inverse, bytes | 6,152,339,456 | 5,961,564,160 |
| Peak RSS at diagnostic completion, bytes | 7,166,689,280 | 5,983,043,584 |
| Diagnostic duration, s | 128.575 | 191.093 |

Both large observed resident-memory increases occur during the first
global inverse application. In the recomputed run, the first internal
reconstruction increases current RSS by 9,568,256 bytes and the first
full form application by 802,816 bytes. The following inverse increases
it by 3,912,318,976 bytes. In the stored run, that inverse increases
current RSS by 4,276,928,512 bytes without increasing the process peak,
because factorization has already reached a higher peak.

This localizes the resident-memory change on degree 6; it does not
identify fresh allocations, factor-page residency or an operating-system
cause. The recomputed diagnostic has a smaller observed peak but also
adds 65.190 s of internal reconstruction and 2.170 s of full form
applications. These partial shared-host observations are not a full-run
performance comparison or a resource guarantee for degree 7. Reducing
the retained boundary arrays alone does not address the operation where
this diagnostic observes the large increase.

The same-version [NGSolve UMFPACK wrapper](https://github.com/NGSolve/ngsolve/blob/v6.2.2606/linalg/umfpackinverse.cpp)
constructs the symbolic and numeric factor during factorization. Its
complex inverse application uses that existing factor and two compressed
complex vectors, totaling 2,542,848 bytes for this coupling dimension.
The visible wrapper does not defer a new factorization to the first
application. This narrows the source inspection, while internal
SuiteSparse workspace and operating-system page residency remain
unmeasured by these observations. No external implementation is copied
into the reference instrument.

The optional `source_progress=True` argument now reports immediately
after each p/s source inverse has completed and its response has been
stored. Each `source_progress` stage records `source_columns_completed`,
`source_columns_total=2` and `polarization`. A caller's resource guard can
therefore stop after the first source, before starting the second source
or assembling any boundary trace. The default False preserves the
existing stage sequence and metadata. This earlier observation cannot
interrupt a native call or lower its peak memory use.

Native forced-evaluation controls stop after source 1 and source 2 under both
storage choices. They confirm one factorization, exactly the requested
number of completed inverses, two assembled volume loads and no assembled
boundary trace. Invalid flags are refused before meshing; the existing
default-stage controls remain unchanged. The 12 focused storage and
affine-scattering tests pass in 31.964 s.

### Allocation and residency observations at the first source

A further degree-6 diagnostic uses that first-source stop and adds three
`vmmap` summaries of its own process. It keeps the same mesh, recomputed
internal blocks, numerical source and 8-GiB peak-RSS ceiling. A small
wrapper control verifies the observations before the degree-6 run. Both
finish intentionally after one factorization, one completed source
inverse, one `ComputeInternal` call and one full form application;
only the two volume loads are assembled, with no boundary traces or
optical result.

| Observation | After factorization | Before first inverse | After first inverse |
| --- | ---: | ---: | ---: |
| Current `ps` RSS, bytes | 1,304,248,320 | 1,323,319,296 | 5,827,772,416 |
| Physical footprint, tool units | 2.5G | 2.5G | 5.5G |
| Peak physical footprint, tool units | 9.3G | 9.3G | 9.3G |
| Malloc-zone total allocated, tool units | 5.7G | 5.7G | 5.7G |
| Malloc-zone total resident, tool units | 801.2M | 816.1M | 5.0G |
| Malloc-zone total swapped, tool units | 1.7G | 1.7G | 545.3M |

The `ps` RSS samples immediately precede the corresponding `vmmap`
captures; these metrics are not sampled simultaneously.

The first inverse increases current RSS by 4,504,453,120 bytes. The
malloc-zone allocated total remains at the same rounded display value,
while its resident total increases and its swapped total decreases.
This is consistent with a substantial change in residency of existing
allocations. It does not prove exact constancy of allocation, identify
individual factor pages, or exclude temporary native allocations between
the snapshots. The independent process-region table reports 563.4M
swapped after the inverse; the writable-region summary reports 556.5M.
These different totals are retained separately, not substituted for the
malloc-zone total.

The diagnostic completes in 159.906 s at peak RSS 5,829,935,104 bytes.
The already higher 9.3G peak *physical footprint* is a different measure
from this RSS ceiling. Each `vmmap` capture takes time and can affect
paging; the before-to-after inverse interval includes the before-snapshot
capture and is not a native-solve benchmark. The extra inverse output
vector still occupies 2,393,664 bytes. These observations support testing
ways to reduce the retained sparse factor and its working set before
retrying degree 7; they do not themselves reduce memory or establish a
degree-7 resource guarantee.

### Small affine comparisons with explicit SuperLU orderings

An independently written adapter restricts the original native matrix to
its exact free-DOF subset, converts that matrix to canonical complex CSC
and calls the existing SciPy `splu` with explicit column ordering and
diagonal pivot threshold 1. No transpose or conjugate transpose is applied.
Each result is embedded into a full vector with zero nonselected entries.
The source elimination, Rayleigh correction, field reconstruction and
original Maxwell residual remain unchanged.

Two known-solution complex matrix controls check noncontiguous free-DOF
selection, complex orientation, zero embedding and an unchanged input
right-hand side. Their largest solution error is below `6.3e-16`. The
five-by-five native fixture has a numerically zero last column, with
structural zeros retained. It does not test a structurally absent last
column. Two earlier fixture setup errors occurred before any factorization
and remain separate from numerical results.

The patterned affine control uses h=.2, degree 3, cut 1, pad=.1,
theta=.31 and phi=.23, with 11,256 full and 9,512 coupling unknowns.
Before factorization, two deterministic complex vectors check the native
operator against the full CSR representation with relative criterion
`1e-12`. SuperLU additionally checks the original selected sparse residual
after every solve at `1e-11`; both final full Maxwell residuals use the
same `1e-11` small-control criterion. Every complete run uses one factor,
40 inverse applications and 38 assembled loads. Full, stored-condensed
and recomputed-condensed COLAMD paths are compared with the native full
solution under the existing `2e-10` observable criterion, including p/s
reflection, transmission, per-order powers, volume fluxes and Robin
complement budgets.
Across the completed comparisons, the largest absolute observable
difference from the native full solution is `4.003e-13`. The largest
full Maxwell residual is `1.159e-12`, and the largest measured selected
SuperLU residual is `1.926e-12`, all below the unchanged criteria.

| Backend and storage | Factor time, s | Diagnostic time, s | Peak RSS, bytes | Reported factor nonzeros |
| --- | ---: | ---: | ---: | ---: |
| Native UMFPACK, full | .430 | 2.520 | 660,717,568 | Not measured |
| Native UMFPACK, stored | .418 | 2.292 | 578,551,808 | Not measured |
| SuperLU COLAMD, full | 7.227 | 8.576 | 830,078,976 | 31,578,377 |
| SuperLU COLAMD, stored | 5.222 | 6.383 | 680,542,208 | 24,161,993 |
| SuperLU COLAMD, recomputed | 5.268 | 7.195 | 664,027,136 | 24,161,993 |
| SuperLU MMD_AT_PLUS_A, stored | 71.570 | 73.997 | 1,883,095,040 | 65,213,972 |

The factor nonzero count uses the scalar `lu.nnz` property. The
[same-version SciPy wrapper](https://github.com/scipy/scipy/blob/v1.18.0/scipy/sparse/linalg/_dsolve/_superluobject.c)
shows that accessing
`lu.L` or `lu.U` would create cached CSC factor copies in this SciPy
version and alter the memory measurement. Native factor nonzeros and
native selected sparse residuals are not measured; missing values are
not zeros. The diagnostic retains an additional selected CSC for
SuperLU's per-solve residual checks and adds native-matrix validation,
copies and observations, so these are instrumented shared-host timings,
not production performance benchmarks.

These unscaled default-pivot alternatives provide no memory improvement on this
control. In particular, the matched stored-mode comparison is larger and
slower for both tested SuperLU orderings. This supports keeping the native
factorization as the reference default and does not justify a high-degree
retry with these settings. All six processes complete within their
180-second parent limit and unchanged 8-GiB peak-RSS guard. The results
qualify the small algebraic comparison, not an optical convergence bound.

### Explicit row-maximum scaling

A separate control changes only the scaling of the selected sparse system.
For its original matrix A, define r_i=max_j |A_ij| and the positive real
diagonal D_i=1/sqrt(r_i). Factor D A D, solve against D b and recover
x=D y. This is an equivalent change of unknowns for a complex,
non-Hermitian matrix; it assumes neither symmetry nor a nonzero diagonal
of A. The same explicit ordering and diagonal pivot threshold 1 remain.
The original selected matrix is retained for every residual check.

The implementation refuses zero rows, nonfinite input or scaling,
malformed CSC indices and nonzero coefficients erased by underflow.
Three independent helper controls include an invertible zero-diagonal
matrix, complex non-Hermitian entries and unequal row magnitudes;
19 invalid controls are rejected. Two complete native-matrix synthetic
controls retain the noncontiguous mask, known solution, unchanged RHS
and zero embedding. Their maximum solution error is below `4.5e-16`.
The four degree-3 FEM controls use the same fixture and unchanged
equation and observable criteria as above.

| Scaled backend and storage | Factor time, s | Diagnostic time, s | Peak RSS, bytes | Reported factor nonzeros |
| --- | ---: | ---: | ---: | ---: |
| SuperLU COLAMD, full | 4.387 | 5.798 | 646,070,272 | 20,587,541 |
| SuperLU COLAMD, stored | 2.651 | 3.747 | 506,150,912 | 14,894,345 |
| SuperLU COLAMD, recomputed | 2.752 | 4.727 | 510,787,584 | 14,894,345 |
| SuperLU MMD_AT_PLUS_A, stored | 9.953 | 11.272 | 638,877,696 | 22,163,980 |

The 11 comparisons include the saved native full result and matching
stored/unscaled controls; no baseline solve is repeated. All matching
original selected CSC and mask hashes are identical. The largest
absolute observable difference from native full is `2.499e-14`; across
all comparisons it is `3.943e-13`. The largest final full Maxwell
residual is `7.027e-14`, and the largest selected sparse residual is
`8.308e-14`. Independent record validation rejects 28 malformed controls.

Scaled COLAMD stored uses about 12.5% less observed peak RSS than native
stored on this small fixture, while its diagnostic takes 3.747 s versus
2.292 s. The scaled CSC copy and diagonal construction are included in
the process peak; the original CSC and per-solve residual overhead also
remain. This is a measured memory/time tradeoff, not a production speed
claim. The predefined small resource gate permits a bounded degree-6
first-source diagnostic. It does not establish that degree 7 fits or
that a complete optical solve converges. Native UMFPACK remains the
reference default.

### Bounded degree-6 scaled-factor attempt

The next diagnostic preserves the qualified scaling, sparse conversion and
inverse-application definitions, and deliberately stops after the first
completed p source. A degree-3 wrapper control confirms the exact original
CSC, free mask, D and D A D hashes against the completed recomputed control.
It performs one factorization, one inverse, one form assembly, two volume
load assemblies, one `ComputeInternal` and one `Apply` call, with no
boundary trace. Its original selected residual is `3.974e-14`. This stop
does not produce a final Maxwell residual or optical powers.

The larger attempt uses the existing vertex-.07 degree-6 setup: h=.18,
edge=.12, pad=.2, cut 4 and bonus 6, with 1,002 tetrahedra and
149,604 full / 79,464 coupling / 2,184 boundary unknowns. The selected
CSC has 27,036,828 nonzeros and occupies 541,054,420 bytes. Native/CSR
operator checks pass before the reported factorization start at 37.045 s.
The recorded peak RSS at that point is 1,782,251,520 bytes.

The parent terminates and waits for the child at its predefined 600-second
limit; recorded parent elapsed time is 600.282 s. No completed factor,
first inverse or source completion is reported. The last recorded RSS is
not the unobserved process peak during the interval after the last record.
Parent and child clocks start separately; subtracting their times does not
give a measured factorization duration.
There is no completed factor fill count, selected source residual for
degree 6 or optical result. The source/probe hashes and closed parent
record are preserved.

Consequently, the small-case memory reduction has not established a usable
degree-6 resource improvement. It does not justify retrying the same
settings or increasing to degree 7. A further attempt needs a different
validated factorization choice or reduction of the retained system.

### Nonzero pivot threshold under the same scaling

Another bounded comparison changes the diagonal pivot threshold from 1
to .1 while retaining D A D scaling and the same original equations.
This changes which diagonal pivots the factorization accepts; the selected
residual, full Maxwell residual and observable criteria remain unchanged.
It does not disable pivoting or replace LU by an approximate factorization.

The initial six controls comprise two known-solution synthetic controls
and four degree-3 solves: COLAMD full/stored/recomputed and MMD stored.
After MMD stored has the lowest observed peak RSS, two separately declared
followups qualify its full and recomputed paths before any larger use.
The following six rows are complete FEM solves, each with one factor,
40 inverse applications and 38 assembled loads.

| Scaled threshold-.1 backend and storage | Factor time, s | Diagnostic time, s | Peak RSS, bytes | Reported factor nonzeros |
| --- | ---: | ---: | ---: | ---: |
| SuperLU COLAMD, full | 2.318 | 3.396 | 523,223,040 | 14,819,080 |
| SuperLU COLAMD, stored | 1.878 | 2.887 | 462,651,392 | 12,353,272 |
| SuperLU COLAMD, recomputed | 1.902 | 3.686 | 448,790,528 | 12,353,272 |
| SuperLU MMD_AT_PLUS_A, full | 2.791 | 3.765 | 444,465,152 | 10,687,044 |
| SuperLU MMD_AT_PLUS_A, stored | 1.878 | 2.789 | 384,696,320 | 8,932,284 |
| SuperLU MMD_AT_PLUS_A, recomputed | 1.932 | 3.638 | 379,633,664 | 8,932,284 |

Independent validation accepts all eight controls and 16 comparisons,
with a largest absolute observable difference of `3.908e-14`. The largest
full Maxwell and selected sparse residuals are `2.051e-13` and
`3.771e-13`, respectively. All matched original CSC, masks, D and D A D
hashes agree. Twenty-seven malformed audit controls are rejected.
The saved MMD threshold-1 baseline covers stored mode only. The full and
recomputed MMD comparisons therefore use COLAMD threshold 1 for those
storage modes and change both ordering and threshold; those comparisons
do not isolate the effect of pivot threshold alone.

The MMD stored observation reduces peak RSS by about 33.5% from native
stored, while taking 2.789 s versus 2.292 s. Both stored orderings pass
the predefined smaller-peak, fewer-factor-nonzeros and shorter-factor-time
criteria relative to the previous scaled COLAMD control. These small
observations justify a separately bounded MMD threshold-.1 first-source
test at degree 6. They do not establish that it finishes within the large
resource limit, nor qualify degree 7 or an optical convergence bound.

### Bounded degree-6 first source with MMD and threshold .1

The separately declared MMD_AT_PLUS_A / threshold-.1 / symmetric-rowmax
experiment uses recomputed local elimination and stops after the first
actual p-source solve and base-column copy. A degree-3 first-source control
first reproduces the completed small recomputed solve's original CSC, free mask,
D and D A D fingerprints, with selected residual `1.193e-13`. Thirty
malformed first-source audit controls are rejected. The controls retain the
600 s parent kill-and-wait limit, 8 GiB observed peak-RSS ceiling and unchanged
`1e-11` selected-system residual criterion.

The vertex-.07 degree-6 diagnostic then completes on the same 1,002-element
mesh, with 149,604 full and 79,464 coupling unknowns. Its original selected
CSC and free mask exactly match the earlier timed-out scaled COLAMD run.
Independent validation confirms one factorization, one inverse application,
one assembled bilinear form, two assembled source loads, one internal
recovery and one full-form application. No boundary trace is assembled.

| Completed first-source observation | Value |
| --- | ---: |
| Factorization time | 429.016 s |
| Reported factor nonzeros | 340,024,111 |
| Child diagnostic time | 502.961 s |
| Parent elapsed time | 503.829 s |
| Process peak RSS at diagnostic completion | 6,777,683,968 bytes |
| Original selected-system relative residual | 3.590e-13 |

The factorized-stage peak is 2,929,623,040 bytes and rises to
6,767,001,600 bytes after the first inverse. A factorization-stage snapshot
therefore does not represent the completed source's memory requirement.
These observations do not identify a particular allocation or prove a
cause for resident-page growth.

The prior native recomputed first-source diagnostic recorded
5,829,935,104 bytes and 159.906 s. Its physical geometry, source and space
metadata agree, but it did not record original CSC fingerprints and the
two diagnostics have different instrumentation. The new observation
provides no evidence of a large-problem memory or time improvement; the
small degree-3 reduction does not transfer to this experiment. Native
UMFPACK remains the reference default. This completed first source has no
full Maxwell residual, boundary response, optical result or degree-7
qualification. Another resource trial needs a distinct validated change.

### One incomplete-factor Krylov candidate

A separate fixed candidate targets factor storage with incomplete LU of
D A D. The factors serve only as a left preconditioner
`M(v) = D P.solve(D v)` for restarted GMRES on the **original** selected
system A x = b. The original selected and full Maxwell residual guards
remain `1e-11`, with `2e-10` required for all matched observables. A
tolerance-stopped Krylov action is not exactly a fixed linear map, so the
Woodbury correction also requires those full residual and observable checks.

The declared candidate uses drop tolerance `1e-4`, fill factor 2,
`basic,area` dropping, MMD_AT_PLUS_A, diagonal pivot threshold .1 and
`RowPerm=NOROWPERM`. The last option avoids requesting the excluded optional
matching implementation. GMRES uses relative tolerance `1e-12`, zero
absolute tolerance and initial guess, restart length 60 and at most ten
cycles. Nonzero exit information always fails; there is no direct-solve
fallback or tolerance adjustment. These are candidate settings, not a
convergence guarantee. Only the already licensed installed SciPy/SuperLU
internal-validation scope is used.

Two known-solution complex synthetic controls, including zero diagonals,
noncontiguous free indices and prescribed positive diagonal rescaling,
pass with largest solution error `4.837e-13`. The same request validator
returns exact zero for zero loads and refuses 16 injected failures,
including a misleading zero preconditioned-residual report.

The single planned degree-3 stored control then **fails its first GMRES
request** after 600 inner iterations and 611 preconditioner actions. Its
original CSC, free mask, D and D A D match the saved same-storage direct
control. The incomplete factor reports 1,997,505 nonzeros and takes
0.311 s to construct. No inverse request completes: the counts are one
factorization, one assembled bilinear form and two source loads, with no
boundary response or internal recovery. The failed return's exact exit
value and original residual were not recorded because the exit-information
guard runs before those observations.

The child fails after 2.469 s with observed peak RSS 283,279,360 bytes;
the closed parent records exit 1 after 2.535 s, without a timeout. This is
an aborted first-source attempt, not a complete low-memory solve. It has
no full Maxwell residual or optical result and supplies no basis for a
degree-6/7 run. The fixed weak preconditioner is not adopted; changing it
would require another declared construction and independent qualification.

### A separately fixed fill-factor-4 control

The next candidate changes only the incomplete-factor fill factor from 2
to 4. All equation, scaling, ordering, dropping, pivoting, GMRES and
acceptance settings above remain fixed. Failed-return telemetry now records
the returned exit information and, when finite, the original-system residual
before applying the unchanged rejection guard. This adds a diagnostic sparse
matrix-vector product; it cannot turn a failed request into an accepted one.
The declared scope is the synthetic controls followed by one stored degree-3
control, with no adaptive retuning or higher-degree solve.

Both known-solution synthetic cases and all 16 injected refusals pass again;
the largest solution error is `4.837e-13`. An injected nonzero exit code
remains rejected even when its returned original relative residual is
`1.58e-16`. Nonfinite answers or exit information also remain failures.

The actual degree-3 first request returns integer `info=10` after 545 inner
iterations and 556 preconditioner actions. Its finite original residual
norm is 252.698 against a right-hand-side norm of 17.4831, giving relative
residual **14.4539**. It fails without completing an inverse request. The
original CSC, free mask, D and D A D exactly match the saved direct stored
control. The incomplete factor contains 3,979,819 nonzeros and takes 0.726 s
to construct. The child closes after 3.425 s with observed peak RSS
312,164,352 bytes; the parent closes with exit 1 after 3.533 s, without a
timeout.

The factorization and two source-load assemblies complete, but internal
recovery, boundary responses and final Maxwell checks do not. This candidate
also supplies no optical result or complete-run memory benefit. The first
candidate did not record its failed original residual, so their convergence
quality cannot be compared from the iteration counts. Neither candidate
qualifies a degree-6/7 run or changes the native direct-solver default.

### Exact additional condensation of independent face blocks

A separate algebraic candidate eliminates independent face-interior blocks
after the existing element-interior condensation. Actual compressed periodic
face DOFs define the groups; top/bottom boundary and edge DOFs remain retained.
A fixed greedy pass checks both directions of the original sparse adjacency,
ignoring only exact zeros. Identical periodic aliases are deduplicated;
overlapping or partially excluded groups are refused.

With eliminated block diagonal H, the retained matrix is
`S = A_RR - A_RB H^-1 A_BR`. Every request forms
`g = b_R - A_RB H^-1 b_B`, solves `S x_R = g`, and reconstructs
`x_B = H^-1 (b_B - A_BR x_R)`, including when `b_B=0`.
These are ordinary complex solves, with no adjoint substitution. A singular
local block is refused even when the original full matrix is invertible.
The retained factor keeps the previously qualified row scaling, MMD ordering
and .1 pivot threshold. Shared face DOFs are not reclassified as element-local.

Two known-solution complex synthetic cases pass, with largest solution error
`2.67e-14` and complex-linearity error `5.94e-14`. Twenty refusal controls
include singular local and retained factors, ambiguous groups and finite
inaccurate solve outputs rejected by their original residual checks.

The single stored degree-3 FEM control also completes. Its original selected
CSC and free mask exactly match the stored controls. All 40 reconstructed
inverse actions pass, with maximum selected residual `4.48e-14` and p/s full
Maxwell residuals `3.85e-14`/`2.89e-14`. All compared outputs, including every
diffraction order and signed Robin budget, agree with the three saved direct
controls within `1.67e-13`.

However, this construction **does not reduce observed memory**. The 197 local
blocks remove 1,576 of 9,512 selected unknowns, leaving 7,936, but the Schur
matrix grows from 1,039,840 to 1,255,680 nonzeros. Its retained LU has
9,114,416 nonzeros versus 8,932,284 for the original scaled MMD control,
in addition to the local factors. Peak process RSS is 758,153,216 bytes,
compared with 578,551,808 for native stored and 384,696,320 for scaled MMD
stored. The child takes 4.887 s, including 1.828 s for the retained factor;
these are shared-host observations, not equal-accuracy performance bounds.

The preallocation estimate is 379,691,640 bytes against the declared 512-MiB
adapter-array limit. It includes streamed nonsymmetric Schur updates and
retained local factors, but excludes opaque sparse-factor workspaces, native
condensation storage and runtime allocators. The separate 8-GiB observed
process guard includes those costs. A smaller retained dimension is therefore
insufficient evidence for a higher-degree run. The default solver and optical
convergence status remain unchanged.

### Complete lower-precision LU with double-precision correction

A further candidate changes the precision of a complete sparse factor, while
retaining the original complex128 matrix, right-hand sides, Krylov vectors,
solutions and Maxwell checks. It forms the same verified `B128 = D A128 D`,
then factors its complex64 conversion with the existing MMD ordering and .1
pivot threshold. The original and both scaled matrices remain available in
this small control. Nonfinite conversion or loss of a nonzero real or
imaginary component to zero is refused; this does not rule out underflow
inside the sparse factorization.

Each approximate right action is
`M(v) = D cast128(LU64.solve(cast64(D v)))`. Fixed `gcrotmk` parameters
`m=30`, `k=10`, `maxiter=15`, `rtol=1e-12`, `atol=0` use this action with the
original `A128`. The installed flexible implementation stores each actual
preconditioned vector before applying `A128`, so it accommodates rounded
actions without assuming exact linearity. Every right-hand side starts from
zero with an empty recycled-vector list. A successful solver flag is
insufficient: every returned solution must also pass the original selected
residual, and the completed fields must pass the full Maxwell and all-output
comparisons. No fallback or parameter search is used.

Two known-solution complex synthetic cases pass, including a pivot-requiring
zero diagonal and positive diagonal rescaling. Their largest solution error
is `1.86e-11`; both require two actual complex64 factor actions. Twenty-two
refusal controls cover precision conversion, invalid outputs and solver
flags, mutated inputs, and fixed action/callback limits. Exact zero right-hand
sides return zero without a factor action.

The single stored degree-3 FEM control completes all 40 original solves.
Each uses three factor actions, giving 120 in total. Maximum selected residual
is `7.28e-14`; p/s full Maxwell residuals are `6.78e-14`/`4.99e-14`. Every
compared output agrees with all three saved direct controls within `1.38e-13`.
The original CSC, free mask, D and B128 hashes match before precision
conversion. All input and solution precision checks pass.

| Complete stored degree-3 control | Peak process RSS, bytes | Child elapsed, s |
| --- | ---: | ---: |
| Native direct | 578,551,808 | 2.29228 |
| Scaled complex128 MMD direct | 384,696,320 | 2.78916 |
| Complete complex64 MMD with flexible correction | 356,499,456 | 3.95168 |

This measurement reduces observed memory by about 7.3% against the scaled
complex128 control, with increased elapsed time. The complex64 factor takes
.91307 s and has 8,932,284 nonzeros, matching that control's factor nonzero count.
These are shared-host observations; they neither guarantee the same pivot
pattern at higher degree nor imply that total memory is halved.

The arrays of the three retained CSC matrices occupy 54,185,836 bytes. The
explicit vector allowance is 30,742,784 bytes, including possible overlap
between successive Krylov bases, and conversion diagnostics have a separate 66,549,760-byte
work allowance. These are array counts and allowances, not an allocation
cap. Native condensation, trace storage, small dense QR, opaque factor
workspaces and runtime allocators also contribute to the observed process
peak. The 8-GiB observed process guard and 180-s parent remain in force.
This qualifies the fixed small precision strategy. The separate larger
control below measures its degree-6 behavior; optical convergence remains open.

### Full degree-6 precision control on the vertex-.070 mesh

The unchanged complete-complex64/flexible-complex128 strategy also completes
the stored global-degree-6 problem at h=.18, edge target .12, vertex target
.070, cut 4, physical pad .2 and quadrature bonus 6. The fixed mesh has
1,002 tetrahedra, 149,604 full and 79,464 selected unknowns, 2,184 boundary
unknowns and all 81 ordered modes. The full trial is bounded by a 1,800-s
parent and the same 8-GiB observed process guard; neither is a prediction
of resource use at another degree.

All 328 original inverse requests complete: two source fields, 324 boundary
responses and two final corrections. There is one factorization, one bilinear
assembly, 326 linear assemblies, two full operator applications and no internal
block recomputation. Of the requests, 173 need three lower-precision factor
actions and 155 need four, totaling 1,139 actions. Every request uses two outer
callbacks. Maximum original selected residual is `6.16e-13`; the p/s full
Maxwell residuals are `1.52e-14` and `1.79e-14`.

Every scalar, diffraction-order power, volume flux and Robin-budget output
agrees with the historical complete stored native degree-6 result within
`4.26e-14`. Transmission is .8521985908020011 for p and .7308750185490156
for s. This agreement qualifies the finite linear-solver replacement at
this case; the nonzero mesh/degree differences remain separate.

| Complete stored degree-6 control | Peak process RSS, bytes | Child elapsed, s |
| --- | ---: | ---: |
| Historical native direct | 5,795,479,552 | 488.61995 |
| Complete complex64 MMD with flexible correction | 5,849,022,464 | 734.46170 |

This larger measurement **does not reduce observed memory** against the
complete native result: the peak is 53,542,912 bytes (about .924%) higher,
and elapsed time is about 50.3% longer. The complete factor has 340,024,111
nonzeros and takes 214.639 s. Retained original/scaled/rounded CSC arrays
occupy 1,406,868,636 bytes; separate vector and conversion-work allowances
are 256,827,648 and 1,730,356,992 bytes. Those counts exclude native
condensation storage, opaque factor workspace and runtime allocations.
The final peak, after reconstructed fields, exceeds the boundary-stage peak.
Passing the process guard is not evidence that degree 7 will fit.

Historical provenance limits remain explicit. The native stored record
contains physical metadata and complete mode/output arrays, but no original
CSC/D/B128 fingerprints or clean-launch numerical-source hash. Only its
documented absent `recompute_internal=False` setting is normalized. The new
run verifies native-CSR orientation, exact free-mask restriction and DAD
identities, records fresh fingerprints before conversion, and checks retained
inputs throughout. Its original/scaled matrices differ from the saved
**recomputed** degree-6 first-source experiment, although the free mask agrees;
that different-storage experiment is not a matched full-run baseline.
Current instrumentation also retains matrices and checks casts, hashes and
iterations. The table therefore reports shared-host observations, not a
controlled timing replication or allocation bound.

The source-frozen parent closes normally after 734.727 s. An independent
standard-library audit verifies the complete stage sequence, all original
requests, physical metadata, every output comparison, budget arithmetic and
source/launch/completion pins without rerunning a solver. No tolerance or
factor setting changes during the run. This degree-6 control alone does
not complete the spatial/degree comparison; the later
[degree-7 control](#degree-7-reference-release-control) supplies that corner.
The optical convergence stopping criterion remains unresolved.

### Degree-7 precision attempt on the vertex-.070 mesh

The retained-reference attempt with this frozen precision method does
**not** complete the global-degree-7 corner. The physical settings remain
h=.18, edge target .12,
vertex target .070, cut 4, pad .2 and quadrature bonus 6, with stored internal
blocks. Its 1,002 tetrahedra have 227,280 full, 107,040 selected and 2,880
boundary unknowns. The explicit readout quadrature would be 20 rather than
degree 6's 18. No degree-6 optical-equality gate or unavailable degree-7
golden is imposed. The fixed parent allows 3,600 s; the observed process
limit remains 8 GiB.

The complete complex64 factor succeeds in 449.451 s and contains 581,066,720
nonzeros. The first source request, for p polarization, completes four factor
actions with GCROT info 0 and original selected residual `4.73e-14`. During
the second source request's second outer callback, observed process peak RSS
reaches 8,960,032,768 bytes (8.345 GiB), exceeding the guard by 370,098,176
bytes. The guard raises before that request returns an accepted iterate or
GCROT info. Its callback residual, `2.16e-14` after four factor actions, is
therefore a partial observation, not a completed inverse or full-field check.

| Observed stage | Peak process RSS, bytes | Child elapsed, s |
| --- | ---: | ---: |
| Complete factor returned | 3,217,162,240 | 573.11325 |
| First original inverse completed | 7,614,627,840 | 591.24939 |
| Second-source callback raised; failure recorded | 8,960,032,768 | 593.81245 |

Exactly one of the required 328 inverse requests completes. There is one
factorization, one bilinear assembly and two linear assemblies, with no
boundary response, full operator application or internal recomputation.
Neither polarization has a completed Maxwell residual or optical output.
The parent reaps the failed child after 594.249 s, without a timeout, and
all frozen source pins remain unchanged. An independent standard-library
audit accepts this closed **failure record**, not a completed reference.
It checks actual partial stages and counters without rerunning the solver.

The selected original CSC has 47,523,840 nonzeros. Retained original/scaled
complex128 and rounded complex64 CSC arrays total 2,472,524,172 bytes;
the diagonal adds 856,320 bytes. The conservative vector allowance is
345,953,280 bytes and the conversion-diagnostic allowance is 3,041,525,760
bytes. These separate live-array counts and workspace allowances are not
an allocation sum or a bound on opaque LU/native storage. Stage readings
are cumulative process peaks observed after native calls, not a continuous
memory trace or a hard allocation cap. A lower observed peak at factor
return does not establish lower eventual resident memory.

New original/scaled matrix and free-mask fingerprints are captured before
casting; the rounded-matrix fingerprint follows conversion. Retained inputs
are checked during the accepted work.
Historical native stored/recomputed degree-7 attempts lack a complete optical
result and do not supply matching original-matrix fingerprints. This attempt
therefore establishes neither a complete resource improvement over them nor
the spatial/degree comparison. This attempt returns no full optical value;
no four-corner increment or convergence bound is computed from its partial
sources. The later completed control below preserves this failure record.

### Releasing scaled matrix references after factorization

A separate small control shortens the lifetime of the scaled construction
matrices without changing the numerical precision method. It retains the
same original complex128 operator, row-max scaling, checked complex64
conversion, complete MMD factor with .1 pivot threshold, fixed flexible
correction and original selected/full Maxwell residual guards. After the
factor returns and all native/original/scaled/rounded/D identities have been
checked, the helper releases its B128, B64 and cast-buffer references before
the first inverse request. The action retains A128, D and the completed
factor. Later identity checks cover A128 and D, with native CSR, free-mask and
right-hand-side checks preserved; the discarded matrices'
hashes are historical construction evidence rather than per-request checks.

The source-frozen tiny controls establish actual Python object lifetime.
In each of the two known-solution cases, all 12 weak-reference targets are
dead before the known and zero right-hand-side solves. These targets cover
two CSC matrix objects, their array views and six distinct owning allocations
occupying 840 bytes per case. A deliberately retained B64 data alias is
refused. The 22 original refusal controls, two new A128/D mutation checks and
that ownership check give 25 refusals in total. Both known-solution controls
still pass, with maximum errors `5.33e-15` and `1.86e-11`; exact zero loads
still return zero without a factor action. Separate phase records preserve
that B128 is retained during factor creation and released afterward.
Weak-reference disappearance establishes reference lifetime, not the amount
of memory returned to the operating system.

The one complete stored degree-3 FEM control also passes independent closed
result review. Its settings remain h=.2, cut 1, pad .1, theta=.31, phi=.23
and quadrature bonus 6, on 436 tetrahedra with 11,256 full and 9,512 selected
unknowns. All 40 original requests complete with three factor actions each,
120 actions and 80 outer callbacks in total. There is one factorization,
one bilinear assembly, 38 linear assemblies, two full operator applications
and no internal block recomputation. Maximum original selected residual is
`7.28e-14`; p/s full Maxwell residuals are `6.78e-14`/`4.99e-14`. Every scalar,
mode power, volume flux and signed Robin-budget output is exactly equal to
the earlier retained-reference mixed-precision result. Comparisons with all
three saved direct controls remain within `1.38e-13`.

| Stored degree-3 lifetime control | Peak process RSS, bytes | Child elapsed, s |
| --- | ---: | ---: |
| Previous method retaining B128 and B64 | 356,499,456 | 3.95168 |
| Same method releasing them after factorization | 357,351,424 | 3.36817 |

This measurement **does not reduce observed memory**: the new peak is
851,968 bytes, about .239%, higher. One elapsed time is shorter, but this
shared-host observation does not establish a general speedup. The factor
has the same 8,932,284 nonzeros and takes .95519 s.

The removed references cover 33,350,984 bytes of B128/B64 arrays. The original
CSC still occupies 20,834,852 bytes and D adds 76,096 bytes. The earlier
54,185,836-byte three-matrix count remains a construction-phase observation.
These byte counts are not measured OS savings or an allocation subtraction
from peak RSS. Native condensation and trace storage, Krylov bases, opaque
factor/workspaces and allocator behavior remain relevant; release cannot
undo a process high-water mark reached earlier. The original 180-s parent
and 8-GiB observed process guard are unchanged.

Both tiny and full degree-3 parents close with exit 0 and unchanged source
pins. The independent p3 audit checks the lifetime transition, all original
requests, physical outputs and comparisons without rerunning a solver.
This independently authored helper remains a separate internal numerical
validation experiment, with no new dependency or product solver integration.
It does not authorize redistribution of the optional reference environment.
These small controls qualify the construction and its finite equations;
they do not predict that the larger problem will fit or complete. The
following full run supplies separate evidence for that outcome.

### Degree-7 reference-release control

The vertex-.070 global-degree-7 run now completes and passes independent
closed-result review. It retains h=.18, edge target .12, cut 4, pad .2,
quadrature bonus 6 and stored internal blocks on the same physical affine
fixture. Its 1,002 tetrahedra have 227,280 full, 107,040 coupling and 2,880
outer-boundary unknowns, with periodic point counts 25/62. Explicit readout
quadrature is 20. The retained boundary arrays occupy 29,859,840 bytes.

The numerical precision method is unchanged: original complex128 equations
and flexible correction use the complete complex64 MMD factor with .1 pivot
threshold. B128/B64 and cast-buffer references are released after the
construction checks and factor return. Native CSR, selected A128, free-mask,
D, B128 and B64 fingerprints match the earlier retained-reference attempt
on the current numerical source. Later identity checks cover retained A128
and D, with native CSR, free-mask and right-hand-side checks preserved;
recorded construction hashes do not become new checks of opaque factor
contents. Earlier native degree-7 failures still lack a complete optical
golden and matching original-matrix fingerprints.

All 328 original inverse requests complete: two sources, 324 boundary
responses and two final corrections. There is one factorization, one
bilinear assembly, 326 linear assemblies, two full operator applications
and no internal recomputation. The inverse requests use 1,164 factor actions
and 656 outer callbacks. The maximum original selected residual is
`4.41588e-13`; the full reconstructed Maxwell residuals and optical
readouts are:

| finite degree-7 output | p | s |
| --- | ---: | ---: |
| T | .8522509294793323 | .7309173918139138 |
| R | .1477490697957023 | .2690826075914002 |
| full Maxwell residual | 5.11783e-14 | 2.39552e-14 |
| modal/volume flux disagreement | 1.68216e-6 | 6.10961e-7 |
| raw energy error | 7.24965e-10 | 5.94686e-10 |
| signed Robin complement total | +6.67916e-10 | +6.30397e-10 |
| signed energy-budget remainder | +5.70498e-11 | -3.57110e-11 |

The independent audit checks the complete stage/request sequence, physical
lattice and propagating modes, all modal sums, volume readouts, signed
Robin-budget identities and source/launch/completion pins without another
solve. Reflection propagates only `(0,0)` and transmission `(-1,-1)`,
`(-1,0)` and `(0,0)`. The small budget remainders remain distinct from the
larger modal/volume disagreement and optical refinement sensitivity.

The factor contains 581,066,720 nonzeros and takes 460.388 s. The child
closes normally after 1,367.095 s and the parent after 1,367.636 s, without
a timeout or source change. Observed peak process RSS is **8,578,875,392
bytes**, only **11,059,200 bytes (about 10.55 MiB) below 8 GiB**. This is
a cumulative process peak checked after native calls, not a continuous
memory trace or hard allocation cap. It leaves very little observed margin
and supplies no general resource guarantee.

The released B128/B64 array references cover 1,521,619,208 bytes; A128 and
D retain 951,761,284 bytes. These are reference-array counts, not measured
OS savings. The tiny weak-reference proof is not repeated as a large-run
deallocation measurement. The earlier retained-reference degree-7 run
failed during the second source and supplies no matched complete timing or
memory baseline. This remains a separate internal numerical validation
experiment, without product integration or an expanded dependency/license
scope. It supplies a complete finite corner for the separately audited
[degree-6/7 comparison](#vertex-and-degree-refinement-at-degrees-6-and-7),
not an optical stopping certificate.

### Native selected action without a retained CSC copy

A later small control removes the last retained SciPy copy of the selected
matrix. The [reference-release control](#releasing-scaled-matrix-references-after-factorization)
still keeps the selected complex128 CSC `A128` for every original residual
and Krylov product. This control applies the same selected operator through
the native stored-condensed NGSolve matrix `M` as `P.T M P`, where `P`
embeds the existing free mask into the full space. Before any release, the
helper compares the native and CSC products on two fixed pseudo-random
vectors at the existing `1e-12` operator threshold. It then releases A128,
B128, B64 and the cast buffer after the factor returns. The action retains
the native matrix with its CSR views, the private and native free masks, D
and the complete complex64 factor.

Before each request, and again after its final residual product, the helper
checks the native object type and dimensions, the captured and freshly
exposed CSR hashes and storage addresses, both free masks and D. The
equations, scaling, conversion, MMD factor with .1 pivot threshold, flexible
correction options, residual thresholds and physical model are unchanged.
Native row accumulation replaces SciPy accumulation, so the control requires
agreement within the existing tolerances rather than bitwise outputs or
identical iteration counts.

The source-frozen tiny controls use two 12-unknown native matrices, each
with an interleaved eight-unknown free mask. In each case, all 16
weak-reference targets are dead before the known and zero right-hand-side
solves. They cover three CSC matrix objects and nine distinct owning
allocations occupying 1,356 bytes. The native action matches the original
CSC exactly on both vectors and an independent dense block within
`1.62e-16`. There are 35 exact refusals: the 22 original controls, seven
native-matrix, mask or D mutations and injections before a request, one D
mutation after the final native product, three invalid native inputs and
two deliberately retained A128 or B64 aliases. The refusal for freshly
exposed storage substitutes a copied CSR; it does not reallocate native
storage. Both known-solution controls pass with maximum errors `5.32e-15`
and `1.85e-11`.

The one complete stored degree-3 control passes independent closed-result
review with unchanged settings: h=.2, cut 1, pad .1, theta=.31, phi=.23,
quadrature bonus 6, 436 tetrahedra, 11,256 full and 9,512 selected
unknowns. Its native CSR, selected matrix, free mask, D, B128 and B64
fingerprints match the earlier runs before release, and the native and CSC
selected products agree exactly. All 40 original requests complete with
three factor actions and two outer callbacks each. Counters, the maximum
original selected residual `7.28e-14` and p/s full Maxwell residuals
`6.78e-14`/`4.99e-14` are unchanged. All 58 p/s scalar, mode-power,
volume-flux and Robin-budget outputs are bitwise equal to both earlier
mixed-precision results. Comparisons with the three saved direct controls
therefore remain within `1.38e-13`.

| Stored degree-3 control | Peak process RSS, bytes | Child elapsed, s |
| --- | ---: | ---: |
| Retaining A128, B128 and B64 | 356,499,456 | 3.95168 |
| Releasing B128 and B64 after factorization | 357,351,424 | 3.36817 |
| Native action, releasing all three | 350,633,984 | 4.82071 |

The native-action peak is 6,717,440 bytes, about 1.88%, below the B128/B64
release control. This is one shared-host observation. It is smaller than
the 20,834,852-byte A128 array that is no longer retained and does not
establish a general memory reduction. The child takes 1.45 s longer, and the
mean request takes .0511 s instead of .0269 s. The added per-request work
includes hashing the full native CSR at entry and exit and building
full-length native vectors for every product; the split of the extra time
was not profiled. The factor again has 8,932,284 nonzeros.

The released references now cover all 54,185,836 bytes of the three
construction CSC matrices, and no selected CSC copy remains. These are
Python reference lifetimes, not measured operating-system savings. At
degree 7 the retained selected CSC is the 950,904,964-byte array of the
[degree-7 reference-release control](#degree-7-reference-release-control).
In that run the observed RSS high-water mark is 4,336,680,960 bytes when
factorization returns. It reaches 7,014,957,056 bytes after the first
request, 8,051,359,744 at the boundary stage and its final 8,578,875,392
only at the solver return, so the array is held while that peak is set.
This small control does not predict whether releasing it lowers that
peak: RSS excludes compressed and swapped memory and allocator reuse is
unmeasured. Per-request hashing of the degree-7 native CSR, with
47,523,840 nonzeros, also adds time. The
[degree-7 control](#native-selected-action-at-degree-7) below measures both.

Both the tiny and the degree-3 parents close with exit 0 and unchanged
source pins. The independent p3 audit checks the lifetime transition, all
original requests, physical outputs and comparisons without rerunning a
solver. This independently authored helper remains a separate internal
numerical validation experiment, with no new dependency, product solver
integration or redistribution scope. It qualifies the small construction
and its finite equations; it does not establish that a finer or
higher-degree problem will fit or complete.

### Native selected action at degree 7

The native action was also run once at the completed vertex-.070
global-degree-7 stored corner, as a separately released resource control.
Its probe, launcher, plan and auditor are the accepted degree-7 sources with
only the native lifetime validators, the native tiny and p3 gates and one
added comparison. Fourteen inherited probe functions are unchanged at the
syntax-tree level, and 17 focused refusals were checked before the launch.
The expected-fingerprint table gains the native CSR and rounded-matrix
entries that the native helper reads; the physical fixture, settings,
factor options and guards are the accepted degree-7 contract.

All 328 original requests complete, and the independent audit accepts the
closed run. Before release the native and CSC selected products agree
exactly on both check vectors. Counters, the 1,164 factor actions, the 656
outer callbacks, the maximum original selected residual `4.41588e-13` and the
p/s full Maxwell residuals `5.11783e-14`/`2.39552e-14` repeat the accepted
release row. All 346 p/s scalar, 81-order and Robin-budget outputs are
bitwise equal to it. The factor again has 581,066,720 nonzeros and takes
467.989 s.

| Stored degree-7 observed peak RSS, bytes | Factor start | Factor return | First request | Final |
| --- | ---: | ---: | ---: | ---: |
| Releasing B128 and B64 | 4,336,680,960 | 4,336,680,960 | 7,014,957,056 | 8,578,875,392 |
| Native action, releasing all three | 3,141,402,624 | 3,521,511,424 | 6,948,880,384 | 7,332,790,272 |

The final observed peak is 1,246,085,120 bytes, about 14.5%, lower, but
**this is not a measured saving**. At factorization start both helpers
still run identical code, and the observations there already differ by
1,195,278,336 bytes. The host was under memory pressure during this run. An
external sampler, reading `top` and `vm_stat` every 30 s without pausing the
solver, recorded up to 8.79 GiB of the solver's memory compressed, a memory
footprint of about 10 GiB during factorization, and system swap between 9.7
and 13.4 GiB. RSS excludes compressed and swapped pages, so both rows partly
track the host state. Across the release point the footprint drops by about
2.4 GiB; the released arrays are 2.30 GiB, but factor workspace is freed at
the same moment. The release run has no footprint record, so no footprint
comparison is possible.

The child takes 1,699.576 s instead of 1,367.095 s, and the mean request
2.948 s instead of 1.938 s. The added per-request work includes hashing the
native CSR at entry and exit; the split was not profiled, and the memory
pressure may also have slowed the run. The observed 8 GiB guard never
tripped, yet it did not bound the solver's real memory: the factorization
footprint exceeded 8 GiB while the observed RSS high-water mark stayed at
most 3,521,511,424 bytes. A separate watchdog on the kernel memory-pressure
level read normal or warning at every 5 s check and never critical.

This control therefore qualifies the native action at full degree-7 scale
for arithmetic and outputs only. It does not establish a memory reduction,
the fit of a finer mesh or a vertex-.065 degree-7 run. A clean memory
attribution needs physical footprint recorded inside both variants under
matched host conditions.

## Completing degree 7 on the smaller edge mesh

The separate edge-.12 branch completes global degree 7 within the same
8-GiB RSS guard. It uses h=.18, physical pad=.2, cut 4, quadrature bonus 6
and no added vertex restriction. Degree is assigned after mesh generation;
all three rows have 780 tetrahedra and periodic point counts 22/36.
Matching historical mesh summaries does not establish an exact topology
fingerprint. Explicit readout quadrature follows degree, with orders
16/18/20 at degrees 5/6/7.

| global degree | p T | s T |
| ---: | ---: | ---: |
| 5 | .852038068565 | .730662922035 |
| 6 | .852147871193 | .730777052860 |
| 7 | .852213454659 | .730845979436 |

| interval | p total T change | s total T change | largest p individual R/T change | largest s individual R/T change |
| --- | ---: | ---: | ---: | ---: |
| 5 to 6 | +1.09803e-4 | +1.14131e-4 | 1.44154e-4 | 1.14144e-4 |
| 6 to 7 | +6.55835e-5 | +6.89266e-5 | 8.38648e-5 | 6.89265e-5 |

Both measured intervals become smaller, but the final individual changes
are still tens of parts per million of incident power. In p transmission,
the degree-6-to-7 changes in `(-1,-1)`, `(-1,0)` and `(0,0)` are
`+2.93487e-6`, `+8.38648e-5` and `-2.12162e-5`; cancellation reduces
the total. The corresponding s changes are `+1.12023e-5`, `+5.42548e-7`
and `+5.71817e-5`. Two finite intervals do not establish an asymptotic
rate or a remaining-error bound. This edge-only result belongs to a
different mesh branch from the separately completed
[vertex-.070 degree-7 control](#degree-7-reference-release-control).

At degree 7, modal/volume flux disagreements are `3.33281e-6` and
`1.54597e-6`, smaller than the degree-6 values `5.32953e-6` and
`2.62282e-6`. Full linear residuals are `1.31802e-14` and `7.24477e-15`.
The signed energy remainders after the Robin complement are
`+1.45691e-11` and `+5.18571e-13`. These algebraic and readout checks
remain distinct from optical degree sensitivity.

Against the saved finite Wavebit `(24,32)` result, p/s total transmission
differences are `-1.67256e-4` / `-1.67239e-4`. For s, the `(-1,0)`
order differs by only `-1.33087e-6`, while `(0,0)` still differs by
`-1.32713e-4`. Agreement in one channel does not qualify the others;
both finite solvers retain unresolved optical discretization error.

The solve has 177,232 full, 83,632 coupling and 2,880 boundary unknowns.
It completes in 818.731 s, including factorization by 252.240 s and
the boundary stage by 800.012 s. Observed peak RSS is 7,055,966,208 bytes.
The callback reports every eight completed trace applications, ending
at 324/324. Counts match preflight and the two numerical source hashes
match the clean launch at `5bc835e` (landed as `ff56d28`). These are
shared-host costs;
small independent-oracle arithmetic checks overlap part of this run.
They are not a general runtime or memory guarantee.

## Separating the film band and exterior at degree 7

On the same edge-.12 mesh branch, base degree 6 with film degree 7 and
band padding .05 completes with 160,774 full and 76,824 coupling unknowns.
It selects 765 edges, 1,129 faces and 587 cells, while the outer boundary
retains 2,184 unknowns. The geometry settings, 780 tetrahedra and 22/36
periodic point counts match the completed global-degree rows. Historical
mesh summaries still do not provide an exact topology fingerprint.

| space | p T | s T |
| --- | ---: | ---: |
| global degree 6 | .852147871193 | .730777052860 |
| base 6, film-band degree 7 | .852203145502 | .730843282952 |
| global degree 7 | .852213454659 | .730845979436 |

| change | p total T | s total T | largest p individual R/T | largest s individual R/T |
| --- | ---: | ---: | ---: | ---: |
| global 6 to film-band 7 | +5.52743e-5 | +6.62301e-5 | 5.52756e-5 | 6.71155e-5 |
| film-band 7 to global 7 | +1.03092e-5 | +2.69648e-6 | 3.06867e-5 | 1.24043e-5 |

The first change also raises explicit readout quadrature from 18 to 20.
The second keeps order 20 and raises the unselected exterior entities
and boundary traces; its boundary space increases from 2,184 to 2,880
unknowns. These are different space increments, not equal refinement steps.
Their signed sums reproduce the completed global-6-to-7 interval.

The exterior-and-boundary contribution remains visible in individual
transmitting orders even when the total is small:

| T order | film-band 7 to global 7, p | film-band 7 to global 7, s |
| --- | ---: | ---: |
| (-1,-1) | -7.40278e-7 | +2.25865e-7 |
| (-1,0) | +3.06867e-5 | +1.24043e-5 |
| (0,0) | -1.96373e-5 | -9.93373e-6 |
| sum of absolute changes | 5.10643e-5 | 2.25639e-5 |

For s `(-1,0)`, film-band enrichment first changes the power by
`-1.18618e-5`; the remaining global increase adds `+1.24043e-5`.
Those contributions nearly cancel in the global-degree interval, whose
change in that order is only `+5.42548e-7`. This finite comparison does
not qualify the local space as optically equivalent to global degree 7,
bound either solver's remaining error or replace the distinct
[vertex-.070 global-degree-7 result](#degree-7-reference-release-control).

The local p/s modal-volume disagreements are `3.45926e-6` / `1.75515e-6`,
and full residuals are `1.22413e-14` / `7.83061e-15`. Raw energy errors
are `3.99176e-9` / `8.80404e-10`; after the reported Robin complement,
the signed remainders are `-1.75517e-9` / `-5.59611e-10`. These checks
do not replace the larger optical space differences. Relative to the
saved finite Wavebit `(24,32)` row, total p/s T differences remain
`-1.77565e-4` / `-1.69936e-4`.

The run completes in 831.270 s with peak RSS 6,464,536,576 bytes,
compared with 818.731 s and 7,055,966,208 bytes for the saved global-7
run. Factorization completes by 218.549 s, boundary responses by
814.522 s, and all 324 traces pass the interval-eight progress guard.
The smaller space uses less observed peak RSS but is not faster in
these two shared-host runs. Small NumPy geometry checks overlap part
of this run. These observations compare different finite approximations
and do not establish equal-accuracy performance. The numerical source
hashes match the clean launch at `747dd25` (landed as `e9801e2`) when checked
after completion.

## Comparing the physical diffraction orders with Wavebit

Both solvers use `k_parallel=k_incident+G`. At the saved Wavebit optical
order `(24,32)`, an FEM label `(m,n)` maps to array index `[n+32,m+24]`,
without a sign reversal. The public `order_kxy()` values over the full
65-by-49 window agree with independently evaluated physical reciprocal
vectors within `2.28e-13` per micrometer. Reflection propagates only
`(0,0)`; transmission propagates `(-1,-1)`, `(-1,0)` and `(0,0)`.
Summing those orders reproduces each full power array within `1.12e-16`.
Both sets of powers are already normalized by incident power.

The Wavebit row uses the independently fitted degree-10 auxiliary field,
coefficient cap 64 and projection quadrature 40 at optical order `(24,32)`.
It still uses Wavebit's Maxwell engine and has its own unresolved Fourier
tail. The FEM row below uses h=.18, degree 6, cut 4 and pad=.1.

| transmission order | Wavebit p | FEM p | Wavebit s | FEM s |
| --- | ---: | ---: | ---: | ---: |
| (-1,-1) | .002546061499 | .002524305231 | .006349820924 | .006288808035 |
| (-1,0) | .407255483011 | .407069924636 | .269989779138 | .270045795985 |
| (0,0) | .442579166466 | .442588490413 | .454673618575 | .454361279445 |
| total | .852380710977 | .852182720280 | .731013218636 | .730695883465 |

The total FEM-minus-Wavebit differences are `-1.97991e-4` for p and
`-3.17335e-4` for s. Apparent total agreement at lower FEM degree can
hide substantially larger redistribution: at degree 4, the p differences
in orders `(-1,0)` and `(0,0)` are `-1.08912e-3` and `+1.06132e-3`,
while the total difference is only `-1.00245e-4`. The corresponding s
differences are `-9.44532e-4` and `+1.39708e-3`, with a total difference
of `+1.74793e-4`. Neither finite solver is assigned as a converged oracle
from this agreement. FEM discretization, outgoing boundary and Wavebit's
Fourier truncation still require separate checks.

Across the completed pad-.2 controls, global degree 6 to 7 on the edge-only
mesh changes p/s total T by `+6.55835e-5`/`+6.89266e-5`. Adding vertex
target .070 at degree 6 changes them by `+5.07196e-5`/`+9.79657e-5`;
the next .070-to-.065 interval changes them by
`-8.71206e-6`/`+4.04727e-5`. In particular, the p spatial increment
reverses sign. These unresolved Maxwell changes exceed the candidate-minus-
fitted-auxiliary-degree-10 total differences at the same `(24,32)` Fourier
order, `-1.52748e-6`/`-3.42893e-6`. Close agreement between those two
finite NVF fields therefore does not establish Maxwell or Fourier convergence.

The completed vertex-.070 global-degree-7 control now closes the separate
no-vertex/.070 by degree-6/7 comparison below. The previous failures remain
historical records; the new finite intervals still supply sensitivity
measurements rather than a remaining-error bound.

## Vertex and degree refinement at degrees 6 and 7

A separately frozen reader uses the accepted complete degree-7 audit and
the three saved native-direct results. It checks common physical settings
and ordered modes, modal sums and signed Robin budgets before computing
the four-corner differences. All rows retain h=.18, edge target .12,
cut 4, pad .2 and quadrature bonus 6. Degree 6/7 uses explicit readout
quadrature 18/20. Within each degree pair, mesh settings, element counts
and periodic point counts agree; historical counts alone do not prove
identical topology. Adding vertex sizing remeshes the cell. The new corner
uses the qualified mixed-precision correction; the other three use native
direct solves. Their original-matrix and forward-error provenance limits
remain, so this readout is not a full solver-performance replication.

Let E6/E7 denote the edge-only rows and V6/V7 the vertex-.070 rows:

| row | tetrahedra | full / coupling unknowns | p T | s T |
| --- | ---: | ---: | ---: | ---: |
| E6 | 780 | 116,690 / 62,090 | .8521478711930495 | .7307770528597978 |
| E7 | 780 | 177,232 / 83,632 | .8522134546586737 | .7308459794359807 |
| V6 | 1,002 | 149,604 / 79,464 | .8521985908019867 | .7308750185489832 |
| V7 | 1,002 | 227,280 / 107,040 | .8522509294793323 | .7309173918139138 |

Each maximum below ranges over individual propagating **R and T** orders.
The mixed difference is `V7 - V6 - E7 + E6`, evaluated on each power as
well as the totals; it is not a remaining-error estimate.

| signed change | p total T | s total T | largest p individual R/T magnitude | largest s individual R/T magnitude |
| --- | ---: | ---: | ---: | ---: |
| degree on edge mesh, E7 − E6 | +6.55834656e-5 | +6.89265762e-5 | 8.38648320e-5 | 6.89265185e-5 |
| degree on vertex mesh, V7 − V6 | +5.23386773e-5 | +4.23732649e-5 | 6.55808985e-5 | 4.23723961e-5 |
| vertex target at degree 6, V6 − E6 | +5.07196089e-5 | +9.79656892e-5 | 5.90348270e-5 | 1.53018200e-4 |
| vertex target at degree 7, V7 − E7 | +3.74748207e-5 | +7.14123779e-5 | 4.07508935e-5 | 1.14585678e-4 |
| mixed vertex/degree difference | -1.32447883e-5 | -2.65533113e-5 | 1.82839335e-5 | 3.84325217e-5 |

On the vertex mesh, degree 6 to 7 changes p transmission in `(-1,-1)`,
`(-1,0)` and `(0,0)` by `+3.33410e-6`, `+6.55809e-5` and
`-1.65763e-5`. Their cancellation fraction,
`1 - |sum(delta)| / sum(|delta|)`, is 38.78%. The three s transmission
changes are positive: `+3.42686e-6`, `+2.01972e-5` and `+1.87492e-5`.
The largest s individual change across R/T is instead the specular
reflection change `-4.23724e-5`; using only the transmitting channels
would understate that maximum.

Adding the vertex target at degree 7 still redistributes s transmission
by `+2.09715e-5`, `-6.41448e-5` and `+1.14585678e-4` in the same order.
The absolute changes sum to `1.99701927e-4`, while their signed total is
`+7.14123779e-5`: 64.24% cancels. The p changes are `+2.08183e-7`,
`+4.07509e-5` and `-3.48426e-6`. Both total-transmission degree and vertex increments are smaller here
than on the corresponding earlier side of this rectangle. This does not
hold for every individual order: the s `(-1,0)` transmission degree
increment grows from `+5.42548e-7` on the edge mesh to `+2.01972e-5`
on the vertex mesh. Individual changes and the nonzero mixed difference
remain visible.
This does not establish an asymptotic rate, monotone accuracy or an optical
stopping criterion.

V6 to V7 reduces p/s modal-volume disagreement from
`2.81115e-6`/`1.08678e-6` to `1.68216e-6`/`6.10961e-7`. All four rows
have full linear residuals below `5.12e-14`; the signed Robin-budget
remainders range from `-1.77116e-9` to `+5.70498e-11`. These verify
finite equations and readout identities without bounding the larger
optical differences. The outgoing-boundary and Wavebit Fourier checks
remain separate, and no missing vertex-.065 degree-7 value is inferred.

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

The existing validation environments also contain SciPy 1.18.0, source
revision `54ef5423f2e4376230ec3bfda6912a07a50958e3`, with its vendored
SuperLU extension. The [same-version bundled license inventory](https://github.com/scipy/scipy/blob/v1.18.0/LICENSES_bundled.txt)
and [bundled SuperLU license](https://github.com/scipy/scipy/blob/v1.18.0/scipy/sparse/linalg/_dsolve/SuperLU/License.txt)
specify BSD-3-Clause. The vendored header identifies 7.0.1, while its
README still says 6.0.1; this is source-header evidence rather than a
separate binary version query. SciPy replaces the optional MC64 routines
with error stubs. The FEM environment links OpenBLAS 0.3.34
(BSD-3-Clause), libgfortran 16.2.0 (GPL-3.0 with GCC runtime exception)
and LLVM OpenMP 22.1.8 (Apache-2.0 with LLVM exception); the main test
environment instead links system Accelerate.

The comparison is independently written internal code that uses explicit
SciPy `splu` ordering; it installs no dependencies and copies no solver
implementation. This inspection does not establish a performance
benefit: the earlier slower and unfinished square SuperLU trials remain
recorded above. Redistribution retains the components' required notices
and needs the separate whole-environment review described here.

## Reproduction

On the current host NGSolve 6.2.2606 is already installed in
the optional NGSolve environment. Run from the
NVF worktree:

Set `NVF_FEM_PYTHON` to the Python executable with the optional NGSolve dependencies.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 "$NVF_FEM_PYTHON" studies/rcwa/nvf_fem_reference/nvf_fem_reference.py --hs .18 .14 --orders 4 --boundary pml
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 "$NVF_FEM_PYTHON" studies/rcwa/nvf_fem_reference/nvf_fem_reference.py --hs .18 --orders 3 4 --uniform 4 --boundary pml
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 "$NVF_FEM_PYTHON" studies/rcwa/nvf_fem_reference/nvf_fem_reference.py --hs .18 .14 --orders 4 --boundary rayleigh --cut 2
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
`python -m unittest tests.rcwa.test_nvf_fem_reference.FEMAlgebraicTests -v`.
The subsequent geometry extension passes all eight affine geometry tests
plus those nine algebraic tests in 0.358 s:

```bash
PYTHONPATH=$PWD "$NVF_FEM_PYTHON" -m unittest tests.rcwa.test_nvf_fem_affine_geometry tests.rcwa.test_nvf_fem_reference.FEMAlgebraicTests -v
```

Five existing square-fixture mesh and periodic-identification tests also
pass in 0.387 s. The five existing scattering-solve tests are not rerun for
the geometry-only change. The new checks establish geometry and space
consistency, not a converged optical reference.

The full affine integration is checked separately, including both
polarizations, the driven degree-4 slab, Robin power and local sizing:

```bash
PYTHONPATH=$PWD "$NVF_FEM_PYTHON" -m unittest tests.rcwa.test_nvf_fem_boundary_power tests.rcwa.test_nvf_fem_affine_scattering tests.rcwa.test_nvf_fem_affine_geometry tests.rcwa.test_nvf_fem_reference.FEMAlgebraicTests -v
```

This selection passes 29 tests in 11.693 s. To reproduce a physical affine
solve through the Python entry point:

```bash
PYTHONPATH=$PWD "$NVF_FEM_PYTHON" - <<'PY'
import json
import ngsolve as ng
from studies.rcwa.nvf_fem_reference.nvf_fem_reference import solve_affine_rayleigh
ng.SetNumThreads(1)
result = solve_affine_rayleigh(h=.18, order=4, cut=2, pad=.1)
print(json.dumps(result, allow_nan=False))
PY
```

Adding `uniform=4., phi=.37` reproduces the slab geometry and incidence;
its reported control above uses `cut=1`. The physical-edge trial uses
`solve_affine_rayleigh(h=.18, order=4, cut=4, pad=.2, edge_h=.10)`.
`quadrature_bonus=14` requests the higher integration setting independently
of mesh and FEM degree. The default command-line entry
point continues to select the historical square fixture.
The boundary-storage and fixed-mesh degree controls are available as
`tests.rcwa.test_nvf_fem_boundary_storage`, `tests.rcwa.test_nvf_fem_space` and
`tests.rcwa.test_nvf_fem_affine_degree`. To raise the film neighborhood on
the target-.12 mesh, use `solve_affine_rayleigh(h=.18, order=5, cut=4,
pad=.2, edge_h=.12, film_degree=6, film_degree_pad=.05)`.
For the separate physical-corner mesh control, use
`solve_affine_rayleigh(h=.18, order=5, cut=4, pad=.2, edge_h=.12,
vertex_h=.07)`; no film-degree enrichment is selected by that call.

For resource monitoring, the optional `progress` callback receives `mesh`,
`factorized` and `boundary` records. Setting `boundary_progress_every=32`
also emits `boundary_progress` after each 32 completed trace inverse
applications and after the last trace. The fields
`boundary_columns_completed` and `boundary_columns_total` count completed
traces, not the percentage of the entire solve. A caller can raise from the
callback to abort before the next trace, including when an observed RSS
limit is exceeded. This is not a hard memory cap: it cannot interrupt
native assembly or inverse calls, and the two final full-field corrections
still follow the boundary stage. The default `None` preserves the original
callback sequence. Boundary-storage tests compare the full and condensed
physical solutions with reporting disabled/enabled, count actual inverse
applications, and verify that a callback exception stops before the next one.

Selected results, with prototype-versus-packaged flux provenance, are retained in
the run record `nvf_fem_reference.log`, archived outside the repository; the
commands above rerun `studies/rcwa/nvf_fem_reference/nvf_fem_reference.py`.
