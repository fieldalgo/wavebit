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
`2*FEM_degree+quadrature_bonus`. Both the result and its settings record
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
polarizations, the driven degree-4 slab, Robin power and local sizing:

```bash
PYTHONPATH=$PWD "$NVF_FEM_PYTHON" -m unittest tests.test_nvf_fem_boundary_power tests.test_nvf_fem_affine_scattering tests.test_nvf_fem_affine_geometry tests.test_nvf_fem_reference.FEMAlgebraicTests -v
```

This selection passes 29 tests in 11.693 s. To reproduce a physical affine
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
its reported control above uses `cut=1`. The physical-edge trial uses
`solve_affine_rayleigh(h=.18, order=4, cut=4, pad=.2, edge_h=.10)`.
`quadrature_bonus=14` requests the higher integration setting independently
of mesh and FEM degree. The default command-line entry
point continues to select the historical square fixture.
Selected results, with prototype-versus-packaged flux provenance, are retained in
[`studies/results/nvf_fem_reference.log`](../studies/results/nvf_fem_reference.log).
