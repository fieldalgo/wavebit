# Stable real radial enrichment of the pair potential space

The explicit corner family can add fixed radial exponents `0.55` and `0.95`
to `0.6` and `0.8`. This increases the radial trial space while keeping its
coordinates real and independent of material values. It does not establish
optical convergence or change ordinary NVF routing.

An explicit [symmetric mesh](nvf-pair-symmetric-potentials.md) additionally
preserves the physical pair's reflection identities. Expanded radial fields
on that mesh use real coordinates formed before physical assembly to retain
accuracy in the original weak equations.

The original exponent subsets retain their original fifth-root evaluation.
An expanded subset uses twentieth-root integration and requires quadrature
at least 48. The working integration order is also at least `20*max(p,4)`. The cubic endpoint taper multiplied by the
linear transverse profile is a degree-four trace potential, even on a
cubic mesh. With `d=max(p,4)`, the radial polynomial stiffness integrand
can have degree `40*d-1` after this substitution. The floor covers both
the nodal and trace polynomial parts. A degree-three metal control at a
small positive gap rejected order 60 with an independent energy-derivative
residual of `3.74e-10`; order 80 passed at `4.55e-14`, with the same original
`1e-10` guard. The working floor is therefore 80 for both supported degrees.
For example, the highest radial power in a quartic trace's gradient is
`r**3`. Squaring it and including the area measure gives `r**7 dr`.
With `r=t**20`, this becomes `20*t**159 dt`; 80 Gauss points cover degree
159. Counting only cubic nodal potentials would stop at degree 119 and
incorrectly suggest 60 points.

The returned `integration_quadrature` reports the actual order, including
this floor and the existing retained-Fourier requirement.
The complete positive-gap equations, moving-boundary terms,
contact response and independent stationary-energy verification retain all
of the enlarged coordinates.

## Why raw powers were insufficient

After the polynomial overlap is removed, nearby real powers have very similar
remainders. Subtracting these sampled powers loses significant digits even
when the response itself remains accurate. In a degree-four dielectric
contact control at Fourier order one, increasing the integration order from
64 to 80 changed the gap derivative by `3.68e-9`, exceeding its `3e-9`
criterion. Increasing the order to 96 did not resolve that failure.

The implementation removes the two overlaps together, before sampling.
For a spatial prefactor of degree `k`, let `r = max(0, p-k)`, where `p` is
the polynomial degree. Normalize the remainder of `s**(alpha-k)` after its
first `r` Taylor terms by its leading Taylor coefficient. Successive divided
differences in the fixed exponent nodes make the leading orders distinct.
This is an invertible real triangular coordinate change within the same
trial space; no field or numerical rank is discarded.

Multiplication by `s**k` makes every such combination a polynomial in
`u = s**(1/20)`. Exact rational arithmetic factors out
`(1-u)**(r + number_of_nodes - 1)`. The remaining coefficients have one sign
for the supported exponent families. Evaluating the factored polynomial
avoids subtracting nearby powers. The small factor `1-u` is obtained as
`(1-s)/(1+u+...+u**19)`, with the complementary coordinates retained
separately near a corner. First and second derivatives follow from the same
factored expression.

A fixed additional real scaling makes each successive leading Taylor
coefficient one. Homogeneous spatial products are evaluated with normalized
neighbor hats before multiplication by the radial power. This avoids
separately overflowing a negative power whose complete field is finite.

## Integration and finite controls

The substitution `s = t**20` resolves the additional rational exponents.
It applies to the ordinary triangle integrals, both ends of collapsing
triangles, clipped moving regions and explicit boundary fluxes. The radial
area Jacobian and its derivative change with the substitution; changing only
the sampled field would not define the same integral or gap derivative.

Two independently evaluated representations of the normalized radial space
passed the original contact response and energy guards. Their order-64/80
contact derivative differences were `2.20e-10` and `2.26e-10`, respectively.
The factored polynomial is used because it represents the fixed rational
powers directly and avoids a matrix exponential at each sampling point.
These checks apply to the stated finite calculations and do not establish
continuum accuracy. The integration comparisons use the stated orders
directly; public closure calls additionally apply the working-order floor.

Further whole-space controls compare two positive-gap quadrature orders and
two fourth-order gap differences, with maximum derivative discrepancies of
`7.46e-10`, `1.22e-10` and `9.21e-11`, respectively. Public finite controls
also retain exact uniform media, complex-linear material directions, length
scaling, nested dielectric energy and invariance under real exponent-node
reordering. The original default field sampler remains bitwise identical. Complete
response compatibility is also checked after the balanced-variation update.

## The final finite metal contact control

The final public implementation uses integration order 80 throughout this
control: period 700, widths `(210,280)`, height 315, offset `(7,-21)`,
materials `(12.25,-10+1j)` in background 1, Fourier order zero, degree three,
all four radial exponents, `junctions=True` and angular order zero. Every
contact and positive state passes the original physical guards.

| Gap | Tensor norm difference from contact | Gap-JVP norm difference from contact |
| --- | ---: | ---: |
| `0.001` | `4.84142648e-04` | `2.78221161e+00` |
| `1e-07` | `2.72395294e-07` | `3.79736169e-03` |
| `1e-09` | `2.72463789e-09` | `1.24437026e-04` |

These checks establish approach to the same finite-family contact response.
They do not assign an optical convergence rate or a continuum error bound.

Expanded radial fields can materially change an unresolved optical
calculation. A metal pair at Fourier order four and angular order two changed
its transmission slopes substantially when all four radial exponents were
included. The expanded field option therefore remains an explicit resolution
control. Mesh, angular, radial and retained Fourier convergence must be checked
together before interpreting a contact slope as an optical prediction.

## Fixed real oscillatory directions

The optional `angular_quadrature` setting separates the angular Gauss rule
from radial integration in two-dimensional corner and clipped-trace charts.
It accepts an integer at least `max(12, 2*max(order)+4)`; the default `None`
uses the effective radial order in both directions. The radial floor and
one-dimensional moving-edge rules are unchanged. Coordinate preparation
uses its fixed geometry-only rules, so selecting a physical integration order
does not reconfigure the prepared field coordinates. Metadata records an
explicit choice as `angular_integration_quadrature`, and the optical study
accepts `--angular-quadrature`.

These are local radial/angular directions around a corner, independent of
the Cartesian x/y Fourier orders. The twentieth-root map needs a high radial
order even when a lower angular order resolves the smooth angular integrand.
The setting is an integration parameter, not an accuracy certificate: check
both directions along with the complete physical response. Exact triangle
moments test the separate rules; physical material and contact/positive
variations retain their original tolerances.

A separate `oscillatory_radial=True` option in
`CartesianPairCornerGalerkinGapCell` adds the real span of
`s**(19/20) * cos(log(s)/2)` and `s**(19/20) * sin(log(s)/2)` to the fixed
`(0.6, 0.8)` real powers. Here `s` is the complementary corner hat. The same
homogeneous neighbor and angular prefactors supply the spatial potentials.
The pair is selected together, remains independent of material values and
uses the existing twentieth-root quadrature floor. Other real exponent lists
are refused with this option. The default is `False`.

Both oscillatory directions enter one real coordinate transformation, rather
than assigning a complex trial function to a real material-independent
space. After inserting the integer polynomial-overlap nodes and the two real
radial nodes, the last block of the real exponent matrix is

```text
[ 19/20  -1/4 ]
[     1  19/20 ]
```

Its eigenvalues are `19/20 +/- i/2`. The first column of the combined
exponential supplies the divided-difference coordinates. Derivatives use
that same exponential multiplied by the exponent matrix and by its product
with the exponent matrix minus identity. The returned fields remain real;
all original physical equations and acceptance criteria still apply.

A compensated scalar evaluation used for raw coordinates and geometry
preparation produces the same rounded real exponent columns. For `h <= 1/4`,
a 96-term series with exact rational high/low
coefficients retains small remainders. Elsewhere, compensated scalar powers
combine the complete polynomial remainders and logarithmic derivatives.
Each scalar exponential uses a scaled 26-term Taylor series and repeated
squaring, with scaled argument magnitude at most `1/4`. The final column
normalization follows the matrix evaluator's rounding order. This avoids
cancellation between close polynomial and radial powers while preserving
bitwise agreement with the complete matrix evaluation. The latter remains
an arithmetic reference; independent Decimal power sums are a separate
accuracy oracle.
Only exactly identical binary `(h, s)` inputs share evaluations. Every row
is restored to its original position. The scalar evaluator uses batches of at most 4096 points, with only linear
scalar-array storage. The matrix reference uses at most 256 points per
batch. The lower-right polynomial suffixes of one exponential
supply all homogeneous families of the chosen polynomial degree. Only the
final radial values and derivatives are retained. Their array and key
payload is bounded by 64 MiB; a profile larger than that budget is evaluated
without retaining it. Exact profiles share this budget rather than each
occupying one equal-sized cache entry. On the unrefined symmetric mesh, all
25 distinct reference profiles occupy about 25.5 MiB for cubic fields at
order 80 and 36.4 MiB at order 96. The corresponding quartic payloads are
33.6 and 47.9 MiB. Larger quadratures and meshes retain the same byte budget. Neither operation removes a point or
field. No numerical rank threshold or material regularization is introduced.

The symmetric mesh needs additional care: its complete coordinate transform
can amplify rounding of otherwise accurate raw fields. It therefore evaluates
the transformed potentials before rounding. If `c_i` is the corner hat at
triangle vertex `i`, define `q_i = b_i*(1-c_i)/s`. Homogeneous angular products
become Bernstein polynomials of degree `K = 2 + angular_order` in `q`. Their
exact rational coefficients and the polynomial overlap are combined with
the complete stored coordinate transform. Each corner group has the form

```text
h * sum_l R_l(s) * B_K(q) @ C_l - B_p(b) @ C_p
```

Here the four real `R_l` are the two real powers and the logarithmic cosine
and sine pair. Every original field column remains. The polynomial term
contains both its Taylor overlap and the common nodal interpolant. Values,
spatial gradients and moving gradients retain high/low components through
this whole expression. A binary range reduction and 25-term odd-power
logarithm series supply the radial kernels. For fixed reference nodes, the
same reference gradient supplies both the physical inverse-Jacobian map and
its variation; moving reference nodes use the full differentiated expression.

Bernstein evaluation has at most 28 columns. Its compensated products split
normalized inputs into two bounded integer chunks and the complete floating
remainder, retain all nine cross products and accumulate their high/low
parts. Leading integer partial sums stay below `2**50`. This small geometry
kernel is separate from the unchanged three-chunk original physical residual
arithmetic. Fixed coefficient tables have a 64 MiB array/key payload budget
per cell. Oversized tables are evaluated without retention, and sampled
field intermediates use batches of at most 256 points. This budget does not
include the complete returned fields or solver matrices. Gradient-only calls
retain the same spatial and moving derivatives without allocating an unused
whole-field value array. Their polynomial coordinate products omit only those
unused value/directional columns. Summation buffers are reused with unchanged
term order; the original residual arithmetic is unaffected.

Independent Decimal potentials check this complete transformed expression
and physical variations, including corner distances down to `2**-200`.
Interior controls use 100 digits; singular-corner controls use 420 digits.
Neither their reference powers nor their high-precision physical difference
stencils use the production polynomial coefficient builder.

Returned metadata identifies the option, power and logarithmic frequency as
`oscillatory_radial`, `oscillatory_radial_power` and
`oscillatory_radial_frequency`; `corner_exponents` continues to report the
ordinary real powers. The optical study accepts `--oscillatory-radial` and
records that choice alongside the complete geometry and source hashes.

Independent 180-digit complex-power sums check all four real radial
coordinates and both derivatives for cubic/quartic spaces and angular
orders through four, including complements as small as `1e-78`. Separate
raw-field controls compare complete spatial and moving gradients with
fourth-order differences. Selecting this enlarged finite space does not
establish joint radial, angular, mesh or Fourier convergence.

An initial raw-coordinate control of the period-700 metal contact passed
through the explicit oscillatory option at Fourier orders `(4, 4)`, degree
three, angular order two, the unrefined symmetric mesh and integration order
80. Its original energy and energy-derivative residuals are `2.77e-13` and
`1.44e-13`. Maximum full Fourier-reflection defects are `1.80e-10` for the
value and `1.57e-9` for the physical gap derivative, within the unchanged
`3e-10` / `3e-9` criteria. The p/s transmissions are `.48734923` /
`.47181436`, and optical directional slopes are `-.04600084` / `.00196346`.
The two optical stencil estimates differ by less than `7.4e-9`; these
constitutive perturbations remain distinct from a physical gap-difference
verification and do not establish optical convergence.

At angular order four, raw-coordinate rounding exceeded the same reflection
criterion: the largest value defect was `1.75e-9`. Complete compensated
evaluation with radial/angular Gauss orders 80/24 reduced the maximum value
and derivative defects to `4.34e-12` and `8.76e-11`, respectively, retaining
the original `3e-10` / `3e-9` criteria. Original energy and energy-derivative
residuals were `2.30e-13` and `1.72e-13`. The p/s transmissions were
`.48868243` / `.47548184`. An independent complete response through the
public 80/32 quadrature option passed the same guards. Increasing the angular
Gauss order from 24 to 32 changed the entire value and derivative by at most
`4.22e-12` and `5.39e-11`; the original energy and energy-derivative residuals
were `2.35e-13` and `1.82e-13`. This is a finite-space numerical accuracy
result; it does not establish joint mesh, radial, angular or Fourier
convergence.

A positive-gap control with unit period, widths and height `3/7`, gap `.003`,
materials `12.25` and `-10+i`, Fourier order zero and angular order zero also
passed the original guards with the complete symmetric oscillatory space at
radial/angular orders 80/16. Its entire value and gap derivative differed from
the independently evaluated raw-coordinate control by at most `1.92e-12` and
`4.21e-10`. This checks evaluation of that finite space; it is not a
mesh-convergence result.

The physical gap-difference CI control uses the original mesh's complete
outer-corner oscillatory space, with materials `4` and `9`, gap `.02` and
radial/angular orders 80/12. Fourth-order stencils at `3e-5` and `1.5e-5`
retain the `3e-9` derivative criterion. Exactly identical complete physical
states are reused between the stencils. The separate coordinate and joint
material/gap controls include junction fields and the symmetric space.
