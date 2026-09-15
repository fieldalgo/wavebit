# Stable real radial enrichment of the pair potential space

The explicit corner family can add fixed radial exponents `0.55` and `0.95`
to `0.6` and `0.8`. This increases the radial trial space while keeping its
coordinates real and independent of material values. It does not establish
optical convergence or change ordinary NVF routing.

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
