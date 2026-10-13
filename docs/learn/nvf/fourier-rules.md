# 03. Direct and inverse rules: which matrix is inverted?

:::{note}
The runnable examples in this imported course explicitly use `wavebit.compat.legacy` to preserve the workflow behind its historical numerical records. For new work, use the [current getting-started guide](../../getting-started/index.md) and [API migration guide](../../migration/index.md). Historical measurements were not regenerated during this documentation import.
:::

RCWA needs material matrices **to express pointwise multiplication of permittivity and electric field as an operation on vectors of Fourier coefficients**. We first examine what is being multiplied, then see why collecting the product by order gives a convolution matrix. Finally, we connect this to the difference between the direct and inverse rules at finite truncation.

## 1. Start with the product of permittivity and electric field

[Chapter 2](boundary-conditions.md) denoted scalar relative permittivity by {math}`\varepsilon` and normalized electric displacement by {math}`d=D/\varepsilon_0`. For one electric-field component {math}`E`, the following relation holds at each position:

```{math}
d(x)=\varepsilon(x)E(x).
```

First consider a distribution periodic along x at a fixed height within a layer. RCWA does not solve for individually stored values of {math}`E(x)` at every position; it solves for **the coefficients of the Fourier orders** making up the field. Our question is therefore:

> Given the Fourier coefficients of permittivity and electric field, how do we obtain each order's coefficient for their product, the electric displacement?

The field symbols below represent the periodic part. At oblique incidence, the common incident phase factor can be separated from both sides, leaving the same relation between material and field orders.

## 2. Multiply a single electric-field order by permittivity

Let the period be {math}`L` and the fundamental spatial wavenumber be {math}`g=2\pi/L`. Choose the following two functions just to examine multiplication. This is not a calculated solution for a physical structure.

```{math}
\varepsilon(x)=2+\cos(gx)
=2+\tfrac12 e^{igx}+\tfrac12 e^{-igx},
\qquad E(x)=e^{igx}.
```

Permittivity contains orders 0 and ±1, whereas the electric field contains only order +1. Multiplying the two functions in space gives

```{math}
\begin{aligned}
d(x)&=\left(2+\tfrac12 e^{igx}+\tfrac12 e^{-igx}\right)e^{igx}\\
&=\underbrace{\tfrac12}_{\text{order 0}}
+\underbrace{2e^{igx}}_{\text{order 1}}
+\underbrace{\tfrac12 e^{i2gx}}_{\text{order 2}}.
\end{aligned}
```

The electric field had one order, but its product with permittivity has three. This happens because **orders add when Fourier waves multiply**. For example, {math}`e^{-igx}e^{igx}=1`, so permittivity order −1 and electric-field order +1 contribute together to output order 0.

| Permittivity order and coefficient | Electric-field order and coefficient | Product order and coefficient |
| --- | --- | --- |
| Order −1, 1/2 | Order +1, 1 | Order 0, 1/2 |
| Order 0, 2 | Order +1, 1 | Order +1, 2 |
| Order +1, 1/2 | Order +1, 1 | Order +2, 1/2 |

Spatial variation in the material distribution creates connections between field orders in this way. Conversely, constant permittivity contains only order 0, so it multiplies every electric-field coefficient by the same constant.

(nvf-convolution)=
## 3. Collecting terms by output order gives convolution

Now suppose the electric field also contains several orders. In one dimension, write the permittivity coefficient at order {math}`\ell` as {math}`\widehat\varepsilon(\ell)` and the electric-field coefficient at order {math}`q` as {math}`\widehat E(q)`:

```{math}
\varepsilon(x)=\sum_{\ell}\widehat\varepsilon(\ell)e^{i\ell gx},
\qquad
E(x)=\sum_q\widehat E(q)e^{iqgx}.
```

Multiplying terms from the two series, **permittivity order ℓ and electric-field order q contribute to output order ℓ+q**. To obtain the coefficient of output order {math}`m`, collect every term satisfying {math}`\ell+q=m`. In other words, for each input order {math}`q`, the required permittivity order is {math}`\ell=m-q`.

```{math}
\widehat d(m)
=\sum_q\widehat\varepsilon(m-q)\,\widehat E(q).
```

This operation—**multiplying by the material coefficient at the difference between output and input orders, then summing over all input orders**—is convolution. It is different from multiplying coefficients with matching order numbers.

For example, to obtain output order +1 while retaining only input orders −1, 0, and +1,

```{math}
\widehat d(1)\approx
\widehat\varepsilon(2)\widehat E(-1)
+\widehat\varepsilon(1)\widehat E(0)
+\widehat\varepsilon(0)\widehat E(1).
```

Connecting input order −1 to output order +1 requires permittivity order +2; input order 0 requires permittivity order +1; input order +1 requires permittivity order 0. The approximation symbol reflects the omitted electric-field orders. If all omitted input coefficients are zero, the equality is exact for the selected output orders.

(nvf-toeplitz)=
## 4. Writing this sum as a matrix reveals a Toeplitz structure

Arrange both input and output orders as −1, 0, +1, and abbreviate the permittivity coefficients as {math}`a_\ell=\widehat\varepsilon(\ell)` for this example only. Writing the preceding sum as one row per output order gives

```{math}
\begin{pmatrix}
\widehat d(-1)\\ \widehat d(0)\\ \widehat d(1)
\end{pmatrix}
\approx
\underbrace{\begin{pmatrix}
a_0 & a_{-1} & a_{-2}\\
a_1 & a_0 & a_{-1}\\
a_2 & a_1 & a_0
\end{pmatrix}}_{C(\varepsilon)}
\begin{pmatrix}
\widehat E(-1)\\ \widehat E(0)\\ \widehat E(1)
\end{pmatrix}.
```

This is a **convolution matrix**. A row corresponds to the **output order** being calculated, and a column to an **input order** contributing to it. Each entry contains the permittivity coefficient at “output order − input order.”

A Toeplitz matrix has **constant entries along each diagonal**. In the matrix above, the main diagonal contains {math}`a_0`, the diagonal immediately above it contains {math}`a_{-1}`, and the one immediately below contains {math}`a_1`. This structure arises because shifting the row and column together leaves their order difference unchanged.

Even when the field retains only orders −1 through +1, the material coefficients are needed through ±2 to cover the order differences. **Distinguish the orders retained in the field from the range of material coefficients required to construct the matrix.** The matrix product also returns only the selected output orders. Components outside that range, such as output order +2 in the earlier example, are absent from the result vector of this 3×3 matrix.

(nvf-convolution-2d)=
## 5. In two dimensions, use the differences of both order indices

In two dimensions, an order is labeled {math}`(m,n)`. If row {math}`p` corresponds to output order {math}`(m_p,n_p)` and column {math}`q` to input order {math}`(m_q,n_q)`, the required material order is {math}`(m_p-m_q,n_p-n_q)`.

For multiplication by any periodic function {math}`f`, not only permittivity, define its **convolution matrix** as

```{math}
C(f)_{pq}=\widehat f(m_p-m_q,\,n_p-n_q).
```

- **Consecutive orders arranged in sequence in one dimension produce a Toeplitz matrix.** The preceding 3×3 example is of this type.
- **A rectangular order set in two dimensions, grouped by the orders along one direction, produces a block Toeplitz matrix with Toeplitz blocks.** The arrangement of blocks depends on order differences in one direction; entries within each block depend on order differences in the other.
- **A selected order set, such as circular truncation, still produces a convolution matrix.** Selecting rows and columns from the rectangular set may remove the regular block Toeplitz arrangement above. The formula for the entries is unchanged.

“Fourier multiplication matrix” means the same thing in this document. Double brackets, {math}`\left[\!\left[f\right]\!\right]`, also denote {math}`C(f)`. The matrix {math}`K=C(\chi_d)` used in [Chapter 14](metal-inverse-rule.md) arranges the coefficients of a **dielectric indicator function** instead of permittivity. That matrix contains only shape information; material values multiply it afterward.

(nvf-direct-inverse)=
## 6. Truncation makes the direct and inverse rules different

We have seen why multiplication in space becomes convolution of Fourier coefficients. Actual RCWA retains finitely many orders, so **the relation chosen to represent material and field discontinuities** also matters. In particular, when both factors are discontinuous at a material boundary, a simple finite convolution can represent the boundary conditions poorly.

Distinguish the matrices constructed from permittivity and reciprocal permittivity:

```{math}
A=C(\varepsilon),\qquad B=C(1/\varepsilon).
```

**Taking the reciprocal of a function before constructing its matrix generally differs from inverting the finite matrix after construction.** Thus {math}`B\ne A^{-1}`, or equivalently {math}`B^{-1}\ne A`. Here we assume the required reciprocals and inverses exist.

Applying the [boundary conditions in Chapter 2](boundary-conditions.md) explains why the two components should be constructed differently.

| Component | Quantity continuous at the boundary | Construction of the coefficient relation |
| --- | --- | --- |
| Tangential | Electric field {math}`E_t` | Construct {math}`\varepsilon E_t` directly: {math}`\widehat{\mathbf d}_t\approx A\widehat{\mathbf E}_t` |
| Normal | Normalized electric displacement {math}`d_n` | First construct {math}`\widehat{\mathbf E}_n\approx B\widehat{\mathbf d}_n`, then solve it for {math}`\widehat{\mathbf d}_n\approx B^{-1}\widehat{\mathbf E}_n` |

The bold, hatted symbols in the table are vectors of retained Fourier coefficients. For the normal component, we begin with **the relation that multiplies continuous electric displacement by reciprocal permittivity**. Solving in the direction from electric field to displacement therefore introduces {math}`B^{-1}`. This is the inverse rule discussed here.

```{figure} ../../figures/learn/nvf/03-rules.svg
:alt: The sequence of operations for the direct and inverse rules, with a zeroth-order hand calculation
:figclass: learning-figure

Figure 3-1. The sequence of operations for the direct and inverse rules, with a zeroth-order hand calculation
```

Represent a periodic function that is half {math}`\varepsilon=1` and half {math}`\varepsilon=4` using only order 0. Each matrix then becomes a 1×1 matrix containing a single average:

```{math}
A=\frac{1+4}{2}=2.5,\qquad
B=\frac{1+1/4}{2}=0.625,\qquad
B^{-1}=1.6.
```

Even for the same material distribution, {math}`A` and {math}`B^{-1}` differ. The difference occurs because **the two constructions use different product rules in the truncated Fourier space**. This hand calculation demonstrates that the operations differ; it is not a generally valid effective-permittivity answer for a two-dimensional structure.

The inverse rule assumes that the required reciprocals and matrix solves are well defined. When ε is very close to zero or a matrix is poorly conditioned, additional numerical care and convergence checks are necessary.

Which component is normal at a slanted edge or along a circle? Fixed x and y directions cannot provide the answer everywhere, so the next step is to bring in a normal direction that varies with position.
