# 14. Small denominators, large responses: metals and the inverse rule

:::{note}
The runnable examples in this imported course explicitly use `wavebit.compat.legacy` to preserve the workflow behind its historical numerical records. For new work, use the [current getting-started guide](../../getting-started/index.md) and [API migration guide](../../migration/index.md). Historical measurements were not regenerated during this documentation import.
:::

[Chapter 13](metal-original-plots.md) showed power sums exceeding 1 and narrow, truncation-dependent features in gold nanohole and disk calculations. We now return to **constructing the material matrices**, before reflectance and transmittance are calculated.

The question is why entering only gold and air permittivities can create much larger responses inside a matrix. Reducing the inverse rule to one scalar denominator reveals the mechanism.

This chapter treats layers with constant scalar permittivity within each region. Separating geometry from materials also applies to three or more materials; the small-denominator eigenvalue analysis focuses on two. Its evidence is the code and analysis of Wavebit's 2026-09-27 pre-correction baseline `6fe4b7b` and correction `f90cf1d`.

## 1. Why NVF forms an inverse

NVF applies different Fourier multiplication rules to electric-field components tangential and normal to an interface, because different quantities are continuous there. [Chapter 3: Fourier multiplication rules](fourier-rules.md) introduced this distinction.

Write {math}`\left[\!\left[ f\right]\!\right]` for the **finite multiplication matrix** constructed from the Fourier coefficients of function {math}`f`. The required matrices are

```{math}
E=\left[\!\left[\varepsilon\right]\!\right],
\qquad
B=\left[\!\left[1/\varepsilon\right]\!\right],
\qquad
Q=B^{-1}
```

Here {math}`E` enters the direct rule and {math}`Q` the inverse rule. Matrix {math}`B` is unrelated to case label B; matrix {math}`E` does not denote an electric-field vector in this section.

The key is that **{math}`\left[\!\left[1/\varepsilon\right]\!\right]^{-1}` generally differs from {math}`\left[\!\left[\varepsilon\right]\!\right]` in a finite Fourier basis**. Taking a reciprocal, forming its Fourier matrix and then inverting does not equal forming the permittivity matrix directly.

For square truncation with per-axis order {math}`M`, the harmonic count is {math}`N=(2M+1)^2`: {math}`M=5` gives 121 and {math}`M=10` gives 441. Changing it changes both matrix size and the representation near interfaces.

## 2. Combining metal and dielectric creates a small denominator

First consider only positive, lossless permittivities. If both materials have positive {math}`1/\varepsilon`, a positively weighted combination cannot vanish.

For metals this changes. In the experiment, gold's permittivity at 807 nm is approximately {math}`-24.6+1.5i`, giving

```{math}
\frac1{\varepsilon_{\mathrm{Au}}}\approx-0.0405-0.0025i
```

Its real part has the opposite sign to air's {math}`1/\varepsilon=1`. Their weighted real parts can cancel. Loss prevents an exactly zero denominator but can still leave a very small magnitude.

### Separating geometry from materials

**Fourier-transform linearity lets us transform each material region's shape separately, multiply by its material value, and add the contributions.** This is the geometry/material separation used here.

We want the **Fourier coefficients and multiplication matrix** of spatial reciprocal permittivity {math}`1/\varepsilon(x,y)`. Call that matrix {math}`B`. Permittivity is constant within each region, although it may vary with wavelength.

- **Geometry:** each region's shape, size and position, encoded by a map equal to 1 inside and 0 elsewhere.
- **Materials:** permittivities such as dielectric {math}`\varepsilon_d` and metal {math}`\varepsilon_m`; the inverse rule uses their reciprocals.

#### ① Give each material a map: 1 in its region, 0 elsewhere

Suppose nonoverlapping dielectric and metal regions fill the unit cell. Define both maps over **the same entire cell**.

| Position | Dielectric map {math}`\chi_d(x,y)` | Metal map {math}`\chi_m(x,y)` |
| --- | --- | --- |
| Dielectric region | 1 | 0 |
| Metal region | 0 | 1 |

A function encoding region membership with 0 and 1 is an **indicator function**. {math}`\chi` is chi; subscripts {math}`d` and {math}`m` denote dielectric and metal.

**When transforming the dielectric map, set the metal region to 0; when transforming the metal map, set the dielectric region to 0.** Compute coefficients over the entire unit cell, including outside the selected region. Zero means that the selected material is absent from that map location; it does not mean the physical background has zero permittivity.

```{figure} ../../figures/learn/nvf/14-geometry-material.svg
:alt: In the same cell, the dielectric map is 1 inside the circle and 0 outside; the metal map is its complement. Fourier-transform both, multiply by their reciprocal permittivities, and sum to obtain total reciprocal-permittivity coefficients.
:figclass: learning-figure

Figure 14-1. Encode each material shape as a 0/1 map, multiply its Fourier coefficients by the material value, and add. Both maps share coordinates and the unit cell. The circle size is illustrative.
```

Weighting the maps by reciprocal permittivities reproduces the physical distribution:

```{math}
\frac1{\varepsilon(x,y)}
=\frac{\chi_d(x,y)}{\varepsilon_d}
+\frac{\chi_m(x,y)}{\varepsilon_m}.
```

At dielectric points, only the first term remains, giving {math}`1/\varepsilon_d`. At metal points, only the second remains, giving {math}`1/\varepsilon_m`. The equation preserves the original value at every position.

#### ② Linearity lets each map be transformed separately

Linearity means **transforming separately and adding equals adding first and transforming**. Spatially constant material values can also be taken outside the transform, so each term can be transformed independently.

The hat in **{math}`\widehat f(m,n)` denotes the {math}`(m,n)` Fourier coefficient of {math}`f`**. As in [Chapter 3](fourier-rules.md), {math}`m,n` are integer orders along the two lattice directions. The {math}`(0,0)` coefficient is the cell average. Below, the subscript {math}`m` in {math}`\chi_m` means metal, while {math}`(m,n)` in parentheses labels orders.

```{math}
\widehat{(1/\varepsilon)}(m,n)
=\underbrace{\frac1{\varepsilon_d}}_{\text{material value}}
 \underbrace{\widehat{\chi_d}(m,n)}_{\text{dielectric shape coefficient}}
+\underbrace{\frac1{\varepsilon_m}}_{\text{material value}}
 \underbrace{\widehat{\chi_m}(m,n)}_{\text{metal shape coefficient}}.
```

This **adds the two materials' contributions at each Fourier order**. Shape coefficients depend only on geometry; reciprocal permittivities are their weights. At fixed geometry, these coefficients can be reused as wavelength or material values change.

:::{dropdown} How does order (m,n) relate to a physical wavevector?

Let {math}`\mathbf b_1,\mathbf b_2` be the reciprocal basis vectors from [Fundamentals Chapter 6](../fundamentals/lattice.md). Order {math}`(m,n)` has wavevector {math}`m\mathbf b_1+n\mathbf b_2`. For a rectangular cell with periods {math}`L_x,L_y`,

```{math}
m\mathbf b_1+n\mathbf b_2
=\left(\frac{2\pi m}{L_x},\frac{2\pi n}{L_y}\right)
```

The corresponding basis function is {math}`e^{i(m\mathbf b_1+n\mathbf b_2)\cdot(x,y)}`, combining sine and cosine in a complex exponential. Order {math}`(0,0)` is spatially constant; in a rectangular cell, {math}`(1,0)` varies once along x. Oblique lattices retain labels {math}`(m,n)` while {math}`\mathbf b_1,\mathbf b_2` determine the physical directions.

{math}`\widehat{\chi_d}(m,n)` multiplies that basis function in the expansion of the dielectric map. This chapter normalizes the {math}`(0,0)` coefficient to the cell average.
:::

#### ③ Since the maps sum to 1, one map's coefficients determine the other's

Assume **exactly one of dielectric and metal occupies each point**. The maps are 1 and 0 in the dielectric, 0 and 1 in the metal. Thus **their sum is 1 everywhere**:

```{math}
\chi_d+\chi_m=1,\qquad \chi_m=1-\chi_d
```

It is **the sum of the spatial maps that equals 1**. Expand the resulting constant function in a Fourier series. For clarity, write the 1D sine/cosine version:

```{math}
1=\underbrace{1}_{\text{order 0}}
+\underbrace{0\cos x+0\sin x}_{\text{order 1}}
+\underbrace{0\cos 2x+0\sin 2x}_{\text{order 2}}
+\cdots
```

**Only the mean 1 remains; every spatially varying component is zero.** The same holds in 2D: the constant function has coefficient 1 only at {math}`(0,0)`, and zero at every other order.

By linearity, **adding the maps' coefficients at the same order gives that order's coefficient of the constant function**. Thus **zeroth-order coefficients sum to 1; coefficients at any other matching order sum to 0**. It does not mean their sum is 1 at every order.

For example, a dielectric {math}`(1,0)` coefficient of 0.2 implies a metal coefficient of −0.2. They sum to zero. Here “cancel” means **coefficients of the same order sum to zero**.

The metal coefficients therefore follow:

| Fourier component | Metal-map coefficient |
| --- | --- |
| Zeroth order, {math}`(m,n)=(0,0)` | {math}`\widehat{\chi_m}(0,0)=1-\widehat{\chi_d}(0,0)` |
| Other orders, {math}`(m,n)\ne(0,0)` | {math}`\widehat{\chi_m}(m,n)=-\widehat{\chi_d}(m,n)` |

**Why is the zeroth coefficient an area fraction?** Under this normalization, **it is the average over the whole cell**. The dielectric map is 1 in dielectric and 0 in metal, so its area-weighted average is

```{math}
\begin{aligned}
\widehat{\chi_d}(0,0)
&=\frac{1\times\text{dielectric area}+0\times\text{metal area}}
        {\text{total cell area}}\\
&=\frac{\text{dielectric area}}{\text{total cell area}}.
\end{aligned}
```

For example, **if dielectric occupies 30% of the cell**, then {math}`\widehat{\chi_d}(0,0)=0.3`. The remaining 70% is metal, giving {math}`\widehat{\chi_m}(0,0)=0.7`. As required, they sum to 1.

Zeroth-order coefficients encode **how much area each material occupies**. Equal area fractions can still have different nonzero-order coefficients when shape or arrangement differs.

One may transform the metal map directly, but **once the dielectric coefficients are known, a separate metal transform is unnecessary**.

#### ④ The maps' convolution matrices are K and I − K

Construct the dielectric map's **convolution matrix** from its Fourier coefficients. Insert {math}`f=\chi_d` into Chapter 3's {math}`C(f)` and call the result {math}`K`:

```{math}
K=C(\chi_d)=\left[\!\left[\chi_d\right]\!\right],\qquad
K_{pq}=\widehat{\chi_d}(m_p-m_q,\,n_p-n_q).
```

Row {math}`p` and column {math}`q` correspond to orders {math}`(m_p,n_p)` and {math}`(m_q,n_q)`. Since no permittivity has yet been applied, {math}`K` contains only dielectric geometry. The {ref}`convolution derivation <nvf-convolution>` and {ref}`matrix example <nvf-toeplitz>` in Chapter 3 explain order differences and the Toeplitz structure.

Arrange the coefficients of constant 1 in the same way. On the diagonal, the order difference is {math}`(m_p-m_q,n_p-n_q)=(0,0)`, supplying coefficient 1; off diagonal, nonzero-order coefficients supply 0. **That is why multiplication by constant 1 becomes the identity matrix {math}`I`.**

Transferring {math}`\chi_m=1-\chi_d` to matrices also uses **linearity**: both Fourier transformation and arranging coefficients into a matrix are linear. With the same order set and ordering, **subtracting functions before matrix construction equals subtracting their separately constructed matrices**.

```{math}
\begin{aligned}
\left[\!\left[\chi_m\right]\!\right]
&=\left[\!\left[1-\chi_d\right]\!\right]\\
&=\left[\!\left[1\right]\!\right]
 -\left[\!\left[\chi_d\right]\!\right]\\
&=I-K.
\end{aligned}
```

The second line applies linearity. The last calls the constant's matrix {math}`I` and the dielectric map's matrix {math}`K`. Weighting and adding material contributions gives

```{math}
\begin{aligned}
B
&=\underbrace{\frac1{\varepsilon_d}K}_{\text{dielectric contribution}}
 +\underbrace{\frac1{\varepsilon_m}(I-K)}_{\text{metal contribution}}\\
&=\frac1{\varepsilon_m}I+
\left(\frac1{\varepsilon_d}-\frac1{\varepsilon_m}\right)K.
\end{aligned}
```

**At fixed geometry and harmonic set, keep {math}`K` and rebuild {math}`B` by changing only {math}`\varepsilon_d,\varepsilon_m`.** The direct-rule matrix uses the same geometry: {math}`E=\varepsilon_dK+\varepsilon_m(I-K)`.

So far, we have assembled Fourier coefficients and material matrices. The next operation, {math}`Q=B^{-1}`, must invert the combined {math}`B`. It is not the sum of inverses of separate terms.

#### ⑤ Geometry/material separation also works for three materials

If three nonoverlapping materials fill the cell, define maps {math}`\chi_1,\chi_2,\chi_3`, each **1 in its own region and 0 in the other two**. Constant permittivity in each region gives, by the same linearity,

```{math}
\frac1{\varepsilon(x,y)}
=\frac{\chi_1}{\varepsilon_1}
+\frac{\chi_2}{\varepsilon_2}
+\frac{\chi_3}{\varepsilon_3},\qquad
K_a=\left[\!\left[\chi_a\right]\!\right]
```

```{math}
B=\frac{K_1}{\varepsilon_1}
+\frac{K_2}{\varepsilon_2}
+\frac{K_3}{\varepsilon_3}
```

Here {math}`a` indexes materials 1, 2 and 3. Since the maps sum to 1,

```{math}
K_1+K_2+K_3=I,\qquad K_3=I-K_1-K_2.
```

After computing two geometry matrices, subtraction gives the third. For more materials, the same sum of contributions applies provided regions do not overlap and fill the entire cell.

However, **the following explanation using a single geometry eigenvalue returns to two materials**. With two materials, only one independent geometry matrix {math}`K` is needed, and {math}`B` shares its eigenvectors. Three or more materials involve several independent geometry matrices; in a finite Fourier basis, they need not share eigenvectors. Distinguish the scope of material-matrix assembly from that of the eigenvalue analysis below.

### Finding small denominators in a two-material geometry matrix

#### An eigenvalue of K is the synthesized function's dielectric weight

We want to identify **which combinations of coefficients the material matrix nearly reduces to zero**, since inversion can greatly amplify those directions. First find eigenvectors of the dielectric-map convolution matrix {math}`K`:

```{math}
K\mathbf v_j=\kappa_j\mathbf v_j.
```

Here {math}`\mathbf v_j` is **the jth eigenvector**, and {math}`\kappa_j` its eigenvalue; {math}`\kappa` is kappa. Rows and columns of {math}`K` correspond to Fourier orders, so **each eigenvector entry can be read as the coefficient of its corresponding order**.

Keeping {math}`N` orders gives an {math}`N\times N` matrix {math}`K` and {math}`N` entries in each eigenvector. With exact indicator coefficients, {math}`K` is Hermitian, admitting {math}`N` orthonormal eigenvectors. **Distinguish the number of eigenvectors from the number of entries in one vector.**

For example, the plotted 3×3 matrix has three eigenvectors:

```{math}
K\mathbf v_1=\kappa_1\mathbf v_1,\qquad
K\mathbf v_2=\kappa_2\mathbf v_2,\qquad
K\mathbf v_3=\kappa_3\mathbf v_3.
```

Panels (a), (b) and (c) show functions synthesized from {math}`\mathbf v_1`, {math}`\mathbf v_2` and {math}`\mathbf v_3`, respectively.

**Fourier basis functions** turn these coefficients into spatial functions. They are sine/cosine functions, combined in complex notation as {math}`e^{i\theta}=\cos\theta+i\sin\theta`. This does not compute new physical propagation or resonances.

For example, normalize 1D position as {math}`t=x/L`, and order the rows and columns of {math}`K` as −1, 0, +1. **Choose one of the three eigenvectors** and write {math}`\mathbf v=(a,b,c)^{\mathsf T}`. Entries {math}`a,b,c` belong to that one vector; they do not name three eigenvectors.

| Matrix position | Fourier order | Basis function | Eigenvector entry |
| --- | --- | --- | --- |
| First | −1 | {math}`e^{-i2\pi t}` | {math}`a` |
| Second | 0 | {math}`1` | {math}`b` |
| Third | +1 | {math}`e^{i2\pi t}` | {math}`c` |

Call **the sum of each entry times its corresponding basis function** {math}`u(t)`:

```{math}
u(t)=a\,e^{-i2\pi t}+b+c\,e^{i2\pi t}.
```

{math}`u` is not a separately solved unknown: it is **the Fourier synthesis of an already computed eigenvector**. Another eigenvector changes the coefficients and therefore the function. Synthesizing the three vectors gives {math}`u_1,u_2,u_3`. Distinguish the Fourier coefficients of map {math}`\chi_d` from eigenvector entries of the matrix {math}`K` constructed from that map.

The same procedure works in 2D. If {math}`\phi_p(x,y)` is the pth selected orthonormal Fourier basis function, the function synthesized from {math}`\mathbf v_j` is

```{math}
u_j(x,y)=\sum_p(\mathbf v_j)_p\,\phi_p(x,y)
```

This spatial representation lets us integrate **how much of its squared magnitude lies in the dielectric**.

**Why does the eigenvalue equal that weight?** Multiply the eigenvalue equation by {math}`\mathbf v_j^\dagger` and divide by {math}`\mathbf v_j^\dagger\mathbf v_j`. The dagger {math}`\dagger` denotes conjugate transpose:

```{math}
\kappa_j
=\frac{\mathbf v_j^\dagger K\mathbf v_j}{\mathbf v_j^\dagger\mathbf v_j}
=\frac{\displaystyle\int_{\text{dielectric region}}|u_j(x,y)|^2\,dA}
       {\displaystyle\int_{\text{entire cell}}|u_j(x,y)|^2\,dA}.
```

The numerator sums **squared magnitude inside dielectric**; the denominator sums it **over the whole cell**. Larger function values receive more weight. Since dielectric is part of the cell, the ratio lies between 0 and 1. The remaining metal weight is {math}`1-\kappa_j`.

- When {math}`\kappa_j` is near 0, most squared magnitude lies in metal.
- When {math}`\kappa_j` is near 1, most lies in dielectric.
- Intermediate values distribute it across both regions.

**In the figure, read these integrals as areas under a curve.** A 1D example makes the calculation visible: dielectric occupies the middle half of the cell, with metal on both sides. Retain only orders −1, 0 and +1 to form {math}`K`. Compute its three eigenvectors, synthesize each Fourier series and plot squared magnitude. In 1D, the integral's {math}`dA` becomes {math}`dx`.

```{figure} ../../figures/learn/nvf/14-geometry-eigenmodes.svg
:alt: The same 1D cell has dielectric in its central half and metal on both sides. Three eigenvector-synthesized squared-magnitude curves concentrate in metal, equally in both regions, and in dielectric. Blue dielectric integral fractions are about 5%, 50% and 95%, matching the eigenvalues.
:figclass: learning-figure

Figure 14-2. Eigenvectors of a 3×3 geometry matrix constructed from exact indicator coefficients. Curves show {math}`|u_j|^2`, the squared magnitude of each synthesized Fourier series. Blue integrates over dielectric, grey over metal; each total area is normalized to 1. Bars on the right show the same fractions. These are not electromagnetic fields of the gold nanostructure.
```

The **dielectric area fraction is 50% in every panel**. Yet (a), large mainly in metal, has about 5% dielectric weight; (c), concentrated in dielectric, has about 95%; (b) has about 50%. **Identical geometry can have different dielectric weights for different synthesized eigenvectors.** Area fraction counts all positions equally; this ratio weights them by squared function magnitude.

Here {math}`u_j` is **a function constructed from a geometry-matrix eigenvector**, not necessarily a physical resonance mode of the full structure. Do not read its {math}`|u_j|^2` ratio as an absorbed-energy fraction: permittivity and material loss have not yet entered.

:::{dropdown} Why does the matrix ratio equal the spatial-integral ratio?

In an orthonormal Fourier basis, distinct basis functions have zero overlap, and each squared-magnitude integral is 1. Therefore

```{math}
\int_{\text{cell}}|u_j|^2\,dA
=\sum_p |(\mathbf v_j)_p|^2
=\mathbf v_j^\dagger\mathbf v_j.
```

With exact coefficients, the indicator convolution matrix also equals

```{math}
K_{pq}=\int_{\text{cell}}\phi_p^*\,\chi_d\,\phi_q\,dA.
```

Multiplying by eigenvector coefficients on both sides and summing gives

```{math}
\begin{aligned}
\mathbf v_j^\dagger K\mathbf v_j
&=\int_{\text{cell}}\chi_d\,|u_j|^2\,dA\\
&=\int_{\text{dielectric}}|u_j|^2\,dA.
\end{aligned}
```

The final equality uses {math}`\chi_d=1` inside dielectric and {math}`\chi_d=0` outside. Dividing these two integrals gives the main equation. Exactly projected {math}`K` is Hermitian, with real eigenvalues between 0 and 1.

A 0/1 map does not require eigenvalues of only 0 or 1. **Finite matrix K multiplies by the indicator and then retains only the selected Fourier orders**. This differs from preserving the full spatial multiplication across every order.
:::

:::{dropdown} The plotted K and its three eigenvalues

With {math}`t=x/L`, the cell is {math}`-1/2\le t\le1/2` and dielectric is {math}`|t|<1/4`. Its 1D Fourier coefficients are

```{math}
\widehat{\chi_d}(\ell)=
\begin{cases}
1/2,&\ell=0,\\
\sin(\pi\ell/2)/(\pi\ell),&\ell\ne0.
\end{cases}
```

Ordering as −1, 0, +1 gives

```{math}
K=\begin{pmatrix}
1/2&1/\pi&0\\
1/\pi&1/2&1/\pi\\
0&1/\pi&1/2
\end{pmatrix}
```

with eigenvalues

```{math}
(\kappa_1,\kappa_2,\kappa_3)
=\left(\frac12-\frac{\sqrt2}{\pi},\frac12,\frac12+\frac{\sqrt2}{\pi}\right)
```

These are approximately 0.04984, 0.50000 and 0.95016. Choosing suitable overall phases, the synthesized functions can be written

```{math}
\begin{aligned}
u_1(t)&=\cos(2\pi t)-1/\sqrt2,\\
u_2(t)&=\sqrt2\sin(2\pi t),\\
u_3(t)&=\cos(2\pi t)+1/\sqrt2.
\end{aligned}
```

Each satisfies {math}`\int_{-1/2}^{1/2}|u_j(t)|^2\,dt=1`; its dielectric-interval integral equals its eigenvalue. The plotting script computes the eigenvectors and also checks that numerical spatial-integral ratios match their eigenvalues.
:::

#### Material values reveal cancellation between two terms

Because {math}`B` combines {math}`K` and {math}`I`, the selected vector is also an eigenvector of {math}`B`. The multiplier applied by {math}`B` is

```{math}
b_j=\underbrace{\frac{\kappa_j}{\varepsilon_d}}_{\text{dielectric contribution}}
    +\underbrace{\frac{1-\kappa_j}{\varepsilon_m}}_{\text{metal contribution}}
```

Spatial indicator values were 0 or 1. Here the weights {math}`\kappa_j` and {math}`1-\kappa_j` instead describe how the synthesized squared magnitude is distributed between regions.

Insert numbers to see cancellation. Using the 807 nm approximations—air {math}`1/\varepsilon_d=1` and gold {math}`1/\varepsilon_m\approx-0.0405-0.0025i`—choose {math}`\kappa_j=0.04` for illustration:

```{math}
b_j\approx 0.04\times1+0.96\times(-0.0405-0.0025i)
=0.00112-0.0024i.
```

The dielectric contribution +0.04 nearly cancels the metal contribution of about −0.0389. Including the imaginary part leaves {math}`|b_j|` near 0.00265. This illustrative choice does not assert that any particular harmonic count actually produced {math}`\kappa_j=0.04`.

The inverse rule uses {math}`Q=B^{-1}`. Its multiplier on the same eigenvector becomes

```{math}
q_j=\frac1{b_j},\qquad
|q_j|\approx\frac1{0.00265}\approx380
```

If {math}`B` strongly shrinks a vector in that direction, {math}`B^{-1}` must strongly expand it to undo the action. We have not replaced the whole matrix by one number; we have **followed its action along one eigenvector**.

Near cancellation in the denominator makes {math}`|q_j|` large. Call this **near-pole amplification** of the inverse rule. With loss in this experiment, it means a large finite response rather than strict divergence.

```{figure} ../../figures/learn/nvf/13-inverse-rule-pole.png
:alt: Inverse-rule magnitude versus dielectric weight kappa, rising sharply near 0.038. The direct-rule response lacks the same peak.
:figclass: learning-figure

Figure 14-3. The two-material gold/air expression versus {math}`\kappa`. This plot uses gold permittivity at 818 nm, so peak position and height differ slightly from the 807 nm example above. Horizontal is dielectric weight {math}`\kappa`, not wavelength; vertical is the matrix multiplier's magnitude. This is not the full structure's reflection/transmission spectrum. The dashed level near 51 is the inverse-amplitude cap chosen by this implementation.
```

Here gold's permittivity magnitude is about 26, but the inverse-rule response reaches about 400. **Exceeding the material permittivity alone does not establish an unphysical response**: real metal/dielectric composites can resonate strongly. The concern is amplification from finite-basis interface modes that shifts with truncation and destabilizes the calculation.

### A matrix pole and an optical spectral peak have different positions

Neglecting loss, the denominator approximately vanishes when

```{math}
\operatorname{Re}\varepsilon_m(\lambda)
\approx\varepsilon_d\left(1-\frac1{\kappa_j}\right)
```

Geometry and the retained harmonic set determine {math}`\kappa_j`; wavelength determines permittivity. Increasing harmonics can therefore move the wavelength of strong amplification.

| Case | Harmonics | Matrix-predicted wavelength | Observed spectral interval |
| --- | --- | --- | --- |
| B disk | 121 | About 845 nm | Around 828–860 nm; maximum power sum at 852 nm |
| B disk | 625 | About 754, 755, 776 nm | Around 755–765 nm; large amplification at 760 nm |
| B disk | 1225 | About 783, 796 nm | Reflectance kink at 788–792 nm |
| A hole | 1369 | About 812, 834 nm | Around 816–819 nm |
| A hole | 1681 | About 820, 832 nm | Narrow feature at 822.5–823 nm |
| A hole | 441 | About 831, 881 nm | 804–808 nm: about 25 nm from even the nearer prediction |

Internal amplification appears in the far-field spectrum only after coupling to propagation and boundary conditions throughout the stack. This table supports a possible causal mechanism, not exact resonance-wavelength prediction. The large discrepancy for A 441 is especially unexplained by this equation alone.

## 3. Why can a large inverse also spoil power balance?

A large inverse and {math}`R+T>1` are distinct. The preceding section explains amplification; assessing a passivity violation also requires the NVF material-matrix assembly.

Let {math}`C` be the Fourier matrix selecting normal directions, and {math}`\Delta=E-Q` the direct/inverse-rule difference. The implementation's symmetric assembly has the block form

```{math}
\mathcal{E}_{\mathrm{sym}}=
E-\frac12(\Delta C+C\Delta).
```

Here {math}`E` and {math}`Q` are block diagonal, with the same matrix for x and y. Matrix {math}`C` collects the normal projector's xx, xy, yx and yy blocks. This is not a scalar product for a single electric-field component.

Symmetrization helps retain Hermitian structure in lossless materials. It does not guarantee a positive-semidefinite absorption matrix with loss: the symmetric product of two positive matrices can have negative eigenvalues when they do not commute.

With time convention {math}`e^{-i\omega t}`, the Hermitian loss matrix is

```{math}
H_{\mathrm{loss}}=
\frac{\mathcal{E}-\mathcal{E}^{\dagger}}{2i}
```

This generally differs from taking each entry's imaginary part. A negative eigenvalue of {math}`H_{\mathrm{loss}}` means the discrete constitutive matrix permits negative absorption along some field direction. Coupling an actual incident wave to such a direction can produce {math}`R+T>1`.

Earlier diagnostics found minimum eigenvalues near −17 in one A condition and −25 in one B condition. The interpretation is that the large inverse-rule response amplifies nonpassivity introduced in assembly. **A negative eigenvalue does not mean every incidence produces a power sum above 1.**

Changing assembly to {math}`E-\Delta C` was also tested. It improved some intervals for A but gave {math}`R+T\approx1.81` elsewhere for B, so an assembly-order change alone was not accepted as a solution.

## 4. Which operation should change?

The argument has followed three steps.

1. Metal and dielectric reciprocal permittivities have opposite real signs and can nearly cancel in some finite-Fourier-basis modes.
2. Inverting that small value greatly amplifies the internal response. Geometry and truncation determine the wavelengths affected.
3. Symmetric NVF assembly supports lossless energy conservation but does not guarantee positive absorption in every field direction when loss is present.

The correction therefore targets **small singular values immediately before inversion**, rather than final {math}`R` or {math}`T`. [Chapter 15](metal-regularization.md) gives the actual correction, its effects and its limitations.
