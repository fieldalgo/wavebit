# 12. Reading results: uppercase R/T and lowercase r/t

:::{note}
The downloadable teaching scripts use the current physical Problem/Job API. Some explanatory snippets and figures retain earlier API spellings; compare the [current getting-started guide](../../getting-started/index.md) and [migration guide](../../migration/index.md). Historical measurement files and their source provenance remain unchanged; they are distinct from validation of the migrated scripts.
:::

The `result` returned by `solver.solve()` contains **power fractions by diffraction order** and **complex amplitude coefficients by diffraction order**. Uppercase `R` and `T` are the reflected and transmitted power in each order divided by incident power. Lowercase `r_s`, `r_p`, `t_s`, and `t_p` are complex amplitude coefficients expressing each reflected or transmitted order in its s and p polarization components.

## Reading reflectance and transmittance by order

A periodic structure divides reflected and transmitted light into diffraction orders. **Each element of `result.R` or `result.T` is the power carried by one order divided by the total incident power.** For example, `T=0.2` for an order means that 20% of the incident power leaves in that transmitted order.

Using the `result` from [the TE grating example in Chapter 4](quickstart.md), this section explains **how to read one order and how to sum all orders**.

### Reflectance and transmittance of one order

`result.order(m, n)` extracts the result for a specified diffraction order. `m` and `n` are the order indices along the two lattice directions; `n=0` in this 1D example. Read the zeroth order as follows:

```python
result = solver.solve()
zero = result.order(0, 0)
R_00 = zero.R
T_00 = zero.T
print(f"Zeroth-order reflectance: {R_00:.4f}")   # 0.2053
print(f"Zeroth-order transmittance: {T_00:.4f}")   # 0.3531
```

**A zeroth-order transmittance of 0.3531 means that about 35.31% of the total incident power is transmitted in the zeroth order.** This value excludes power transmitted in orders −1 and +1.

### Total reflectance and transmittance over all orders

Sum each array to obtain total reflectance and transmittance.

```python
R_total = result.R.sum()
T_total = result.T.sum()
print(f"Total reflectance: {R_total:.4f}")   # 0.2053
print(f"Total transmittance: {T_total:.4f}")   # 0.7947
```

In this example, orders −1, 0, and +1 carry transmitted power, with transmittances of approximately 0.2208, 0.3531, and 0.2208. The total is therefore:

```{math}
T_{\mathrm{total}}
=T_{-1,0}+T_{0,0}+T_{1,0}
\simeq 0.2208+0.3531+0.2208=0.7947.
```

About 79.47% of the incident power is transmitted in these three orders, and about 20.53% is reflected. On the reflected side, only the zeroth order carries power, so its reflectance equals the total reflectance. The figure shows **totals over all orders on the left** and **transmittance resolved by order on the right**.

```{figure} ../../figures/learn/learn/11-result.svg
:alt: TE incidence on the first grating: total reflectance and transmittance on the left, transmittance by order on the right
:figclass: learning-figure

Figure 12-1. Results for TE incidence on the first grating. Left: total reflectance and transmittance. Right: transmittance by order.
```

### Identifying the orders represented by array rows and columns

The result array’s **columns correspond to order `m`, and its rows to order `n`.** `orders_u` identifies the order index of each column, and `orders_v` that of each row.

In Chapter 4, `rcwa.Numerics(orders=(10, 0))` retains 21 orders from `m=-10` to `+10`, with `n=0`. The result array therefore has **1 row × 21 columns**.

```python
print(result.T.shape)    # (1, 21)
print(result.orders_u)   # m for each column: -10, -9, ..., 0, ..., 9, 10
print(result.orders_v)   # n for each row: [0]

T_00_by_index = result.T[0, 10]
T_00_by_order = result.order(0, 0).T   # The same value
```

Python indices start at zero, so the zeroth order is at `[0, 10]` in this array. With `result.order(0, 0)`, you can select by order index without calculating its array position yourself.

(diffraction-amplitude-coefficients)=
## Complex amplitude coefficients by order

For single-plane-wave incidence, four arrays provide the complex amplitude coefficients.

| Result array | Coefficient in each diffraction order |
| --- | --- |
| `result.r_s` | s component of the reflected wave |
| `result.r_p` | p component of the reflected wave |
| `result.t_s` | s component of the transmitted wave |
| `result.t_p` | p component of the transmitted wave |

`result.incident` holds the reflection and transmission result for the current incidence. The four properties above provide direct access to that result: for example, `result.r_s` and `result.incident.r_s` refer to the same array.

If `rcwa(answers="efficiency", ...)` requests only power results such as reflectance and transmittance, these four amplitude coefficients are not returned; all four values are `None`.

Each coefficient is a **ratio to the reference electric-field amplitude of the incident wave**. Calling that reference amplitude {math}`E_{\mathrm{in}}`, the transmitted s component of order {math}`(m,n)` is:

```{math}
t_s^{(m,n)}=\frac{E_{t,s}^{(m,n)}}{E_{\mathrm{in}}}.
```

When the incident reference amplitude is 1, the coefficient itself is the complex amplitude of that output component. Each output is expressed in an s/p basis aligned with its own order’s propagation direction. Reflection phases are referenced to the incident-side interface, and transmission phases to the opposite exit interface.

**A complex amplitude coefficient describes both magnitude and phase.** For example, if {math}`t_s^{(m,n)}=0.5e^{i\pi/4}`, that transmitted component has an electric-field amplitude half the incident reference amplitude and a phase difference of 45°. The complex modulus gives the amplitude magnitude; the complex argument gives the phase.

```python
import numpy as np

r_s = result.r_s            # Reflected s coefficients for every order
zero = result.order(0, 0)    # Result for order (0, 0)
amplitude = np.abs(zero.t_s)
phase_deg = np.angle(zero.t_s, deg=True)
```

Power depends on both electric and magnetic fields, so converting an amplitude coefficient to transmittance requires factors involving the medium and propagation direction. **The formula and its derivation** appear in {ref}`Chapter 13 <amplitude-to-efficiency>`. Read the already calculated per-order power fractions from `result.R` and `result.T`.

## Distinguishing incident and output polarization

The name of a result object indicates which **incident polarization** was used.

| Result object | Incidence |
| --- | --- |
| `result.incident` | The incidence configured in the solver |
| `result.s_inc` | Pure s polarization from the same incident direction |
| `result.p_inc` | Pure p polarization from the same incident direction |

Omitting `tetm`, `psi`, and `polarization` selects the default s-polarized incidence, whose result is stored in `result.incident`. `result.R` and `result.T` also describe this incidence.

In contrast, the suffixes `s` and `p` in `r_s`, `t_p`, `R_s`, and `T_p` describe **output polarization**. For example, `result.s_inc.t_p` is the complex amplitude coefficient of the transmitted p component under s-polarized incidence.

`s_inc` and `p_inc` are provided for calculations treating both polarizations with `answers="amplitude"` or `"field"`. If only one polarization is calculated with `tetm="te"` or `"tm"`, or if `answers="efficiency"` is selected, both objects are `None`; read the calculated incidence from `result.incident`.

## Zero-transmittance orders and evanescent waves

An order’s transmittance `T` is the power it delivers through the exit interface divided by incident power. **Zero transmittance alone does not imply that the order has no electric or magnetic field.**

First determine whether each order can propagate away from the surface. Consider a lossless isotropic exit medium with positive real refractive index {math}`n_{\mathrm{out}}`, and real tangential wavevectors {math}`k_x` and {math}`k_y`. For vacuum wavelength {math}`\lambda` and vacuum wavenumber {math}`k_0=2\pi/\lambda`, the normal wavevector {math}`k_z` satisfies the following equation. All three components {math}`k_x`, {math}`k_y`, and {math}`k_z` belong to **the particular output order**.

```{math}
k_z^2=(n_{\mathrm{out}}k_0)^2-k_x^2-k_y^2.
```

Since {math}`k_x` and {math}`k_y` differ among orders, the same exit medium can contain orders with real {math}`k_z` and others with purely imaginary {math}`k_z`.

| Right-hand side above | This order’s {math}`k_z` | Outgoing wave |
| --- | --- | --- |
| Positive | Nonzero real | Propagates away from the surface |
| Negative | Purely imaginary | Evanescent: amplitude decays exponentially away from the surface |
| Zero | Zero | Limiting case of propagation parallel to the surface |

**An evanescent wave has electric and magnetic fields near the surface.** In the lossless exit medium above, a single evanescent wave decaying into the exit medium carries zero time-averaged power normal to the surface. Its transmittance can therefore be zero while its complex amplitude coefficient is nonzero.

For example, **the zeroth order can have real {math}`k_z`, while the other orders, including ±1, have purely imaginary {math}`k_z`**. The zeroth order then propagates to the far field, while ±1 and other orders may appear as evanescent near-surface fields. The label “zeroth order” does not itself guarantee propagation; use the right-hand side of the equation to decide for each order.

RCWA includes evanescent components when enforcing the electric and magnetic boundary conditions at material interfaces. Omitting these components can leave the near-interface fields insufficiently represented and **make even the propagating zeroth-order reflectance and transmittance inaccurate**.

Do not exclude an order merely because its reflectance or transmittance is zero. As explained in [Chapter 10](fourier-orders.md), choose the retained Fourier orders by increasing the truncation until the quantity of interest is sufficiently stable.

## Absorption of the complete finite stack

The total power absorbed by the layers with specified thicknesses is **the net power entering at the incident-side interface minus the power leaving at the opposite interface**. This section expresses each power as a ratio to the incident wave’s power alone at the incident-side interface.

```python
R_total = result.R.sum()           # Reflected-wave power fraction
T_total = result.T.sum()           # Transmitted power fraction at the opposite side
P_enter = result.power_entering    # Net power fraction entering the stack
A = result.absorption             # P_enter - T_total
```

`power_entering` is the net entering-power fraction obtained after summing the incident and reflected electric and magnetic fields and calculating power from the total field. `absorption` sums absorption in all finite-thickness layers; it excludes power absorbed in the semi-infinite `top` and `bot` media.

For a lossless incident medium and a propagating incident wave, the incident/reflected cross terms cancel in the total power. Thus `P_enter = 1 - R_total`, and absorption is:

```{math}
A=1-R_{\mathrm{total}}-T_{\mathrm{total}}.
```

**Electric and magnetic fields still interfere in a lossless medium.** What cancels above is the cross term in the time-averaged normal power; the field interference pattern does not disappear.

For an absorbing incident medium, power cross terms can remain, so `1 - R_total` alone cannot give the net entering power. `result.power_entering` includes this contribution, and `result.absorption` subtracts `T_total` from it. **The interference term can have either sign; its sign does not indicate whether the medium absorbs or amplifies.**

The origin of interference terms in the total field and the comparison between lossless and lossy media are explained in {ref}`the net entering-power derivation in Chapter 13 <incident-reflected-power>`.
