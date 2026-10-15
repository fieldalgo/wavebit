# Multimode incidence — several orders lit at once

:::{note}
This detailed guide documents the retained `wavebit.compat.legacy` interface and its measured scope. For the current problem/numerics lifecycle, start with [the common guide](../common/lifecycle.md). Advanced capabilities are preserved here with explicit compatibility calls; their limits are unchanged.
:::

**Code**: `src/wavebit/_common/conventions.py` — `lit_orders`, `incident_column`;
`src/wavebit/rcwa/_engine/solve_1d.py` — the two matchings; `src/wavebit/observations/power.py` —
`incident_power`; `src/wavebit/rcwa/_engine/scattering.py` — the SMM twins;
`src/wavebit/rcwa/_engine/engine.py` — `excitation`.
**Tests**: `tests/rcwa/test_excitation.py`.

```python
solver = wavebit.rcwa(..., excitation={0: 1.0, 1: 0.6 - 0.8j})     # 1D: by the x order
solver = wavebit.rcwa(..., excitation={(0, 0): 1.0, (1, 0): 0.5j})  # 2D: by the pair
```

The illumination arrives on more than one diffraction order, coherently, in one solve. The
weights are *spatial*: `psi` / `polarization` / `tetm` still say which polarization arrives,
and every lit order carries that same polarization, **each in its own s/p frame**: a weight
multiplies its order's `s_hat` / `p_hat`. On a 1D solve at `phi = 0` that is
`sign(kx) * y_hat` for s, so a TE beam on an order with `kx < 0` is polarized along `-y`. The
pure 1D `tetm` path has meant the same since 2026-09-27 (before, it multiplied `+y` into every
lit order, which is different light once the lit set straddles `kx = 0`: `sum(R)` 0.4933
against 0.3701 on `tests/rcwa/test_excitation.py`'s scene). Each weight is its beam's amplitude at
`x = y = 0` on the lit face, the cell centre -- the same weights `OrderExcitation(orders=...)`
takes, and the same light. That too dates from 2026-09-27: until then the weights were
referenced at the cell corner, which lit every order with `m + n` odd negated relative to the
rest, and the same scene's `sum(R)` moved a second time, 0.3701 -> 0.1953 (§6).

## 1. What it is for, and what it is not for

**Not** for lighting a single non-specular order. Exciting order `m` at `theta` is the *same
plane wave* as exciting the specular order at the angle whose `kx` matches — the order label is
a bookkeeping choice, not a physical one. Verified by the shift identity: solving at `theta`
and reading order `+1` against solving at `arcsin(sin(theta) + lambda/period)` and reading the
specular order agrees to **~1e-9** at every basis size tried, the residue being nothing but the
two solves keeping slightly different finite windows. So a one-order excitation buys a
relabelling. (It was 9e-1 → 2.5e-4 before §4's widening, which is itself a demonstration of
what the widening does: lighting order +1 now retains `[-truncation_order, truncation_order + 1]`, all but
one order of what the shifted-angle solve keeps.) One caveat on the `ky == 0` row: amplitudes
there are quoted in `sign(kx) * y_hat`, so an order whose `kx` is exactly `0` in one spelling
and `+-1e-17` in the other (a rounding) can come back with the opposite sign in the two, while
its efficiency agrees.

**For** the coherent superposition, which no single ordinary solve can express. Concretely,
coherent beam combining: a grating run as an N-to-1 combiner takes N beams at the angles the
grating equation fixes, which is exactly "order 0 and order +1 at once". A diffractive splitter
run backwards is the same problem.

Also **not** for composition. Light arriving from another periodic structure above is a
superposition on this grid, but joining two stacks is matrix algebra on their scattering
matrices (`ScatteringMatrix.cascade`, `connecting-algorithms.md` §9), not an excitation handed
to a solve.

## 2. What it buys

The alternative without this is one ordinary solve per beam, at the angle that makes each beam
specular, with the order indices realigned and the amplitudes added. That works — the solve is
linear in its incident field — and it is **as accurate**:

What it buys is the solve count — measured **2.95x** for three beams — and two bookkeeping
steps the caller no longer performs, each of which is a place to be silently wrong: computing
`arcsin(sin(theta) + m*lambda/period)` for every beam, and shifting each result by the right
number of order indices before adding.

It is also *more* accurate, but only because of the window widening in §4. On a window that
stayed centred on the specular order the two are a measured tie — the winner alternates
arbitrarily across basis sizes, and their difference is smaller than each one's own truncation
error from `truncation_order` 8 up. The widening is what turns the tie into a win.

## 3. Every lit order must propagate

An order past the light line carries no time-averaged z-flux. Lighting one is refused, by name
and one order at a time:

```
the excitation lights order 5, which is evanescent or carries no positive incident power in
this medium; a zero-power near-field excitation cannot be normalized as an incident plane wave.
```

This is not fussiness. Every efficiency is `outgoing power / incident power`, and an evanescent
input drives the structure while paying nothing into the denominator — so `R + T` exceeds one
on a passive stack (**8.87** on a thirteen-order excitation, measured before the refusal
existed) and "efficiency" quietly stops meaning *fraction of incident power*. An evanescent
excitation is a near-field source, which is a different problem than an incident plane wave and
one this library does not model.

The practical consequence is a ceiling on how many beams can be combined: an order propagates
only while `|kx| < n_inc`, so roughly `2 * period / wavelength` orders are available. A grating
fine enough to have few propagating orders cannot combine many beams — which is physics, not a
limit of this code.

## 4. The retained window widens around the lit orders

Every lit order needs `truncation_order` neighbours on each side, or a beam sitting off-centre is
worse resolved than it would be solved alone. So the retained set spans
`[m_min - truncation_order, m_max + truncation_order]` rather than staying centred on the specular order.

That set is carried as `(centre, half_width)` — `solver.order_window` — rather than as a range,
because `n_orders = 2*half_width + 1` then stays true and every size computation downstream is
untouched. The cost is that a symmetric window cannot straddle a half-integer centre, so an
odd span buys one order more than the union strictly needs.

**This is what makes the feature better than the workaround rather than equal to it.** Three
beams at orders 0, +1, +2, error on orders -2..+2 against a converged reference:

| truncation_order | union window | orders | three solves apart | orders |
| --- | --- | --- | --- | --- |
| 6 | 4.01e-1 | 15 | 3.48e-1 | 39 |
| 8 | **2.51e-1** | 19 | 4.85e-1 | 51 |
| 12 | **2.14e-2** | 27 | 3.07e-2 | 75 |
| 16 | **8.30e-3** | 35 | 1.00e-2 | 99 |
| 20 | **4.34e-3** | 43 | 5.05e-3 | 123 |
| 24 | **2.09e-3** | 51 | 2.47e-3 | 147 |

From `truncation_order` 8 up it wins every time, on a third of the orders. (Row 6 is the unconverged
regime — both errors are ~0.4 — and means nothing.) Without the widening the two are a measured
tie, which is what an earlier draft of this feature had.

The extra orders are cheap and get cheaper: `m_max - m_min` of them, a **fixed** number, against
an `n_orders` that grows. Measured **3.6–9.1%**, falling with N.

**The lit orders themselves must stay inside the declared `truncation_order`.** Without that rule the
widening would be unbounded — `{0: 1.0, 9: 0.5}` at `truncation_order=6` would quietly retain 22
orders — and `truncation_order` would stop meaning the orders being modelled. With it the widening is
capped at twice the declared half-width, and lighting an order outside is refused by name.

Everything the widening touches is a *size* or an *offset*: the four `kx` builders, the two
factorization entry points, and `lit_orders`'s mapping. **The factorization does not care where
the window sits** — a convolution matrix's `(i, j)` entry is the harmonic `i - j`, so two solves
at one `truncation_order` and very different angles build bit-identical slots (measured 0.0). Only the
width travels into it.

## 5. Where it enters

`delta` — the incident column — was built at four sites, one per matching, each as a unit
vector on the specular order:

| | |
| --- | --- |
| `solve_1d.py` stage 3 | the pure 1D TE/TM matching |
| `solve_1d.py::final_matching` | conical, 2D and first-order, which share it |
| `scattering.py::smm_1d` | the SMM twin of the first |
| `scattering.py::_smm_sp` | the SMM twin of the second |

All four take `incident_column` now. `lit_orders` resolves the declaration once at the
boundary, the way `incident_sp_amplitudes` resolves the polarization — and returns `None` when
nothing was declared, which is what keeps the ordinary path untouched.

**The keys are structure and the values are not.** Which entries of the column are non-zero
decides array shapes, so keys are read as Python integers and never compared against a traced
value; the amplitudes beside them are ordinary design variables and carry a gradient
(checked against a difference quotient, not against autodiff agreeing with itself).

## 6. Per-order polarization: through the matrix, not through this keyword

`excitation` carries an amplitude per order and **one polarization for all of them** — whatever
`psi` / `polarization` / `tetm` says, each order in its own s/p frame. That is a scope
decision, not a limitation of the formulation, and the capability is already reachable another
way.

**Through `solver.scattering`.** On the coupled paths the incident column is
`[s-block ; p-block]`, each of length `n_orders`, so filling both halves independently *is*
per-order polarization. The s half is the s amplitude itself; the p half is a **port
variable**, the p wave's magnetic field `-1j * (n/mu) * a_p` in the lit medium, and the p half
that comes back is the reflected magnetic field, which turns over for the up-going wave:

```python
solver = wavebit.rcwa(layer_algorithm='SMM', answers='amplitude', ...)
solver.solve()
S = solver.scattering
h = n_lit / mu_lit                            # the lit medium's admittance, Z0 units

column = np.zeros((2 * n_orders, 1), dtype=complex)
column[n_orders // 2] = 1.0                   # order 0: s, amplitude 1
column[n_orders + n_orders // 2 + 1] = -1j * h * 0.5j   # order +1: p, amplitude 0.5j
reflected = S.S11 @ column                    # both halves respond, from one solve
r_s = reflected[:n_orders]                    # the s amplitudes
r_p = reflected[n_orders:] / (1j * h)         # the p amplitudes, out of the port variable
```

Checked: with a *common* polarization this route and `excitation` agree to 6e-17 on the s
half, and to 4e-17 on the p half once it is read as above (`S11 @ delta_p` is `-r_p` per unit
of the column's own p variable, `S21 @ delta_p` is `t_p` times `(n/mu)_exit / (n/mu)_lit`;
`docs/reference/common/conventions.md` states it once for every path). What the matrix route does not give is
efficiencies — it hands back amplitudes and the caller normalizes — and it is SMM-only. The pure
1D path's matrix is the s-s (TE) or p-p (TM) block of this, in the same variables, and its ports
say which (`S.entry_port.channel` is `'te'` / `'tm'` there, `'sp'` here).

**The two doors agree (since 2026-09-27).** `excitation`'s weights and `OrderExcitation`'s on
the coherent-source path are both referenced at `x = y = 0`, the cell **centre** the public API
measures from, so one dict lights both doors with one light. On `tests/rcwa/test_excitation.py`'s
scene at `phi = 0`, `polarization=(1, 0)`, `{0: 1, 1: 0.6-0.8j}` gives `sum(R)` 0.151782
through either, 4.2e-16 apart (0.241294 lit from the glass). Until that day the ordinary door
referenced its weights at the cell **corner**, the Fourier origin the solves integrate from,
which lit order `(m, n)` with an extra `(-1)**(m + n)` relative to the rest -- different physics
once two orders of opposite parity are lit, not a global phase: the same dict gave 0.452653
there. Pinned in `tests/rcwa/test_centre_reference.py` (and, as the reciprocal-shift identity, in
`tests/rcwa/test_coherent_illumination.py`); the record is `docs/theory/rcwa/amplitude-sp-frame.md` §7.

**Why not in the keyword.** The s/p basis is *per order*: `order_azimuth` gives each retained
order its own azimuth, and `s_hat = z_hat x k_hat_par` is built from **that order's**
wavevector. So "s" for order 0 and "s" for order +1 point in different directions in space, and
`(a_s, a_p)` per order would mean each beam specified in its own frame rather than a common
one. That is this repo's most sign-error-prone corner — `order_azimuth`'s docstring records two
fixed bugs there (the `kx < 0` half-turn and the origin's `phi`), and the suite is blind to a
per-order amplitude sign by construction. Putting that machinery on the *input* side too needs
its own external oracle, which is a separate piece of work.

If it is ever added, S4's choice is the one to follow: a fixed Cartesian `'x'` / `'y'` frame
rather than per-order s/p, which sidesteps the frame question entirely.

## 7. What the reference solvers do

Surveyed against local copies of RETICOLO V10, S4 and fmmax.

| | multimode incidence | per-order polarization | how |
| --- | --- | --- | --- |
| **fmmax** | yes | yes | no incidence concept at all — it ships the S matrix and the caller fills a `(2n, k)` amplitude array. `amplitude_poynting_flux` computes flux *from the supplied amplitudes*, so the normalization question never arises |
| **S4** | yes | yes (`'x'` / `'y'`) | `SetExcitationExterior{{G-index, pol, {re, im}}, ...}`, documented as a **"low level"** escape hatch beside `SetExcitationPlanewave`. One list covers front-forward and back-backward, so it spans both sides |
| **RETICOLO** | no | no | enumerates fixed incidences (TE/TM x top/bottom) and reports each; `einc = [a_TE, a_TM]` is one wave's polarization |
| **wavebit** | yes, `excitation=` | through `solver.scattering` | a keyword for the common case, the matrix for the rest |

Three things follow.

**Both peers that support it reached it through the matrix**, and one of them has no incidence
concept at all. That is the same conclusion this repo arrived at from the other direction: the
scattering matrix is the general answer, and `excitation` is the convenience on top of it for
the case where efficiencies are wanted back.

**S4 calls its version low level and keeps a plane-wave call beside it.** The split here is the
same shape — `excitation` is the high-level door and returns efficiencies; the matrix is the low
one and returns amplitudes.

**RETICOLO cannot be the oracle for any of this**, which matters because it is the oracle this
repo leans on most. The checks in `tests/rcwa/test_excitation.py` are internal (the SMM cascade
against the ETM matching) for that reason; fmmax or S4 would have to be the external one.

## 8. Incident power and real angles

The incident order's normal wavevector is read from its half-space mode. It is
`n_inc*cos(theta)` for a real angle in a lossless positive-index medium. For absorbing or
negative-index real-angle incidence, use the [real-angle
construction](../../reference/common/conventions.md#real-angle-incidence); the same formula with complex `n_inc` is
not its normal wavevector.

Each illuminated order is weighted by its own real power admittance: `Re(kz/mu)`
for electric s amplitudes and `Re(kz/epsilon)` for magnetic p amplitudes, with the
electric-to-magnetic p scale included. The incident power is the sum of these
weights times the squared amplitude magnitudes. With no explicit `excitation`,
the single central order is illuminated.

For real transverse wavevectors, finite-stack absorption is
`power_entering - T.sum()`. Incident/reflected interference is included in
`power_entering`, so replacing it by `1-R.sum()` requires a lossless incident medium.
