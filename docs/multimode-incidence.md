# Multimode incidence — several orders lit at once

**Code**: `wavebit/_core/conventions.py` — `lit_orders`, `incident_column`;
`wavebit/_core/solve_1d.py` — `_incident_flux`, the two matchings;
`wavebit/_core/scattering.py` — the SMM twins; `wavebit/_core/engine.py` — `excitation`.
**Tests**: `tests/test_excitation.py`.

```python
solver = wavebit.rcwa(..., excitation={0: 1.0, 1: 0.6 - 0.8j})     # 1D: by the x order
solver = wavebit.rcwa(..., excitation={(0, 0): 1.0, (1, 0): 0.5j})  # 2D: by the pair
```

The illumination arrives on more than one diffraction order, coherently, in one solve. The
weights are *spatial*: `psi` / `polarization` / `tetm` still say which polarization arrives,
and every lit order carries that same one.

## 1. What it is for, and what it is not for

**Not** for lighting a single non-specular order. Exciting order `m` at `theta` is the *same
plane wave* as exciting the specular order at the angle whose `kx` matches — the order label is
a bookkeeping choice, not a physical one. Verified by the shift identity: solving at `theta`
and reading order `+1` against solving at `arcsin(sin(theta) + lambda/period)` and reading the
specular order agrees to **~1e-9** at every basis size tried, the residue being nothing but the
two solves keeping slightly different finite windows. So a one-order excitation buys a
relabelling. (It was 9e-1 → 2.5e-4 before §4's widening, which is itself a demonstration of
what the widening does: lighting order +1 now retains `[-truncation_order, truncation_order + 1]`, all but
one order of what the shifted-angle solve keeps.)

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
the excitation lights order 5, which is evanescent in the incidence medium -- it carries no
time-averaged power, so it cannot be part of an incident flux and every efficiency would be
measured against the rest.
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
| `solve_1d.py::final_matching` | conical, 2D and 4ff, which share it |
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
`psi` / `polarization` / `tetm` says. That is a scope decision, not a limitation of the
formulation, and the capability is already reachable another way.

**Through `solver.scattering`.** On the coupled paths the incident column is
`[s-block ; p-block]`, each of length `n_orders`, so filling both halves independently *is*
per-order polarization:

```python
solver = wavebit.rcwa(layer_algorithm='SMM', answers='amplitude', ...)
solver.solve()
S = solver.scattering

column = np.zeros((2 * n_orders, 1), dtype=complex)
column[n_orders // 2] = 1.0                 # order 0, s
column[n_orders + n_orders // 2 + 1] = 0.5j  # order +1, p
reflected = S.S11 @ column                   # both halves respond, from one solve
```

Checked: with a *common* polarization this route and `excitation` agree to 1.6e-17, so the two
are one contract seen from two sides. What the matrix route does not give is efficiencies — it
hands back amplitudes and the caller normalizes — and it is SMM-only.

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
repo leans on most. The checks in `tests/test_excitation.py` are internal (the SMM cascade
against the ETM matching) for that reason; fmmax or S4 would have to be the external one.

## 8. Incident power and phase angles

The incident order's normal wavevector is read from its half-space mode. It is
`n_inc*cos(theta)` for a real angle in a lossless medium. For absorbing real-angle
incidence, use the [phase-direction construction](conventions.md#phase-incidence);
the same formula with complex `n_inc` is not its normal wavevector.

Each illuminated order is weighted by its own real power admittance: `Re(kz/mu)`
for electric s amplitudes and `Re(kz/epsilon)` for magnetic p amplitudes, with the
electric-to-magnetic p scale included. The incident power is the sum of these
weights times the squared amplitude magnitudes. With no explicit `excitation`,
the single central order is illuminated.

For real transverse wavevectors, finite-stack absorption is
`power_entering - T.sum()`. Incident/reflected interference is included in
`power_entering`, so replacing it by `1-R.sum()` requires a lossless incident medium.
