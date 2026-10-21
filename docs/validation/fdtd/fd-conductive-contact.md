# C8: two exactly touching conductive films

Written: 2026-10-18

Targets: `src/wavebit/fdtd/_prepare/conductive_contact.py`,
`src/wavebit/fdtd/_prepare/conductive_stack.py`, `src/wavebit/fdtd/_prepare/checks.py`,
`src/wavebit/fdtd/_prepare/junction.py`, `tests/fdtd/test_fd_conductive_contact.py`,
`tests/fixtures/fdtd/conductive_contact.py`.

The local harmonic constitutive law of two passive conductors in one cell has
an exact realization as a DC conductivity plus one induced Debye relaxation.
This gives a passive cell update for an exactly touching planar pair. The local
constitutive and energy identities below are verified. Wave accuracy is
**not** fully accepted; see [Accuracy status](#accuracy-status) and
[issue #51](https://github.com/fieldalgo/wavebit/issues/51).

## Bounded contract

| Input | Contract |
| --- | --- |
| Geometry | Exactly two unturned, unranged voxel `Box` structures, uniform across two grid axes, sharing the third (normal) axis. Their computed physical face coordinates must be exactly equal. |
| Thickness | Each film is at least one grid cell thick, so a Yee averaging cell contains at most two materials. |
| Material | `ConductiveMaterial` (or `Conductor`) with positive isotropic real permittivity, nonnegative isotropic conductivity and unit permeability. Conductivity is in S/m in the public API. |
| Host | One plain positive, isotropic, lossless, nonmagnetic `Material`. |
| Boundaries | Both film supports stay inside the allowed interior region, clear of absorbing layers and normal domain walls. |
| Exclusions | Curved or rotated interfaces, three-material cells, lossy hosts, declared dispersion, initial fields, material or geometry differentiation and GPU qualification. |

There is no contact tolerance. A tiny positive gap or overlap is not silently
converted to contact. When the padded supports are disjoint, the existing
[separated-stack route](fd-conductive-stack.md) still applies. Build the two
boxes from one common, exactly representable face coordinate and inspect
`fdtd.check`. Non-default `Structure(junction=...)` rules do not select this
material path. Planning emits a `SubpixelAccuracyWarning` that names this page.

## Constitutive derivation

Use normalized solver units and `s = -i omega`. Let the two media be
`a + u/s` and `b + v/s`, with cell fractions `f` and `g = 1-f`. The tangential
response is the arithmetic average. For the normal component define

```text
A = f*b + g*a
B = f*v + g*u
1 / (f/(a+u/s) + g/(b+v/s))
    = eps_inf + sigma_dc/s + delta/(1+s*tau)
eps_inf = a*b/A
sigma_dc = u*v/B
tau = A/B
delta = f*g*(a*v-b*u)^2 / (A*B^2)
```

Properties of this decomposition:

- For `B > 0`, every storage and dissipation coefficient is nonnegative.
- If both conductivities vanish, the response reduces to the harmonic real
  permittivity.
- A pure cell keeps its original material.
- If `a*v = b*u`, the induced relaxation vanishes, though the DC conductivity
  may remain.
- Setting one conductivity to zero recovers the conductor–dielectric formula.

The split polarization stores `P_tilde = P - c*E`, where
`c = delta*dt/(2*tau+dt)`. The update uses `eps_inf + c`, the DC conductivity
and the trapezoidal Debye recurrence. One memory term covers the union of the
two films, including the two exterior host interfaces. A cut cell receives
exactly one relaxation, so its storage is never double-counted. The energy
observer uses `1/delta` and `c` explicitly. The conservative material CFL
contribution is the maximum local `f*g*(a*v-b*u)^2/(A^2*a*b)`, including the
exterior interfaces.

The prepared signature binds both original members, their order, geometry,
material values and the full memory set. Missing, foreign or stale memories
are rejected before lowering. The global field storage is counted once.

## Verified checks

`tests/fdtd/test_fd_conductive_contact.py` checks the actual memory advance and
injected current against the independent complex series/parallel law on all
three axes and five subcell placements. It also covers:

- zero loss, identical materials and equal relaxation ratios;
- wrong-coefficient controls and prepared-plan refusals;
- public SI material lowering and the allocated memory;
- the discrete energy budget, with physical storage derived by rational
  polynomial long division, verifying conductive plus Debye dissipation on
  NumPy, JAX and PyTorch.

These are local CPU checks. They are not evidence of general three-dimensional
wave accuracy or differentiation.

## Accuracy status

**Registered wave fixture.** `tests/fixtures/fdtd/conductive_contact.py` places
two films on a fixed transverse wavenumber at 35 degrees:

- eps 4 / sigma 8 above eps 2.25 / sigma 4, each 0.125 L0 thick;
- three wavelengths .9 / 1 / 1.1 L0, both polarizations;
- comparison against an independent continuum characteristic-matrix
  reference.

Its registered p-polarized 40/80/160-cell campaign, at grid phase .375, with
120 periods and time-step halving, did **not** pass as a whole. That result is
kept:

- **Reflection order.** The direct convergence order of the complex
  reflection coefficient is 2.50–2.63, above the pre-registered window
  [1.8, 2.3].
- **Transmission order.** 2.00–2.01, inside the window.
- **Time step.** Halving the time step passes on every channel.
- **Wide band.** The df = .1 pulse cohort fails its registered criteria.

The narrower df = .07 cohort was registered separately. It stopped at the same
reflection-order gate.

**Cross-tool comparison.** The same physical case at n80 (dx = L0/80,
dt = .2 dx/√3, 24-cell PML, 120 periods), maximum over the three wavelengths:

| | abs r error | abs t error | 60→120-period change, df .1 |
|---|---|---|---|
| Wavebit | 2.5e-4 | 1.8e-3 | 1.8e-6 |
| Meep 1.34.0, `eps_averaging` on | 2.5e-3 | 1.5e-3 | 2.4e-7 |
| Meep 1.34.0, `eps_averaging` off | 1.75e-2 | 1.6e-3 | 2.4e-7 |
| FDTDx 0.6.2 and 0.6.3, voxel staircase | 3.2e-2 | 1.9e-3 | 2.2e-7 |

Reading the table:

- **Errors at n80.** Wavebit's reflection error is the smallest of the three
  tools. Its transmission error is at the same level as the others.
- **Placement sensitivity.** The external results depend strongly on grid
  placement and smoothing; Meep changes by up to 4× under a half-cell shift.
  They are context, not a reference, and a similar error in another tool does
  not establish a common limit.
- **Wide-band residual.** The df = .1 late-time change is 5–12× larger in
  Wavebit. The external runs inject a soft current sheet while Wavebit uses its
  plane-wave source. Whether this is a solver defect or a property of the
  source was unresolved in that original comparison; see the follow-up below.

These comparisons were measured by the maintainers and have not been
independently reviewed.

**Open questions.** Both the reflection order and the wide-band residual are
tracked in [issue #51](https://github.com/fieldalgo/wavebit/issues/51). Until it
is resolved, validate spatial, time and boundary convergence for your own case.
No tolerance, golden value or acceptance window was changed for this route.

## Original-bandwidth follow-up (2026-10-22)

The current differentiated-current p launch was tested at the original
`df=.1, cutoff=1e-6`, using the same films, fixed transverse wavevector and
continuum reference. A separately registered asymptotic criterion passed on
both s/p polarizations and both grid phases .375/.125. This does not rewrite
the original coarse-grid failure or establish general wave support.

Each group used 40/80/160/320 cells at 60 periods, plus an n80 120-period
record, doubled PML and halved time step. Two additional n320 controls used
120 periods with 24 and 48 PML cells: 36 new CPU records in total. All retained
the original fixture and numerical budgets.

The asymptotic criterion requires a stable complex h² coefficient (within 5%
between the 80/160 and 160/320 h²+h³ fits), a resolved finest-pair order interval
inside [1.8, 2.3], and separate time/boundary uncertainty and half-step checks.
The original 40→80 order is still reported, even when outside that window.

For each of the six r/t channels, let e_n = |q_n − q_ref| and
d_ab = |q_a − q_b|. The registered noise proxy is
eta = |q80,T120 − q80,T60| + |q80,T120,PML48 − q80,T120,PML24|.
All errors and consecutive differences must exceed 1e-10, and
2 eta <= .1 min(e320, d160,320). The n80 half-step difference must be
<= .1 min(d80,160, d160,320). The fine-grid extension applies the same
2 eta_fine bound using the n320 time and PML pairs. Use max(eta, eta_fine)
in the interval [log2((e160−eta)/(e320+eta)),
log2((e160+eta)/(e320−eta))]; nonpositive arguments fail. All rows must
be stable with finite positive energy, monitor decay below 1e-5, complete
post-source qualification and no growth warning. Both phases must pass.

| Polarization / grid phase | Finest reflection-order interval, all wavelengths | Largest h² coefficient change | Result |
|---|---:|---:|---|
| s / .375 | 1.891–2.139 | 1.180% | PASS |
| s / .125 | 1.997–2.002 | .177% | PASS |
| p / .375 | 1.958–2.175 | 1.470% | PASS |
| p / .125 | 1.997–2.001 | .168% | PASS |

Transmission also passed. Every row had complete post-source evidence and met
the original finite-energy, stable-run and decay requirements. The additional
fine-grid time-plus-boundary uncertainty consumed at most .148 of its fixed
budget; using the larger coarse/fine uncertainty still passed the order window.
The high coarse-grid reflection order is consistent with cancellation of the
leading h² term and a pre-asymptotic h³ contribution, rather than loss of
second-order convergence. These observations cover these two phases and this
material/angle fixture only. They are new-source Linux JAX CPU evidence, not a
fresh external-solver run, other-platform validation or independent review.

See [PlaneWave acceptance](fd-planewave-acceptance.md) for source qualification
and the separate broad-pulse PML limitation. The original failed campaign and
its external-tool measurements above remain historical evidence. No material
algorithm, default boundary, golden or numerical tolerance changed in this follow-up.
