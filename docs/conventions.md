# Conventions — the contract

Written: 2026-08-12, for the first release. This page is the normative statement of every
sign, axis, name and unit a user's script can depend on. Each row is pinned by a test; the
measurement documents behind the choices are linked at the end.

If a future change would falsify a sentence on this page, that change is breaking, whatever
the diff looks like.

---

## Time and loss

| | convention |
| --- | --- |
| time factor | `e^{-iωt}` — a physical field is `Re[E(r) e^{-iωt}]` |
| loss | `Im(n) > 0`, `Im(eps) > 0` — indices as databases tabulate them (`n + 1j*k`) |
| gain | `Im(n) < 0`; a solve may then legitimately return `R + T > 1` (the `UnphysicalResultWarning` is a false positive there — filter it) |

Same statement as fmmax's; RETICOLO's efficiencies and amplitudes agree with this
convention with no conversion (pinned to 4e-14 by the RETICOLO goldens). All complex
amplitudes this library returns are in this convention; results quoted from an
`e^{+jωt}` source (older papers, meent-era scripts writing `n - jk`) are the complex
conjugates of ours.

## Geometry and propagation

Layers are listed **top to bottom**, and `+z` points down, into the stack: `ucell[0]` is
the layer light meets first, `n_top` the medium it arrives from, `n_bot` the medium it
exits into. A wave travelling `+z` carries `exp(+i*k0*kz*z)` with `Re(kz) > 0`; decay
toward `+z` is `Im(kz) > 0`.

"Top/bottom" name the *scene*, not the illumination: light enters from the top today, and
the parameter slot for back-side incidence (`incidence_side`) is reserved so that adding
it will not rename anything on this page.

Within a layer the array follows the image convention: row `0` of `ucell` is `y = 0`, for
every builder and for the reconstructed field cell alike. The flip that renders a field
the conventional way up belongs to plotting (`field_plot`), not to the data.

## Units and angles

`wavelength`, `period`, `thickness` and the smooth-path shape coordinates (`cx`, `lx`,
…) share **one length unit of your choosing** — nothing in the solver assumes nm. Shape
coordinates are absolute lengths, not fractions of the period.

All angles are **radians**.

| angle | meaning |
| --- | --- |
| `theta` | polar incidence angle, measured from the stack normal in the incidence medium |
| `phi` | azimuth of the plane of incidence, from the `+x` axis |
| `psi` | polarization: the angle of **E** from the p direction in the plane transverse to **k**. `psi = pi/2` is pure s (TE-like), `psi = 0` pure p (TM-like) |
| `pol` (1D paths) | `0` = TE, `1` = TM — an alias for `psi = pi/2` / `0` on the pure path |

## Diffraction orders

The retained orders are `m = -fto_x..fto_x`, `n = -fto_y..fto_y`, with in-plane
wavevectors `kx_m = n_top·sinθ·cosφ + m·λ/Λx` (and correspondingly `ky_n`). Per-order
arrays have shape `(ff_y, ff_x)` — **`y` first, the transpose of the `fto=[x, y]` input
order** — with the incident order at the centre `[fto_y, fto_x]` and `m` increasing along
the last axis.

`result.order(m, n)` indexes all per-order quantities by the order indices themselves, so
user code never needs the previous paragraph.

## The result surface

**Uppercase is power, lowercase is amplitude.**

| name | meaning |
| --- | --- |
| `R`, `T` | per-order diffraction efficiency: power returned to the incidence medium / leaving through the far side, as a fraction of incident power |
| `R_s`, `R_p`, `T_s`, `T_p` | the same, resolved by outgoing polarization; `R = R_s + R_p` |
| `r_s`, `r_p`, `t_s`, `t_p` | complex **Fresnel amplitudes**: E-field ratios against a unit incident E-field, directly comparable to the textbook Fresnel/Airy coefficients (pinned to 1e-12 against them, absorbing media included) |
| `res`, `res_te_inc`, `res_tm_inc` | the `psi`-specified incidence, plus the pure-s and pure-p variants the coupled solves produce for free; top-level `R`/`T` average the two variants when no `psi` solve exists (unpolarized illumination) |

Amplitude fine print, stated once so it never has to be discovered:

- the p sign convention is the one where a single interface at normal incidence gives
  `r_p = (n2 - n1)/(n2 + n1) = -r_s`;
- phase reference: `r` at the top interface, `t` at the bottom one;
- with an absorbing **superstrate**, incident power is convention-dependent, so `R`/`T`
  are not comparable to 1 — the amplitudes are the meaningful quantities there. Concretely
  so: the pure-1D and coupled paths normalize that undefined quantity *differently* (TM
  efficiencies differ by exactly `|n_top|²·Re(n_top⁻²)`; TE by exactly 1), which is pinned
  as deliberate notice rather than hidden — compare `r`/`t`, never `R`/`T`, in that regime;
- there are no raw "boundary variables" in the public surface. (Before the first release
  the p amplitudes were exposed as magnetic-field boundary variables needing a
  `/(-i*n_top)` footnote; they were renamed and normalized outright, with no aliases —
  including the meent-era efficiency names `de_ri`/`de_ti`.)

## Where each row is pinned

| rows | tests |
| --- | --- |
| time/loss, amplitudes | `tests/test_lossy_media.py` (Airy oracle, absorbing media both sides) |
| order layout, order-index sign | `tests/references/test_reticolo_golden.py` (mirrors orders if wrong, conserving energy) |
| result naming, `order()`, delegation | `tests/test_result_api.py` |
| square-root branches behind `kz` | `tests/test_branch_selection.py` |
| the whole surface, per backend | `tests/test_golden_legacy_parity.py` |

Why these choices: [`time-convention-audit.md`](time-convention-audit.md) (the branch
rules and the flip residue), [`formulation.md`](formulation.md) (the full derivation in
this convention), [`branch-cuts.md`](branch-cuts.md) (single precision limits).
