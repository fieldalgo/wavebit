# Examples

Wavebit applied to problems, in plain scripts. No datasets or GPU are required.
Some examples use optional JAX/PyTorch backends.

Install `wavebit[plot]` for the FDTD image example; the shape examples use the
optional JAX/PyTorch backends.

| Entry | Question it answers |
| --- | --- |
| [`nvf_method_selection.py`](rcwa/nvf_method_selection.py) | How do I choose ordinary NVF, explicit Cartesian Li, or the experimental auxiliary pair solver for the same structure? |

| [`fdtd_multilayer_optimization.py`](fdtd/fdtd_multilayer_optimization.py) | How can three coating thicknesses and permittivities be optimized together while maintaining minimum lengths and checking the final spectrum independently? |
| [`fdtd_backends.py`](fdtd/fdtd_backends.py) | Evaluate one FDTD objective on NumPy/JAX/Torch and compare automatic derivatives with finite differences. |
| [`fdtd_bloch_gradient.py`](fdtd/fdtd_bloch_gradient.py) | Differentiate oblique plane power through both the source phase and the transverse Bloch boundary. |
| [`fdtd_dispersive_gradient.py`](fdtd/fdtd_dispersive_gradient.py) | Differentiate Lorentz oscillator strength while vacuum-end CPML absorbs outgoing waves. |
| [`fdtd_layer_optimization.py`](fdtd/fdtd_layer_optimization.py) | Optimize a coating thickness in a typed Stack with optional C1 averaging, then check smoothed and sharp designs at twice the resolution and against quarter-wave optics. |
| [`fdtd_frequency_planes.py`](fdtd/fdtd_frequency_planes.py) | Measure six frequency-domain fields and integrated slab R/T without keeping time histories. |
| [`fdtd_grid_gradient.py`](fdtd/fdtd_grid_gradient.py) | Differentiate spectral transmission through a native Yee grid with JAX; uses a graded absorber, not CPML. |
| [`fdtd_poynting.py`](fdtd/fdtd_poynting.py) | How do you measure instantaneous energy flow at a point with staggered E/H records? Saves a Sz time trace inside a film; requires the plot extra. |
| [`fdtd_record.py`](fdtd/fdtd_record.py) | How do you define a typed FDTD scene, record a conical pulse and save an Ex slice? Run `python fdtd_record.py --output /path/to/field.png`; add `--chi 0.7853981633974483` for circular incidence and `--component hy` to view the magnetic field. FDTD is experimental; see the [research-alpha scope](../docs/guide/fdtd/fdtd-alpha.md). |
| [`shapes_1d.py`](rcwa/shapes_1d.py), [`shapes_2d.py`](rcwa/shapes_2d.py) | How do you describe a structure as drawn shapes instead of a raster? The `_verification.py` file beside each answers the follow-up: does that path agree with the raster it replaces? |

[`rectilinear_contact.py`](rcwa/rectilinear_contact.py) selects exact Cartesian shape Li
and prints JIT transmission, gap derivatives, and material derivatives through contact.
It requires the JAX extra. Unlike the default NVF examples, it uses fixed zero-angle
rectangles wholly inside the cell.

[`nvf_contact.py`](rcwa/nvf_contact.py) keeps `factorization='nvf'` and uses the
rectangular circular projector for contact and material gradients, including a
frozen plan at contact. It requires JAX. See
[shape NVF projector details](../docs/guide/rcwa/parametric-modeling.md#complex-nvf-projection-for-rectangular-contacts).

[`nvf_curved_contact.py`](rcwa/nvf_curved_contact.py) exercises the same NVF options on
tangent ellipses and compares the contact derivative with an eager one-sided
difference. See [curved shape support](../docs/guide/rcwa/parametric-modeling.md#disjoint-circles-and-ellipses).

[`nvf_curved_painting.py`](rcwa/nvf_curved_painting.py) uses circular NVF on overlapping
curves and checks the live width derivative. The projector follows the polygon
facets used by the painting; see [scope and approximation limits](../docs/guide/rcwa/parametric-modeling.md#overlapping-circles-and-ellipses).

## How maintained is this?

**Covered by CI, as a smoke job only.** If something here disagrees with
[`../tests/`](../tests), the tests are right — the tests check the physics, and this only checks
that the examples still run.

The `examples smoke` job in [`../.github/workflows/tests.yml`](../.github/workflows/tests.yml)
runs every script from its own directory with `MPLBACKEND=Agg` and fails on a non-zero exit or
on `Traceback` in the output. This is a separate smoke job; runtime depends on the machine and installed backends.

**It cannot tell you the numbers are right.** A green tick means the code ran; it says nothing
about what was printed. It used to also fail on an `UnphysicalResultWarning`, but that check
was removed from the solver — see [`../README.md`](../README.md) for why.

Nor does it check what a verification example is *for*. Until 2026-08-11 the 2D one compared a
shape `structure` against a raster that was the vertical mirror of it — a defect that survived
because a mirrored structure radiates into the specular order exactly like the original, so the
two paths agreed to 1e-15 for as long as the only propagating order was the zeroth. Energy was
conserved throughout. If you add a case here, give it a propagating higher order.

The verification pair also stopped being able to lean on a bit-identity. They compared the
**block** path against a commensurate raster, and on such a grating the block path *was* the
raster's arithmetic — Li's rule over the same segments — so it agreed to the last bit while
saying nothing about whether either was right. That path was removed on 2026-08-26. The shape
path runs NVF off an analytic normal field instead, so the same commensurate geometry now gives
two answers that converge on each other rather than one number twice, and each file ends with
the truncation sweep that shows it.

**It cannot see a material read as the wrong quantity either**, and that is worth stating next
to the mirror, because it is the same shape of defect. `structure` is **relative
permittivity** everywhere — raster pixels, `Material(epsilon)`, `top` and `bot`. A refractive
index left in one of those slots is a legal medium, so the run exits 0, prints numbers and
conserves energy; only the structure is different. The pre-rename spellings (`ucell`, `n_top`,
`n_bot`, `n_base`, `n_index`) were refused by name for that reason until the refusals were
deleted on 2026-08-26 — nothing has been released, so there is no recipe in the world still
using them — and a *value* nobody renamed was always invisible here anyway.
`wavebit.eps_from_index(n)` is the conversion, and these scripts write it out at the point the
datasheet number arrives rather than folding it into a literal.

The same check runs locally, from the repository root:

```bash
python tools/run_examples_smoke.py
```

It takes a path to narrow the run — `python tools/run_examples_smoke.py examples/rcwa/shapes_2d.py`.
