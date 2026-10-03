# Examples

Wavebit applied to problems, in plain scripts. Everything here is pure Wavebit — no ML
frameworks, no vendored dependencies, nothing that needs a dataset or a GPU.

| Entry | Question it answers |
| --- | --- |
| [`nvf_method_selection.py`](nvf_method_selection.py) | How do I choose ordinary NVF, explicit Cartesian Li, or the experimental auxiliary pair solver for the same structure? |
| [`shapes_1d.py`](shapes_1d.py), [`shapes_2d.py`](shapes_2d.py) | How do you describe a structure as drawn shapes instead of a raster? The `_verification.py` file beside each answers the follow-up: does that path agree with the raster it replaces? |

[`rectilinear_contact.py`](rectilinear_contact.py) selects exact Cartesian shape Li
and prints JIT transmission, gap derivatives, and material derivatives through contact.
It requires the JAX extra. Unlike the default NVF examples, it uses fixed zero-angle
rectangles wholly inside the cell.

[`nvf_contact.py`](nvf_contact.py) keeps `factorization='nvf'` and uses the
rectangular circular projector for contact and material gradients, including a
frozen plan at contact. It requires JAX. See
[shape NVF projector details](../docs/parametric-modeling.md#complex-nvf-projection-for-rectangular-contacts).

[`nvf_curved_contact.py`](nvf_curved_contact.py) exercises the same NVF options on
tangent ellipses and compares the contact derivative with an eager one-sided
difference. See [curved shape support](../docs/parametric-modeling.md#disjoint-circles-and-ellipses).

[`nvf_curved_painting.py`](nvf_curved_painting.py) uses circular NVF on overlapping
curves and checks the live width derivative. The projector follows the polygon
facets used by the painting; see [scope and approximation limits](../docs/parametric-modeling.md#overlapping-circles-and-ellipses).

## How maintained is this?

**Covered by CI, as a smoke job only.** If something here disagrees with
[`../tests/`](../tests), the tests are right — the tests check the physics, and this only checks
that the examples still run.

The `examples smoke` job in [`../.github/workflows/tests.yml`](../.github/workflows/tests.yml)
runs every script from its own directory with `MPLBACKEND=Agg` and fails on a non-zero exit or
on `Traceback` in the output. The whole set takes well under a minute, which is why it is a
separate job from the verification suite rather than part of it.

**It cannot tell you the numbers are right.** A green tick means the code ran; it says nothing
about what was printed. It used to also fail on an `UnphysicalResultWarning`, but that check
was removed from the solver — see [`../docs/result-guards.md`](../docs/result-guards.md) for why.

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
python tests/run_examples_smoke.py
```

It takes a path to narrow the run — `python tests/run_examples_smoke.py examples/shapes_2d.py`.
