# Examples

Wavebit applied to problems, in plain scripts. Everything here is pure Wavebit — no ML
frameworks, no vendored dependencies, nothing that needs a dataset or a GPU.

| Entry | Question it answers |
| --- | --- |
| [`block_1d.py`](block_1d.py), [`block_2d.py`](block_2d.py) | How do you describe a structure as drawn shapes instead of a raster? The `_verification.py` file beside each answers the follow-up: does that path agree with the raster it replaces? |

## How maintained is this?

**Covered by CI, as a smoke job only.** If something here disagrees with
[`../tests/`](../tests), the tests are right — the tests check the physics, and this only checks
that the examples still run.

The `examples smoke` job in [`../.github/workflows/tests.yml`](../.github/workflows/tests.yml)
runs every script from its own directory with `MPLBACKEND=Agg` and fails on a non-zero exit, on
`Traceback` in the output, or on an `UnphysicalResultWarning`. That last one matters because an
example can run to completion and still be returning R+T > 1. The whole set takes well under a
minute, which is why it is a separate job from the verification suite rather than part of it.

**It cannot tell you the numbers are right.** All four scripts import Wavebit, but the result
guards are NumPy-only, and `block_1d.py` solves on PyTorch while `block_2d.py` solves on JAX. So
the `UnphysicalResultWarning` condition reaches **two scripts**, the two `_verification.py`
files. For the other two a green tick means the code ran, and says nothing about what it
printed.

Nor does it check what a verification example is *for*. Until 2026-08-11 the 2D one compared a
block `structure` against a raster that was the vertical mirror of it — a defect that survived
because a mirrored structure radiates into the specular order exactly like the original, so the
two paths agreed to 1e-15 for as long as the only propagating order was the zeroth. Energy was
conserved throughout. If you add a case here, give it a propagating higher order.

**It cannot see a material read as the wrong quantity either**, and that is worth stating next
to the mirror, because it is the same shape of defect. `structure` is **relative
permittivity** everywhere — raster pixels, `Material(epsilon)`, `top` and `bot`. A refractive
index left in one of those slots is a legal medium, so the run exits 0, prints numbers and
conserves energy; only the structure is different. The retired spellings (`ucell`, `n_top`,
`n_bot`, `n_base`, `n_index`) are refused by name for that reason, but a *value* nobody
renamed is invisible here. `wavebit.eps_from_index(n)` is the conversion, and these scripts
write it out at the point the datasheet number arrives rather than folding it into a literal.

The same check runs locally, from the repository root:

```bash
python tests/run_examples_smoke.py
```

It takes a path to narrow the run — `python tests/run_examples_smoke.py examples/block_2d.py`.
