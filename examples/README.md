# Examples

FieldWave applied to problems, in plain scripts. Everything here is pure FieldWave — no ML
frameworks, no vendored dependencies, nothing that needs a dataset or a GPU.

| Entry | Question it answers |
| --- | --- |
| [`block_1d.py`](block_1d.py), [`block_2d.py`](block_2d.py) | How do you describe a structure as drawn shapes instead of a raster? The `_verification.py` file beside each answers the follow-up: does that path agree with the raster it replaces? |
| [`levelset_2d.py`](levelset_2d.py) | How do you describe a structure whose shape *and topology* are still unknown? A level-set layer, and eight ascent steps on `dJ/d phi` — the shape derivative, straight out of autodiff. |

## How maintained is this?

**Covered by CI, as a smoke job only.** If something here disagrees with
[`../tests/`](../tests), the tests are right — the tests check the physics, and this only checks
that the examples still run.

The `examples smoke` job in [`../.github/workflows/tests.yml`](../.github/workflows/tests.yml)
runs every script from its own directory with `MPLBACKEND=Agg` and fails on a non-zero exit, on
`Traceback` in the output, or on an `UnphysicalResultWarning`. That last one matters because an
example can run to completion and still be returning R+T > 1. The whole set takes well under a
minute, which is why it is a separate job from the verification suite rather than part of it.

**It cannot tell you the numbers are right.** All five scripts import FieldWave, but the result
guards are NumPy-only, and `block_1d.py` and `levelset_2d.py` solve on PyTorch while
`block_2d.py` solves on JAX. So the `UnphysicalResultWarning` condition reaches **two scripts**,
the two `_verification.py` files. For the other three a green tick means the code ran, and says
nothing about what it printed.

Nor does it check what a verification example is *for*. Until 2026-08-11 the 2D one compared a
block `ucell` against a raster that was the vertical mirror of it — a defect that survived
because a mirrored structure radiates into the specular order exactly like the original, so the
two paths agreed to 1e-15 for as long as the only propagating order was the zeroth. Energy was
conserved throughout. If you add a case here, give it a propagating higher order.

The same check runs locally, from the repository root:

```bash
python tests/run_examples_smoke.py
```

It takes a path to narrow the run — `python tests/run_examples_smoke.py examples/block_2d.py`.
