# Examples

FieldWave applied to problems, in plain scripts. Everything here is pure FieldWave — no ML
frameworks, no vendored dependencies, nothing that needs a dataset or a GPU.

| Entry | Question it answers |
| --- | --- |
| [`rcwa/`](rcwa) | What do the textbook cases look like in FieldWave? Moharam and Lalanne's 1D TE/TM, conical and 2D gratings — the standard results any RCWA implementation is checked against, written out step by step for someone studying the method. |
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

The same check runs locally, from the repository root:

```bash
python tests/run_examples_smoke.py
```

It takes a path to narrow the run — `python tests/run_examples_smoke.py examples/rcwa`.
