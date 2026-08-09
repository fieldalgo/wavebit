# Examples

FieldWave applied to problems, in plain scripts. Everything here is pure FieldWave — no ML
frameworks, no vendored dependencies, nothing that needs a dataset or a GPU.

| Entry | Question it answers |
| --- | --- |
| [`rcwa/`](rcwa) | What do the textbook cases look like in FieldWave? Moharam and Lalanne's 1D TE/TM, conical and 2D gratings — the standard results any RCWA implementation is checked against, written out step by step for someone studying the method. |
| [`vector_1d.py`](vector_1d.py), [`vector_2d.py`](vector_2d.py) | How do you describe a structure as drawn shapes instead of a raster? The `_verification.py` file beside each answers the follow-up: does that path agree with the raster it replaces? |
| [`ocd/`](ocd) | Can FieldWave drive an optical critical-dimension fit, and which optimizer suits it? A sweep across five algorithms, from the published study. |

## What is not here any more

The two machine-learning examples — a Fourier neural operator for field prediction, and
metasurface design by reinforcement learning — were removed on 2026-08-10. Between them they
bundled forks of two third-party packages (`neuraloperator` and `sheeprl`), which came to **341
files, 69% of this repository**, none of it FieldWave code and none of it exercised by anything.

They remain in full in [meent](https://github.com/kc-ml2/meent), the project FieldWave was
forked from, under `examples/electric-field-fno/` and `examples/metasurface-design/` — which is
the right home for them: they belong to that lineage and to the paper published from it. They
are also still in this repository's git history.

Removing them was not a judgement on the work. It is that FieldWave is a solver, and carrying a
reinforcement-learning framework inside `examples/` made every repository-wide search answer for
somebody else's code.

## How maintained is this?

**Not covered by CI at the time of writing.** If something here disagrees with
[`../tests/`](../tests), the tests are right. The entries above are small and fast enough that a
smoke job is cheap — see the open decisions in
[`../docs/repository-layout.md`](../docs/repository-layout.md).
