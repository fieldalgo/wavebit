# Examples

FieldWave applied to problems, in plain scripts. Everything here is pure FieldWave — no ML
frameworks, no vendored dependencies, nothing that needs a dataset or a GPU.

| Entry | Question it answers |
| --- | --- |
| [`rcwa/`](rcwa) | What do the textbook cases look like in FieldWave? Moharam and Lalanne's 1D TE/TM, conical and 2D gratings — the standard results any RCWA implementation is checked against, written out step by step for someone studying the method. |
| [`vector_1d.py`](vector_1d.py), [`vector_2d.py`](vector_2d.py) | How do you describe a structure as drawn shapes instead of a raster? The `_verification.py` file beside each answers the follow-up: does that path agree with the raster it replaces? |
| [`ocd/`](ocd) | Can FieldWave drive an optical critical-dimension fit, and which optimizer suits it? A sweep across five algorithms, from the published study. |

## How maintained is this?

**Not covered by CI at the time of writing.** If something here disagrees with
[`../tests/`](../tests), the tests are right. The entries above are small and fast enough that a
smoke job is cheap; it has not been set up yet.
