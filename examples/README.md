# Examples

FieldWave applied to problems, in plain scripts. Everything here is pure FieldWave — no ML
frameworks, no vendored dependencies, nothing that needs a dataset or a GPU.

| Entry | Question it answers |
| --- | --- |
| [`rcwa/`](rcwa) | What do the textbook cases look like in FieldWave? Moharam and Lalanne's 1D TE/TM, conical and 2D gratings — the standard results any RCWA implementation is checked against, written out step by step for someone studying the method. |
| [`block_1d.py`](block_1d.py), [`block_2d.py`](block_2d.py) | How do you describe a structure as drawn shapes instead of a raster? The `_verification.py` file beside each answers the follow-up: does that path agree with the raster it replaces? |

## How maintained is this?

**Not covered by CI at the time of writing.** If something here disagrees with
[`../tests/`](../tests), the tests are right. The entries above are small and fast enough that a
smoke job is cheap; it has not been set up yet.

If one is added, it should fail on an `UnphysicalResultWarning` as well as on a non-zero exit.
An example can run to completion and still be returning R+T > 1.
