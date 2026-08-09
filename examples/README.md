# Examples

FieldWave applied to outside problems. **CI does not run any of this**, so treat it as a
starting point rather than as a guarantee — if something here disagrees with
[`../tests_verification/`](../tests_verification), the tests are right.

| Entry | Question it answers |
| --- | --- |
| [`rcwa/`](rcwa) | What do the textbook cases look like in FieldWave? Moharam and Lalanne's 1D TE/TM, conical and 2D gratings — the standard results a new RCWA implementation is checked against, written out step by step for someone studying the method. |
| [`vector_1d.py`](vector_1d.py), [`vector_2d.py`](vector_2d.py) | How do you describe a structure as drawn shapes instead of a raster, and how does that path compare against the raster? The `*_verification.py` pair beside each is the comparison. |
| [`ocd/`](ocd) | Can FieldWave drive an optical critical-dimension fit, and which optimizer suits it? Reproduction material for the published sweep across five algorithms. |
| [`electric-field-fno/`](electric-field-fno) | Can a Fourier neural operator predict the field a solver would compute? Training and evaluation for the FNO surrogate. |
| [`metasurface-design/`](metasurface-design) | Can reinforcement learning design a metasurface with FieldWave in the loop? DQN and DreamerV3 agents. |

## Two things to know before relying on these

**The two ML examples vendor a fork of a third-party package** — `neuraloperator` under
`electric-field-fno/`, and `sheeprl` under `metasurface-design/dv3/`. Together they are 318 of
the files in this repository. They are forks rather than pins because each carries local
modifications; both are scheduled to move to their own repositories, and their licences
(MIT and Apache-2.0 respectively) travel with them.

**They also predate the rewrite.** `electric-field-fno/data.py` still imports
`fieldwave.on_numpy.modeler.modeling`, which no longer exists — that example cannot run as
committed. `rcwa/`, `vector_*.py` and `ocd/` are pure FieldWave and current.

Nothing in this directory is covered by CI, which is how the above went unnoticed. See
[`../docs/repository-layout.md`](../docs/repository-layout.md) for what is planned about it.
