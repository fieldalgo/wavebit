# Reference

The dry side. What every name *is*, rather than how to choose between them — for that, the
[user guide](../guide/README.md) takes one subject at a time.

| Page | What it holds |
| --- | --- |
| [Conventions](../conventions.md) | **Normative.** Every sign, axis, name and unit a script can depend on. A change that falsifies a sentence there is breaking by definition |
| [Keywords](keywords.md) | All 27 `rcwa()` keywords: what each accepts, what it defaults to, what it refuses |
| [Python API](api.md) | Generated from the docstrings, so it cannot drift from the code |

One more reference is not a page at all. `python -m wavebit.manifest` writes the capabilities
and constraints as JSON, for a program that has to know what the solver supports without
reading the source — see [the capability manifest](../capability-manifest.md).
