# Guide

**The user manual.** Written for someone who knows the API exists and needs the contract for a
particular keyword — what it accepts, what it defaults to, and which spellings are refused.

The boundary against the two neighbouring folders is **audience**, not subject:

| | Reader | Answers |
| --- | --- | --- |
| [`tutorials/`](https://github.com/fieldalgo/wavebit/tree/develop/tutorials) | has not used the API yet | "how do I get started" — lessons, in order, read once and outgrown |
| **`guide/`** (here) | knows the API, needs the contract | "what does this keyword do, and what does it refuse" — looked up, not read through |
| [`docs/`](https://github.com/fieldalgo/wavebit/tree/develop/docs) | knows the API, questions the design | "why is the default what it is" — measurements, cited from code comments |

Overlap in subject between the three is expected and fine. A page here may state a default
that `../` spends four thousand words justifying; it says what the default *is* and links.

## House style

Different from the documents in [`docs/`](https://github.com/fieldalgo/wavebit/tree/develop/docs), deliberately. A page here carries **no `Written:`
date and no `Targets:` list** — those exist so a measurement can be cited and so a decision
can be dated, and a manual page is neither. It states what is true now.

What it shares with `../` is the outward-pointing rule: nothing here may name a path outside
the checkout, and `tests/test_docs_contract.py` fails the build on one. Every code block should
be runnable as written.

## Pages

### Getting started

| Page | Question it answers |
| --- | --- |
| [`what-is-wavebit.md`](what-is-wavebit.md) | What does this solve, what is it bad at, and what state is it in? |
| [`installation.md`](installation.md) | How do I install it, and why are the version floors so high? |
| [`quickstart.md`](quickstart.md) | What does a first solve look like, and how do I read the result? |

### Fundamentals and usage

| Page | Question it answers |
| --- | --- |
| [`fd-radiation.md`](fd-radiation.md) | How do continuum and Yee propagation differ for the general engine's FarField monitor, and which readings and derivatives are supported? |
| [`scene.md`](scene.md) | What is the stack made of, and what cell does it repeat in? |
| [`light.md`](light.md) | Which polarization arrives, on which orders, from which side? |
| [`numerics.md`](numerics.md) | Which factorization and normal field, and what do they cost? |
| [`fields.md`](fields.md) | How do I get a near field out, why is it off by default, and what shape does it come back in? |
| [`autodiff.md`](autodiff.md) | How do I differentiate a solve, and what does `jax.jit` need that `jax.grad` does not? |
| [`backends.md`](backends.md) | What can NumPy, JAX and PyTorch each do here, which should I run, and how do I ask for a device or a precision? |
| [`time-domain.md`](time-domain.md) | When is the *other* solver the right one, and what does a run in the time domain need that a harmonic one does not? |
| [`fd-differentiation.md`](fd-differentiation.md) | How do I differentiate the general FDTD engine -- material values, a ranged body's shape, both together -- and why might a traced run return NaN? |
| [`fd-metasurface.md`](fd-metasurface.md) | How do I solve a periodic metasurface cell with the general FDTD engine, read its diffraction orders, compare them with RCWA, and how fine a grid does it need? |

### Advanced

| Page | Question it answers |
| --- | --- |
| [`specs.md`](specs.md) | How do I describe a run as **data** rather than as a solver object — for a sweep, a queue, or a service? |

**Not here: the keyword tables.** Every keyword, its default and what it refuses is
[`../reference/keywords.md`](../reference/keywords.md) — a lookup, deliberately separate from
the pages that explain how to choose. That split is the one this folder gets wrong most easily:
a page here says what a choice *means*, and the reference says what the argument *is*.
