# Guide

**The user manual.** Written for someone who knows the API exists and needs the contract for a
particular keyword — what it accepts, what it defaults to, and which spellings are refused.

The boundary against the two neighbouring folders is **audience**, not subject:

| | Reader | Answers |
| --- | --- | --- |
| [`../../tutorials/`](../../tutorials) | has not used the API yet | "how do I get started" — lessons, in order, read once and outgrown |
| **`guide/`** (here) | knows the API, needs the contract | "what does this keyword do, and what does it refuse" — looked up, not read through |
| [`../`](..) | knows the API, questions the design | "why is the default what it is" — measurements, cited from code comments |

Overlap in subject between the three is expected and fine. A page here may state a default
that `../` spends four thousand words justifying; it says what the default *is* and links.

## House style

Different from the documents in [`../`](..), deliberately. A page here carries **no `Written:`
date and no `Targets:` list** — those exist so a measurement can be cited and so a decision
can be dated, and a manual page is neither. It states what is true now.

What it shares with `../` is the outward-pointing rule: nothing here may name a path outside
the checkout, and `tests/test_docs_contract.py` fails the build on one. Every code block should
be runnable as written.

## Pages

| Page | Question it answers |
| --- | --- |
| [`scene.md`](scene.md) | What does every `rcwa()` keyword mean, what does it default to, and which spellings are refused? |
| [`backends.md`](backends.md) | What can NumPy, JAX and PyTorch each do here, which should I run, and how do I ask for a device or a precision? |
| [`fields.md`](fields.md) | How do I get a near field out, why is it off by default, and what shape does it come back in? |
| [`autodiff.md`](autodiff.md) | How do I differentiate a solve, and what does `jax.jit` need that `jax.grad` does not? |
| [`specs.md`](specs.md) | How do I describe a run as **data** rather than as a solver object — for a sweep, a queue, or a service? |
