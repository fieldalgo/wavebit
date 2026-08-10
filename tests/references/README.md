# Reference cases

Validation against solvers that are not ours. The expected answer is known — agreement to
~1e-13 — so these are tests, not studies. What makes them their own directory is that
*producing* the reference needs software this project does not ship.

The arrangement is the same for every oracle: the expensive half runs once, by hand, on a
machine that has the tools; what CI checks on every push is the committed numbers.

**Regenerating a golden is not a way to make a failing test pass.** These numbers did not come
from this codebase, which is exactly what makes them worth having.

## The oracles

| Oracle | Reaches | Generator needs | Golden |
| --- | --- | --- | --- |
| **RETICOLO** (Li/Lalanne lineage) | diffraction *orders*; uniform layers with a complex 3x3 tensor — **but only up to one off-diagonal pair, at classical mounting**, see below | Octave + the V10 snapshot in `<a reference collection outside this repository>` | `golden_reticolo.npz`, `golden_reticolo_anisotropic.npz` |
| **fmmax** (Liu & Fan lineage, MIT) | uniform complex tensors, **patterned** tensors, **permeability** | the `fwref` env — see below | `golden_fmmax.npz`, `golden_fmmax_magnetic.npz` |
| **S4** (same lineage, C++) | real tensors, patterned | the S4 extension, already in the project env | `golden_s4.npz` |

A fourth oracle lives outside this directory: the Berreman 4x4 in
`tests/test_anisotropic_berreman.py` is an independent *formulation* — not RCWA at all — but our
own implementation, so it sits a tier below these three.

### Where RETICOLO stops being an oracle

Measured 2026-08-11, not taken from the manual. Feed a **real symmetric** tensor — Hermitian,
therefore lossless, therefore `R + T = 1` by theorem — through `interface/reti_aniso.m`:

| tensor | `delta0` | RETICOLO's own `R + T` |
|---|---|---|
| diagonal | 0 and 40 | 1 to 1e-15 |
| one off-diagonal pair | 0 | 1 to 1e-15 |
| one off-diagonal pair | 1 / 40 | off by 5.8e-6 / 6.3e-3 |
| two or three off-diagonal pairs | 0 | off by 2.5e-3 to 1.2e-2 |

The deviation is equal and opposite between TE and TM and does not move when `nn` goes from 0 to
8, so it is converged rather than truncated. It is not a convention difference on our side
either: a real symmetric matrix stays real symmetric under transposition and under any
permutation of axes, so nothing this bridge could misread makes the medium lossy. RETICOLO's own
`exemple_V9_0D_anisotrope.m` uses a single `xz` pair at `delta0 = 0`, inside the region that
holds.

So a **tilted magnetisation — the ordinary state of a real magnetised film — has no external
oracle here**, and `tests/test_anisotropic_berreman.py` is still the only judge of it. The cases
that would have covered it were written, measured, and removed rather than frozen; the
generator's energy guard is what caught them. This says nothing about RETICOLO's internals,
which have not been read — only about the numbers it returns through this bridge.

Two things to weigh when reading agreement between them:

* **fmmax generalises the S4 paper**, so those two share a formulation lineage. Agreement between
  *them* is weaker evidence than agreement with RETICOLO or Berreman.
* **RETICOLO validates individual orders**; the others are compared on totals per incident
  polarisation. For anisotropy that is deliberate — `<a survey of the available solvers>` records that
  S4 is silently wrong when polarisation decomposition and anisotropy are combined.

The GRCWA and TORCWA bridges in `interface/` predate all of this and have no frozen case. Neither
supports anisotropy.

## The `fwref` environment

fmmax carries its own JAX floor and the project env sits well ahead of it, so installing fmmax
alongside would drag JAX back and break the backend the suite tests. It gets its own environment,
used by nothing but the generators here:

```bash
conda create -y -n fwref python=3.11
conda run -n fwref pip install fmmax
```

Two consequences worth knowing before touching those generators:

* **The case list cannot live in the test module** the way the RETICOLO pair's does — neither env
  can import the other's module, since `fwref` has no `fieldwave` and the project env has no
  `fmmax`. It lives in `_fmmax_cases.py`, which imports numpy and nothing else.
* **JAX defaults to float32.** Without `jax.config.update('jax_enable_x64', True)` an fmmax
  generator agrees with an analytic slab only to ~3e-8, which reads as a mediocre solver and is
  really a mediocre configuration.

## Which file does what

| Piece | Needs |
| --- | --- |
| `interface/` — bridges to RETICOLO, GRCWA and TORCWA | Octave or MATLAB, and the external solver |
| `gen_goldens_*.py` — freeze the reference numbers | the oracle; run by hand, rarely |
| `golden_*.npz` — the frozen numbers, committed | nothing |
| `test_*_golden.py` — the assertions | nothing; **run in CI everywhere** |
| `reti_1D.py`, `reti_1Dc.py`, `reti_2D.py` — live side-by-side comparisons | Octave; exploratory, not asserted |

## A golden frozen ahead of its consumer

`golden_fmmax_magnetic.npz` holds rows nothing in this solver can consume — permeability has no
slot anywhere (`docs/anisotropic-materials.md` §5). It was frozen anyway, on purpose: **an oracle
chosen after the work is an oracle chosen to agree.**

What keeps that honest is that every case is generated twice, once with its permeability and once
at `mu = 1`, and the `mu = 1` half *is* held to the solver today. If the harness, units or
conventions were wrong, that subset would fail now, while the answer is independently knowable.
The magnetic half carries a physical guard instead of a comparative one: every case is Hermitian
in both tensors, hence lossless, so `R + T = 1` is enforced on the frozen numbers.
