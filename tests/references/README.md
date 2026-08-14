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
| **RETICOLO** (Li/Lalanne lineage) | diffraction *orders*; uniform layers with a complex 3x3 tensor — **but only up to one off-diagonal pair, at classical mounting**, see below | Octave + a licensed RETICOLO V10 tree (`RETICOLO_PATH`) | `golden_reticolo.npz`, `golden_reticolo_anisotropic.npz` |
| **RETICOLO**, swept over `(kx, ky)` | conical incidence across the light line, **including complex (evanescent) angles** | nothing — frozen from a 2024-09-09 study, see its generator | `golden_reticolo_conical_sweep.npz` |
| **fmmax** (Liu & Fan lineage, MIT) | uniform complex tensors, **patterned** tensors, **permeability** | the `fwref` env — see below | `golden_fmmax.npz`, `golden_fmmax_magnetic.npz` |
| **S4** (same lineage, C++) | real tensors, patterned | the S4 extension, already in the project env | `golden_s4.npz` |
| **S4**, the in-cell **field** | the six field components on *uniform* stacks — every reconstruction path, all angles and azimuths, exact at ~1e-15 | same | `golden_s4_field.npz` |

A fourth oracle lives outside this directory: the Berreman 4x4 in
`tests/test_anisotropic_berreman.py` is an independent *formulation* — not RCWA at all — but our
own implementation, so it sits a tier below these three.

### Why the field row exists, and what it cost to not have it

Until 2026-08-14 every oracle here checked efficiencies or amplitudes, and **field
reconstruction had none at all**. Its only reference was a golden captured from this project's
own earlier implementation — a change detector wearing an oracle's name. What it froze was a
wrong sign convention (`Ez`, `Hx`, `Hy` in the frame whose `+z` points *out* of the stack), and
the suite stayed green for as long as the file existed, because a self-captured golden agrees
with whatever produced it. It is gone; `golden_s4_field.npz` replaces the claim it was making,
and `tests/data/golden_field.npz` keeps the drift-detection job under its own honest name.

S4 stops at **uniform** stacks, and that is not a limitation for this purpose: conventions and
frames are decided there, and neither solver truncates, so the comparison is exact rather than
converged. On a patterned layer the two disagree at 1e-3 to 1e-2 (and up to 7.6e-1 on the
discontinuous `Ez`) purely from differing truncation and factorization — a convergence study,
not a golden, and deliberately not attempted.

Two conventions had to be measured rather than assumed, and both are recorded in
`gen_goldens_s4_field.py`: S4's p amplitude is a **global** `-1` against ours (not a frame
parity — it moves all six components together), and the comparison must sample layer
**interiors**, because `Ez` is discontinuous at an interface and the two codes may legitimately
report different one-sided limits there.

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

### The other three, and why none of them fills that gap

Checked 2026-08-11, because "no oracle exists" is a claim that has to be earned rather than
assumed:

Each was run, at its current release, against two checks that need no agreement with anything
else: does a lossless tensor come back with `R + T = 1`, and does zeroing `xz/zx/yz/zy` change
the answer.

| solver | version | z-coupling | how it was established |
|---|---|---|---|
| fmmax | 1.7.1 | none | `eigensolve_general_anisotropic_media` takes `xx/xy/yx/yy/zz` for each of eps and mu, with no `xz/yz/zx/zy` — "general" there means *magnetic*. API signature. |
| inkstone | 0.3.15 | **accepted, silently dropped** | `AddMaterial` takes a `(3,3)` array; output is bit-identical to the same tensor with the z row zeroed |
| EMpy | 2.2.3 | **read, but unusable** | `AnisotropicTransferMatrix` does use the z row, and returns `R + T = 1.6e13` for a plain Faraday tensor at normal incidence |
| S4 | — | none | same five-slot shape; not re-run here |

inkstone deserves the detail because it is the trap. It has the only full-3×3 API of the eight,
it conserves energy to 1.3e-15 across thirty lossless configurations, and it reproduces our 4ff
to 1e-15 on every z-*decoupled* case including conical incidence — so the harness is sound and
nothing looks wrong. On z-coupled tensors it is 4e-4 to 3e-2 out, unmoved by `num_g` from 1 to
21, and on the four cases inside RETICOLO's validated region **RETICOLO and our 4ff agree to
1e-15 while inkstone alone departs**. Energy conservation cannot catch this: the tensor it
actually solves is a valid lossless one, which is the same reason our own 2ff path returns
`R + T = 1` on a Voigt tensor while being wrong (`test_anisotropic_berreman.py`).

EMpy fails the opposite way, and its normal-incidence blowup is what makes that readable: at
`theta = 0` there is no angle convention to get wrong, and the same harness returns the exact
Airy answer for isotropic and diagonal tensors, so `1.6e13` is the solver rather than the
bridge. It is a pure T-matrix, which its own documentation calls unstable.

**The pattern worth carrying forward: "supports a full 3×3 tensor" was wrong or badly incomplete
for three of the four.** An API that accepts an argument is not an API that uses it correctly.

inkstone is AGPL, so its implementation has not been read; every statement above is black-box —
public API signatures and returned numbers. EMpy and fmmax are MIT.

Two things to weigh when reading agreement between them:

* **fmmax generalises the S4 paper**, so those two share a formulation lineage. Agreement between
  *them* is weaker evidence than agreement with RETICOLO or Berreman.
* **RETICOLO validates individual orders**; the others are compared on totals per incident
  polarisation. For anisotropy that is deliberate: S4 is silently wrong when polarisation decomposition
  and anisotropy are combined, so totals are the only sound comparison against it.

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
  can import the other's module, since `fwref` has no `wavebit` and the project env has no
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
