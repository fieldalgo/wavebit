"""What a slab add costs per FDTD step on NumPy, JAX and Torch, against dense full-volume adds.

The general-purpose FDTD design keeps a small kernel and lets every additive term (a source,
a dispersive polarisation, a TFSF face) add its value into a SLAB -- a small sub-box -- of the
field arrays each step, instead of adding a full-volume array. That only works if a slab add
is cheap on every backend: if XLA's fusion of `E.at[slab].add(v)` inside a `lax.scan` secretly
copies the whole volume, or if Torch's in-place `add_` on a slice breaks autograd, the term
mechanism has to change. This script measures both on raw array-library operations, without
importing either existing wavebit FDTD engine.

Stand-in kernel: `H -= c*curl(E); E += c*curl(H)` with periodic one-sided differences by axis
rolls -- the memory traffic of a Yee update, not its physics -- on (3, N, N, N) float64 fields,
N = 64, 96, 128. A term is a (3, s, s, s) block scaled by its own per-step amplitude. Variants:
`baseline` (no term); `slab x1 origin` (one s=8 block at the corner (0, 0, 0)); `slab x12
interior` (twelve s=8 blocks at distinct cells of a 3x3x3 block of slab cells one slab in from
the low faces -- the same absolute origins at every N, so N changes nothing but N); `slab x12
face` (twelve s=8 blocks with one coordinate 0, four on each zero face, the others interior);
`slab x12 s=16` (the interior layout with s=16 blocks, so their origins 8+16i are multiples of
8 but not of 16); `dense x12` (the interior terms as full-volume patterns, `E += a_k*pattern_k`);
`dense x1` (one full-volume multiply-add). `slab x12 interior` and `dense x12` compute the same
field, and every backend must agree on it to roundoff. NumPy: in-place ops on preallocated
arrays. JAX: `jax.jit` over one `lax.scan` of all steps, x64 enabled before any array exists.
Torch: a Python loop with in-place `add_` on slices under `torch.no_grad()`, `alpha=` giving
the dense variants a fused multiply-add.

Protocol. The JAX scan runs 1000/200/60 steps at N = 64/96/128, the eager loops 250/60/20 (a
run of seconds each; JAX reproduces the loop-length field for the cross-backend check). Every
runner is warmed once (the compile, on JAX); the five timed rounds then go over all of a
backend's variants in turn, so a slow phase of the machine hits every variant alike; a cell is
the median ms/step and the half-range of its five samples. Twelve slab adds are far below the
eager loops' noise, so the eager terms are also timed alone: the twelve interior slab adds, the
twelve dense adds and one full in-place pass, repeated for 0.3 s with no kernel. Each JAX slab
program is lowered and compiled: the optimized scan body is counted for dynamic-update-slice
fusions and full-volume copies, XLA's buffer assignment read for whether the DUS outputs share
the loop carry's allocation, and the LLVM IR dumped for the DUS kernels read for loop trip
counts and for `.us` blocks, LLVM's unswitched loop clones. At N=64 one slab per step is timed
at the eight {0, 8}^3 corners, the high corner, two origins that are not multiples of 8, and
s=16 at (8, 8, 8) and at (16, 16, 16), interleaved with the baseline re-timed as the drift
control. Torch differentiated (N=64 for 50 steps, N=128 for 10, three interleaved rounds):
forward with autograd on and backward timed apart for the in-place `add_`, a custom
`autograd.Function` that adds in place under `mark_dirty` and whose backward slices the
cotangent, an out-of-place `pad` reference and dense, each gradient checked against pad. The
contract test (N=32, 10 steps): both in-place primitives must reproduce the pad gradient when
the fresh E is written before anything reads it, and must raise when a view of E is read into
the loss first. A JAX-only section, "JAX slab primitives" (`--section scatter` runs it alone, in
minutes), times at every N, with the scan lengths and rounds above, the interior s=8 and s=16
layouts and each moved by (1, 3, 2) off every multiple of 8, through three primitives -- `dus`,
the `E.at[slices].add` above; `scatter4`, an index scatter-add on the 4-D carry,
`E.at[(c, z, y, x)].add(v, indices_are_sorted=True, unique_indices=True,
mode='promise_in_bounds')` with the four index arrays built once per term and closed over; and
`scatter_flat`, the same scatter at flat indices on E reshaped to 1-D inside the scan body and
back -- holds every final field to the layout's `dus` field within 1e-12, counts the scan body's
DUS fusions, scatter fusions and full-volume copies, and at every N, for 50/20/10 scan steps,
checks the gradient of sum(E_final^2) with respect to a (12,) amplitude leaf (and e0) through
`scatter4` against `dus`, with twelve terms and with one, times forward+backward through the five
programs (the baseline, each primitive with one interior s=8 term and with twelve) in the rounds
above, and counts the transposed scan body's full-volume copies, gathers and dynamic slices.

Numbers are CPU-only, on this machine, 2026-09-18 (12 logical CPUs, Zen 4, 32 GB; NumPy 2.5.1,
JAX 0.11.0 on one CpuDevice, Torch 2.13.0+cpu on 6 threads; no OMP/MKL/OPENBLAS variables
set). `time.process_time` under-reports on this Windows build (a pure-Python busy loop shows a
fifth of its wall time), so no cpu/wall figure is given. Run from the repo root; an optional
argument names the JSON output (default `studies/results/fdtd_slab_add_cost.json`, or
`fdtd_slab_add_cost.scatter.json` under `--section scatter`); the checked-in log beside it is
the printed output: the full run first, then the primitives section from its own runs, a first
one and the one tabulated here, run after the reverse mode was extended to every N.
Measured:

| N=64, 1000 scan steps (jax) / 250 loop steps (numpy, torch), ms/step (x baseline) +-half-range of 5 runs | numpy | jax | torch |
| --- | --- | --- | --- |
| baseline | 20.247 (1.00x) +-35% | 0.375 (1.00x) +-0% | 2.718 (1.00x) +-2% |
| slab x1 origin | 23.816 (1.18x) +-16% | 0.648 (1.73x) +-3% | 2.692 (0.99x) +-6% |
| slab x12 interior | 24.121 (1.19x) +-18% | 0.537 (1.43x) +-1% | 2.962 (1.09x) +-8% |
| slab x12 face | 23.523 (1.16x) +-24% | 0.539 (1.43x) +-2% | 2.886 (1.06x) +-4% |
| slab x12 s=16 | 17.608 (0.87x) +-28% | 4.036 (10.75x) +-3% | 3.162 (1.16x) +-2% |
| dense x12 | 30.466 (1.50x) +-30% | 4.321 (11.51x) +-3% | 5.277 (1.94x) +-2% |
| dense x1 | 25.397 (1.25x) +-20% | 0.546 (1.45x) +-2% | 3.168 (1.17x) +-4% |
| noise: baseline half-range, us/step | 7081 | 2 | 57 |
| us over baseline: origin slab / interior slab / face slab / s=16 slab / dense term / full pass | 3569 / 323 / 273 / -220 / 852 / 5150 | 272 / 13 / 14 / 305 / 329 / 171 | -26 / 20 / 14 / 37 / 213 / 450 |
| terms alone, no kernel, us per call: slab add / dense add / in-place pass | 7 / 1644 / 72 | - | 10 / 117 / 26 |

| N=96, 200 scan steps (jax) / 60 loop steps (numpy, torch), ms/step (x baseline) +-half-range of 5 runs | numpy | jax | torch |
| --- | --- | --- | --- |
| baseline | 60.742 (1.00x) +-2% | 2.125 (1.00x) +-5% | 16.611 (1.00x) +-4% |
| slab x1 origin | 61.458 (1.01x) +-2% | 3.216 (1.51x) +-6% | 16.733 (1.01x) +-9% |
| slab x12 interior | 61.064 (1.01x) +-3% | 2.477 (1.17x) +-5% | 17.695 (1.07x) +-6% |
| slab x12 face | 60.277 (0.99x) +-2% | 2.445 (1.15x) +-9% | 17.243 (1.04x) +-5% |
| slab x12 s=16 | 61.654 (1.02x) +-2% | 12.552 (5.91x) +-4% | 17.776 (1.07x) +-5% |
| dense x12 | 110.826 (1.82x) +-2% | 13.071 (6.15x) +-1% | 28.830 (1.74x) +-3% |
| dense x1 | 64.953 (1.07x) +-7% | 2.765 (1.30x) +-4% | 19.998 (1.20x) +-8% |
| noise: baseline half-range, us/step | 1065 | 113 | 585 |
| us over baseline: origin slab / interior slab / face slab / s=16 slab / dense term / full pass | 715 / 27 / -39 / 76 / 4174 / 4211 | 1091 / 29 / 27 / 869 / 912 / 640 | 122 / 90 / 53 / 97 / 1018 / 3387 |
| terms alone, no kernel, us per call: slab add / dense add / in-place pass | 5 / 4068 / 232 | - | 11 / 647 / 73 |

| N=128, 60 scan steps (jax) / 20 loop steps (numpy, torch), ms/step (x baseline) +-half-range of 5 runs | numpy | jax | torch |
| --- | --- | --- | --- |
| baseline | 147.781 (1.00x) +-3% | 17.122 (1.00x) +-1% | 61.988 (1.00x) +-1% |
| slab x1 origin | 148.355 (1.00x) +-5% | 20.132 (1.18x) +-1% | 62.574 (1.01x) +-2% |
| slab x12 interior | 148.364 (1.00x) +-3% | 17.724 (1.04x) +-0% | 61.966 (1.00x) +-2% |
| slab x12 face | 146.353 (0.99x) +-2% | 17.684 (1.03x) +-1% | 62.516 (1.01x) +-2% |
| slab x12 s=16 | 148.918 (1.01x) +-2% | 50.281 (2.94x) +-2% | 61.998 (1.00x) +-3% |
| dense x12 | 276.599 (1.87x) +-1% | 39.491 (2.31x) +-2% | 97.153 (1.57x) +-3% |
| dense x1 | 159.659 (1.08x) +-2% | 18.593 (1.09x) +-1% | 65.559 (1.06x) +-2% |
| noise: baseline half-range, us/step | 4913 | 164 | 514 |
| us over baseline: origin slab / interior slab / face slab / s=16 slab / dense term / full pass | 574 / 49 / -119 / 95 / 10735 / 11877 | 3009 / 50 / 47 / 2763 / 1864 / 1470 | 586 / -2 / 44 / 1 / 2930 / 3570 |
| terms alone, no kernel, us per call: slab add / dense add / in-place pass | 6 / 10764 / 1655 | - | 9 / 2459 / 1169 |

Slab origins, every N: origin (0, 0, 0); interior (8, 8, 8), (16, 16, 16), (24, 24, 24), (8, 16, 24),
(16, 24, 8), (24, 8, 16), (8, 24, 16), (16, 8, 24), (24, 16, 8), (8, 8, 24), (24, 24, 8), (16, 16, 8);
face (0, 8, 8), (0, 16, 16), (0, 24, 24), (0, 8, 24), (8, 0, 8), (16, 0, 16), (24, 0, 24), (8, 0, 24),
(8, 8, 0), (16, 16, 0), (24, 24, 0), (8, 24, 0); the s=16 layout is the interior one with 16 for 8
(origins 8, 24, 40). The misaligned layouts of the primitives section move every origin of those
two by (1, 3, 2), off every multiple of 8 on every axis: s=8 (9, 11, 10), (17, 19, 18), (25, 27, 26),
(9, 19, 26), ...; s=16 (9, 11, 10), (25, 27, 26), (41, 43, 42), (9, 27, 42), ...

JAX HLO, every slab variant at every N: 12 (or 1) `stablehlo.scatter` ops become 12 (1)
dynamic-update-slice fusions in the scan body, 0 full-volume copies in the body (the program's 2
are the loop-entry copies of e0 and h0), and all DUS outputs live in one allocation shared with
the loop carry (a `(phi)` allocation): in place. Every DUS kernel dumped is a loop nest with trip
count N. Kernels LLVM unswitched (`.us` clones), out of 12 or 1: interior 12, face 12, origin 0,
s=16 at origins 8+16i 0, at every N.

JAX one slab per step at N=64, us/step over the baseline (0.375 ms/step) +-half-range, and whether
LLVM unswitched the kernel (U) or left the full loop nest (F): baseline again +5 +-6, (0, 0, 0) s=8
+257 +-13 F, (0, 0, 8) s=8 +20 +-1 U, (0, 8, 0) s=8 +60 +-1 U, (0, 8, 8) s=8 +26 +-6 U, (8, 0, 0) s=8
+54 +-7 U, (8, 0, 8) s=8 +25 +-1 U, (8, 8, 0) s=8 +24 +-2 U, (8, 8, 8) s=8 +22 +-2 U, (9, 9, 9) s=8
+257 +-7 F, (12, 12, 12) s=8 +256 +-4 F, (8, 8, 8) s=16 +267 +-9 F, (16, 16, 16) s=16 +44 +-3 U,
(56, 56, 56) s=8 +23 +-7 U. (The previous run of this script: +3 +-1, +287, +16, +60, +23, +53,
+20, +21, +22, +268, +275, +290, +42, +23, the same U/F on every entry.)

| JAX slab primitives at N=64, 1000 scan steps, ms/step (+us over the baseline 0.326 +-5%) +-half-range of 5 runs | dus | scatter4 | scatter_flat |
| --- | --- | --- | --- |
| interior s=8 | 0.396 (+70) +-3% | 0.280 (-46) +-6% | 0.310 (-16) +-6% |
| misaligned s=8 | 3.185 (+2859) +-1% | 0.275 (-51) +-1% | 0.311 (-14) +-8% |
| interior s=16 | 3.228 (+2902) +-0% | 0.468 (+143) +-1% | 0.389 (+63) +-9% |
| misaligned s=16 | 3.231 (+2905) +-1% | 0.425 (+99) +-4% | 0.391 (+65) +-6% |
| final field vs dus, max relative gap, the four layouts | 0e+00 / 0e+00 / 0e+00 / 0e+00 | 2e-13 / 1e-13 / 2e-13 / 2e-13 | 2e-13 / 1e-13 / 2e-13 / 2e-13 |
| scan body: DUS fusions / scatter fusions / full-volume copies | 12/0/0 in every layout | 0/12/0 in every layout | 0/12/1 in every layout |

| JAX slab primitives at N=96, 200 scan steps, ms/step (+us over the baseline 1.909 +-2%) +-half-range of 5 runs | dus | scatter4 | scatter_flat |
| --- | --- | --- | --- |
| interior s=8 | 2.189 (+280) +-3% | 1.983 (+73) +-3% | 3.774 (+1865) +-3% |
| misaligned s=8 | 12.360 (+10451) +-2% | 2.003 (+94) +-5% | 3.817 (+1907) +-3% |
| interior s=16 | 11.729 (+9820) +-1% | 2.552 (+643) +-4% | 4.189 (+2280) +-3% |
| misaligned s=16 | 12.385 (+10476) +-1% | 2.618 (+709) +-9% | 4.349 (+2440) +-2% |
| final field vs dus, max relative gap, the four layouts | 0e+00 / 0e+00 / 0e+00 / 0e+00 | 1e-14 / 1e-14 / 9e-15 / 9e-15 | 1e-14 / 1e-14 / 9e-15 / 9e-15 |
| scan body: DUS fusions / scatter fusions / full-volume copies | 12/0/0 in every layout | 0/12/0 in every layout | 0/12/1 in every layout |

| JAX slab primitives at N=128, 60 scan steps, ms/step (+us over the baseline 17.381 +-2%) +-half-range of 5 runs | dus | scatter4 | scatter_flat |
| --- | --- | --- | --- |
| interior s=8 | 17.856 (+475) +-1% | 17.311 (-70) +-2% | 20.253 (+2872) +-1% |
| misaligned s=8 | 48.315 (+30934) +-1% | 17.356 (-25) +-2% | 20.327 (+2946) +-1% |
| interior s=16 | 48.135 (+30754) +-2% | 18.299 (+918) +-4% | 20.797 (+3417) +-1% |
| misaligned s=16 | 47.829 (+30448) +-1% | 18.596 (+1215) +-1% | 20.900 (+3519) +-2% |
| final field vs dus, max relative gap, the four layouts | 0e+00 / 0e+00 / 0e+00 / 0e+00 | 9e-16 / 8e-16 / 6e-16 / 6e-16 | 9e-16 / 8e-16 / 6e-16 / 6e-16 |
| scan body: DUS fusions / scatter fusions / full-volume copies | 12/0/0 in every layout | 0/12/0 in every layout | 0/12/1 in every layout |

| JAX reverse mode, interior s=8: grad of sum(E_final^2) w.r.t. the (12,) amplitude leaf and e0, forward+backward ms/step (+ms over the baseline) +-half-range of 5 runs | baseline | dus x1 | scatter4 x1 | dus x12 | scatter4 x12 |
| --- | --- | --- | --- | --- | --- |
| N=64, 50 scan steps | 1.26 (+0.00) +-5% | 1.40 (+0.15) +-3% | 1.36 (+0.10) +-4% | 1.45 (+0.19) +-5% | 1.32 (+0.06) +-4% |
| N=96, 20 scan steps | 6.20 (+0.00) +-5% | 8.30 (+2.10) +-6% | 8.33 (+2.13) +-5% | 8.38 (+2.18) +-2% | 8.27 (+2.07) +-4% |
| N=128, 10 scan steps | 32.83 (+0.00) +-2% | 36.12 (+3.29) +-7% | 36.18 (+3.35) +-3% | 36.61 (+3.78) +-2% | 35.91 (+3.08) +-3% |
| scatter4 gradient vs its dus twin, amplitudes / e0, at N=64, 96, 128 | - | - | 4e-16 / 8e-16; 0e+00 / 1e-16; 1e-16 / 3e-17 | - | 1e-16 / 9e-16; 2e-16 / 3e-16; 2e-16 / 9e-17 |
| transposed scan body: full-volume copies / gather fusions / dynamic-slice fusions | 0/0/0 in every N | 1/0/1 in every N | 1/1/0 in every N | 1/0/1 in every N | 1/12/0 in every N |

(The primitives section ran alone, after the full run above, twice; the tables here are the second
run's. The first run's N=64/96/128 baselines 0.328/2.031/17.372 and interior s=8 DUS cells
0.392/2.330/17.633, and this run's 0.326/1.909/17.381 and 0.396/2.189/17.856, stand against
0.375/2.125/17.122 and 0.537/2.477/17.724 in the full run: both section runs sit 12-27% under the
full run at N=64 and 4-12% under at N=96, within 2% at N=128. The first run timed the reverse mode at
N=64 only, three rounds, twelve terms: dus +0.25, scatter4 +0.13 over a 1.22 baseline. Every
comparison is within its own interleaved rounds.)

| Torch autograd at N=64, 50 steps, ms/step | forward, grad on | backward | over baseline | grad vs pad |
| --- | --- | --- | --- | --- |
| slab x12 baseline | 2.75 | 4.48 | +0.00 / +0.00 | - |
| slab x12 in-place add_ | 3.68 | 5.50 | +0.92 / +1.02 | 0.0e+00 |
| slab x12 Function | 3.74 | 5.30 | +0.99 / +0.82 | 0.0e+00 |
| slab x12 pad | 4.78 | 5.58 | +2.03 / +1.10 | - |
| slab x12 dense | 6.39 | 7.46 | +3.64 / +2.98 | 1.1e-16 |

| Torch autograd at N=128, 10 steps, ms/step | forward, grad on | backward | over baseline | grad vs pad |
| --- | --- | --- | --- | --- |
| slab x12 baseline | 64.17 | 195.88 | +0.00 / +0.00 | - |
| slab x12 in-place add_ | 65.07 | 233.38 | +0.90 / +37.51 | 0.0e+00 |
| slab x12 Function | 63.72 | 199.85 | -0.44 / +3.98 | 0.0e+00 |
| slab x12 pad | 133.57 | 199.62 | +69.40 / +3.74 | - |
| slab x12 dense | 145.97 | 261.46 | +81.81 / +65.58 | 1.1e-16 |

Final fields: `slab x12 interior` equals `dense x12` exactly on every backend, and JAX and
Torch match NumPy within 3e-16 at every N (the gaps are in the JSON; the assertion allows 1e-9
of the field's magnitude).
Torch autograd, correct order, in-place add_: PASS (relative gradient error 0.00e+00 against
out-of-place pad). Torch autograd, wrong order, in-place add_: PASS (RuntimeError: "one of the
variables needed for gradient computation has been modified by an inplace operation:
[torch.DoubleTensor [3, 32, 32]], which is output 0 of AsStrided, is at version 12; expected
version 0 instead"). Torch autograd, correct order, Function: PASS (relative gradient error
0.00e+00). Torch autograd, wrong order, Function: PASS (the same RuntimeError, at version 24).

Reading. JAX: the slab add is in place in XLA's buffer at every size and layout -- 0 full-volume
copies in the scan body, the DUS outputs in the loop carry's allocation -- but in place is not
O(slab). Every DUS kernel is a loop nest over the whole volume that stores `select(in_slab, E +
a*v, E)` back to the same address; it is cheap only where LLVM unswitched the outer loops on the
slab's x and y range tests and deleted the self-copy. It did that for every 8-wide slab whose x
and y starts are multiples of 8 except the origin corner (two of them, (0, 8, 0) and (8, 0, 0), at
2.5x the interior price), and for a 16-wide slab at (16, 16, 16); it did not for an 8-wide slab at
(9, 9, 9) or (12, 12, 12), a 16-wide slab at (8, 8, 8), or the origin. An unswitched slab costs
13-14 us at N=64, 27-29 at N=96, 47-50 at N=128 -- 8%, 4%, 3% of a full in-place pass, growing
with N because the nest still visits 3N^2 rows -- and the same on a zero face as inside; a slab
that is not unswitched costs 1.4-2.1 full passes, the order of a dense term. So twelve interior
s=8 terms add 0.43x, 0.17x and 0.04x of the stand-in kernel at N=64/96/128 (twelve dense terms:
10.5x, 5.2x, 1.3x), while a term placed at an arbitrary cell, or of a width the heuristics
dislike, silently costs a full pass. That is LLVM's cost model on XLA's full-volume kernel, not a
slab loop, so it is version-dependent and must be re-measured when JAX moves; the design can keep
it cheap by snapping slab origins to multiples of the slab width (measured for widths 8 and 16 at
N=64) and must not budget O(s^3).
The index scatter-add removes the dependence on the origin. `scatter4` -- the (c, z, y, x) index
arrays built once per term, sorted, unique, promised in bounds, the scatter on the 4-D carry --
lowers to twelve scatter fusions beside the baseline's two full-volume fusions, with no copy, and
costs the same at the interior origins and moved off them by (1, 3, 2): twelve s=8 terms -46/-51 us
at N=64, +73/+94 at N=96, -70/-25 at N=128 (DUS +70, +280, +475 aligned and +2859, +10451, +30934
misaligned), twelve s=16 terms +143/+99, +643/+709, +918/+1215 (DUS +2902-2905, +9820-10476,
+30448-30754). That is cheaper than the unswitched DUS at every N. Twelve s=8 terms cost -70 to +94
us at every N, 0-8 us per term, inside the baseline's half-range (16, 38 and 348 us) except at
N=96, where they clear it by twice and still stand at a quarter to a third of the aligned DUS; s=16, eight
times the cells, costs 8-12, 54-59 and 77-101 us per term, 5-9% of a full in-place pass, a price
that follows the cells and the cache (in L3 at N=64; from memory at N=96 and 128, where it grows
1.4-1.7x for 2.4x the volume), not N^3. The negative cells are not a saving: pinned to one core
(psutil affinity, 300 steps) the twelve s=8 scatters cost +45 us over a 0.860 ms baseline at N=64
(DUS +142), so on all cores the extra small ops between the two big fusions move the runtime's
thread scheduling by more than the scatters cost; read them as zero. The final field agrees with
DUS to 2e-13 at 1000 steps, 1e-14 at 200 and 9e-16 at 60: at one step `scatter4` reproduces
NumPy's `E + (a*v)` in every cell while the DUS fusion contracts it to a fused multiply-add (1219
cells off by one ulp), a one-rounding difference the leapfrog carries. Its transpose is a gather
at the same indices -- twelve gather fusions in the transposed scan body, where DUS has its
dynamic slices -- and the amplitude and e0 gradients agree with DUS to 1e-15 at every N. But a
differentiated slab program is not O(slab), for either primitive, and its cost is not per term:
forward+backward with twelve terms costs +0.06/+0.19 ms/step (scatter4/DUS) over the baseline's
1.26 at N=64, +2.07/+2.18 over 6.20 at N=96 and +3.08/+3.78 over 32.83 at N=128, and with one term
+0.10/+0.15, +2.13/+2.10 and +3.35/+3.29: the first term brings all of it, the other eleven sit
inside the half-ranges (0.04-0.07 ms at N=64, 0.2-0.5 at N=96, 0.7-2.5 at N=128). The transposed
scan body shows what the first term brings: one full-volume `copy` of the E cotangent that the
baseline's body lacks, there with one term or twelve and with either primitive, a copy op outside
every fusion at 2.2-3.3 times the fused full-volume pass (640/1470 us at N=96/128) -- read as XLA
keeping the cotangent the term's gather (or dynamic slice) reads apart from the kernel's in-place
update of it. At N=64 the volume sits in cache and the whole effect is 0.1-0.2 ms/step, DUS and
scatter4 about a half-range apart (+0.19 and +0.06 at twelve terms here, +0.25 and +0.13 in the
first section run's three rounds): scatter4 has not come out dearer, but the margin is the noise,
not a property. The design's JAX slab_add primitive is therefore `scatter4`; alignment no longer
matters, so the origin-snapping rule above is not needed, and the DUS numbers stay as the record
of why the primitive is not `E.at[slices].add`. A differentiated run is a separate line in the
budget, and the primitive does not move it: one full-volume copy per backward step as soon as the
program holds any slab term, whatever their number -- a third of the baseline's differentiated
step at N=96, a tenth at N=128; whether another form of the term escapes the copy (a custom_vjp,
one scatter of all terms) is unmeasured. The one forward pitfall: reshaping the carry to 1-D
inside the body (`scatter_flat`) puts one full-volume copy per step into the body, the reshape
back into the carry -- +1.9 and +2.9 ms/step at N=96/128, a read and a write of the whole volume,
invisible at N=64 where the volume sits in cache -- so the carry stays 4-D and the indices 4-D.
Like the DUS finding this is XLA's CPU scatter lowering and must be re-measured when JAX moves.
Torch: a slab add costs 9-11 us alone and 15-20 us in the loop where the loop resolves it (N=64),
so twelve terms cost 0-9% of the stand-in kernel under `no_grad` here and 20% at N=64 in the first
run of the day (dense: 57-94%). Under autograd the in-place `add_` on a view installs a CopySlices
node whose backward allocates and fills a full field cotangent per term: +1.0 ms/step at N=64 (23%
of the kernel's backward, in cache) and +37.5 ms/step at N=128 (3.1 ms per term, one 50 MB pass
each, 19% of the kernel's backward; the previous run +36.5, a reviewer's probe +45.6). The
`mark_dirty` Function gives the same gradient bit for bit and costs +0.8 and +4.0 ms/step in
backward at N=64/128 (the previous run +0.5 and +1.9), the latter 2% of the kernel's backward; the
Torch primitive should therefore be such a Function (or one batched in-place op per step,
unmeasured), not a bare `add_`. The contract holds for both: an in-place write into a fresh E
before anything reads it reproduces the out-of-place gradient exactly, and a view read first
raises at backward (version 12 for `add_`, 24 for the Function, which bumps twice per term).
NumPy: a slab add costs 5-7 us alone at every N, a dense add 1.6, 4.1 and 10.8 ms, an in-place
pass 0.07, 0.23 and 1.7 ms; the in-loop slab rows are below this stand-in's noise and say nothing
more.
Noise: JAX cells reproduce to a few percent across four runs of the day (sweep drift control +3 to
+5 us, every sweep entry within 30 us of the previous run). The eager loops allocate every
temporary afresh, and their absolute values move with the machine's state in ways this study did
not resolve: this run's eager N=64 block was hit from its third round (NumPy baseline samples
16.9, 17.1, 20.2, 31.1, 28.4 ms/step, while a fresh process right after the run held 17.3-17.8 for
80 s), so the NumPy N=64 ratios above are the machine, not the code -- the previous run of this
script had that block at 17.4 ms/step with every s=8 slab row at 1.01-1.03x and dense x12 at
1.87x, and its N=96 block disturbed instead (82 +-22% against 60.7 +-2% here). Torch's N=64
baseline moved 1.83-2.72 across the four runs while its N=96 and N=128 held within 4%. The ratios
keep their ordering in every run: NumPy dense x12 1.5-1.9x and dense x1 1.0-1.3x, Torch dense x12
1.5-2.8x, every resolved s=8 slab row within 1.2x of its baseline. Trust the ratios, the orderings
and the terms-alone line, not the third digit of an eager cell.
"""
import argparse
import glob
import json
import os
import platform
import re
import shutil
import tempfile
import time

DUMP = tempfile.mkdtemp(prefix='slab_add_xla_')
os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '') + ' --xla_dump_to=' + DUMP).strip()
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)  # before any array exists
import jax.numpy as jnp
import torch
import torch.nn.functional as tnn
from jax import lax

C, S, TERMS, RUNS = .4, 8, 12, 5
STEPS = {64: (1000, 250), 96: (200, 60), 128: (60, 20)}  # (JAX scan length, NumPy/Torch loop length)
GRAD_STEPS = {64: 50, 128: 10}  # the Torch forward+backward timing
JAX_GRAD_STEPS = {64: 50, 96: 20, 128: 10}  # the JAX reverse mode of the primitives section, at every N
REVERSE_ARMS = {'baseline': ('dus', 0), 'dus x1': ('dus', 1), 'scatter4 x1': ('scatter4', 1),
                'dus x12': ('dus', TERMS), 'scatter4 x12': ('scatter4', TERMS)}  # (primitive, terms of the interior s=8 layout)
# twelve distinct cells of a 3x3x3 block of slabs that starts one slab in from the low faces
INTERIOR = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (0, 1, 2), (1, 2, 0), (2, 0, 1),
            (0, 2, 1), (1, 0, 2), (2, 1, 0), (0, 0, 2), (2, 2, 0), (1, 1, 0)]
# the JAX slab primitives section: the interior layouts of both widths, each also moved by SHIFT off every multiple of 8
SHIFT = (1, 3, 2)
LAYOUTS = {'interior s=8': (S, False), 'misaligned s=8': (S, True), 'interior s=16': (2*S, False), 'misaligned s=16': (2*S, True)}
PRIMITIVES = ('dus', 'scatter4', 'scatter_flat')
SCATTER = dict(indices_are_sorted=True, unique_indices=True, mode='promise_in_bounds')  # what `slab_indices` guarantees
FACE = [(a, o) for a in range(3) for o in ((0, 0), (1, 1), (2, 2), (0, 2))]  # four slabs on each zero face
# one JAX slab per step: the eight {0, S}^3 corners, two origins that are not multiples of S, and s=2S
# at an origin that is a multiple of S only and at one that is a multiple of 2S (the high corner is added at run time)
SWEEP = [((x, y, z), S) for x in (0, S) for y in (0, S) for z in (0, S)] + \
        [((9, 9, 9), S), ((12, 12, 12), S), ((S, S, S), 2*S), ((2*S, 2*S, 2*S), 2*S)]
VARIANTS = {'baseline': None, 'slab x1 origin': ('slab', 'origin', S), 'slab x12 interior': ('slab', 'interior', S),
            'slab x12 face': ('slab', 'face', S), 'slab x12 s=16': ('slab', 'interior', 2*S),
            'dense x12': ('dense', TERMS, S), 'dense x1': ('dense', 1, S)}
BACKENDS = ('numpy', 'jax', 'torch')
# roll, stack, copy, in-place `v += a*x`, in-place scale, and the array constructor of each eager library
NUMPY = (np.roll, np.stack, np.copy, lambda v, x, a: np.add(v, a*x, out=v),
         lambda E: np.multiply(E, 1.000001, out=E), np.asarray)
TORCH = (torch.roll, torch.stack, torch.clone, lambda v, x, a: v.add_(x, alpha=a),
         lambda E: E.mul_(1.000001), torch.from_numpy)


def curl(F, roll, stack, forward):
    """Stand-in for the Yee curl: periodic one-sided differences built from axis rolls."""
    shift, sign = (-1, 1.) if forward else (1, -1.)

    def d(c, ax):
        return sign*(roll(F[c], shift, ax) - F[c])
    return stack([d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1)])


def origins(layout, s):
    """Slab origins: one at the corner, twelve one slab in from the low faces, or twelve on zero faces."""
    if layout == 'origin':
        return [(0, 0, 0)]
    if layout == 'interior':
        return [tuple(S + s*i for i in cell) for cell in INTERIOR]
    return [tuple(0 if ax == a else S + s*o[ax - (ax > a)] for ax in range(3)) for a, o in FACE]


def slab(origin, s):
    return (slice(None),) + tuple(slice(o, o + s) for o in origin)


def problem(n, steps, seed=0):
    rng = np.random.default_rng(seed)
    values = {s: rng.standard_normal((TERMS, 3, s, s, s)) for s in (S, 2*S)}
    patterns = np.zeros((TERMS, 3, n, n, n))
    for k, origin in enumerate(origins('interior', S)):
        patterns[k][slab(origin, S)] = values[S][k]
    t = np.arange(steps)[:, None]
    return dict(e0=1e-3*rng.standard_normal((3, n, n, n)), h0=np.zeros((3, n, n, n)), values=values,
                amps=1e-3*np.sin(2*np.pi*(t+1)*(np.arange(TERMS)+1)/steps), patterns=patterns)


def terms(variant, p):
    """A variant's additive terms: (slab slice, block) pairs, or (None, full-volume pattern) for dense."""
    if VARIANTS[variant] is None:
        return []
    kind, layout, s = VARIANTS[variant]
    if kind == 'slab':
        return [(slab(o, s), p['values'][s][k]) for k, o in enumerate(origins(layout, s))]
    return [(None, x) for x in (p['patterns'] if layout == TERMS else [p['patterns'].sum(0)])]


def eager_runner(ts, p, lib):
    """NumPy and Torch share one Python loop; `lib` supplies the copy and the in-place add."""
    roll, stack, copy, add, _, wrap = lib
    e0, h0, xs, amps = wrap(p['e0']), wrap(p['h0']), [wrap(x) for _, x in ts], p['amps'].tolist()

    def run():
        with torch.no_grad():
            E, H = copy(e0), copy(h0)
            for a in amps:
                H -= C*curl(E, roll, stack, True)
                E += C*curl(H, roll, stack, False)
                for k, ((sl, _), x) in enumerate(zip(ts, xs)):
                    add(E[sl] if sl else E, x, a[k])
        return E
    return run


def eager_term_cost(p, lib):
    """The terms alone, no kernel: microseconds per slab add, per dense add and per full in-place pass."""
    roll, stack, copy, add, scale, wrap = lib
    E = wrap(np.copy(p['e0']))
    kinds = {'slab add': [(sl, wrap(x)) for sl, x in terms('slab x12 interior', p)],
             'dense add': [(sl, wrap(x)) for sl, x in terms('dense x12', p)]}

    def run(ts):
        for sl, x in ts:
            add(E[sl] if sl else E, x, 1e-6)
    out = {}
    for label, fn in [(k, lambda ts=ts: run(ts)) for k, ts in kinds.items()] + [('in-place pass', lambda: scale(E))]:
        with torch.no_grad():
            fn()
            reps, t0 = 0, time.perf_counter()
            while reps < 5 or time.perf_counter() - t0 < .3:
                fn()
                reps += 1
            out[label] = 1e6*(time.perf_counter() - t0)/reps/(TERMS if 'add' in label else 1)
    return out


def jax_program(name, ts, p):
    def program(e0, h0, amps, xs):
        def body(carry, a):
            E, H = carry
            H = H - C*curl(E, jnp.roll, jnp.stack, True)
            E = E + C*curl(H, jnp.roll, jnp.stack, False)
            for k, ((sl, _), x) in enumerate(zip(ts, xs)):
                E = E.at[sl].add(a[k]*x) if sl else E + a[k]*x
            return (E, H), None
        return lax.scan(body, (e0, h0), amps)[0][0]
    program.__name__ = re.sub(r'\W', '_', name)  # names the XLA dump files
    args = tuple(jnp.asarray(p[k]) for k in ('e0', 'h0', 'amps')) + (tuple(jnp.asarray(x) for _, x in ts),)
    return jax.jit(program), args


def dus_kernels(name):
    """XLA's buffer assignment and the optimized LLVM IR of the DUS kernels, for the latest module compiled as `name`."""
    dump = max(glob.glob(os.path.join(DUMP, f'*jit_{name}.*buffer-assignment.txt')), key=os.path.getmtime)
    prefix = os.path.basename(dump).split('.')[0]
    files = glob.glob(os.path.join(DUMP, f'{prefix}.jit_{name}.*dynamic-update-slice*ir-with-opt.ll'))
    return open(dump).read(), [open(f).read() for f in files]


def hlo_evidence(program, args, n):
    """Counts in the optimized HLO of the scan body, plus XLA's buffer assignment and what LLVM made of the kernels."""
    lowered = program.lower(*args)
    text = lowered.compile().as_text()
    blocks = dict(re.findall(r'^(?:ENTRY )?(\S+) \([^\n]*\{\n(.*?)\n\}', text, re.M | re.S))
    body = blocks[re.search(r'while\(.*?body=(\S+),', text).group(1)]
    full = re.escape(f'f64[3,{n},{n},{n}]') + r'\{[^}]*\} '
    called = [blocks.get(c, '') for c in re.findall(r'calls=(\S+?)[,\s]', body)]
    assignment, kernels = dus_kernels(program.__name__)
    allocations = re.split(r'^allocation ', assignment, flags=re.M)[1:]
    volume = [a for a in allocations if int(re.match(r'\d+: size (\d+)', a).group(1)) >= 24*n**3]
    holding = [a for a in volume if 'dynamic-update-slice' in a]
    # a loop trip count of n means a full-volume loop nest; `.us` blocks are LLVM's unswitched clones of it
    trips = [{int(t) for t in re.findall(r'icmp eq i64 %\S+, (\d+)', k)} for k in kernels]
    return dict(stablehlo_scatter_ops=lowered.as_text().count('"stablehlo.scatter"('),
                dus_kernels_dumped=len(kernels), dus_kernels_looping_to_n=sum(n in t for t in trips),
                dus_kernels_unswitched=sum('.us' in k for k in kernels),
                hlo_dynamic_update_slice_ops=text.count('dynamic-update-slice('),
                body_dus_fusions=sum('dynamic-update-slice(' in c for c in called),
                body_full_volume_fusions=len(re.findall(full + r'fusion\(', body)),
                body_full_volume_copies=len(re.findall(full + r'copy\(', body)),
                total_full_volume_copies=len(re.findall(full + r'copy\(', text)),
                allocations_of_at_least_one_volume=len(volume),
                allocations_holding_dus_outputs=len(holding),
                dus_outputs_share_the_loop_carry=bool(holding) and all('(phi)' in a for a in holding))


def jax_slab_sweep(p, n, steps):
    """One slab per step at the SWEEP origins and sizes, interleaved with the baseline re-timed as the drift control."""
    runs, unswitched = {}, {}
    for origin, s in [(None, S)] + SWEEP + [((n-S,)*3, S)]:
        ts = [] if origin is None else [(slab(origin, s), p['values'][s][0])]
        program, args = jax_program('sweep', ts, p)
        program(*args).block_until_ready()
        key = 'baseline again' if origin is None else f'{origin} s={s}'
        runs[key] = lambda program=program, args=args: program(*args).block_until_ready()
        if origin is not None:
            unswitched[key] = '.us' in dus_kernels('sweep')[1][0]
    stats = timed(runs, steps)[1]
    for key, flag in unswitched.items():
        stats[key]['unswitched'] = flag
    return stats


def timed(runs, steps):
    """One untimed warm-up per runner (the compile, on JAX), then RUNS rounds over all runners in turn, so a slow
    phase of the machine hits every variant alike: per runner its final field, median ms/step and half-range."""
    finals, wall = {k: np.asarray(r()) for k, r in runs.items()}, {k: [] for k in runs}
    for _ in range(RUNS):
        for k, r in runs.items():
            t0 = time.perf_counter()
            r()
            wall[k].append(time.perf_counter() - t0)
    stats = {k: dict(ms_per_step=1e3*float(np.median(w))/steps, spread=(max(w) - min(w))/2/float(np.median(w)),
                     samples_ms_per_step=[1e3*x/steps for x in w]) for k, w in wall.items()}
    return finals, stats


def layout_origins(s, misaligned):
    """The interior layout with s-wide slabs, every origin moved by SHIFT when misaligned."""
    return [tuple(o + d for o, d in zip(origin, SHIFT if misaligned else (0, 0, 0))) for origin in origins('interior', s)]


def slab_indices(origin, s):
    """The (c, z, y, x) index arrays of a slab's 3*s^3 cells in the array's row-major order: sorted and unique."""
    return tuple(g.ravel() for g in np.meshgrid(np.arange(3), *(np.arange(o, o + s) for o in origin), indexing='ij'))


def jax_primitive_program(primitive, origin_list, s, p, n):
    """The study's JAX scan with the terms at `origin_list` added by one primitive. 'dus' is `jax_program` itself, the
    `E.at[slices].add` of the tables above; 'scatter4' an index scatter-add on the 4-D carry at (c, z, y, x) index arrays
    built here once per term, sorted, unique and promised in bounds; 'scatter_flat' the same scatter at flat indices on
    E reshaped to 1-D inside the body and back. The index arrays are closed over: constants of the compiled program."""
    if primitive == 'dus':
        return jax_program('dus', [(slab(o, s), p['values'][s][k]) for k, o in enumerate(origin_list)], p)
    index = [slab_indices(o, s) for o in origin_list]
    if primitive == 'scatter_flat':
        index = [(np.ravel_multi_index(i, (3, n, n, n)),) for i in index]
    index = [tuple(jnp.asarray(a) for a in i) for i in index]

    def program(e0, h0, amps, xs):
        def body(carry, a):
            E, H = carry
            H = H - C*curl(E, jnp.roll, jnp.stack, True)
            E = E + C*curl(H, jnp.roll, jnp.stack, False)
            if primitive == 'scatter_flat':
                E = E.reshape(-1)
            for k, x in enumerate(xs):
                E = E.at[index[k]].add(a[k]*x, **SCATTER)
            if primitive == 'scatter_flat':
                E = E.reshape(H.shape)
            return (E, H), None
        return lax.scan(body, (e0, h0), amps)[0][0]
    xs = tuple(jnp.asarray(x.ravel()) for x in p['values'][s][:len(origin_list)])
    return jax.jit(program), tuple(jnp.asarray(p[k]) for k in ('e0', 'h0', 'amps')) + (xs,)


def body_counts(program, args, n, backward=False):
    """In the optimized HLO of the scan body (the first while loop's, or with `backward` the last one's: the transposed
    scan of a differentiated program): fusions holding a dynamic-update-slice, a scatter, a gather or a dynamic-slice,
    and copies of a full volume (4-D or flat)."""
    text = program.lower(*args).compile().as_text()
    blocks = dict(re.findall(r'^(?:ENTRY )?(\S+) \([^\n]*\{\n(.*?)\n\}', text, re.M | re.S))
    body = blocks[re.findall(r'while\(.*?body=(\S+),', text)[-1 if backward else 0]]
    called = [blocks.get(c, '') for c in re.findall(r'calls=(\S+?)[,\s]', body)]
    full = r'f64\[(?:3,%d,%d,%d|%d)\]\{[^}]*\} ' % (n, n, n, 3*n**3)
    return dict(dus_fusions=sum('dynamic-update-slice(' in c for c in called), scatter_fusions=sum(' scatter(' in c for c in called),
                gather_fusions=sum(' gather(' in c for c in called), dynamic_slice_fusions=sum(' dynamic-slice(' in c for c in called),
                full_volume_copies=len(re.findall(full + r'copy\(', body)))


def jax_primitives(p, n, steps):
    """The baseline and the four layouts through the three primitives, timed together under the study's protocol; every
    final field is held to the layout's dus field, and the scan body's op counts are read."""
    runs, counts = {}, {}
    program, args = jax_primitive_program('dus', [], S, p, n)
    runs['baseline'] = lambda program=program, args=args: program(*args).block_until_ready()
    for layout, (s, misaligned) in LAYOUTS.items():
        for primitive in PRIMITIVES:
            program, args = jax_primitive_program(primitive, layout_origins(s, misaligned), s, p, n)
            runs[f'{layout} {primitive}'] = lambda program=program, args=args: program(*args).block_until_ready()
            counts[f'{layout} {primitive}'] = body_counts(program, args, n)
    finals, stats = timed(runs, steps)
    for key, t in stats.items():
        t['us_over_baseline'] = 1e3*(t['ms_per_step'] - stats['baseline']['ms_per_step'])
        if key != 'baseline':
            ref = finals[key.rsplit(' ', 1)[0] + ' dus']
            t['final_rel_gap_vs_dus'] = float(np.max(np.abs(finals[key] - ref))/np.max(np.abs(ref)))
            t['hlo'] = counts[key]
        print(f'N={n} jax   {key:28s} {t["ms_per_step"]:8.3f} ms/step +-{100*t["spread"]:.0f}%', flush=True)
    gaps = {k: t['final_rel_gap_vs_dus'] for k, t in stats.items() if k != 'baseline'}
    assert max(gaps.values()) < 1e-12, gaps
    return stats


def jax_reverse_mode(p, n, steps):
    """jax.grad of sum(E_final**2) with respect to a (12,) amplitude leaf that scales every step's amplitudes (and to e0,
    so the baseline's backward walks the kernel too) through the REVERSE_ARMS: the baseline, and each primitive with one
    term and with twelve of the interior s=8 layout. Each scatter4 gradient must agree with its dus twin's; forward+backward
    is timed interleaved after a warm-up, RUNS rounds, median and half-range; the transposed scan body's op counts are read."""
    short, grads, out = dict(p, amps=p['amps'][:steps]), {}, {}
    for arm, (primitive, k) in REVERSE_ARMS.items():
        program, args = jax_primitive_program(primitive, layout_origins(S, False)[:k], S, short, n)

        def loss(theta, e0, h0, amps, xs, program=program):
            return jnp.sum(program(e0, h0, amps*theta, xs)**2)
        grads[arm] = jax.jit(jax.grad(loss, argnums=(0, 1))), (jnp.ones(TERMS),) + args
    wall = {k: [] for k in grads}
    for r in range(RUNS + 1):  # the first round is the warm-up (the compile)
        for arm, (g, a) in grads.items():
            t0 = time.perf_counter()
            out[arm] = jax.block_until_ready(g(*a))
            if r:
                wall[arm].append(time.perf_counter() - t0)
    stats = {k: dict(fwd_bwd_ms_per_step=1e3*float(np.median(w))/steps, spread=(max(w) - min(w))/2/float(np.median(w)),
                     samples_ms_per_step=[1e3*x/steps for x in w], hlo=body_counts(*grads[k], n, backward=True))
             for k, w in wall.items()}
    for arm, t in stats.items():
        t['ms_over_baseline'] = t['fwd_bwd_ms_per_step'] - stats['baseline']['fwd_bwd_ms_per_step']
        if arm.startswith('scatter4'):
            for i, leaf in enumerate(('amplitudes', 'e0')):
                d, ref = np.asarray(out[arm][i]), np.asarray(out[arm.replace('scatter4', 'dus')][i])
                t[f'grad_rel_err_vs_dus_{leaf}'] = float(np.linalg.norm(d - ref)/np.linalg.norm(ref))
        print(f'N={n} jax   grad {arm:23s} {t["fwd_bwd_ms_per_step"]:8.3f} ms/step +-{100*t["spread"]:.0f}%', flush=True)
    errs = {f'{arm} {k}': v for arm, t in stats.items() for k, v in t.items() if k.startswith('grad_rel_err')}
    assert max(errs.values()) < 1e-12, errs
    return stats


def primitives_table(n, steps, stats):
    base = stats['baseline']

    def cell(layout, primitive):
        t = stats[f'{layout} {primitive}']
        return f'{t["ms_per_step"]:.3f} ({t["us_over_baseline"]:+.0f}) +-{100*t["spread"]:.0f}%'

    lines = [f'| JAX slab primitives at N={n}, {steps} scan steps, ms/step (+us over the baseline {base["ms_per_step"]:.3f} '
             f'+-{100*base["spread"]:.0f}%) +-half-range of {RUNS} runs | ' + ' | '.join(PRIMITIVES) + ' |',
             '| --- | --- | --- | --- |']
    lines += [f'| {layout} | ' + ' | '.join(cell(layout, pr) for pr in PRIMITIVES) + ' |' for layout in LAYOUTS]
    lines.append('| final field vs dus, max relative gap, the four layouts | '
                 + ' | '.join(' / '.join(f'{stats[f"{layout} {pr}"]["final_rel_gap_vs_dus"]:.0e}' for layout in LAYOUTS)
                              for pr in PRIMITIVES) + ' |')
    lines.append('| scan body: DUS fusions / scatter fusions / full-volume copies | ' + ' | '.join(
        counts([stats[f'{layout} {pr}']['hlo'] for layout in LAYOUTS], ('dus_fusions', 'scatter_fusions', 'full_volume_copies'), 'layout')
        for pr in PRIMITIVES) + ' |')
    lines.append(f'origins at N={n}: ' + '; '.join(f'{layout} {layout_origins(s, m)}' for layout, (s, m) in LAYOUTS.items()))
    return lines


def counts(hlo, keys, over):
    """One cell of op counts, collapsed when every entry of `hlo` (one per `over`: layout or N) agrees."""
    text = ['/'.join(str(h[k]) for k in keys) for h in hlo]
    return f'{text[0]} in every {over}' if len(set(text)) == 1 else '; '.join(text)


def reverse_table(reverse):
    """The JAX reverse mode at every N, one row per N over the REVERSE_ARMS, then the gradient checks and the op counts."""
    arms, sizes = list(REVERSE_ARMS), list(reverse)
    lines = ['| JAX reverse mode, interior s=8: grad of sum(E_final^2) w.r.t. the (12,) amplitude leaf and e0, forward+backward '
             f'ms/step (+ms over the baseline) +-half-range of {RUNS} runs | ' + ' | '.join(arms) + ' |', '| --- |' + ' --- |'*len(arms)]
    for n in sizes:
        t = reverse[n]['timings']
        lines.append(f'| N={n}, {reverse[n]["steps"]} scan steps | ' + ' | '.join(
            f'{t[a]["fwd_bwd_ms_per_step"]:.2f} ({t[a]["ms_over_baseline"]:+.2f}) +-{100*t[a]["spread"]:.0f}%' for a in arms) + ' |')
    lines.append('| scatter4 gradient vs its dus twin, amplitudes / e0, at N=' + ', '.join(str(n) for n in sizes) + ' | ' + ' | '.join(
        '; '.join(f'{reverse[n]["timings"][a]["grad_rel_err_vs_dus_amplitudes"]:.0e} / {reverse[n]["timings"][a]["grad_rel_err_vs_dus_e0"]:.0e}'
                  for n in sizes) if a.startswith('scatter4') else '-' for a in arms) + ' |')
    lines.append('| transposed scan body: full-volume copies / gather fusions / dynamic-slice fusions | ' + ' | '.join(
        counts([reverse[n]['timings'][a]['hlo'] for n in sizes], ('full_volume_copies', 'gather_fusions', 'dynamic_slice_fusions'), 'N')
        for a in arms) + ' |')
    return lines


class SlabAdd(torch.autograd.Function):
    """In-place slab add whose backward slices the cotangent instead of copying the whole volume."""
    @staticmethod
    def forward(ctx, E, v, sl):
        ctx.sl = sl
        E[sl].add_(v)
        ctx.mark_dirty(E)
        return E

    @staticmethod
    def backward(ctx, g):
        return g, g[ctx.sl], None


def slab_add(mode, E, v, sl, n):
    """Three ways of adding a slab into E under autograd; the in-place ones return E itself."""
    if mode == 'in-place add_':
        E[sl].add_(v)
        return E
    if mode == 'Function':
        return SlabAdd.apply(E, v, sl)
    return E + tnn.pad(v, tuple(b for s in reversed(sl[1:]) for b in (s.start, n - s.stop)))  # 'pad', out of place


def torch_grad_forward(mode, p, n, steps, wrong=False):
    """One differentiated run; `wrong` reads a view of the fresh E into the loss before the slab adds."""
    ts = [(sl, torch.from_numpy(x)) for sl, x in terms('slab x12 interior', p)]
    dense = [torch.from_numpy(x) for _, x in terms('dense x12', p)]
    amps = torch.tensor(p['amps'][:steps], requires_grad=True)
    E = torch.from_numpy(p['e0']).clone().requires_grad_()  # so the baseline's backward walks the kernel too
    H, acc = torch.from_numpy(p['h0']).clone(), torch.zeros((), dtype=torch.float64)
    for a in amps:
        H = H - C*curl(E, torch.roll, torch.stack, True)
        E = E + C*curl(H, torch.roll, torch.stack, False)  # a fresh tensor every step
        if wrong:
            acc = acc + (E[:, :, :, 5]**2).sum()  # a view of E, saved for backward
        for k, (sl, v) in enumerate(ts if mode != 'baseline' else []):
            E = E + a[k]*dense[k] if mode == 'dense' else slab_add(mode, E, a[k]*v, sl, n)
    return (E**2).sum() + acc, amps


def torch_grad_cost(p, n, steps, runs=3):
    """Forward with autograd on and backward, timed apart and interleaved, for four ways of adding twelve slabs."""
    modes = ('baseline', 'in-place add_', 'Function', 'pad', 'dense')
    fwd, bwd, grads = {m: [] for m in modes}, {m: [] for m in modes}, {}
    for r in range(runs + 1):  # the first round is the warm-up
        for mode in modes:
            t0 = time.perf_counter()
            loss, amps = torch_grad_forward(mode, p, n, steps)
            t1 = time.perf_counter()
            loss.backward()
            if r:
                fwd[mode].append(t1 - t0)
                bwd[mode].append(time.perf_counter() - t1)
            grads[mode] = amps.grad
    out = {m: dict(fwd_ms_per_step=1e3*float(np.median(fwd[m]))/steps, bwd_ms_per_step=1e3*float(np.median(bwd[m]))/steps)
           for m in modes}
    for mode in ('in-place add_', 'Function', 'dense'):
        out[mode]['grad_rel_err_vs_pad'] = float((grads[mode] - grads['pad']).norm()/grads['pad'].norm())
    return out


def torch_contract(n=32, steps=10):
    """In-place slab adds on a fresh E must reproduce the out-of-place gradient; a view read first must raise."""
    p = problem(n, steps, seed=1)

    def gradient(mode, wrong=False):
        loss, amps = torch_grad_forward(mode, p, n, steps, wrong)
        loss.backward()
        return amps.grad.numpy()
    ref, out = gradient('pad'), {}
    for mode in ('in-place add_', 'Function'):
        rel = float(np.linalg.norm(gradient(mode) - ref)/np.linalg.norm(ref))
        try:
            gradient(mode, wrong=True)
            wrong = dict(passed=False, message='no RuntimeError was raised')
        except RuntimeError as err:
            wrong = dict(passed=True, message=str(err).splitlines()[0])
        out[mode] = dict(correct=dict(passed=rel < 1e-12, relative_error=rel), wrong=wrong)
    return out


def table(n, steps, size):
    def cell(b, v):
        t, base = size[b, v], size[b, 'baseline']['ms_per_step']
        return f'{t["ms_per_step"]:.3f} ({t["ms_per_step"]/base:.2f}x) +-{100*t["spread"]:.0f}%'

    def per_term(b):
        base = size[b, 'baseline']['ms_per_step']
        d = {v: 1e3*(size[b, v]['ms_per_step'] - base) for v in VARIANTS if v != 'baseline'}
        return ' / '.join(f'{x:.0f}' for x in (d['slab x1 origin'], d['slab x12 interior']/TERMS, d['slab x12 face']/TERMS,
                                                d['slab x12 s=16']/TERMS, d['dense x12']/TERMS, d['dense x1']))

    def alone(b):
        return '-' if b == 'jax' else ' / '.join(f'{x:.0f}' for x in size[b, 'terms alone'].values())
    rows = [(v, cell) for v in VARIANTS] + [
        ('noise: baseline half-range, us/step', lambda b, v: f'{1e3*size[b, "baseline"]["ms_per_step"]*size[b, "baseline"]["spread"]:.0f}'),
        ('us over baseline: origin slab / interior slab / face slab / s=16 slab / dense term / full pass', lambda b, v: per_term(b)),
        ('terms alone, no kernel, us per call: slab add / dense add / in-place pass', lambda b, v: alone(b))]
    lines = ([f'| N={n}, {steps[0]} scan steps (jax) / {steps[1]} loop steps (numpy, torch), ms/step (x baseline) '
              f'+-half-range of {RUNS} runs | ' + ' | '.join(BACKENDS) + ' |',
              '| --- | --- | --- | --- |']
             + [f'| {v} | ' + ' | '.join(f(b, v) for b in BACKENDS) + ' |' for v, f in rows]
             + [f'origins at N={n}: ' + '; '.join(f'{lay} {origins(lay, S)}' for lay in ('origin', 'interior', 'face'))]
             + [f'JAX {v} HLO at N={n}: ' + json.dumps(size["jax", v]["hlo"]) for v in VARIANTS if v.startswith('slab')])
    if ('jax', 'sweep') in size:
        base = size['jax', 'baseline']['ms_per_step']
        lines.append(f'JAX one slab per step at N={n}, us/step over the baseline ({base:.3f} ms/step) +-half-range, '
                     'and whether LLVM unswitched the kernel (U) or left the full loop nest (F): '
                     + ', '.join(f'{k} {1e3*(t["ms_per_step"]-base):+.0f} +-{1e3*t["ms_per_step"]*t["spread"]:.0f}'
                                 + (' U' if t.get('unswitched') else ' F' if 'unswitched' in t else '')
                                 for k, t in size['jax', 'sweep'].items()))
    if ('torch', 'autograd') in size:
        g = size['torch', 'autograd']
        lines += [f'| Torch autograd at N={n}, {GRAD_STEPS[n]} steps, ms/step | forward, grad on | backward | over baseline | grad vs pad |',
                  '| --- | --- | --- | --- | --- |']
        for mode, t in g.items():
            over = ' / '.join(f'{t[k] - g["baseline"][k]:+.2f}' for k in ('fwd_ms_per_step', 'bwd_ms_per_step'))
            err = f'{t["grad_rel_err_vs_pad"]:.1e}' if 'grad_rel_err_vs_pad' in t else '-'
            lines.append(f'| slab x12 {mode} | {t["fwd_ms_per_step"]:.2f} | {t["bwd_ms_per_step"]:.2f} | {over} | {err} |')
    return lines


def variants_section(report):
    """The three backends over the variants at every N, the JAX sweep, the Torch autograd timings and the contract."""
    lines = []
    for n, steps in STEPS.items():
        p, size, finals = problem(n, steps[0]), {}, {}
        short = dict(p, amps=p['amps'][:steps[1]])  # the loop-length waveform: the field every backend must reproduce
        for backend in BACKENDS:
            lib = NUMPY if backend == 'numpy' else TORCH
            runs, extra = {}, {v: {} for v in VARIANTS}
            for variant in VARIANTS:
                ts = terms(variant, p)
                if backend == 'jax':
                    program, args = jax_program(variant, ts, p)
                    t0 = time.perf_counter()
                    program(*args).block_until_ready()
                    extra[variant]['compile_and_first_call_seconds'] = time.perf_counter() - t0
                    if variant.startswith('slab'):
                        extra[variant]['hlo'] = hlo_evidence(program, args, n)
                    check, cargs = jax_program(variant, ts, short)  # the loop-length field, for the check below
                    finals[backend, variant] = np.asarray(check(*cargs))
                    runs[variant] = lambda program=program, args=args: program(*args).block_until_ready()
                else:
                    runs[variant] = eager_runner(ts, short, lib)
            fields, stats = timed(runs, steps[0] if backend == 'jax' else steps[1])
            for variant in VARIANTS:
                size[backend, variant] = dict(stats[variant], **extra[variant])
                if backend != 'jax':
                    finals[backend, variant] = fields[variant]
                print(f'N={n} {backend:5s} {variant:18s} {stats[variant]["ms_per_step"]:8.3f} ms/step '
                      f'+-{100*stats[variant]["spread"]:.0f}%', flush=True)
            if backend != 'jax':
                size[backend, 'terms alone'] = eager_term_cost(p, lib)
        if n == 64:
            size['jax', 'sweep'] = jax_slab_sweep(p, n, steps[0])
        if n in GRAD_STEPS:
            size['torch', 'autograd'] = torch_grad_cost(p, n, GRAD_STEPS[n])
        gaps = {f'{b} slab-vs-dense': float(np.max(np.abs(finals[b, 'slab x12 interior'] - finals[b, 'dense x12'])))
                for b in BACKENDS}
        gaps.update({f'{b}-vs-numpy {v}': float(np.max(np.abs(finals[b, v] - finals['numpy', v])))
                     for b in ('jax', 'torch') for v in VARIANTS})
        assert max(gaps.values()) < 1e-9*np.max(np.abs(finals['numpy', 'baseline'])), gaps
        lines += table(n, steps, size)
        report['sizes'][n] = dict(steps=steps, max_abs_gaps=gaps, origins={lay: origins(lay, S) for lay in ('origin', 'interior', 'face')},
                                  timings={f'{b} {v}': t for (b, v), t in size.items()})
    report['torch_contract'] = contract = torch_contract()
    for mode, r in contract.items():
        lines.append(f'Torch autograd, correct order, {mode}: {"PASS" if r["correct"]["passed"] else "FAIL"} '
                     f'(relative gradient error {r["correct"]["relative_error"]:.2e} against out-of-place pad)')
        lines.append(f'Torch autograd, wrong order, {mode}: {"PASS" if r["wrong"]["passed"] else "FAIL"} ({r["wrong"]["message"]})')
    return lines


def primitives_section(report):
    """The JAX slab primitives and the reverse mode at every N."""
    lines, report['jax_primitives'], report['jax_reverse_mode'] = [], {}, {}
    for n, steps in STEPS.items():
        p = problem(n, steps[0])
        report['jax_primitives'][n] = stats = jax_primitives(p, n, steps[0])
        report['jax_reverse_mode'][n] = dict(steps=JAX_GRAD_STEPS[n], layout='interior s=8',
                                             timings=jax_reverse_mode(p, n, JAX_GRAD_STEPS[n]))
        lines += primitives_table(n, steps[0], stats)
    return lines + reverse_table(report['jax_reverse_mode'])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('out', nargs='?', help='the JSON output (default studies/results/fdtd_slab_add_cost[.scatter].json)')
    parser.add_argument('--section', choices=('all', 'scatter'), default='all',
                        help='"scatter" runs the JAX slab primitives section alone; the default runs everything')
    opts = parser.parse_args()
    out_path = opts.out or f'studies/results/fdtd_slab_add_cost{"" if opts.section == "all" else ".scatter"}.json'
    report = dict(machine=dict(
        platform=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count(),
        python=platform.python_version(), numpy=np.__version__, jax=jax.__version__,
        torch=torch.__version__, jax_devices=[str(d) for d in jax.devices()],
        torch_threads=(torch.get_num_threads(), torch.get_num_interop_threads()),
        env={k: v for k, v in os.environ.items() if re.match(r'(OMP|MKL|OPENBLAS|XLA|JAX)_', k)}),
        constants=dict(c=C, slab=S, terms=TERMS, runs=RUNS, steps=STEPS, grad_steps=GRAD_STEPS, jax_grad_steps=JAX_GRAD_STEPS,
                       shift=SHIFT, layouts={k: layout_origins(s, m) for k, (s, m) in LAYOUTS.items()}, reverse_arms=REVERSE_ARMS),
        sizes={})
    lines = variants_section(report) if opts.section == 'all' else []
    lines += primitives_section(report)
    print('\n'.join(lines))
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=1, allow_nan=False)
    shutil.rmtree(DUMP, ignore_errors=True)


if __name__ == '__main__':
    main()
