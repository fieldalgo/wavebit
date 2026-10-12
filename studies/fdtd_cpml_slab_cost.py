"""What a CPML face costs per FDTD step: psi over the slab against psi over the full volume.

A CPML face is, in the general-purpose FDTD's kernel, a zero face rule plus a slab of 12 cells
inside that face where the derivative along the face's axis is stretched: per psi,
``psi <- b psi + a d`` and ``d_stretched = d/kappa + psi``. The difference along one axis
appears in two of the three curl components, on two components of the field, in each of the two
curls, so one face carries FOUR psi arrays. Where they live is a free choice with a factor of
``N/24`` in memory behind it:

* **slab** -- psi is ``(12, <full transverse>)``, one array per face per psi: the step reads the
  slab of the bare difference, advances psi and writes the stretched value back (a gather and a
  scatter per psi), and nothing outside the slab holds psi at all;
* **full volume** -- psi is ``(N, N, N)`` with its coefficients zero outside the slabs, so both
  ends of an axis share one array (four per AXIS, not per face): one full-volume multiply-add
  per psi for the recurrence and one for the stretch, most of it on zeros.

Nothing here imports either FDTD engine. The CPML coefficients are ported from
``src/wavebit/_core/fdtd.py::cpml_coefficients`` (a kernel seed), and they only fix the arithmetic
per cell -- two fused multiply-adds -- not the answer, which this study never checks against
physics. What it does check is that every arm computes the same field: the slab layouts are held
to their full-volume twin at every size, and the three backends to each other.

Stand-in kernel: the slab study's ``H -= c*curl(E); E += c*curl(H)`` with periodic one-sided
differences by axis rolls -- the memory traffic of a Yee update, not its physics -- on
``(3, N, N, N)`` float64 fields, N = 64, 96, 128, with the stretch hooked where
``src/wavebit/fd/_kernel/curl.py`` says stage 1 will put it: on each bare difference, before the
curl components are assembled. Arms: ``baseline`` (no absorber); ``full psi`` and ``slab psi``
at ``2 faces (x)`` (one axis, both ends, on the LAST array axis) and at ``6 faces``. Three ways
of writing the slab layout are timed on JAX -- ``dus``, ``d.at[slices].add``; ``scatter``, an
index scatter-add on the 3-D difference with the index arrays built once per face and shared by
its four psi; ``merged``, one 4-D scatter per face per curl carrying both curl components'
corrections at once, on the assembled curl -- and two on the eager backends (``--section eager``,
in minutes): ``inplace``, the gather-correct-add-back above, and ``fused``, the recurrence and
the stretch as four in-place operations on the slab view, gathering nothing and allocating
nothing. psi is a separate entry of the JAX scan carry in every arm; a recurrence cannot be
recomputed instead of carried, so there is no third option there.

Protocol. The JAX scan runs 200 steps at every N, the eager loops 200/100/50 at N = 64/96/128.
Every runner is warmed once (the compile, on JAX); five timed rounds then go over all of a
backend's arms in turn, so a slow phase of the machine hits every arm alike, and a cell is the
median ms/step with the half-range of its five samples. Each JAX program is lowered and
compiled: the optimized scan body is counted for dynamic-update-slice, scatter, gather and
full-volume fusions and for full-volume copies, and the compiled executable's own memory
analysis is read for its argument, output and temporary bytes. The psi and index bytes are also
computed exactly from the shapes. At N=128 the same two-face question is asked of each axis in
turn (a face slab on axis 0 restricts the outermost loop of a full-volume kernel, one on axis 2
the innermost). The cross-backend check runs every arm of every backend at N=64 for 50 steps.
The JAX reverse mode (N=64 for 50 steps, N=128 for 10) differentiates ``sum(E_final^2)`` with
respect to a scalar on the initial field and to the field itself, holds every slab gradient to
its full-volume twin's, and counts the transposed scan body's ops.

Numbers are CPU-only, on this machine, 2026-09-18 (12 logical CPUs, Zen 4, 32 GB; NumPy 2.5.1,
JAX 0.11.0 on one CpuDevice, Torch 2.13.0+cpu on 6 threads; no OMP/MKL/OPENBLAS/XLA variables
set). GPU is stage 6 and NOTHING here transfers to it: the DUS-versus-scatter answer below is
LLVM's cost model on XLA's CPU lowering, the eager rows are a Python loop over CPU kernels, and
a GPU would change the arithmetic-to-bandwidth ratio that decides full against slab. Re-measure
there, in this order: the three slab writings on JAX, the eager ``fused`` writing on Torch, and
the reverse-mode copy counts. ``time.process_time`` under-reports on this Windows build, so no
cpu/wall figure is given. Run from the repo root; an optional argument names the JSON output
(default ``studies/results/fdtd_cpml_slab_cost.json``, or ``.eager.json`` under
``--section eager``); the checked-in log beside it is the printed output, the full run first and
the eager section from its own run after it. Measured:

| N=64, 200 scan steps (jax) / 200 loop steps (numpy, torch), ms/step (x baseline) +-half-range of 5 runs | numpy | jax | torch | psi MB | index MB |
| --- | --- | --- | --- | --- | --- |
| baseline | 20.740 (1.00x) +-19% | 0.390 (1.00x) +-2% | 4.050 (1.00x) +-7% | 0.00 | 0.00 |
| full psi, 2 faces (x) | 27.255 (1.31x) +-19% | 0.780 (2.00x) +-5% | 5.611 (1.39x) +-3% | 8.00 | 0.00 |
| slab psi inplace, 2 faces (x) | 22.606 (1.09x) +-11% | - | 7.152 (1.77x) +-1% | 3.00 | 0.00 |
| slab psi dus, 2 faces (x) | - | 0.679 (1.74x) +-8% | - | 3.00 | 0.00 |
| slab psi scatter, 2 faces (x) | - | 1.236 (3.17x) +-12% | - | 3.00 | 1.12 |
| slab psi merged, 2 faces (x) | - | 2.951 (7.57x) +-20% | - | 3.00 | 3.00 |
| full psi, 6 faces | 41.820 (2.02x) +-21% | 2.826 (7.25x) +-4% | 7.965 (1.97x) +-1% | 24.00 | 0.00 |
| slab psi inplace, 6 faces | 25.319 (1.22x) +-19% | - | 8.997 (2.22x) +-3% | 9.00 | 0.00 |
| slab psi dus, 6 faces | - | 2.074 (5.32x) +-18% | - | 9.00 | 0.00 |
| slab psi scatter, 6 faces | - | 3.284 (8.42x) +-4% | - | 9.00 | 3.38 |
| slab psi merged, 6 faces | - | 7.267 (18.63x) +-6% | - | 9.00 | 9.00 |
| noise: baseline half-range, us/step | 3863 | 9 | 277 | - | - |
| us over baseline per psi array, full / slab: 2 faces; 6 faces | 1629 / 233; 1757 / 191 | 97 / 106; 203 / 121 | 390 / 388; 326 / 206 | - | - |
| slab field vs its full-psi twin, max relative gap | 1e-15/1e-15 | 1e-15/1e-15/1e-15/2e-15/2e-15/2e-15 | 1e-15/1e-15 | - | - |

JAX scan body at N=64 (DUS / scatter / gather / full-volume fusions / full-volume copies): baseline 0/0/0/2/0; full psi, 2 faces (x) 0/0/0/10/0; slab psi dus, 2 faces (x) 6/0/0/12/0; slab psi scatter, 2 faces (x) 0/8/0/16/0; slab psi merged, 2 faces (x) 0/4/0/12/0; full psi, 6 faces 0/0/0/26/0; slab psi dus, 6 faces 14/0/0/32/0; slab psi scatter, 6 faces 0/24/0/44/0; slab psi merged, 6 faces 0/12/0/28/0
JAX compiled bytes at N=64, argument / output / temp, MB: baseline 12/6/6; full psi, 2 faces (x) 12/6/18; slab psi dus, 2 faces (x) 12/6/15; slab psi scatter, 2 faces (x) 12/6/17; slab psi merged, 2 faces (x) 12/6/21; full psi, 6 faces 12/6/42; slab psi dus, 6 faces 12/6/29; slab psi scatter, 6 faces 12/6/39; slab psi merged, 6 faces 12/6/35

| N=96, 200 scan steps (jax) / 100 loop steps (numpy, torch), ms/step (x baseline) +-half-range of 5 runs | numpy | jax | torch | psi MB | index MB |
| --- | --- | --- | --- | --- | --- |
| baseline | 73.680 (1.00x) +-6% | 2.098 (1.00x) +-7% | 21.693 (1.00x) +-2% | 0.00 | 0.00 |
| full psi, 2 faces (x) | 103.799 (1.41x) +-6% | 6.780 (3.23x) +-5% | 25.809 (1.19x) +-1% | 27.00 | 0.00 |
| slab psi inplace, 2 faces (x) | 86.313 (1.17x) +-7% | - | 24.729 (1.14x) +-2% | 6.75 | 0.00 |
| slab psi dus, 2 faces (x) | - | 5.789 (2.76x) +-6% | - | 6.75 | 0.00 |
| slab psi scatter, 2 faces (x) | - | 8.332 (3.97x) +-6% | - | 6.75 | 2.53 |
| slab psi merged, 2 faces (x) | - | 17.237 (8.22x) +-3% | - | 6.75 | 6.75 |
| full psi, 6 faces | 154.737 (2.10x) +-2% | 17.412 (8.30x) +-1% | 36.640 (1.69x) +-1% | 81.00 | 0.00 |
| slab psi inplace, 6 faces | 94.727 (1.29x) +-2% | - | 28.171 (1.30x) +-2% | 20.25 | 0.00 |
| slab psi dus, 6 faces | - | 12.467 (5.94x) +-2% | - | 20.25 | 0.00 |
| slab psi scatter, 6 faces | - | 15.924 (7.59x) +-3% | - | 20.25 | 7.59 |
| slab psi merged, 6 faces | - | 26.655 (12.70x) +-1% | - | 20.25 | 20.25 |
| noise: baseline half-range, us/step | 4110 | 152 | 434 | - | - |
| us over baseline per psi array, full / slab: 2 faces; 6 faces | 7530 / 1579; 6755 / 877 | 1171 / 779; 1276 / 576 | 1029 / 380; 1246 / 270 | - | - |
| slab field vs its full-psi twin, max relative gap | 9e-16/1e-15 | 1e-15/1e-15/1e-15/2e-15/2e-15/1e-15 | 9e-16/1e-15 | - | - |

JAX scan body at N=96: the same counts as N=64, arm for arm.
JAX compiled bytes at N=96, argument / output / temp, MB: baseline 40/20/20; full psi, 2 faces (x) 40/20/61; slab psi dus, 2 faces (x) 40/20/47; slab psi scatter, 2 faces (x) 40/20/55; slab psi merged, 2 faces (x) 40/20/64; full psi, 6 faces 40/20/142; slab psi dus, 6 faces 40/20/88; slab psi scatter, 6 faces 40/20/122; slab psi merged, 6 faces 40/20/105

| N=128, 200 scan steps (jax) / 50 loop steps (numpy, torch), ms/step (x baseline) +-half-range of 5 runs | numpy | jax | torch | psi MB | index MB |
| --- | --- | --- | --- | --- | --- |
| baseline | 167.270 (1.00x) +-8% | 17.483 (1.00x) +-1% | 73.354 (1.00x) +-3% | 0.00 | 0.00 |
| full psi, 2 faces (x) | 227.042 (1.36x) +-3% | 24.298 (1.39x) +-1% | 95.207 (1.30x) +-2% | 64.00 | 0.00 |
| slab psi inplace, 2 faces (x) | 187.085 (1.12x) +-1% | - | 79.056 (1.08x) +-3% | 12.00 | 0.00 |
| slab psi dus, 2 faces (x) | - | 23.890 (1.37x) +-1% | - | 12.00 | 0.00 |
| slab psi scatter, 2 faces (x) | - | 29.498 (1.69x) +-3% | - | 12.00 | 4.50 |
| slab psi merged, 2 faces (x) | - | 43.220 (2.47x) +-2% | - | 12.00 | 12.00 |
| full psi, 6 faces | 345.982 (2.07x) +-2% | 48.704 (2.79x) +-3% | 137.918 (1.88x) +-1% | 192.00 | 0.00 |
| slab psi inplace, 6 faces | 211.101 (1.26x) +-1% | - | 86.308 (1.18x) +-2% | 36.00 | 0.00 |
| slab psi dus, 6 faces | - | 39.693 (2.27x) +-3% | - | 36.00 | 0.00 |
| slab psi scatter, 6 faces | - | 49.502 (2.83x) +-1% | - | 36.00 | 13.50 |
| slab psi merged, 6 faces | - | 62.783 (3.59x) +-1% | - | 36.00 | 36.00 |
| noise: baseline half-range, us/step | 13207 | 200 | 1882 | - | - |
| us over baseline per psi array, full / slab: 2 faces; 6 faces | 14943 / 2477; 14893 / 1826 | 1704 / 1502; 2602 / 1334 | 5463 / 713; 5380 / 540 | - | - |
| slab field vs its full-psi twin, max relative gap | 7e-16/7e-16 | 1e-15/1e-15/1e-15/1e-15/1e-15/1e-15 | 7e-16/7e-16 | - | - |

JAX scan body at N=128 (DUS / scatter / gather / full-volume fusions / full-volume copies): baseline 0/0/0/2/0; full psi, 2 faces (x) 0/0/0/6/0; slab psi dus, 2 faces (x) 6/0/0/10/0; slab psi scatter, 2 faces (x) 0/8/0/14/0; slab psi merged, 2 faces (x) 0/4/0/8/0; full psi, 6 faces 0/0/0/22/0; slab psi dus, 6 faces 14/0/0/30/0; slab psi scatter, 6 faces 0/24/0/42/0; slab psi merged, 6 faces 0/12/0/24/0
JAX compiled bytes at N=128, argument / output / temp, MB: baseline 96/48/96; full psi, 2 faces (x) 96/48/160; slab psi dus, 2 faces (x) 96/48/108; slab psi scatter, 2 faces (x) 96/48/126; slab psi merged, 2 faces (x) 96/48/146; full psi, 6 faces 96/48/352; slab psi dus, 6 faces 96/48/212; slab psi scatter, 6 faces 96/48/260; slab psi merged, 6 faces 96/48/240

| exact bytes, float64 psi and int32 indices, 12-cell faces | 2 faces | 6 faces | fields E and H |
| --- | --- | --- | --- |
| N=64 | slab 3.0 / full 8.0 MB (2.7x); slab indices 1.1 (scatter) / 3.0 (merged) MB | slab 9.0 / full 24.0 MB (2.7x); slab indices 3.4 (scatter) / 9.0 (merged) MB | 12.0 MB |
| N=96 | slab 6.8 / full 27.0 MB (4.0x); slab indices 2.5 (scatter) / 6.8 (merged) MB | slab 20.2 / full 81.0 MB (4.0x); slab indices 7.6 (scatter) / 20.2 (merged) MB | 40.5 MB |
| N=128 | slab 12.0 / full 64.0 MB (5.3x); slab indices 4.5 (scatter) / 12.0 (merged) MB | slab 36.0 / full 192.0 MB (5.3x); slab indices 13.5 (scatter) / 36.0 (merged) MB | 96.0 MB |

Cross-backend check at N=64, 50 steps: every arm of every backend within 8e-16 of the NumPy
field of the same arm and of its full-psi twin (largest of 29 comparisons).

| JAX per-axis face pair at N=128, 200 scan steps, us over the baseline (17.48 ms/step +-3%) +-half-range | z pair (axis 0) | y pair (axis 1) | x pair (axis 2) |
| --- | --- | --- | --- |
| full psi | +12211 +-1093 | +12953 +-948 | +7061 +-281 |
| slab psi dus | +7238 +-1384 | +8161 +-2173 | +6585 +-1109 |
| slab psi scatter | +9503 +-918 | +10468 +-1938 | +12910 +-499 |
| slab psi merged | +15581 +-1414 | +16829 +-1237 | +25774 +-1077 |

scan body per axis pair at N=128 (DUS / scatter / gather / full-volume fusions / copies): the z and y
pairs give 0/0/0/10/0 (full), 6/0/0/12/0 (dus), 0/8/0/16/0 (scatter), 0/4/0/12/0 (merged); the x pair
carries fewer full-volume fusions in each -- 6, 10, 14 and 8 against 10, 12, 16 and 12.

| JAX reverse mode, 2 faces (x): grad of sum(E_final^2) w.r.t. a scalar on the initial field and the field itself, forward+backward ms/step (+ms over the baseline) +-half-range of 5 runs | baseline | full psi | slab psi dus | slab psi scatter | slab psi merged |
| --- | --- | --- | --- | --- | --- |
| N=64, 50 scan steps | 1.41 (+0.00) +-3% | 3.07 (+1.66) +-6% | 3.27 (+1.86) +-7% | 3.33 (+1.92) +-9% | 4.75 (+3.34) +-10% |
| N=128, 10 scan steps | 33.56 (+0.00) +-3% | 54.35 (+20.79) +-1% | 58.32 (+24.76) +-2% | 60.38 (+26.82) +-2% | 74.00 (+40.44) +-2% |
| gradient vs the full-psi twin, at N=64, 128 | - | - | 1e-15; 2e-16 | 1e-15; 2e-16 | 1e-15; 2e-16 |
| transposed scan body: full-volume copies / gather / dynamic-slice / scatter fusions | 0/0/0/0; 0/0/0/0 | 5/0/0/0; 5/0/0/0 | 1/0/0/0; 1/0/0/0 | 0/8/0/0; 0/8/0/0 | 0/4/0/0; 0/4/0/0 |

| eager slab writings at N=64, 200 loop steps, ms/step (x baseline) +-half-range of 5 runs | numpy | torch | psi MB |
| --- | --- | --- | --- |
| baseline | 19.867 (1.00x) +-1% | 3.637 (1.00x) +-2% | 0.00 |
| full psi, 2 faces (x) | 26.916 (1.35x) +-1% | 5.538 (1.52x) +-3% | 8.00 |
| slab psi inplace, 2 faces (x) | 22.255 (1.12x) +-1% | 7.040 (1.94x) +-5% | 3.00 |
| slab psi fused, 2 faces (x) | 22.098 (1.11x) +-1% | 6.732 (1.85x) +-2% | 3.00 |
| full psi, 6 faces | 40.838 (2.06x) +-0% | 7.963 (2.19x) +-3% | 24.00 |
| slab psi inplace, 6 faces | 24.887 (1.25x) +-1% | 9.233 (2.54x) +-4% | 9.00 |
| slab psi fused, 6 faces | 24.817 (1.25x) +-1% | 7.865 (2.16x) +-3% | 9.00 |
| us over baseline per psi array, full / inplace / fused: 2 faces; 6 faces | 1762 / 298 / 279; 1748 / 209 / 206 | 475 / 425 / 387; 361 / 233 / 176 | - |
| slab field vs its full-psi twin, max relative gap | 1e-15/0e+00/1e-15/0e+00 | 1e-15/0e+00/1e-15/0e+00 | - |

| eager slab writings at N=128, 50 loop steps, ms/step (x baseline) +-half-range of 5 runs | numpy | torch | psi MB |
| --- | --- | --- | --- |
| baseline | 165.289 (1.00x) +-1% | 71.442 (1.00x) +-2% | 0.00 |
| full psi, 2 faces (x) | 223.274 (1.35x) +-1% | 93.989 (1.32x) +-1% | 64.00 |
| slab psi inplace, 2 faces (x) | 183.254 (1.11x) +-0% | 77.187 (1.08x) +-4% | 12.00 |
| slab psi fused, 2 faces (x) | 175.890 (1.06x) +-1% | 77.563 (1.09x) +-2% | 12.00 |
| full psi, 6 faces | 337.670 (2.04x) +-1% | 135.681 (1.90x) +-1% | 192.00 |
| slab psi inplace, 6 faces | 205.252 (1.24x) +-1% | 84.200 (1.18x) +-5% | 36.00 |
| slab psi fused, 6 faces | 183.571 (1.11x) +-1% | 80.742 (1.13x) +-3% | 36.00 |
| us over baseline per psi array, full / inplace / fused: 2 faces; 6 faces | 14496 / 2246 / 1325; 14365 / 1665 / 762 | 5637 / 718 / 765; 5353 / 532 / 387 | - |
| slab field vs its full-psi twin, max relative gap | 7e-16/0e+00/7e-16/0e+00 | 7e-16/0e+00/7e-16/0e+00 | - |

(The eager section ran alone after the full run. Its baselines stand at 19.867 and 165.289 ms/step
against the full run's 20.740 and 167.270 on NumPy, and 3.637 and 71.442 against 4.050 and 73.354
on Torch: 1-10% apart, the machine rather than the code, and every comparison above is inside its
own interleaved rounds. The ``fused`` arm reproduces its full-psi twin's field EXACTLY at every
size -- it evaluates the same ``d/kappa + psi`` -- while ``inplace``, which adds the correction
``d (1/kappa - 1) + psi``, differs by one rounding per cell per step, 7e-16 to 1e-15 of the field
after 50-200 steps.)

Reading. **The slab layout wins, on time and on memory, and it is not close.** Memory first,
because it is exact and N-dependent: psi over the full volume is ``4 N^3`` doubles per absorbing
axis against ``4 * 12 N^2`` per face, a factor ``N/24`` -- 2.7x at N=64, 4.0x at 96, 5.3x at 128.
Six faces at N=128 is 192 MB of psi against 36 MB, where the two fields together are 96 MB: the
full-volume layout costs twice the entire field state to hold the memory of a boundary, and the
slab layout costs a third of it. XLA's own memory analysis agrees on the whole program: 352 MB
of temporaries against 212 MB (baseline 96 MB) for the same scene.

Time says the same thing everywhere except one corner. Six faces at N=128 cost 2.04-2.07x the
no-absorber baseline on NumPy with full psi and 1.11x with slab psi (fused), 1.88-1.90x against
1.13x on Torch, and 2.79x against 2.27x on JAX; at N=96, 2.10x against 1.29x, 1.69x against
1.30x, 8.30x against 5.94x. Per psi array the full-volume cost tracks the volume and the slab
cost tracks the slab: at N=128, 14.4-14.9 ms per full-volume psi against 0.8-1.3 ms per slab psi
on NumPy (a 11-19x ratio where the cells differ by 5.3x -- the full layout pays two full-volume
passes per psi where the slab pays two slab passes plus the loop overhead), and 5.4-5.6 against
0.4-0.8 on Torch. The one corner where the full layout wins is a SMALL box with FEW faces on the
eager backends: at N=64 with two faces Torch costs 1.52x (full) against 1.85x (slab, fused),
because eight small in-place operations per step lose to four large ones when a 12 MB field sits
in cache and the per-operation dispatch is the cost. That corner closes by N=96 (1.19x against
1.14x) and inverts hard by N=128, and it never opens on NumPy. It is not a reason to carry a
second layout.

**On JAX the slab must be written as the slice add, not as the index scatter -- the opposite of
the stage-0 term primitive, for a reason that does not contradict it.** ``d.at[slices].add`` on
the difference array is the cheapest slab writing at every size and on every axis: +6585 to
+8161 us over the baseline at N=128 for a face pair, against +9503 to +12910 for the scatter and
+15581 to +25774 for the merged 4-D scatter, and it is the only writing that beats the full
layout at every N (2.27x against 2.79x at six faces at N=128, 5.32x against 7.25x at N=64). The
stage-0 study rejected the slice add because a small cube at an arbitrary origin lowers to a
full-volume select loop that LLVM unswitches only when the origin is a multiple of the width; a
CPML slab is a different shape entirely -- full extent on two axes, contiguous on the third,
anchored at 0 or at N-12 -- so the loop bounds are exact on every axis and there is nothing to
unswitch away. The scatter loses here because its index traffic is proportional to the slab:
three int32 per cell is 1.5x the float64 psi it addresses (4.5 MB of indices per face pair at
N=128, 13.5 MB at six faces), and the merged writing's four int32 over two components is 36 MB
at six faces -- as many bytes as all the psi. So the design's two slab primitives are not one
primitive: ``rt.slab_add`` for a small term stays the scatter, and a CPML face wants a slice
add. Both are XLA's CPU lowering and must be re-measured when JAX moves.

**The axis matters, and it matters in opposite directions for the two layouts.** At N=128 the
full layout costs +12211/+12953 us on the z and y face pairs but only +7061 on x, where XLA
fuses four full-volume operations away (6 full-volume fusions in the scan body against 10); the
slab scatter and the merged scatter go the other way, +9503/+10468 on z and y against +12910 and
+25774 on x, where the slab is the innermost, strided axis. The slice add is nearly flat
(+7238/+8161/+6585), which is the second reason to prefer it: a six-face scene pays the same on
every face.

**Differentiated, the slab does not cost the stage-0 copy, and which writing decides that.** The
transposed scan body of the baseline holds no full-volume copy; with full-volume psi it holds
FIVE, with the slab written as a slice add exactly ONE -- the same single copy per backward step
the slab study found for a term -- and with the slab written as either scatter, NONE at all (8
and 4 gather fusions instead, the scatters' transpose). The reason the CPML term can reach zero
where a stage-0 source term could not is that it writes into the per-step difference, a
temporary, not into the carried field, so XLA has no carry cotangent to keep apart. The times do
not follow the copy counts, though: forward+backward at N=128 costs 54.35 ms/step with full psi,
58.32 with the slice add, 60.38 with the scatter and 74.00 with the merged scatter, over a 33.56
baseline, and at N=64 3.07 / 3.27 / 3.33 / 4.75 over 1.41 -- so differentiating a two-face CPML
costs +1.7 to +1.9 ms at N=64 and +21 to +27 ms at N=128 whatever the layout, and the five
copies the full layout pays are bought back by the work the slab layout's gathers and small ops
do. Read the copy counts as the reason the slab layout does not get WORSE under reverse mode,
not as a saving; the differentiated run is its own line in the budget either way. Every slab
gradient agrees with its full-volume twin to 1e-15.

**On the eager backends the writing matters as much as the layout.** Holding the layout fixed
and changing only how it is written -- ``fused``, which runs ``psi <- b psi + a d`` and
``d <- d/kappa + psi`` as four in-place operations on the slab view, against ``inplace``, which
gathers the slab, forms the correction and adds it back -- takes NumPy from 1.24x to 1.11x and
Torch from 1.18x to 1.13x at six faces at N=128, and Torch from 2.54x to 2.16x at N=64, where it
is what pulls the slab layout back level with the full one (2.16x against 2.19x). Per psi array
at N=128 it is 0.76 ms against 1.67 (NumPy) and 0.39 against 0.53 (Torch). It gathers nothing
and allocates nothing -- NumPy needs one slab-shaped scratch for the ``a*d`` product, Torch needs
none (``addcmul_``) -- and it reproduces the full-volume field bit for bit, which the gather form
does not. Under autograd neither eager form applies unchanged: the stage-0 study already settled
that an in-place slab write on Torch must go through a ``mark_dirty`` ``autograd.Function``, and
a CPML psi is read as well as written, so its Function's backward is not the slab-add one. That
is unmeasured here.

Noise. The JAX cells reproduce to a few percent and their half-ranges are 1-8% except where the
absolute cost is small. The eager loops allocate every temporary afresh and their absolute values
move with the machine: the NumPy N=64 block of the full run carries +-19-21% half-ranges against
+-1% in the eager section run 40 minutes later, and its baseline moved 20.740 -> 19.867 ms/step.
Trust the ratios within a block, the orderings, and the per-psi rows; not the third digit of an
eager cell.
"""
import argparse
import json
import os
import platform
import re
import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)  # before any array exists
import jax.numpy as jnp
import torch
from jax import lax

C, CELLS, RUNS = .4, 12, 5
WAVELENGTH, RESOLUTION = 0.55, 20.0
DX = WAVELENGTH/RESOLUTION
DT = 0.99*DX/np.sqrt(3)                       # the stage-0 vacuum bound, at courant 0.99
PML_ORDER, PML_KAPPA_MAX, PML_ALPHA_RELATIVE, PML_TARGET = 4.0, 2.0, 1.5, 1e-8
STEPS = {64: (200, 200), 96: (200, 100), 128: (200, 50)}  # (JAX scan length, NumPy/Torch loop length)
CHECK = (64, 50)                              # the cross-backend check: one size, one step count
GRAD_STEPS = {64: 50, 128: 10}                # the JAX reverse mode, at the two sizes asked for
SWEEP_N = 128                                 # the per-axis face sweep, JAX only
EAGER_SIZES = (64, 128)                       # the second eager writing, --section eager
BACKENDS = ('numpy', 'jax', 'torch')
WRITINGS = {'numpy': ('inplace',), 'torch': ('inplace',), 'jax': ('dus', 'scatter', 'merged')}
EAGER_WRITINGS = ('inplace', 'fused')         # what --section eager compares, NumPy and Torch
SCATTER = dict(indices_are_sorted=True, unique_indices=True, mode='promise_in_bounds')
#: Arrays are ``[component, z, y, x]``, so axis 0 is z and axis 2 is x (``src/wavebit/fd/__init__.py``).
FACESETS = {'2 faces (x)': ((2, '-'), (2, '+')),
            '6 faces': tuple((a, s) for a in range(3) for s in '-+')}
AXIS_PAIRS = {'z pair (axis 0)': ((0, '-'), (0, '+')), 'y pair (axis 1)': ((1, '-'), (1, '+')),
              'x pair (axis 2)': ((2, '-'), (2, '+'))}
# roll, stack, copy and the array constructor of each eager library
NUMPY = (np.roll, np.stack, np.copy, np.asarray)
TORCH = (torch.roll, torch.stack, torch.clone, torch.from_numpy)


def cpml(depth, thickness, dt, wavelength, eps_r=1.0, target=PML_TARGET):
    """``(b, a, kappa)`` of the CFS-PML at ``depth`` into a layer (Roden & Gedney 2000).

    The formula, the three constants and the units audit are ported from
    ``src/wavebit/_core/fdtd.py::cpml_coefficients``, which the stage-1 design names as a kernel
    seed: copied with the comment that says what was measured, never imported. ``sigma`` here
    lives in a stretched coordinate, not in a material, so a fixed round-trip target needs
    ``sigma_max`` proportional to ``1/sqrt(eps_r)`` -- Taflove & Hagness eq. 7.61 carries it the
    other way up because there it is a material conductivity. ``alpha`` is a frequency, set
    relative to the operating one, and grades opposite to ``sigma``: that is what makes this
    complex-frequency-shifted rather than plain PML. Nothing in this study depends on the
    values being right, only on their shape, which fixes the arithmetic per cell: one fused
    multiply-add for the recurrence and one for the stretch.
    """
    sigma_max = -(PML_ORDER + 1.0)*np.log(target)/(2.0*np.sqrt(eps_r)*thickness)
    alpha_max = PML_ALPHA_RELATIVE*2.0*np.pi/wavelength
    f = np.clip(depth/thickness, 0.0, 1.0)
    sigma = sigma_max*f**PML_ORDER
    kappa = 1.0 + (PML_KAPPA_MAX - 1.0)*f**PML_ORDER
    alpha = alpha_max*(1.0 - f)
    b = np.exp(-(sigma/kappa + alpha)*dt)
    denom = kappa*(sigma + kappa*alpha)
    a = np.where(denom == 0.0, 0.0, sigma*(b - 1.0)/np.where(denom == 0.0, 1.0, denom))
    return b, a, kappa


def face_profile(side, field, cells=CELLS, dx=DX):
    """``(b, a, 1/kappa)`` of one face as ``(cells,)`` arrays, at the E points or the H points.

    The H points sit half a cell further in on a low face and half a cell further out on a high
    one, so a face's two profiles are genuinely different arrays -- the same arithmetic either
    way, which is all this study measures.
    """
    i = np.arange(cells)
    half = 0.5 if field == 'H' else 0.0
    depth = (cells - i - 0.5 - half)*dx if side == '-' else (i + 0.5 + half)*dx
    b, a, kappa = cpml(np.clip(depth, 0.0, None), cells*dx, DT, WAVELENGTH)
    return b, a, 1.0/kappa


def pml_pair(axis):
    """The two ``(component of F, component of curl F, sign)`` that ``D_axis`` feeds.

    ``(curl F)_c = D_a F_b - D_b F_a`` with ``a = c+1``, ``b = c+2`` (mod 3), so the difference
    along one axis appears in exactly two of the three curl components, on two different
    components of F. Two of those per curl and two curls is the design's four psi per face.
    """
    return (((axis + 1) % 3, (axis + 2) % 3, +1.0), ((axis + 2) % 3, (axis + 1) % 3, -1.0))


def slab_of(axis, side, n, cells=CELLS):
    """The ``cells``-deep slab inside one face, as one slice per axis of a single component."""
    lo = 0 if side == '-' else n - cells
    return tuple(slice(lo, lo + cells) if ax == axis else slice(None) for ax in range(3))


def slab_indices(sl, n, comps=()):
    """The sorted, unique index arrays of a slab, over ``comps`` (a 4-D scatter) or without."""
    rng = [np.arange(s.start or 0, n if s.stop is None else s.stop) for s in sl]
    grid = [g.reshape(-1).astype(np.int32) for g in np.meshgrid(*rng, indexing='ij')]
    if not comps:
        return tuple(grid)
    size = grid[0].size
    return (np.concatenate([np.full(size, c, np.int32) for c in comps]),
            *(np.concatenate([g]*len(comps)) for g in grid))


def build_plan(layout, writing, faces, n, cells=CELLS):
    """Everything one arm's step needs: the psi terms, their coefficients and their indices.

    A **slab** term is one psi array of ``(cells, <full transverse>)`` per face, per field, per
    component of F -- four per face -- gathered out of the bare difference and written back into
    it. A **full** term is one psi array of ``(n, n, n)`` per axis, per field, per component of
    F, its coefficients zero outside the slabs, so both ends of an axis share one array: four
    per axis, not per face.

    Index arrays are built once per face and shared by that face's four psi (and by both
    curls), because they are not free: three int32 per cell is 1.5 times the float64 psi array
    they address, and the 4-D indices of the merged writing are four int32 per cell over two
    components -- as many bytes as all four psi arrays of the face.
    """
    plan = dict(layout=layout, writing=writing, terms=[], merged={}, psi={}, index={})
    if layout is None:
        return plan
    for axis in sorted({a for a, _ in faces}):
        sides = [s for a, s in faces if a == axis]
        for compF, compC, sign in pml_pair(axis):
            for field in 'EH':
                shape = [1, 1, 1]
                if layout == 'full':
                    shape[axis] = n
                    b, a, ik = np.zeros(n), np.zeros(n), np.ones(n)
                    for side in sides:
                        lo = 0 if side == '-' else n - cells
                        b[lo:lo + cells], a[lo:lo + cells], ik[lo:lo + cells] = \
                            face_profile(side, field, cells)
                    key = f'{axis}{field}{compF}'
                    plan['terms'].append(dict(
                        key=key, field=field, axis=axis, compF=compF, compC=compC, sign=sign,
                        sl=None, writing=None,
                        b=b.reshape(shape), a=a.reshape(shape), ik=ik.reshape(shape)))
                    plan['psi'][key] = (n, n, n)
                    continue
                shape[axis] = cells
                for side in sides:
                    sl = slab_of(axis, side, n, cells)
                    key = f'{axis}{side}{field}{compF}'
                    b, a, ik = (v.reshape(shape) for v in face_profile(side, field, cells))
                    if writing == 'scatter' and (axis, side) not in plan['index']:
                        plan['index'][axis, side] = slab_indices(sl, n)
                    plan['terms'].append(dict(
                        key=key, field=field, axis=axis, side=side, face=(axis, side),
                        compF=compF, compC=compC, sign=sign, sl=sl, writing=writing,
                        b=b, a=a, ik=ik))
                    plan['psi'][key] = tuple(cells if ax == axis else n for ax in range(3))
    if writing == 'merged':
        for axis, side in faces:
            comps = sorted(compC for _, compC, _ in pml_pair(axis))
            plan['merged'][axis, side] = slab_indices(slab_of(axis, side, n, cells), n, comps)
    return plan


def psi_bytes(plan, itemsize=8):
    """The exact peak bytes of a layout's psi arrays: one live copy of each."""
    return sum(itemsize*int(np.prod(s)) for s in plan['psi'].values())


def index_bytes(plan, itemsize=4):
    """The exact bytes of the index arrays a writing needs, shared per face."""
    return itemsize*sum(int(a.size) for group in (plan['index'], plan['merged'])
                        for idx in group.values() for a in idx)


def curl_of(F, field, psi, plan, ops):
    """The stand-in curl of ``F`` with every face stretch applied; returns ``(curl, psi)``.

    The kernel is the slab study's stand-in -- periodic one-sided differences by axis rolls, the
    memory traffic of a Yee curl and not its physics -- with the one hook stage 1 needs: each
    bare difference along an absorbing axis is stretched before the curl components are
    assembled, which is where ``src/wavebit/fd/_kernel/curl.py`` says the stretch will go. Per psi
    the CFS recurrence is ``psi <- b psi + a d`` and the stretched difference is
    ``d/kappa + psi``, written in the slab layouts as the correction ``d (1/kappa - 1) + psi``
    so that one slab add carries it.
    """
    roll, stack = ops['roll'], ops['stack']
    scale, shift = (1.0, -1) if field == 'E' else (-1.0, 1)
    d = {(c, ax): scale*(roll(F[c], shift, ax) - F[c])
         for ax in range(3) for c in range(3) if c != ax}
    psi, merged = dict(psi), {}
    for t in plan['terms']:
        if t['field'] != field:
            continue
        k, key, (b, a, ik) = (t['compF'], t['axis']), t['key'], (t['b'], t['a'], t['ik'])
        if plan['layout'] == 'full':
            psi[key] = b*psi[key] + a*d[k]
            d[k] = d[k]*ik + psi[key]
            continue
        if t['writing'] == 'fused':
            # The recurrence and the stretch run in place on the slab view, in the order the
            # formula is written, so nothing is gathered and nothing is allocated.
            d[k], psi[key] = ops['fused'](d[k], t, psi[key])
            continue
        ds = d[k][t['sl']]
        psi[key] = b*psi[key] + a*ds
        corr = ds*(ik - 1.0) + psi[key]
        if t['writing'] == 'merged':
            merged.setdefault(t['face'], []).append((t['compC'], t['sign']*corr))
        else:
            d[k] = ops['write'](d[k], t, corr, plan)
    out = stack([d[(c + 2) % 3, (c + 1) % 3] - d[(c + 1) % 3, (c + 2) % 3] for c in range(3)])
    for face, parts in merged.items():
        out = ops['merge'](out, plan['merged'][face], parts)
    return out, psi


def numpy_write(d, t, corr, plan):
    view = d[t['sl']]
    np.add(view, corr, out=view)
    return d


def torch_write(d, t, corr, plan):
    d[t['sl']].add_(corr)
    return d


def numpy_fused(d, t, psi):
    """``psi <- b psi + a d`` then ``d <- d/kappa + psi``, both in place, on the slab view.

    The bare difference is read out of the view and the stretched one written back into it
    without a gather and without a full-volume temporary: the one scratch is slab-shaped and
    belongs to the term.
    """
    view = d[t['sl']]
    np.multiply(psi, t['b'], out=psi)
    np.add(psi, np.multiply(t['a'], view, out=t['tmp']), out=psi)
    np.multiply(view, t['ik'], out=view)
    np.add(view, psi, out=view)
    return d, psi


def torch_fused(d, t, psi):
    """The same four in-place operations on Torch, with no scratch at all."""
    view = d[t['sl']]
    psi.mul_(t['b']).addcmul_(t['a'], view)
    view.mul_(t['ik']).add_(psi)
    return d, psi


def jax_write(d, t, corr, plan):
    """``dus`` is ``d.at[slices].add``; ``scatter`` the index scatter-add stage 0 chose."""
    if t['writing'] == 'dus':
        return d.at[t['sl']].add(corr)
    return d.at[plan['index'][t['face']]].add(corr.reshape(-1), **SCATTER)


def jax_merge(out, idx, parts):
    """One 4-D scatter per face per curl: both components' corrections in a single op."""
    vals = jnp.concatenate([v.reshape(-1) for _, v in sorted(parts, key=lambda p: p[0])])
    return out.at[idx].add(vals, **SCATTER)


def problem(n, seed=0):
    return dict(e0=1e-3*np.random.default_rng(seed).standard_normal((3, n, n, n)),
                h0=np.zeros((3, n, n, n)))


def eager_runner(spec, p, n, steps, lib):
    """NumPy and Torch share one Python loop; ``lib`` supplies the copy and the in-place write."""
    roll, stack, copy, wrap = lib
    plan = build_plan(*spec, n)
    for t in plan['terms']:
        t['b'], t['a'], t['ik'] = (wrap(np.ascontiguousarray(t[k])) for k in ('b', 'a', 'ik'))
    ops = dict(roll=roll, stack=stack, merge=None,
               write=numpy_write if lib is NUMPY else torch_write,
               fused=numpy_fused if lib is NUMPY else torch_fused)
    if lib is NUMPY and plan['writing'] == 'fused':
        for t in plan['terms']:
            t['tmp'] = np.empty(plan['psi'][t['key']])
    e0, h0, shapes = wrap(p['e0']), wrap(p['h0']), plan['psi']

    def run():
        with torch.no_grad():
            E, H = copy(e0), copy(h0)
            psi = {k: wrap(np.zeros(s)) for k, s in shapes.items()}
            for _ in range(steps):
                cE, psi = curl_of(E, 'E', psi, plan, ops)
                H = H - C*cE
                cH, psi = curl_of(H, 'H', psi, plan, ops)
                E = E + C*cH
        return E
    return run


def jax_program(name, spec, p, n, steps):
    """The study's kernel as one ``lax.scan``, psi carried beside the fields as its own entry."""
    plan = build_plan(*spec, n)
    for t in plan['terms']:
        t['b'], t['a'], t['ik'] = (jnp.asarray(t[k]) for k in ('b', 'a', 'ik'))
    for group in ('index', 'merged'):
        plan[group] = {f: tuple(jnp.asarray(i) for i in idx) for f, idx in plan[group].items()}
    ops = dict(roll=jnp.roll, stack=jnp.stack, write=jax_write, merge=jax_merge)
    shapes = plan['psi']

    def program(e0, h0):
        def body(carry, _):
            E, H, psi = carry
            cE, psi = curl_of(E, 'E', psi, plan, ops)
            H = H - C*cE
            cH, psi = curl_of(H, 'H', psi, plan, ops)
            E = E + C*cH
            return (E, H, psi), None
        psi0 = {k: jnp.zeros(s) for k, s in shapes.items()}
        return lax.scan(body, (e0, h0, psi0), None, length=steps)[0][0]
    program.__name__ = re.sub(r'\W', '_', name)
    return jax.jit(program), (jnp.asarray(p['e0']), jnp.asarray(p['h0']))


def body_counts(program, args, n, backward=False):
    """Op counts in the optimized HLO of the scan body (with ``backward``, the transposed one),
    and the compiled program's own memory analysis."""
    compiled = program.lower(*args).compile()
    text = compiled.as_text()
    blocks = dict(re.findall(r'^(?:ENTRY )?(\S+) \([^\n]*\{\n(.*?)\n\}', text, re.M | re.S))
    loops = re.findall(r'while\(.*?body=(\S+),', text)
    body = blocks[loops[-1 if backward else 0]]
    called = [blocks.get(c, '') for c in re.findall(r'calls=(\S+?)[,\s]', body)]
    full = r'f64\[(?:3,%d,%d,%d|%d,%d,%d)\]\{[^}]*\} ' % ((n,)*6)
    out = dict(dus_fusions=sum('dynamic-update-slice(' in c for c in called),
               scatter_fusions=sum(' scatter(' in c for c in called),
               gather_fusions=sum(' gather(' in c for c in called),
               dynamic_slice_fusions=sum(' dynamic-slice(' in c for c in called),
               full_volume_fusions=len(re.findall(full + r'fusion\(', body)),
               full_volume_copies=len(re.findall(full + r'copy\(', body)))
    mem = compiled.memory_analysis()
    if mem is not None:
        out.update(argument_bytes=int(mem.argument_size_in_bytes),
                   output_bytes=int(mem.output_size_in_bytes),
                   temp_bytes=int(mem.temp_size_in_bytes))
    return out


def timed(runs, steps):
    """One untimed warm-up per runner (the compile, on JAX), then RUNS rounds over all runners in
    turn, so a slow phase of the machine hits every arm alike: per runner its final field, its
    median ms/step and the half-range of its samples."""
    finals, wall = {k: np.asarray(r()) for k, r in runs.items()}, {k: [] for k in runs}
    for _ in range(RUNS):
        for k, r in runs.items():
            t0 = time.perf_counter()
            r()
            wall[k].append(time.perf_counter() - t0)
    stats = {k: dict(ms_per_step=1e3*float(np.median(w))/steps,
                     spread=(max(w) - min(w))/2/float(np.median(w)),
                     samples_ms_per_step=[1e3*x/steps for x in w]) for k, w in wall.items()}
    return finals, stats


def arms(backend=None):
    """The arms: the no-absorber baseline, then each layout at each face count.

    Every backend runs both layouts. The three ways of writing the slab layout are a JAX
    question -- the eager write is an in-place add into the difference array and has no second
    candidate -- so ``backend=None`` returns the union, for the table's rows.
    """
    out = {'baseline': (None, None, ())}
    for label, faces in FACESETS.items():
        out[f'full psi, {label}'] = ('full', None, faces)
        for w in (WRITINGS[backend] if backend else ('inplace',) + WRITINGS['jax']):
            out[f'slab psi {w}, {label}'] = ('slab', w, faces)
    return out


def size_section(n, report):
    """Every backend over its arms at one N, with the JAX evidence and the byte figures."""
    scan_steps, loop_steps = STEPS[n]
    p, stats = problem(n), {}
    plans = {a: build_plan(*spec, n) for a, spec in arms().items()}
    for backend in BACKENDS:
        runs, extra, finals = {}, {}, {}
        for arm, spec in arms(backend).items():
            if backend == 'jax':
                program, args = jax_program(arm, spec, p, n, scan_steps)
                t0 = time.perf_counter()
                program(*args).block_until_ready()
                extra[arm] = dict(compile_and_first_call_seconds=time.perf_counter() - t0,
                                  hlo=body_counts(program, args, n))
                runs[arm] = lambda program=program, args=args: program(*args).block_until_ready()
            else:
                runs[arm] = eager_runner(spec, p, n, loop_steps,
                                         NUMPY if backend == 'numpy' else TORCH)
        finals, timing = timed(runs, scan_steps if backend == 'jax' else loop_steps)
        scale = float(np.max(np.abs(finals['baseline'])))
        for arm in runs:
            t = stats[backend, arm] = dict(timing[arm], **extra.get(arm, {}))
            t['us_over_baseline'] = 1e3*(t['ms_per_step'] - timing['baseline']['ms_per_step'])
            if arm.startswith('slab'):
                twin = finals[re.sub(r'slab psi \w+', 'full psi', arm)]
                t['rel_gap_vs_full_psi'] = float(np.max(np.abs(finals[arm] - twin))/scale)
            print(f'N={n} {backend:5s} {arm:28s} {t["ms_per_step"]:9.3f} ms/step '
                  f'+-{100*t["spread"]:.0f}%', flush=True)
    gaps = {f'{b} {a}': t['rel_gap_vs_full_psi'] for (b, a), t in stats.items()
            if 'rel_gap_vs_full_psi' in t}
    assert max(gaps.values()) < 1e-9, gaps
    report['sizes'][n] = dict(
        steps=STEPS[n], slab_vs_full_psi_rel_gaps=gaps,
        psi_bytes={a: psi_bytes(pl) for a, pl in plans.items()},
        index_bytes={a: index_bytes(pl) for a, pl in plans.items()},
        timings={f'{b} {a}': t for (b, a), t in stats.items()})
    return table(n, stats, plans)


def table(n, stats, plans):
    scan_steps, loop_steps = STEPS[n]

    def cell(backend, arm):
        t = stats.get((backend, arm))
        if t is None:
            return '-'
        base = stats[backend, 'baseline']['ms_per_step']
        return f'{t["ms_per_step"]:.3f} ({t["ms_per_step"]/base:.2f}x) +-{100*t["spread"]:.0f}%'

    rows = list(arms())
    lines = [f'| N={n}, {scan_steps} scan steps (jax) / {loop_steps} loop steps (numpy, torch), '
             f'ms/step (x baseline) +-half-range of {RUNS} runs | ' + ' | '.join(BACKENDS)
             + ' | psi MB | index MB |', '| --- | --- | --- | --- | --- | --- |']
    for arm in rows:
        lines.append(f'| {arm} | ' + ' | '.join(cell(b, arm) for b in BACKENDS)
                     + f' | {psi_bytes(plans[arm])/2**20:.2f} | {index_bytes(plans[arm])/2**20:.2f} |')
    lines.append('| noise: baseline half-range, us/step | ' + ' | '.join(
        f'{1e3*stats[b, "baseline"]["ms_per_step"]*stats[b, "baseline"]["spread"]:.0f}'
        for b in BACKENDS) + ' | - | - |')

    def per_psi(backend):
        slab = 'scatter' if backend == 'jax' else 'inplace'
        return '; '.join(' / '.join(
            f'{stats[backend, a]["us_over_baseline"]/len(plans[a]["psi"]):.0f}'
            for a in (f'full psi, {lab}', f'slab psi {slab}, {lab}')) for lab in FACESETS)
    lines.append('| us over baseline per psi array, full / slab: 2 faces; 6 faces | '
                 + ' | '.join(per_psi(b) for b in BACKENDS) + ' | - | - |')
    lines.append('| slab field vs its full-psi twin, max relative gap | ' + ' | '.join(
        '/'.join(f'{stats[b, a]["rel_gap_vs_full_psi"]:.0e}' for a in arms(b) if a.startswith('slab'))
        for b in BACKENDS) + ' | - | - |')
    keys = ('dus_fusions', 'scatter_fusions', 'gather_fusions', 'full_volume_fusions', 'full_volume_copies')
    lines.append(f'JAX scan body at N={n} (DUS / scatter / gather / full-volume fusions / full-volume copies): '
                 + '; '.join(f'{a} ' + '/'.join(str(stats["jax", a]["hlo"][k]) for k in keys)
                             for a in arms('jax')))
    lines.append(f'JAX compiled bytes at N={n}, argument / output / temp, MB: ' + '; '.join(
        f'{a} ' + '/'.join(f'{stats["jax", a]["hlo"].get(k, 0)/2**20:.0f}'
                           for k in ('argument_bytes', 'output_bytes', 'temp_bytes'))
        for a in arms('jax')))
    return lines


def cross_backend_check(report):
    """Every arm of every backend at one size and one step count: the three backends must make
    the same field, and every writing of the slab layout the same field as the full layout."""
    n, steps = CHECK
    p, fields = problem(n), {}
    for backend in BACKENDS:
        for arm, spec in arms(backend).items():
            if backend == 'jax':
                program, args = jax_program(f'check_{arm}', spec, p, n, steps)
                fields[backend, arm] = np.asarray(program(*args))
            else:
                fields[backend, arm] = np.asarray(
                    eager_runner(spec, p, n, steps, NUMPY if backend == 'numpy' else TORCH)())
    scale = float(np.max(np.abs(fields['numpy', 'baseline'])))
    gaps = {}
    for (backend, arm), f in fields.items():
        ref = fields['numpy', re.sub(r'slab psi \w+', 'slab psi inplace', arm)]
        gaps[f'{backend} {arm} vs numpy'] = float(np.max(np.abs(f - ref))/scale)
        if arm.startswith('slab'):
            twin = fields[backend, re.sub(r'slab psi \w+', 'full psi', arm)]
            gaps[f'{backend} {arm} vs its full-psi twin'] = float(np.max(np.abs(f - twin))/scale)
    assert max(gaps.values()) < 1e-9, gaps
    report['cross_backend_check'] = dict(n=n, steps=steps, max_rel_gaps=gaps)
    return [f'Cross-backend check at N={n}, {steps} steps: every arm of every backend within '
            f'{max(gaps.values()):.0e} of the NumPy field of the same arm and of its full-psi twin '
            f'(largest of {len(gaps)} comparisons).']


def eager_section(report):
    """The second way of writing the slab layout on the eager backends.

    The design's rule is that if the slab layout loses on time the LAYOUT stays (memory wins at
    128^3) and the writing changes, so this section holds the layout fixed and changes only how
    it is written: ``inplace`` gathers the slab of the bare difference, advances psi, forms the
    correction and adds it back (the writing the main section timed); ``fused`` runs the
    recurrence and the stretch as four in-place operations on the slab view itself, gathering
    nothing and allocating nothing. NumPy and Torch only -- on JAX the arithmetic is XLA's to
    fuse and the question is which write op, which the main section answers -- at the two
    extreme sizes, both face counts.
    """
    lines, report['eager_writings'] = [], {}
    specs = {'baseline': (None, None, ())}
    for label, faces in FACESETS.items():
        specs[f'full psi, {label}'] = ('full', None, faces)
        for w in EAGER_WRITINGS:
            specs[f'slab psi {w}, {label}'] = ('slab', w, faces)
    for n in EAGER_SIZES:
        steps, p, stats = STEPS[n][1], problem(n), {}
        plans = {a: build_plan(*s, n) for a, s in specs.items()}
        for backend in ('numpy', 'torch'):
            lib = NUMPY if backend == 'numpy' else TORCH
            runs = {a: eager_runner(s, p, n, steps, lib) for a, s in specs.items()}
            finals, timing = timed(runs, steps)
            scale = float(np.max(np.abs(finals['baseline'])))
            for arm in runs:
                t = stats[backend, arm] = dict(timing[arm])
                t['us_over_baseline'] = 1e3*(t['ms_per_step'] - timing['baseline']['ms_per_step'])
                if arm.startswith('slab'):
                    twin = finals[re.sub(r'slab psi \w+', 'full psi', arm)]
                    t['rel_gap_vs_full_psi'] = float(np.max(np.abs(finals[arm] - twin))/scale)
                print(f'N={n} {backend:5s} eager {arm:30s} {t["ms_per_step"]:9.3f} ms/step '
                      f'+-{100*t["spread"]:.0f}%', flush=True)
        gaps = {f'{b} {a}': t['rel_gap_vs_full_psi'] for (b, a), t in stats.items()
                if 'rel_gap_vs_full_psi' in t}
        assert max(gaps.values()) < 1e-9, gaps
        report['eager_writings'][n] = dict(steps=steps, slab_vs_full_psi_rel_gaps=gaps,
                                           timings={f'{b} {a}': t for (b, a), t in stats.items()})
        lines += [f'| eager slab writings at N={n}, {steps} loop steps, ms/step (x baseline) '
                  f'+-half-range of {RUNS} runs | numpy | torch | psi MB |',
                  '| --- | --- | --- | --- |']
        for arm in specs:
            lines.append(f'| {arm} | ' + ' | '.join(
                f'{stats[b, arm]["ms_per_step"]:.3f} '
                f'({stats[b, arm]["ms_per_step"]/stats[b, "baseline"]["ms_per_step"]:.2f}x) '
                f'+-{100*stats[b, arm]["spread"]:.0f}%' for b in ('numpy', 'torch'))
                + f' | {psi_bytes(plans[arm])/2**20:.2f} |')
        lines.append('| us over baseline per psi array, full / inplace / fused: 2 faces; 6 faces | '
                     + ' | '.join('; '.join(' / '.join(
                         f'{stats[b, f"{lay}, {lab}"]["us_over_baseline"]/len(plans[f"{lay}, {lab}"]["psi"]):.0f}'
                         for lay in ('full psi', 'slab psi inplace', 'slab psi fused'))
                         for lab in FACESETS) for b in ('numpy', 'torch')) + ' | - |')
        lines.append('| slab field vs its full-psi twin, max relative gap | ' + ' | '.join(
            '/'.join(f'{stats[b, a]["rel_gap_vs_full_psi"]:.0e}' for a in specs if a.startswith('slab'))
            for b in ('numpy', 'torch')) + ' | - |')
    return lines


def jax_axis_sweep(n, report):
    """The same two-face question per axis. A face slab on axis 0 restricts the outermost loop
    of a full-volume kernel and one on axis 2 the innermost, which is what decided whether a
    dynamic-update-slice was cheap in the slab study."""
    p, runs, counts = problem(n), {}, {}
    steps = STEPS[n][0]
    layouts = [('full', None)] + [('slab', w) for w in WRITINGS['jax']]
    program, args = jax_program('sweep_baseline', (None, None, ()), p, n, steps)
    program(*args).block_until_ready()
    runs['baseline'] = lambda program=program, args=args: program(*args).block_until_ready()
    for label, faces in AXIS_PAIRS.items():
        for layout, writing in layouts:
            key = f'{layout} psi{" " + writing if writing else ""}, {label}'
            program, args = jax_program(key, (layout, writing, faces), p, n, steps)
            program(*args).block_until_ready()
            runs[key] = lambda program=program, args=args: program(*args).block_until_ready()
            counts[key] = body_counts(program, args, n)
    _, stats = timed(runs, steps)
    for key, t in stats.items():
        t['us_over_baseline'] = 1e3*(t['ms_per_step'] - stats['baseline']['ms_per_step'])
        if key in counts:
            t['hlo'] = counts[key]
        print(f'N={n} jax   sweep {key:34s} {t["ms_per_step"]:9.3f} ms/step '
              f'+-{100*t["spread"]:.0f}%', flush=True)
    report['jax_axis_sweep'] = dict(n=n, steps=steps, timings=stats)
    base = stats['baseline']
    lines = [f'| JAX per-axis face pair at N={n}, {steps} scan steps, us over the baseline '
             f'({base["ms_per_step"]:.2f} ms/step +-{100*base["spread"]:.0f}%) +-half-range | '
             + ' | '.join(AXIS_PAIRS) + ' |', '| --- | --- | --- | --- |']
    for layout, writing in layouts:
        name = f'{layout} psi{" " + writing if writing else ""}'
        lines.append(f'| {name} | ' + ' | '.join(
            f'{stats[f"{name}, {label}"]["us_over_baseline"]:+.0f} '
            f'+-{1e3*stats[f"{name}, {label}"]["ms_per_step"]*stats[f"{name}, {label}"]["spread"]:.0f}'
            for label in AXIS_PAIRS) + ' |')
    lines.append(f'scan body per axis pair at N={n} (DUS / scatter / gather / full-volume fusions / copies): '
                 + '; '.join(f'{k} ' + '/'.join(str(counts[k][f]) for f in (
                     'dus_fusions', 'scatter_fusions', 'gather_fusions', 'full_volume_fusions',
                     'full_volume_copies')) for k in counts))
    return lines


def jax_reverse_mode(n, steps, report):
    """``jax.grad`` of ``sum(E_final^2)`` with respect to a scalar on the initial field and to
    the field itself, for the baseline and both layouts: forward+backward per step, and the
    transposed scan body's op counts. The slab study found that the first slab term brings one
    full-volume cotangent copy per backward step; the question here is whether a CPML face,
    whose slab is read as well as written, brings another."""
    p, grads, out = problem(n), {}, {}
    faces = FACESETS['2 faces (x)']
    specs = {'baseline': (None, None, ()), 'full psi': ('full', None, faces)}
    specs.update({f'slab psi {w}': ('slab', w, faces) for w in WRITINGS['jax']})
    for arm, spec in specs.items():
        program, args = jax_program(f'grad_{arm}', spec, p, n, steps)

        def loss(theta, e0, h0, program=program):
            return jnp.sum(program(e0*theta, h0)**2)
        grads[arm] = jax.jit(jax.grad(loss, argnums=(0, 1))), (jnp.asarray(1.0),) + args
    wall = {k: [] for k in grads}
    for r in range(RUNS + 1):                     # the first round is the warm-up (the compile)
        for arm, (g, a) in grads.items():
            t0 = time.perf_counter()
            out[arm] = jax.block_until_ready(g(*a))
            if r:
                wall[arm].append(time.perf_counter() - t0)
    stats = {k: dict(fwd_bwd_ms_per_step=1e3*float(np.median(w))/steps,
                     spread=(max(w) - min(w))/2/float(np.median(w)),
                     hlo=body_counts(*grads[k], n, backward=True)) for k, w in wall.items()}
    ref = np.asarray(out['full psi'][1])
    for arm, t in stats.items():
        t['ms_over_baseline'] = t['fwd_bwd_ms_per_step'] - stats['baseline']['fwd_bwd_ms_per_step']
        if arm.startswith('slab'):
            t['grad_rel_err_vs_full'] = float(
                np.linalg.norm(np.asarray(out[arm][1]) - ref)/np.linalg.norm(ref))
        print(f'N={n} jax   grad {arm:24s} {t["fwd_bwd_ms_per_step"]:9.3f} ms/step '
              f'+-{100*t["spread"]:.0f}%', flush=True)
    errs = [t['grad_rel_err_vs_full'] for t in stats.values() if 'grad_rel_err_vs_full' in t]
    assert max(errs) < 1e-9, errs
    report['jax_reverse_mode'][n] = dict(steps=steps, faces='2 faces (x)', timings=stats)
    return stats


def reverse_table(reverse):
    sizes = list(reverse)
    names = list(reverse[sizes[0]])
    lines = ['| JAX reverse mode, 2 faces (x): grad of sum(E_final^2) w.r.t. a scalar on the initial '
             'field and the field itself, forward+backward ms/step (+ms over the baseline) '
             f'+-half-range of {RUNS} runs | ' + ' | '.join(names) + ' |',
             '| --- |' + ' --- |'*len(names)]
    for n in sizes:
        t = reverse[n]
        lines.append(f'| N={n}, {GRAD_STEPS[n]} scan steps | ' + ' | '.join(
            f'{t[a]["fwd_bwd_ms_per_step"]:.2f} ({t[a]["ms_over_baseline"]:+.2f}) '
            f'+-{100*t[a]["spread"]:.0f}%' for a in names) + ' |')
    lines.append('| gradient vs the full-psi twin, at N=' + ', '.join(str(n) for n in sizes)
                 + ' | ' + ' | '.join('; '.join(f'{reverse[n][a]["grad_rel_err_vs_full"]:.0e}'
                                                for n in sizes) if a.startswith('slab') else '-'
                                      for a in names) + ' |')
    keys = ('full_volume_copies', 'gather_fusions', 'dynamic_slice_fusions', 'scatter_fusions')
    lines.append('| transposed scan body: full-volume copies / gather / dynamic-slice / scatter '
                 'fusions | ' + ' | '.join('; '.join(
                     '/'.join(str(reverse[n][a]['hlo'][k]) for k in keys) for n in sizes)
                     for a in names) + ' |')
    return lines


def bytes_table():
    """The psi and index bytes of each layout, computed exactly, beside the two field arrays."""
    lines = [f'| exact bytes, float64 psi and int32 indices, {CELLS}-cell faces | 2 faces | 6 faces '
             '| fields E and H |', '| --- | --- | --- | --- |']
    for n in STEPS:
        row = []
        for _, faces in FACESETS.items():
            mb = {lay: psi_bytes(build_plan(lay, None, faces, n))/2**20 for lay in ('slab', 'full')}
            idx = {w: index_bytes(build_plan('slab', w, faces, n))/2**20 for w in ('scatter', 'merged')}
            row.append(f'slab {mb["slab"]:.1f} / full {mb["full"]:.1f} MB ({mb["full"]/mb["slab"]:.1f}x); '
                       f'slab indices {idx["scatter"]:.1f} (scatter) / {idx["merged"]:.1f} (merged) MB')
        lines.append(f'| N={n} | ' + ' | '.join(row) + f' | {2*3*n**3*8/2**20:.1f} MB |')
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('out', nargs='?', help='the JSON output '
                        '(default studies/results/fdtd_cpml_slab_cost[.eager].json)')
    parser.add_argument('--section', choices=('all', 'eager'), default='all',
                        help='"eager" runs the second eager writing alone; the default runs everything')
    opts = parser.parse_args()
    out_path = opts.out or (f'studies/results/fdtd_cpml_slab_cost'
                            f'{"" if opts.section == "all" else ".eager"}.json')
    report = dict(machine=dict(
        platform=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count(),
        python=platform.python_version(), numpy=np.__version__, jax=jax.__version__,
        torch=torch.__version__, jax_devices=[str(d) for d in jax.devices()],
        torch_threads=(torch.get_num_threads(), torch.get_num_interop_threads()),
        env={k: v for k, v in os.environ.items() if re.match(r'(OMP|MKL|OPENBLAS|XLA|JAX)_', k)}),
        constants=dict(c=C, cells=CELLS, runs=RUNS, steps=STEPS, check=CHECK, grad_steps=GRAD_STEPS,
                       dx=DX, dt=DT, wavelength=WAVELENGTH,
                       facesets={k: list(v) for k, v in FACESETS.items()}),
        sizes={}, jax_reverse_mode={})
    lines = []
    if opts.section == 'eager':
        lines += eager_section(report)
    else:
        for n in STEPS:
            lines += size_section(n, report)
        lines += bytes_table()
        lines += cross_backend_check(report)
        lines += jax_axis_sweep(SWEEP_N, report)
        for n, steps in GRAD_STEPS.items():
            jax_reverse_mode(n, steps, report)
        lines += reverse_table({n: report['jax_reverse_mode'][n]['timings'] for n in GRAD_STEPS})
    print('\n'.join(lines))
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=1, allow_nan=False)


if __name__ == '__main__':
    main()
