"""Freeze ``docs/guide/fdtd/fdtd.md`` §2's two tables -- **all three columns** -- into
``golden_fdtd_table.npz``.

    conda run -n fieldwave python tests/references/gen_goldens_fdtd_table.py

**Why a table needs a golden at all.** §2 prints a three-way agreement on one grating and
``README.md`` quotes its worst per-order gaps as the FDTD path's headline validation.
Until 2026-09-07 those digits lived in a markdown file and nowhere else. One of the three
columns had gone stale -- it was a ``pad_scale=2.0`` run and the shipped clearance had
since been halved -- and nothing noticed, because nothing in the test suite read a number
out of a document. This file makes the document a *transcription* of a committed artefact,
and ``test_fdtd_table_golden.py`` reads §2's markdown back and compares it cell by cell.

**What is frozen, and where each column comes from.**

* **RCWA**, the reference both gaps are measured against: solved here, at
  ``_meep_1d_cases.py::RCWA_ORDERS``. Its own uncertainty is measured against
  ``RCWA_ORDERS_CONVERGED`` and frozen beside it, because a gap quoted against a reference
  is only as good as the reference.
* **wavebit's FDTD**: solved here, through ``WAVEBIT_CALL`` -- which is a resolution and
  nothing else. The settings that call inherits are written out by value in
  ``WAVEBIT_SHIPPED``, and this generator **refuses to write** if any of them has stopped
  being what the module ships. That refusal is the gate: halving a clearance again fails
  here and in CI, naming this table, instead of stranding it in silence.
* **Meep**: *copied* from ``golden_meep_1d.npz``, not recomputed -- meep is GPL and lives
  in another environment, and that file is its own oracle with its own calibration gate.
  The copy carries the source file's digest so a stale copy is a failing test rather than
  a quiet disagreement between two committed artefacts.

**This is a regression pin for two of the three columns, and that is the point.** A golden
holding a solver's own answer cannot arbitrate that answer -- ``golden_meep_1d.npz`` and
the RCWA path are what do that, and both comparisons live in ``test_meep_1d_golden.py``.
What *this* file arbitrates is whether the published table is still the table the code
produces, which is a different question and the one that was going unasked.
"""

from wavebit.rcwa._factory import rcwa as _legacy_wavebit_rcwa

import sys as _repository_sys
from pathlib import Path as _RepositoryPath
if __package__ in (None, ""):
    _repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))
from tools.repository_paths import repository_path
import hashlib
import sys
import time

import numpy as np

from wavebit.fdtd._periodic import legacy as fdtd_module
from wavebit.fdtd._periodic.legacy import solve_fdtd

HERE = repository_path('tests/references')
sys.path.insert(0, str(HERE))
from tests.references._meep_1d_cases import CASES, FDTD_CONVERGENCE, FDTD_PUBLISHED, LEGS, RCWA_ORDERS, RCWA_ORDERS_CONVERGED, UNITS, WAVEBIT_BUILD, WAVEBIT_CALL, WAVEBIT_DERIVED, WAVEBIT_SHIPPED, eps_of

OUT = HERE / 'golden_fdtd_table.npz'
MEEP_GOLDEN = HERE / 'golden_meep_1d.npz'
CASE_NAME = 'binary_grating'

#: How closely the recorded derived grid has to match the one the solver actually builds.
#: These are lengths in :data:`UNITS` accumulated by ``cumsum``, so the last bits move;
#: 1e-9 is nine orders below the coarsest cell here and still refuses a moved plane.
DERIVED_TOL = 1e-9


def _shipped(name):
    """The live value of one recorded determinant, from wherever it actually lives.

    Three of them bind differently and the difference is not cosmetic: ``PML_CELLS`` and
    ``COURANT`` are bound into ``solve_fdtd``'s signature at import, so the module
    attribute and the default can disagree and only the default is what a call gets.
    Both spellings are checked.
    """
    import inspect

    params = inspect.signature(solve_fdtd).parameters
    if name in params and name.islower():
        return params[name].default
    return getattr(fdtd_module, name)


def check_the_record_is_still_the_shipped_configuration():
    """Refuse to freeze a table taken at settings the record does not describe.

    **This is the check whose absence produced the defect this file exists for.**
    ``PAD_WAVELENGTHS`` went 1.0 -> 0.5, §2's phrase "at the shipped defaults" stayed
    true-looking, and the printed column silently became a configuration the library no
    longer had -- measured 7.93e-5 per order, against a published agreement of 6.6e-4.
    A record that names values can be checked against the module; a record that names
    "the defaults" cannot, because it is true by construction.
    """
    wrong = []
    for name, recorded in WAVEBIT_SHIPPED.items():
        got = _shipped(name)
        if got != recorded and not (got is None and recorded is None):
            wrong.append(f'  {name}: the record says {recorded!r}, the module ships {got!r}')
    if wrong:
        raise SystemExit(
            'the recorded conditions are not the shipped ones any more:\n'
            + '\n'.join(wrong)
            + '\n\ndocs/guide/fdtd/fdtd.md section 2 is quoted at these values. Decide which is '
              'right, update _meep_1d_cases.py::WAVEBIT_SHIPPED, re-run this generator '
              'and re-type the table -- all four, or the document goes stale the way it '
              'did before 2026-09-07.')


def _orders(case):
    return list(case['orders_u'])


def _grid(result, case):
    """``(R, T)`` on the case's own order window, aligned on the order LABELS."""
    sub = result.incident
    u = list(np.asarray(result.orders_u).ravel())
    i = [u.index(m) for m in _orders(case)]
    return (np.asarray(sub.R).ravel()[i].astype(float),
            np.asarray(sub.T).ravel()[i].astype(float))


def _rcwa(case, leg, orders):
    return _legacy_wavebit_rcwa(
        backend='numpy', wavelength=case['wavelength'], period=[case['period'][0]],
        truncation_order=dict(x=orders, y=0), top=eps_of(case['n_top']),
        bot=eps_of(case['n_bot']), theta=0.0, phi=0.0, tetm=leg,
        structure=eps_of(case['index']), thickness=list(case['thickness'])).solve()


def _fdtd(case, leg, resolution, diagnostics=False):
    call = dict(WAVEBIT_CALL, resolution=resolution)
    return solve_fdtd(
        structure=eps_of(case['index']), thickness=list(case['thickness']),
        top=eps_of(case['n_top']), bot=eps_of(case['n_bot']),
        period=[case['period'][0]], wavelength=case['wavelength'], tetm=leg,
        return_diagnostics=diagnostics, **call)


def check_the_recorded_grid_is_the_one_the_solver_builds(diagnostics):
    """The derived half of the record, against the run's own diagnostics.

    A determinant list of *inputs* is not enough. Two spellings of the clearance are
    indistinguishable in any result, so the record states the clearance, the absorber and
    the run in **lengths and counts** -- and those are only a record if something checks
    them against the grid the solver actually builds. This is that something, and it also
    catches a change that moves the grid without touching any named constant.
    """
    want = {'nx': WAVEBIT_DERIVED['nx'], 'nz': WAVEBIT_DERIVED['nz'],
            'dt': WAVEBIT_DERIVED['dt'], 'n_steps': WAVEBIT_DERIVED['n_steps'],
            'dz': WAVEBIT_DERIVED['dz']}
    for key, value in want.items():
        got = float(np.asarray(diagnostics[key]).ravel()[0])
        if abs(got - float(value)) > DERIVED_TOL * max(1.0, abs(float(value))):
            raise SystemExit(f'the record says {key} = {value}, the solver built {got}')
    pml = np.asarray(diagnostics['pml_thickness'], dtype=float)
    if np.abs(pml - WAVEBIT_DERIVED['pml_thickness']).max() > 1e-9:
        raise SystemExit(f'the record says the PML is {WAVEBIT_DERIVED["pml_thickness"]} '
                         f'{UNITS} at each end; the run built {pml.tolist()}')


def main():
    if not MEEP_GOLDEN.exists():
        raise SystemExit(f'{MEEP_GOLDEN.name} is missing. §2 has three columns and this '
                         f'file freezes all three; regenerate the meep half first, in an '
                         f'environment that has meep.')
    check_the_record_is_still_the_shipped_configuration()

    case = CASES[CASE_NAME]
    orders = _orders(case)
    meep = np.load(MEEP_GOLDEN)
    out = {'orders_u': np.asarray(orders, dtype=int),
           'units': np.array(UNITS),
           'rcwa_orders': np.array(RCWA_ORDERS),
           'rcwa_orders_converged': np.array(RCWA_ORDERS_CONVERGED),
           'fdtd_resolution': np.array(FDTD_PUBLISHED),
           'fdtd_convergence_resolutions': np.asarray(FDTD_CONVERGENCE, dtype=int),
           'meep_resolution': np.array(int(case['meep_resolution'])),
           'meep_golden_sha256': np.array(
               hashlib.sha256(MEEP_GOLDEN.read_bytes()).hexdigest()),
           'meep_version': meep['meep_version'],
           }
    for key, value in WAVEBIT_BUILD.items():
        out[f'wavebit_build__{key}'] = np.array(str(value))
    for name, value in WAVEBIT_SHIPPED.items():
        out[f'record__shipped__{name}'] = np.array(str(value))
    for name, value in WAVEBIT_DERIVED.items():
        out[f'record__derived__{name}'] = np.asarray(value, dtype=float)
    for name, value in WAVEBIT_CALL.items():
        out[f'record__call__{name}'] = np.asarray(value)

    t_start = time.time()
    print(f'{CASE_NAME}: orders {orders}, lengths in {UNITS}')

    # ---- the reference, and its own uncertainty -----------------------------------
    for leg in LEGS:
        ref_r, ref_t = _grid(_rcwa(case, leg, RCWA_ORDERS), case)
        con_r, con_t = _grid(_rcwa(case, leg, RCWA_ORDERS_CONVERGED), case)
        out[f'rcwa__{leg}__R'], out[f'rcwa__{leg}__T'] = ref_r, ref_t
        out[f'rcwa__{leg}__sigma'] = np.array(float(ref_r.sum() + ref_t.sum()))
        unc = float(max(np.abs(ref_r - con_r).max(), np.abs(ref_t - con_t).max()))
        out[f'rcwa__{leg}__uncertainty'] = np.array(unc)
        print(f'  rcwa {leg}: R0 {ref_r[orders.index(0)]:.6f}  '
              f'T0 {ref_t[orders.index(0)]:.6f}  '
              f'T(1) {ref_t[orders.index(1)]:.6f}  Sigma {ref_r.sum() + ref_t.sum():.6f}  '
              f'own uncertainty vs truncation_order {RCWA_ORDERS_CONVERGED}: {unc:.2e}')

    # ---- wavebit's own column, plus the convergence series ------------------------
    for leg in LEGS:
        ref_r, ref_t = out[f'rcwa__{leg}__R'], out[f'rcwa__{leg}__T']
        for res in sorted(set(FDTD_CONVERGENCE) | {FDTD_PUBLISHED}):
            t0 = time.time()
            if res == FDTD_PUBLISHED:
                result, diagnostics = _fdtd(case, leg, res, diagnostics=True)
                check_the_recorded_grid_is_the_one_the_solver_builds(diagnostics)
                out[f'fdtd__{leg}__residual'] = np.array(float(diagnostics['residual']))
            else:
                result = _fdtd(case, leg, res)
            wall = time.time() - t0
            r, t = _grid(result, case)
            worst = float(max(np.abs(r - ref_r).max(), np.abs(t - ref_t).max()))
            out[f'fdtd__{leg}__{res}__R'], out[f'fdtd__{leg}__{res}__T'] = r, t
            out[f'fdtd__{leg}__{res}__sigma'] = np.array(float(r.sum() + t.sum()))
            out[f'fdtd__{leg}__{res}__worst'] = np.array(worst)
            # The wall time is frozen because §2 prints it, and it is the ONE column in
            # that table no test can hold: it is a property of the machine rather than of
            # the method. Recorded so a reader knows which machine, not so CI can assert it.
            out[f'fdtd__{leg}__{res}__wall'] = np.array(float(wall))
            print(f'  fdtd {leg} resolution {res:3d} ({1000.0 / res:4.1f} nm cell): '
                  f'worst {worst:.3e}  Sigma {r.sum() + t.sum():.6f}  wall {wall:.1f} s')
        for a, b in zip(FDTD_CONVERGENCE, FDTD_CONVERGENCE[1:]):
            wa = float(out[f'fdtd__{leg}__{a}__worst'])
            wb = float(out[f'fdtd__{leg}__{b}__worst'])
            out[f'fdtd__{leg}__{b}__order'] = np.array(np.log(wa / wb) / np.log(2.0))

    # ---- the meep column, copied rather than recomputed ---------------------------
    for leg in LEGS:
        r = np.asarray(meep[f'{CASE_NAME}__{leg}__fine__R'], dtype=float).ravel()
        t = np.asarray(meep[f'{CASE_NAME}__{leg}__fine__T'], dtype=float).ravel()
        frozen = list(np.asarray(meep[f'{CASE_NAME}__orders_u']).ravel())
        i = [frozen.index(m) for m in orders]
        out[f'meep__{leg}__R'], out[f'meep__{leg}__T'] = r[i], t[i]
        out[f'meep__{leg}__sigma'] = np.array(float(r.sum() + t.sum()))
        ref_r, ref_t = out[f'rcwa__{leg}__R'], out[f'rcwa__{leg}__T']
        out[f'meep__{leg}__worst'] = np.array(
            float(max(np.abs(r[i] - ref_r).max(), np.abs(t[i] - ref_t).max())))
        print(f'  meep {leg}: R0 {r[i][orders.index(0)]:.6f}  '
              f'T0 {t[i][orders.index(0)]:.6f}  Sigma {r.sum() + t.sum():.6f}  '
              f'worst gap {float(out[f"meep__{leg}__worst"]):.3e}')

    # ---- the per-entry gaps the table prints in its two "gap" columns --------------
    for column in ('fdtd', 'meep'):
        for leg in LEGS:
            for key in ('R', 'T'):
                got = (out[f'fdtd__{leg}__{FDTD_PUBLISHED}__{key}'] if column == 'fdtd'
                       else out[f'meep__{leg}__{key}'])
                out[f'gap__{column}__{leg}__{key}'] = np.abs(got - out[f'rcwa__{leg}__{key}'])

    np.savez(OUT, **out)
    print(f'\nwrote {OUT} ({len(out)} arrays) in {time.time() - t_start:.0f} s')
    print('now re-type docs/guide/fdtd/fdtd.md section 2 from it; '
          'test_fdtd_table_golden.py reads the markdown back and compares.')


if __name__ == '__main__':
    main()
