"""Analyse ``orders_raw.json`` (written by ``orders_run.py``) into ``orders.json`` and a text table.

Run from the repository root::

    python studies/rcwa/isotropic_integrated/orders_analyze.py

References, per case: ``ref_scalar`` / ``ref_integrated`` = that arm's highest-M solve;
``ref_joint`` = their mean (the two differ by ``top_gap``, which bounds what any comparison
below it can resolve). Errors at order M are measured against each reference after padding
the M solve's order array with zeros to the reference window, so orders the M solve lacks
count fully. Convergence rate: least-squares slope ``p`` of ``log err`` against ``log M``
for ``|sum T - ref_joint|`` over M in [2, M_top/1.5], i.e. ``err ~ M**-p``.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..'))
from studies.rcwa.isotropic_integrated import cases as C                   # noqa: E402

RAW = os.path.join(HERE, 'orders_raw.json')
OUT = os.path.join(HERE, 'orders.json')
TXT = os.path.join(HERE, 'orders.txt')


def unpack(v):
    if isinstance(v, dict):
        return np.asarray(v['re']) + 1j * np.asarray(v['im'])
    return np.asarray(v, dtype=float)


def pad(a, n):
    """Order array ``(1, 2m+1, 2m+1)`` zero-padded, centred, to ``(1, 2n+1, 2n+1)``."""
    a = np.asarray(a)[0]
    m = (a.shape[0] - 1) // 2
    out = np.zeros((2 * n + 1, 2 * n + 1), dtype=a.dtype)
    out[n - m:n + m + 1, n - m:n + m + 1] = a
    return out


def fit_rate(ms, errs, m_top):
    ms, errs = np.asarray(ms, float), np.asarray(errs, float)
    sel = (ms >= 2) & (ms <= m_top / 1.5) & (errs > 0)
    if sel.sum() < 3:
        return None
    p = np.polyfit(np.log(ms[sel]), np.log(errs[sel]), 1)
    return float(-p[0])


def extrapolate(ms, vals, m_min=6):
    """``sum T`` at M -> inf from a least-squares fit ``v = v_inf + c M**-p`` over M >= m_min,
    for p = 1 and p = 2 (the spread brackets the extrapolation; oscillating sequences make
    neither exact)."""
    ms, vals = np.asarray(ms, float), np.asarray(vals, float)
    sel = ms >= m_min
    if sel.sum() < 3:
        return None
    res = {}
    for p in (1, 2):
        A = np.stack([np.ones(sel.sum()), ms[sel] ** -p], 1)
        coef, *_ = np.linalg.lstsq(A, vals[sel], rcond=None)
        res[f'p{p}'] = float(coef[0])
    return res


def nonmonotone(errs):
    """Number of steps where the error grows as M grows, and sign changes of the signed error."""
    e = np.asarray(errs)
    grows = int(np.sum(np.abs(e[1:]) > np.abs(e[:-1])))
    signs = np.sign(e[np.abs(e) > 0])
    flips = int(np.sum(signs[1:] != signs[:-1]))
    return grows, flips


def main():
    with open(RAW) as f:
        raw = json.load(f)
    rows = {}
    for k, v in raw.items():
        case, arm, m, grid = k.split('|')
        if 'error' in v:
            print('ERROR', k, v['error'])
            continue
        rows.setdefault((case, grid), {}).setdefault(arm, {})[int(m)] = (unpack(v['R']), unpack(v['T']))
    out, lines = {}, []
    for (case, grid), arms in rows.items():
        if set(arms) != set(C.ARMS):
            continue
        ms = sorted(set(arms['scalar']) & set(arms['integrated']))
        top = ms[-1]
        lossless = C.CASES[case][2]
        ref = {a: arms[a][top] for a in C.ARMS}
        ref['joint'] = tuple((ref['scalar'][i] + ref['integrated'][i]) / 2 for i in (0, 1))
        gap_R = float(np.max(np.abs(ref['scalar'][0] - ref['integrated'][0])))
        gap_T = float(np.max(np.abs(ref['scalar'][1] - ref['integrated'][1])))
        gap_sumT = float(abs(ref['scalar'][1].sum() - ref['integrated'][1].sum()))
        gap_sumR = float(abs(ref['scalar'][0].sum() - ref['integrated'][0].sum()))
        rec = {'grid': grid, 'M': ms, 'M_top': top, 'lossless': lossless,
               'top_gap': {'max_dR': gap_R, 'max_dT': gap_T, 'd_sumT': gap_sumT, 'd_sumR': gap_sumR},
               'arms': {}}
        for a in C.ARMS:
            ar = {'sumR': [], 'sumT': [], 'energy': [], 'own': {}, 'joint': {}, 'other': {}}
            for which, rname in (('own', a), ('joint', 'joint'),
                                 ('other', 'integrated' if a == 'scalar' else 'scalar')):
                rR, rT = ref[rname]
                d = {'max_dR': [], 'max_dT': [], 'dsumT': [], 'dsumR': []}
                for m in ms:
                    r, t = arms[a][m]
                    d['max_dR'].append(float(np.max(np.abs(pad(r, top) - rR[0]))))
                    d['max_dT'].append(float(np.max(np.abs(pad(t, top) - rT[0]))))
                    d['dsumT'].append(float(t.sum() - rT.sum()))
                    d['dsumR'].append(float(r.sum() - rR.sum()))
                ar[which] = d
            for m in ms:
                r, t = arms[a][m]
                ar['sumR'].append(float(r.sum()))
                ar['sumT'].append(float(t.sum()))
                ar['energy'].append(float(abs(r.sum() + t.sum() - 1)))
            ar['rate_sumT_joint'] = fit_rate(ms, np.abs(ar['joint']['dsumT']), top)
            ar['extrap_sumT'] = extrapolate(ms, ar['sumT'])
            ar['rate_maxdT_joint'] = fit_rate(ms, ar['joint']['max_dT'], top)
            ar['rate_sumT_own'] = fit_rate(ms, np.abs(ar['own']['dsumT']), top)
            g, fl = nonmonotone(ar['joint']['dsumT'][:-1])
            ar['nonmonotone_steps_sumT_joint'] = g
            ar['sign_flips_sumT_joint'] = fl
            # successive differences, reference free
            st = np.asarray(ar['sumT'])
            ar['succ_diff_sumT'] = [float(v) for v in np.diff(st)]
            # reference-free tail spread: max-min of sum T over the top three M
            ar['tail_spread_sumT'] = float(st[-3:].max() - st[-3:].min())
            msa = np.asarray(ms)
            mid = (msa >= 3) & (msa < top)
            ar['geomean_dsumT_joint_M3_to_top'] = float(np.exp(np.mean(np.log(
                np.abs(np.asarray(ar['joint']['dsumT'])[mid]) + 1e-16))))
            ar['geomean_maxdT_joint_M3_to_top'] = float(np.exp(np.mean(np.log(
                np.asarray(ar['joint']['max_dT'])[mid] + 1e-16))))
            ar['max_energy_M_ge_4'] = float(np.max(np.asarray(ar['energy'])[msa >= 4]))
            rec['arms'][a] = ar
        # which arm is closer to the joint reference, per M (resolved only above top gap)
        closer = []
        for i, m in enumerate(ms):
            es = abs(rec['arms']['scalar']['joint']['dsumT'][i])
            ei = abs(rec['arms']['integrated']['joint']['dsumT'][i])
            if max(es, ei) < gap_sumT:
                closer.append('unresolved')
            else:
                closer.append('scalar' if es < ei else 'integrated')
        rec['closer_sumT_joint'] = closer
        rec['closer_count'] = {k: closer.count(k) for k in ('scalar', 'integrated', 'unresolved')}
        # robustness of the verdict to the reference: closer in |dsumT| and in max|dT| against
        # each arm's own top-M solve (M < top only; each reference favours its own arm)
        rec['closer_count_by_ref'] = {}
        for refname in C.ARMS:
            for metric in ('dsumT', 'max_dT'):
                cnt = {'scalar': 0, 'integrated': 0}
                for i, m in enumerate(ms[:-1]):
                    e = {a: abs(rec['arms'][a]['own' if a == refname else 'other'][metric][i])
                         for a in C.ARMS}
                    cnt['scalar' if e['scalar'] < e['integrated'] else 'integrated'] += 1
                rec['closer_count_by_ref'][f'ref_{refname}:{metric}'] = cnt
        out[f'{case}' + ('' if grid == 'None' else f'@grid{grid}')] = rec
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    for name, rec in out.items():
        lines.append(f'== {name}  M_top={rec["M_top"]} lossless={rec["lossless"]}  arms at top: '
                     f'max|dR|={rec["top_gap"]["max_dR"]:.2e} max|dT|={rec["top_gap"]["max_dT"]:.2e} '
                     f'|dsumT|={rec["top_gap"]["d_sumT"]:.2e}')
        hdr = ('   M | ' + ' | '.join(f'{a[:5]}: sumT        |dsT|own  |dsT|jnt  mxdT_jnt  mxdR_jnt  |R+T-1|'
                                        for a in C.ARMS) + ' | closer')
        lines.append(hdr)
        for i, m in enumerate(rec['M']):
            cells = []
            for a in C.ARMS:
                ar = rec['arms'][a]
                cells.append(f'{ar["sumT"][i]:.10f} {abs(ar["own"]["dsumT"][i]):.2e} '
                             f'{abs(ar["joint"]["dsumT"][i]):.2e} {ar["joint"]["max_dT"][i]:.2e} '
                             f'{ar["joint"]["max_dR"][i]:.2e} {ar["energy"][i]:.2e}')
            lines.append(f'  {m:2d} | ' + ' | '.join(cells) + f' | {rec["closer_sumT_joint"][i]}')
        for a in C.ARMS:
            ar = rec['arms'][a]
            rs = ar['rate_sumT_joint']
            rm = ar['rate_maxdT_joint']
            lines.append(f'  {a:10s} rate p (|dsumT| jnt)={rs if rs is None else round(rs, 2)} '
                         f'(max|dT| jnt)={rm if rm is None else round(rm, 2)}  '
                         f'grow-steps={ar["nonmonotone_steps_sumT_joint"]} sign-flips={ar["sign_flips_sumT_joint"]}  '
                         f'extrap sumT(M>=6) {ar["extrap_sumT"]}')
            lines.append(f'  {a:10s} geomean|dsumT|(3<=M<top)={ar["geomean_dsumT_joint_M3_to_top"]:.2e} '
                         f'geomean max|dT|={ar["geomean_maxdT_joint_M3_to_top"]:.2e} '
                         f'tail spread sumT(top 3 M)={ar["tail_spread_sumT"]:.2e} '
                         f'max|R+T-1|(M>=4)={ar["max_energy_M_ge_4"]:.2e}')
        lines.append(f'  closer counts (joint ref, |dsumT|, resolved above top gap) {rec["closer_count"]}')
        lines.append(f'  closer counts by reference (M<top) {rec["closer_count_by_ref"]}')
    with open(TXT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
