"""Part (3) of quasistatic_demo with 16 offsets and a least-squares slope."""
import numpy as np
from quasistatic_demo import solve
REF = 1.570562408117   # aligned-grid Richardson value (quasistatic_demo part 1, N<=512)
rng = np.random.default_rng(7)
offs = rng.random((16, 2))
NN = np.array([33, 65, 129, 257, 513])
for rule in ['smooth', 'cell']:
    rms, mx = [], []
    for N in NN:
        e = np.array([solve(N, 0.25 + ox, 0.25 + oy, rule=rule)['eps_eff'] - REF for ox, oy in offs])
        rms.append(np.sqrt(np.mean(e**2))); mx.append(np.max(abs(e)))
    rms = np.array(rms); mx = np.array(mx)
    s_rms = -np.polyfit(np.log(NN), np.log(rms), 1)[0]
    s_max = -np.polyfit(np.log(NN), np.log(mx), 1)[0]
    print(f"rule={rule:6s} RMS={np.array2string(rms, precision=4)}  max={np.array2string(mx, precision=4)}")
    print(f"             LS slope: RMS {s_rms:.3f}, max {s_max:.3f}", flush=True)
