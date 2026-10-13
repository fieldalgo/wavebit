"""Part (1) of quasistatic_demo extended to N=1024 (aligned square, contrast 12)."""
import time
import numpy as np
from quasistatic_demo import solve
Ns = [128, 256, 512, 1024]
res = []
for N in Ns:
    t = time.time(); res.append(solve(N)); print(f"N={N} done {time.time()-t:.1f}s", flush=True)
for key in ['eps_eff', 'J_box', 'E_box']:
    v = [r[key] for r in res]
    orders = [np.log2((v[k-2]-v[k-1])/(v[k-1]-v[k])) for k in range(2, len(v))]
    print(f"{key}: values {np.array2string(np.array(v), precision=12)}")
    print(f"   three-grid orders (ending N=512, 1024): {np.array2string(np.array(orders), precision=4)}   2*nu = 1.4438")
