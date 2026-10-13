"""Is the mirror family's centre-z discrepancy a tie at a node plane? Torch/JAX vs NumPy FD."""
import sys, warnings
import numpy as np
sys.path.insert(0, sys.argv[1])
import wavebit.fd as fd
import fdtd_traced_partner_probe as P
warnings.filterwarnings('ignore')
p = fd.plan(P.scene('mirror'))
for zc in (0.1, 0.1013, 0.0987):
    point = np.array((.12, zc))
    h = 3e-6
    d = np.array((0., h))
    fd_grad = (P.numpy_values('mirror', point + d, p) - P.numpy_values('mirror', point - d, p)) / (2 * h)
    vals, J = P.jacobian('mirror', 'torch', p, point)
    vj, JJ = P.jacobian('mirror', 'jax', p, point)
    print(f'zc {zc}: FD {fd_grad}  torch {J[:, 1]}  jax {JJ[:, 1]}', flush=True)
