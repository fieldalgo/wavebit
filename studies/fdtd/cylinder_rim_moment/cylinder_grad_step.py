"""test_fd_cylinder_grad[y]: AD against central, forward and backward differences at five steps.

The y-axis probe puts a cap 0.002 = eps dx inside a voxel: the end of the cap ramp.
Run with WAVEBIT_STUDY_CYLINDER_MOMENT=split (and _CAP_STEP=c1|c2)."""
import sys, warnings, numpy as np
sys.path.insert(0,'/home/user/wavebit/tests/fdtd'); sys.path.insert(0,'/home/user/wavebit')
warnings.simplefilter('ignore')
import jax; jax.config.update('jax_enable_x64', True); import jax.numpy as jnp
import wavebit.fdtd._legacy as fd
from wavebit.fdtd._runtime import runtime
from test_fd_cylinder_grad import scene, objective, POINT
runtime('jax'); p = fd.plan(scene(axis='y'))
g = np.asarray(jax.grad(lambda v: objective(v,'y','jax',p))(jnp.asarray(POINT)))
print('AD', g)
e = np.eye(5)[3]
for h in (3e-6, 1e-6, 3e-7, 1e-7, 3e-8):
    cd = (objective(POINT+h*e,'y','numpy',p)-objective(POINT-h*e,'y','numpy',p))/(2*h)
    one = (objective(POINT+h*e,'y','numpy',p)-objective(POINT,'y','numpy',p))/h
    two = (objective(POINT,'y','numpy',p)-objective(POINT-h*e,'y','numpy',p))/h
    print(f'h {h:.0e} central {cd:.9f} forward {one:.9f} backward {two:.9f}  AD-central rel {abs(g[3]-cd)/abs(cd):.1e}')
