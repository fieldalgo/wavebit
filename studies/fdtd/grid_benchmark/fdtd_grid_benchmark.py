"""Compare equivalent NumPy/JAX grid updates; includes warm timings and parity."""
import json
import time

import numpy as np
import jax
import jax.numpy as jnp
from wavebit.fdtd._periodic.grid import fdtd_grid as wavebit_fdtd_grid

jax.config.update('jax_enable_x64', True)
epsilon=np.ones((32,8,8))
electric=np.zeros((3,32,8,8),complex)
electric[1]=np.exp(2j*np.pi*np.arange(32)/32)[:,None,None]
kw=dict(spacing=(1.,1.,1.),dt=.2,steps=256,electric=electric,
        wavelengths=[20.,30.],monitor_z=[8,24])
start=time.perf_counter()
reference=wavebit_fdtd_grid(epsilon,**kw)
numpy_seconds=time.perf_counter()-start
run=jax.jit(lambda eps:wavebit_fdtd_grid(eps,**kw,backend='jax'))
start=time.perf_counter()
result=run(jnp.asarray(epsilon))
result['E'].block_until_ready()
compile_and_first=time.perf_counter()-start
samples=[]
for _ in range(5):
    start=time.perf_counter()
    result=run(jnp.asarray(epsilon))
    result['E'].block_until_ready()
    samples.append(time.perf_counter()-start)
gap=float(np.max(abs(np.asarray(result['E'])-reference['E'])))
assert gap<1e-12
print(json.dumps(dict(devices=[str(d) for d in jax.devices()],shape=list(epsilon.shape),steps=256,
                     numpy_seconds=numpy_seconds,jax_compile_and_first_seconds=compile_and_first,
                     jax_warm_median_seconds=float(np.median(samples)),
                     warm_speedup=numpy_seconds/float(np.median(samples)),field_gap=gap),indent=2))
