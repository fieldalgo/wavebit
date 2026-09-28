"""Public ranged-sphere objective: backend gradients against NumPy central differences."""
import json
from pathlib import Path
import time
import warnings
import numpy as np
import wavebit.fd as fd
from wavebit.fd._runtime import runtime
from studies.fdtd_sphere_range_probe import scene, POINTS


def objective(v, backend, plan):
    result = fd.run(scene(v[0], tuple(v[1:])), backend, plan=plan)
    return (result.final.E**2).sum()


def measure():
    warnings.filterwarnings('ignore')
    v = np.array((POINTS[0][0], *POINTS[0][1]))
    p = fd.plan(scene(v[0], tuple(v[1:])))
    rows = {'point': v.tolist(), 'finite_differences': []}
    rows['numpy_value'] = float(objective(v, 'numpy', p))
    for h in (1e-4, 3e-5, 1e-5, 3e-6, 1e-6):
        derivatives = []
        for i in range(4):
            delta = np.eye(4)[i]*h
            derivatives.append(float((objective(v+delta, 'numpy', p)-objective(v-delta, 'numpy', p))/(2*h)))
        row = dict(step=h, gradient=derivatives)
        rows['finite_differences'].append(row)
        print(row, flush=True)
    for backend in ('jax', 'torch'):
        runtime(backend)
        started = time.perf_counter()
        if backend == 'jax':
            import jax
            import jax.numpy as jnp
            fn = lambda x: objective(x, backend, p)
            value, gradient = jax.value_and_grad(fn)(jnp.asarray(v))
            rows[backend] = dict(value=float(value), gradient=np.asarray(gradient).tolist())
        else:
            import torch
            x = torch.tensor(v, dtype=torch.float64, requires_grad=True)
            value = objective(x, backend, p)
            gradient, = torch.autograd.grad(value, x)
            rows[backend] = dict(value=float(value.detach()), gradient=gradient.detach().numpy().tolist())
            try:
                primal, tangent = torch.func.jvp(lambda x: objective(x, backend, p),
                                                (x.detach(),), (torch.ones_like(x),))
                rows['torch_jvp'] = dict(value=float(primal), tangent=float(tangent))
            except Exception as exc:
                rows['torch_jvp'] = dict(error=str(exc), type=type(exc).__name__)
        rows[backend]['seconds'] = time.perf_counter()-started
        print(backend, rows[backend], flush=True)
    print('torch_jvp', rows['torch_jvp'], flush=True)
    monitors = (fd.Record('r', at=(.2, .03, -.02)),
                fd.Field('f', box=fd.Region(size=(.2,)*3, center=(.2, 0., 0.)),
                         components=('Ey',), wavelengths=(3.,)))
    mp = fd.plan(scene(*POINTS[0]).replace(monitors=monitors))
    def monitor_objective(v, backend):
        result = fd.run(scene(v[0], tuple(v[1:])).replace(monitors=monitors), backend, plan=mp)
        return (abs(result['f'].fields['Ey'])**2).sum() + (result['r'].values**2).sum()
    v = np.array((POINTS[1][0], *POINTS[1][1]))
    x = torch.tensor(v, dtype=torch.float64, requires_grad=True)
    value = monitor_objective(x, 'torch')
    gradient, = torch.autograd.grad(value, x)
    fd_gradient = []
    for delta in np.eye(4)*3e-6:
        fd_gradient.append(float((monitor_objective(v+delta, 'numpy')
                                  - monitor_objective(v-delta, 'numpy'))/6e-6))
    rows['monitors'] = dict(point=v.tolist(), gradient=gradient.detach().numpy().tolist(),
                            finite_difference=fd_gradient, step=3e-6)
    print('monitors', rows['monitors'], flush=True)
    return rows


if __name__ == '__main__':
    destination = Path(__file__).resolve().parent/'results/fdtd_sphere_grad_probe.json'
    destination.write_text(json.dumps(measure(), indent=2)+'\n', encoding='utf-8')
