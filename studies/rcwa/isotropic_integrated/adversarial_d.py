"""Adversarial re-check D: polygon T(x) jaggedness on the integrated arm (other x, other polygon, orders (2,2))."""
import json, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import jax, jax.numpy as jnp
from wavebit import rcwa, Material
from studies.rcwa.isotropic_integrated import cases as C

def tri(x):
    v = [(x*np.cos(2*np.pi*k/3 + .5), .28*np.sin(2*np.pi*k/3 + .5)) for k in range(3)]
    return rcwa.Problem(
        stack=rcwa.Stack(layers=(rcwa.Layer(thickness=C.THICKNESS, background=Material(1.5),
            regions=(rcwa.Region(geometry=rcwa.Polygon(vertices=v), material=Material(4.)),)),),
            bottom=Material(C.SUBSTRATE), period=(1., 1.)),
        excitation=rcwa.Excitation(wavelength=C.WAVELENGTH,
                                   sources=(rcwa.PlaneWave(theta=C.THETA, phi=C.PHI, polarization='s'),)))
builders = {'pentagon': lambda x: C.problem('polygon', x), 'triangle': tri}
O = (2, 2)
H = 5e-6
out = {}
which = sys.argv[1:] or ['pentagon|integrated', 'triangle|integrated', 'pentagon|scalar']
for key in which:
    shape, a = key.split('|')
    x0 = .30
    with C.arm(a):
        Ts = []
        for k in range(9):
            r = rcwa.run(builders[shape](x0 + k*H), numerics=C.numerics(O), backend='numpy')
            Ts.append(float(np.sum(C.to_np(r.T))))
        with jax.enable_x64(True):
            f = lambda x: jnp.sum(rcwa.run(builders[shape](x), numerics=C.numerics(O), backend='jax').T)
            ad = float(jax.grad(f)(jnp.asarray(x0)))
    slopes = list(np.diff(Ts)/H)
    out[key] = dict(T=Ts, slopes=slopes, ad=ad, fd_wide=(Ts[-1]-Ts[0])/(8*H))
    print(key, 'AD', ad, 'slopes', np.round(slopes, 4).tolist(), 'wide', out[key]['fd_wide'], flush=True)
p = pathlib.Path(__file__).with_name('adversarial_d.json')
old = json.loads(p.read_text()) if p.exists() else {}
old.update(out); p.write_text(json.dumps(old, indent=1))
