"""Reproduce N5's six five-lane vmaps inside one JIT or independent calls."""

import argparse, faulthandler, hashlib, json, os, platform, resource, signal, sys, time
from pathlib import Path

faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True)
p = argparse.ArgumentParser()
p.add_argument("--root", default=str(Path(__file__).resolve().parents[3]))
p.add_argument("--output", required=True)
p.add_argument("--execution", choices=["joined", "separate"], default="joined")
p.add_argument("--mode", choices=["value", "jacobian"], default="value")
p.add_argument("--offset", type=float, default=0.1)
p.add_argument("--parameter", type=int, default=2)
p.add_argument("--repeat", type=int, default=1)
a = p.parse_args()
sys.path[:0] = [a.root + "/src", a.root]
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import wavebit
from wavebit import rcwa
from studies.rcwa.nvf_fitting_grid.nvf_fitting_grid import TRUE, problem

assert Path(wavebit.__file__).is_relative_to(a.root)
q = np.array(TRUE)
q[a.parameter] += a.offset
qtuple = tuple(float(x) for x in q)
w = jnp.asarray(np.tile(np.linspace(800.0, 1200.0, 15), 2))
j = jnp.asarray(np.repeat(np.eye(2), 15, axis=0))
record = dict(
    completed=False,
    execution=a.execution,
    mode=a.mode,
    parameters=q.tolist(),
    versions=dict(
        python=platform.python_version(), jax=jax.__version__, numpy=np.__version__
    ),
    platform=platform.platform(),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    thread_environment={
        k: os.environ.get(k)
        for k in (
            "XLA_FLAGS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        )
    },
    repeats=[],
)
out = Path(a.output)
out.parent.mkdir(parents=True, exist_ok=True)
assert not out.exists()


def save():
    out.write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


for repeat in range(a.repeat):
    t = time.monotonic()
    print("prepare", repeat, qtuple, flush=True)
    prepared = rcwa.prepare(
        problem(qtuple, 800.0, (1.0, 0.0)),
        backend="jax",
        numerics=rcwa.Numerics(
            orders=(4, 4),
            shape_nvf_grid_size=128,
            **({"topology_check": "host"} if a.mode == "jacobian" else {}),
        ),
    )
    prepared.report.raise_if_invalid()

    def lane(q, wl, jo):
        r = prepared.run(problem(q, wl, (jo[0], jo[1])))
        checks = {
            name: jnp.all(jnp.asarray(flag)) for name, flag in r.validity.checks.items()
        }
        return jnp.real(r.reflection.order(0, 0).efficiency)[0], checks

    def values(q, w, j):
        parts = [
            jax.vmap(lambda wl, jo: lane(q, wl, jo))(w[i : i + 5], j[i : i + 5])
            for i in range(0, len(w), 5)
        ]
        return jax.tree_util.tree_map(lambda *leaves: jnp.concatenate(leaves), *parts)

    def with_value(q, w, j):
        v, flags = values(q, w, j)
        return v, (v, flags)

    f = (
        jax.jit(lambda w, j: values(qtuple, w, j))
        if a.mode == "value"
        else jax.jit(jax.jacfwd(with_value, has_aux=True))
    )
    n = 30 if a.execution == "joined" else 5
    args = (w[:n], j[:n]) if a.mode == "value" else (jnp.asarray(q), w[:n], j[:n])
    print("compile", repeat, flush=True)
    lowered = f.lower(*args)
    compiled = lowered.compile()
    row = dict(
        compile_seconds=time.monotonic() - t,
        temp_bytes=compiled.memory_analysis().temp_size_in_bytes,
    )
    record["repeats"].append(row)
    save()
    print("run", repeat, row, flush=True)
    t = time.monotonic()
    parts = []
    for start in range(0, 30, n):
        args = (
            (w[start : start + n], j[start : start + n])
            if a.mode == "value"
            else (jnp.asarray(q), w[start : start + n], j[start : start + n])
        )
        parts.append(jax.tree_util.tree_map(np.asarray, compiled(*args)))
        print("part completed", start, flush=True)
    result = jax.tree_util.tree_map(lambda *xs: np.concatenate(xs), *parts)
    if a.mode == "value":
        v, flags = result
        jac = None
    else:
        jac, (v, flags) = result
    assert all(np.all(x) for x in flags.values()) and np.isfinite(v).all()
    if jac is not None:
        assert np.isfinite(jac).all()
    row.update(
        run_seconds=time.monotonic() - t,
        values=v.tolist(),
        flags={k: x.tolist() for k, x in flags.items()},
        jacobian=None if jac is None else jac.tolist(),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
    )
    save()
    print("completed repeat", repeat, "seconds", row["run_seconds"], flush=True)
    del compiled, lowered, f, prepared
    jax.clear_caches()
    import gc

    gc.collect()
record["completed"] = True
save()
