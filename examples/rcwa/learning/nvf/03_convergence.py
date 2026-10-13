"""Two separate resolution sweeps; this short ladder is not a certificate."""
import importlib.util
import json
from pathlib import Path

path = Path(__file__).with_name('01_shape_nvf.py')
spec = importlib.util.spec_from_file_location('shape_example', path)
shape_example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shape_example)


def run(orders, grid):
    result = shape_example.make_solver(orders, grid).solve()
    r, t = float(result.R.sum()), float(result.T.sum())
    return dict(orders=list(orders), shape_nvf_grid_size=grid, R_p=r, T_p=t,
                energy_balance_residual=abs(r+t-1.))


if __name__ == '__main__':
    # Hold the other resolution fixed in each sweep.
    rows = [dict(sweep='Fourier', **run((n, n), 128)) for n in (1, 2, 3)]
    rows += [dict(sweep='shape_nvf_grid_size', **run((3, 3), g)) for g in (64, 128, 256)]
    print(json.dumps(rows, indent=2))
