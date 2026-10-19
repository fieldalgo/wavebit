"""Rebuild the bundled refractive-index tables from the refractiveindex.info database.

The database is public domain (CC0 1.0). Each table below comes from one
published dataset; ``src/wavebit/data/refractiveindex/README.md`` cites it.
Tabulated datasets are copied with wavelengths converted to nanometres.
Sellmeier datasets (formula 1) are evaluated on a 5 nm grid over the range the
publication states, with k = 0. Silicon joins two crystalline-silicon datasets:
Green (2008) up to its last row at 1450 nm, then Li (1980, 293 K) beyond it with
k = 0; at 1450 nm they differ by 5e-4 in n and Green's k is 1.4e-13.

    git clone https://github.com/polyanskiy/refractiveindex.info-database DB
    git -C DB checkout c5c2f188e848453def5970e347399d653df2ffc2
    python tools/build_material_tables.py DB

The command rewrites the four tables; ``git diff`` then shows any change.
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

OUTPUT = Path(__file__).resolve().parents[1]/'src'/'wavebit'/'data'/'refractiveindex'
SOURCES = {
    'Si': ('Si/nk/Green-2008.yml', 'Si/nk/Li-293K.yml'),
    'SiO2': 'SiO2/nk/Malitson.yml',
    'Si3N4': 'Si3N4/nk/Luke.yml',
    'Al2O3': 'Al2O3/nk/Malitson.yml',
}


def sellmeier(coefficients, wavelength_um):
    """refractiveindex.info formula 1: n**2 = 1 + C0 + sum B*l**2/(l**2 - C**2)."""
    c = np.asarray(coefficients, dtype=float)
    square = 1 + c[0] + np.zeros_like(wavelength_um)
    for b, resonance in zip(c[1::2], c[2::2]):
        square += b*wavelength_um**2/(wavelength_um**2 - resonance**2)
    return np.sqrt(square)


def table(entry):
    (block,) = entry['DATA']
    if block['type'] == 'tabulated nk':
        rows = np.array([line.split() for line in block['data'].strip().splitlines()], dtype=float)
        return rows[:, 0]*1e3, rows[:, 1], rows[:, 2]
    if block['type'] == 'tabulated n':
        rows = np.array([line.split() for line in block['data'].strip().splitlines()], dtype=float)
        return rows[:, 0]*1e3, rows[:, 1], np.zeros(len(rows))
    if block['type'] == 'formula 1':
        low, high = (float(v)*1e3 for v in block['wavelength_range'].split())
        nm = np.arange(np.ceil(low/5)*5, high + 1e-9, 5.)
        nm = np.unique(np.r_[low, nm, high])
        coefficients = [float(v) for v in block['coefficients'].split()]
        return nm, sellmeier(coefficients, nm*1e-3), np.zeros_like(nm)
    raise ValueError(f"unsupported dataset type {block['type']!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('database', type=Path, help='refractiveindex.info-database checkout')
    args = parser.parse_args()
    root = args.database/'database'/'data'/'main'
    for name, relatives in SOURCES.items():
        relatives = (relatives,) if isinstance(relatives, str) else relatives
        nm, n, k = (np.array([]),)*3
        for relative in relatives:
            more = table(yaml.safe_load((root/relative).read_text(encoding='utf-8')))
            keep = more[0] > (nm[-1] if len(nm) else -np.inf)
            nm, n, k = (np.r_[old, new[keep]] for old, new in zip((nm, n, k), more))
        relative = ' + '.join(relatives)
        lines = ['Wavelength(nm)\tn\tk'] + [f'{w:.6g}\t{a:.6f}\t{b:.6g}' for w, a, b in zip(nm, n, k)]
        (OUTPUT/f'{name}.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
        print(f'{name}: {relative}, {nm[0]:g}-{nm[-1]:g} nm, {len(nm)} rows')


if __name__ == '__main__':
    main()
