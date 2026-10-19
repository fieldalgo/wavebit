# Bundled refractive-index tables

Each table lists wavelength in nanometres, the real refractive index n and the
extinction coefficient k (`Im(n) > 0` is loss). `wavebit.materials.load` reads
them; the identifier names the publications the numbers come from.

The numbers are taken from the [refractiveindex.info database](https://github.com/polyanskiy/refractiveindex.info-database),
commit `c5c2f188e848453def5970e347399d653df2ffc2` (2026-09-04), retrieved on
2026-10-19. The database is in the public domain: its author waived copyright and
related rights through the CC0 1.0 Universal dedication. The measurements belong
to the publications below; cite them when you use these tables.
`tools/build_material_tables.py` in the source repository rebuilds the four files
from that commit.

| File | Identifier | Material | Source | Range (nm) | How the table was made |
| --- | --- | --- | --- | --- | --- |
| `Si.txt` | `Green2008-Li1980/Si` | crystalline silicon, 300 K and 293 K | M. A. Green, *Sol. Energ. Mat. Sol. Cells* 92, 1305-1310 (2008), doi:10.1016/j.solmat.2008.06.009, for 250-1450 nm; H. H. Li, *J. Phys. Chem. Ref. Data* 9, 561-658 (1980), doi:10.1063/1.555624, beyond 1450 nm | 250-14000 | Green's n and k copied; Li's n copied with k = 0. At 1450 nm the two differ by 5e-4 in n, and Green's k is 1.4e-13 |
| `SiO2.txt` | `Malitson1965/SiO2` | fused silica, 20 °C | I. H. Malitson, *J. Opt. Soc. Am.* 55, 1205-1208 (1965), doi:10.1364/JOSA.55.001205 | 210-6700 | Sellmeier formula evaluated every 5 nm, k = 0 |
| `Si3N4.txt` | `Luke2015/Si3N4` | 340 nm Si3N4 film on thermal SiO2 | K. Luke et al., *Opt. Lett.* 40, 4823-4826 (2015), doi:10.1364/OL.40.004823 | 310-5504 | Sellmeier formula evaluated every 5 nm, k = 0 |
| `Al2O3.txt` | `Malitson1962/Al2O3` | synthetic sapphire, ordinary ray, 24 °C | I. H. Malitson, *J. Opt. Soc. Am.* 52, 1377-1379 (1962), doi:10.1364/JOSA.52.001377 | 265.2-5577 | Sellmeier formula evaluated every 5 nm, k = 0 |

Until 2026-10-19 the package shipped tables named `filmetrics/Al2O3`,
`filmetrics/Si`, `filmetrics/Si3N4` and `filmetrics/SiO2` with no recorded
source, terms or retrieval date. They were removed because their terms could not
be confirmed; loading those names now raises an error that lists these
replacements.
