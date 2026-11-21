"""A pillar phase library away from one wavelength and normal incidence, and where it stops.

The scene is the one in ``metalens_phase_library.py``: TiO2 pillars (n = 2.43) on fused silica
(n = 1.46), hexagonal pitch 350 nm, height 600 nm, light arriving from the air side. Here the
diameter sweep is repeated at 450, 532 and 633 nm, for s and p, at incidence angles from 0 to
35 degrees in air and two azimuths: 0 degrees, along a nearest-neighbour direction, and 90
degrees, along a shortest reciprocal vector. Both are mirror planes of the array.

Design decisions, and what they leave out:

* The geometry is kept rather than re-pitched, so the 0 degree, 532 nm column is the table of
  the normal-incidence script and the rest shows what that table does not cover. A pitch
  designed for 532 nm at normal incidence is not single-order elsewhere: the substrate gains a
  second propagating order at 17.2 degrees (azimuth 90) or 20.7 degrees (azimuth 0) at 532 nm,
  and at 1.4 or 1.6 degrees at 450 nm; at 633 nm it stays single-order to 38.9 degrees. Lighting from the substrate side would not change this:
  the in-plane wavevector, and so the set of propagating orders, is the same.
* Both indices are constants. The wavelength dependence printed here is therefore geometric
  only (the same pillar measured in a different wavelength); material dispersion is not in it.
* Each entry is still one infinite periodic array. Placing pillars from the table (the local
  periodic approximation), propagating to a focus and writing a layout are not part of Wavebit.

For every (wavelength, direction, polarization) the script prints the co-polarized zero-order
transmitted amplitude against diameter, the number of propagating orders in each half-space,
and the fraction of the transmitted power the zero order carries. Where that fraction is below
one, a single complex number per pillar no longer describes what the array transmits.

The checks hold for this scene; they are not an accuracy bound for other structures.

* Lossless power balance at every point.
* The propagating orders the solver reports are the ones the pitch condition predicts from the
  lattice, order by order, and where only one propagates it carries all of T.
* At 0 degrees and 532 nm, 289 orders reproduce the normal-incidence script's transmittances.
* Reciprocity: the amplitude from the substrate side equals the reversed one from the air side.
* The lattice mirror that exchanges azimuths 30 and 90 degrees, and inversion (azimuth 180).
* Each point is saved as a run record and the table is read back from the records alone.

Storage: one run record holds one ``Job``, and a ``Job`` holds one geometry and one direction,
so the table is a directory of records, one per (diameter, direction, polarization), each with
the three wavelengths, named by ``Job.key``. That makes an interrupted sweep resumable (a
record that exists is loaded, not solved again) but nothing stores the grid itself: its axes
live in this script. Pass ``--output DIR`` to keep the records; the default uses a temporary
directory. ``--full`` runs every 10 nm diameter, five angles and 289 orders (about 30 minutes
on four cores); the default is a coarser grid at 121 orders.
"""
import argparse
from pathlib import Path
import tempfile
import time

import numpy as np

from wavebit import Job, Lattice, Material, Units, eps_from_refractive_index, rcwa
from wavebit.records import load_run, run_recorded

PITCH = 350.0            # nm, hexagonal
HEIGHT = 600.0           # nm
N_PILLAR = 2.43          # TiO2, held constant over wavelength
N_SUBSTRATE = 1.46       # fused silica, held constant over wavelength
N_AIR = 1.0
WAVELENGTHS = (450.0, 532.0, 633.0)        # nm
AZIMUTHS = (0.0, 90.0)   # degrees; at normal incidence only the first is solved
POLARIZATIONS = ("s", "p")
MIN_TRANSMISSION = 0.9   # as in the normal-incidence script: below this an entry is left out

DEFAULT_GRID = dict(diameters=np.arange(90.0, 241.0, 30.0), angles=(0.0, 20.0, 35.0), orders=(5, 5))
FULL_GRID = dict(diameters=np.arange(80.0, 261.0, 10.0), angles=(0.0, 10.0, 20.0, 30.0, 35.0),
                 orders=(8, 8))

# Normal incidence, 532 nm, s, 289 orders: the numbers metalens_phase_library.py prints.
REFERENCE_ORDERS = (8, 8)
REFERENCE_T = {150.0: 0.96643, 240.0: 0.99905}
REFERENCE_LEFT_OUT = [180.0, 210.0, 220.0]
REFERENCE_SPAN = 6.649   # rad over the 16 usable entries; needs the 10 nm grid, so --full only


def problem(diameter, theta, phi, polarization, *, side="top", wavelength=WAVELENGTHS):
    """One pillar per primitive hexagonal cell; ``theta`` and ``phi`` in degrees, in the lit medium."""
    # Datasheet indices are converted explicitly to relative permittivity.
    pillar = Material(epsilon=eps_from_refractive_index(N_PILLAR))
    substrate = Material(epsilon=eps_from_refractive_index(N_SUBSTRATE))
    return rcwa.Problem(
        units=Units(length="nm"),
        stack=rcwa.Stack(lattice=Lattice.hexagonal(PITCH), bottom=substrate, layers=(
            rcwa.Layer(thickness=HEIGHT, regions=(
                rcwa.Region(geometry=rcwa.Circle(radius=diameter/2), material=pillar),)),)),
        excitation=rcwa.Excitation(wavelength=wavelength, sources=(
            rcwa.PlaneWave(side=side, polarization=polarization,
                           theta=float(np.radians(theta)), phi=float(np.radians(phi))),)),
    )


def directions(angles):
    """(theta, phi) pairs in degrees: every azimuth for oblique angles, one for normal incidence."""
    return [(theta, phi) for theta in angles for phi in (AZIMUTHS if theta else AZIMUTHS[:1])]


# --- The pitch condition, from the lattice alone -------------------------------------------

def reciprocal_basis():
    """Columns b1, b2 with a_i . b_j = 2*pi*delta_ij, for the cell the problems are solved on."""
    cell = np.asarray(Lattice.hexagonal(PITCH), dtype=float)   # columns a1, a2
    return 2*np.pi*np.linalg.inv(cell).T


def in_plane_wavevector(wavelength, theta, phi):
    k0 = 2*np.pi/wavelength
    return k0*np.sin(np.radians(theta))*np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi))])


def order_wavevectors(wavelength, theta, phi, reach):
    """``k_par + m*b1 + n*b2`` for |m|, |n| <= reach, indexed [n, m, xy]."""
    basis, m = reciprocal_basis(), np.arange(-reach, reach+1)
    return (in_plane_wavevector(wavelength, theta, phi)
            + m[None, :, None]*basis[:, 0] + m[:, None, None]*basis[:, 1])


def predicted_orders(wavelength, theta, phi, index, reach):
    """Which orders (m, n), |m|, |n| <= reach, propagate in a medium of this index.

    Order (m, n) propagates when ``|k_par + m*b1 + n*b2| < k0*index``. Returns the boolean
    (2*reach+1, 2*reach+1) mask, indexed [n, m] as the solver's order arrays are, and the smallest distance of any order from that threshold in
    units of k0: a small value means a Rayleigh anomaly is close and the count is fragile.
    """
    k0 = 2*np.pi/wavelength
    q = np.linalg.norm(order_wavevectors(wavelength, theta, phi, reach), axis=-1)/k0
    return q < index, float(np.abs(q-index).min())


def second_order_onset(wavelength, phi, index):
    """Smallest incidence angle in air (degrees) at which a second order propagates, or None.

    For each reciprocal vector G, ``|k*s + G| = k0*index`` is a quadratic in the in-plane
    wavenumber k along the unit vector s of the azimuth.
    """
    basis, k0 = reciprocal_basis(), 2*np.pi/wavelength
    s = np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi))])
    onset = None
    for m in range(-3, 4):
        for n in range(-3, 4):
            g = m*basis[:, 0] + n*basis[:, 1]
            if (m, n) == (0, 0):
                continue
            if g @ g < (k0*index)**2:
                return 0.0
            discriminant = (s @ g)**2 - g @ g + (k0*index)**2
            if discriminant < 0:
                continue
            k = -(s @ g) - np.sqrt(discriminant)   # the smaller root; the larger leaves the disc again
            if 0 < k < k0 and (onset is None or k < onset):
                onset = k
    return None if onset is None else float(np.degrees(np.arcsin(onset/k0)))


# --- One table entry, from a live result or from a loaded record ----------------------------

def _field(orders, name):
    # A live DiffractionOrders has attributes; a loaded one is a mapping of the same names.
    return np.asarray(orders[name] if hasattr(orders, "keys") else getattr(orders, name))


def entry(transmitted, reflected, polarization):
    """What the library keeps of one solve, per wavelength."""
    co, cross = (("amplitude_ratio_s", "amplitude_ratio_p") if polarization == "s"
                 else ("amplitude_ratio_p", "amplitude_ratio_s"))
    orders_u = list(_field(transmitted, "orders_u"))
    orders_v = list(_field(transmitted, "orders_v"))
    iu, iv = orders_u.index(0), orders_v.index(0)
    efficiency = _field(transmitted, "efficiency")
    total = efficiency.sum(axis=(1, 2))
    return {
        "co": _field(transmitted, co)[:, iu, iv].copy(),
        "cross": _field(transmitted, cross)[:, iu, iv].copy(),
        "T": total,
        "R": _field(reflected, "efficiency").sum(axis=(1, 2)),
        "zero_fraction": efficiency[:, iu, iv]/total,
        # A propagating order has a real normal wavenumber; an evanescent one an imaginary one.
        "propagating_air": np.abs(_field(reflected, "k_normal").imag) == 0,
        "propagating_substrate": np.abs(_field(transmitted, "k_normal").imag) == 0,
        "k_parallel": _field(transmitted, "k_parallel").real.copy(),
    }


def solve_point(directory, diameter, theta, phi, polarization, numerics):
    """The record of one grid point: loaded if it exists, otherwise solved and saved.

    Returns the entry read from the record on disk and whether a solve was needed. A directory
    left incomplete by an interrupted write is refused by ``load_run``; delete it and rerun.
    """
    job = Job(solver="rcwa", problem=problem(diameter, theta, phi, polarization), numerics=numerics)
    path = Path(directory)/job.key
    live = {}
    if path.exists():
        record = load_run(path)
    else:
        def sample(result):
            live.update(entry(result.transmission, result.reflection, polarization))
            return {"library_entry": dict(live)}
        # The source snapshot is left out to keep hundreds of records small; its identity stays.
        record = run_recorded(job, path, sample=sample, include_source=False)
    assert record.job.key == job.key
    restored = entry(record.outputs["transmitted"], record.outputs["reflected"], polarization)
    # Round trip: the entry recomputed from the loaded outputs is the saved sample, and (when
    # this call ran the solver) the live values, bit for bit.
    for name, value in restored.items():
        np.testing.assert_array_equal(value, record.samples["library_entry"][name])
        if live:
            np.testing.assert_array_equal(value, live[name])
    return restored, bool(live)


def sweep(directory, grid):
    """The whole table, indexed [wavelength, direction, polarization, diameter]."""
    numerics = rcwa.Numerics(orders=grid["orders"])
    cases = directions(grid["angles"])
    shape = (len(WAVELENGTHS), len(cases), len(POLARIZATIONS), len(grid["diameters"]))
    table = {name: np.empty(shape, dtype=dtype) for name, dtype in (
        ("co", complex), ("cross", complex), ("T", float), ("R", float), ("zero_fraction", float))}
    table["orders_air"] = np.empty(shape[:2], dtype=int)
    table["orders_substrate"] = np.empty(shape[:2], dtype=int)
    table["threshold_margin"] = np.empty(shape[:2])
    solved = 0
    reach = grid["orders"][0]
    assert grid["orders"][0] == grid["orders"][1]
    for c, (theta, phi) in enumerate(cases):
        predicted = [[predicted_orders(wavelength, theta, phi, index, reach)
                      for index in (N_AIR, N_SUBSTRATE)] for wavelength in WAVELENGTHS]
        for w, wavelength in enumerate(WAVELENGTHS):
            # Nothing the truncation dropped propagates: a wider window predicts the same count.
            for (mask, _), index in zip(predicted[w], (N_AIR, N_SUBSTRATE)):
                assert predicted_orders(wavelength, theta, phi, index, reach+4)[0].sum() == mask.sum()
            table["orders_air"][w, c] = predicted[w][0][0].sum()
            table["orders_substrate"][w, c] = predicted[w][1][0].sum()
            table["threshold_margin"][w, c] = min(predicted[w][0][1], predicted[w][1][1])
        for p, polarization in enumerate(POLARIZATIONS):
            for d, diameter in enumerate(grid["diameters"]):
                found, ran = solve_point(directory, float(diameter), theta, phi, polarization, numerics)
                solved += ran
                for name in ("co", "cross", "T", "R", "zero_fraction"):
                    table[name][:, c, p, d] = found[name]
                for w, wavelength in enumerate(WAVELENGTHS):
                    # The solver's order arrays sit at the wavevectors the prediction used, so
                    # the two masks index the same orders.
                    np.testing.assert_allclose(found["k_parallel"][w],
                                               order_wavevectors(wavelength, theta, phi, reach),
                                               atol=1e-12, rtol=0)
                    # The order count against the pitch condition, order by order.
                    np.testing.assert_array_equal(found["propagating_air"][w], predicted[w][0][0])
                    np.testing.assert_array_equal(found["propagating_substrate"][w], predicted[w][1][0])
    return table, solved


def zero_order(result, polarization):
    order = result.transmission.order(0, 0)
    co = order.amplitude_ratio_s if polarization == "s" else order.amplitude_ratio_p
    cross = order.amplitude_ratio_p if polarization == "s" else order.amplitude_ratio_s
    return np.asarray(co).reshape(-1), np.asarray(cross).reshape(-1)


# --- Reporting and checks -------------------------------------------------------------------

def report(table, grid):
    cases = directions(grid["angles"])
    diameters = grid["diameters"]
    print("\nPropagating orders from the pitch condition (confirmed against every solve), the zero")
    print("order's smallest share of T over diameters and polarizations, and the largest change of")
    print("the relative phase (against the smallest diameter) from the normal-incidence table.")
    print("wavelength  angle  azimuth  air  substrate  onset of a 2nd      min zero-order  max phase")
    print("   (nm)     (deg)   (deg)                   substrate order(deg)  share of T    change (rad)")
    for w, wavelength in enumerate(WAVELENGTHS):
        normal = table["co"][w, 0]                              # [polarization, diameter]
        for c, (theta, phi) in enumerate(cases):
            onset = second_order_onset(wavelength, phi, N_SUBSTRATE)
            change = np.angle(table["co"][w, c]*np.conj(normal))
            change = np.angle(np.exp(1j*(change-change[:, :1])))
            print(f"  {wavelength:5.0f}    {theta:5.1f}  {phi:5.0f}    {table['orders_air'][w, c]:2d}"
                  f"     {table['orders_substrate'][w, c]:2d}          "
                  + (f"{onset:6.2f}" if onset is not None else "  none")
                  + f"            {table['zero_fraction'][w, c].min():.4f}        {np.abs(change).max():.3f}")
    for w, wavelength in enumerate(WAVELENGTHS):
        for p, polarization in enumerate(POLARIZATIONS):
            print(f"\n{wavelength:.0f} nm, {polarization}: co-polarized zero-order amplitude, "
                  "|t| (electric-field ratio) and phase (rad)")
            print("  angle,azimuth:" + "".join(f"  {theta:4.0f},{phi:3.0f}    " for theta, phi in cases))
            for d, diameter in enumerate(diameters):
                amplitude = table["co"][w, :, p, d]
                print(f"  d = {diameter:3.0f} nm   " + "".join(
                    f"  {abs(a):.3f} {np.angle(a):+.3f}" for a in amplitude))


def check_table(table, grid):
    balance = np.abs(table["R"]+table["T"]-1).max()
    print(f"\npower balance: max |R + T - 1| = {balance:.1e}")
    assert balance < 1e-10                                       # measured 7.4e-13; 2.6e-12 with --full

    single = table["orders_substrate"] == 1                      # [wavelength, direction]
    fraction = table["zero_fraction"]
    deficit = np.abs(fraction[single]-1).max()
    print(f"single-order cases: max |zero-order share - 1| = {deficit:.1e}; "
          f"multi-order cases: share from {fraction[~single].min():.4f} to {fraction[~single].max():.4f}")
    assert deficit < 1e-12                                       # measured 0.0
    assert fraction.max() <= 1+1e-12
    print(f"nearest Rayleigh threshold: {table['threshold_margin'].min():.4f} k0 "
          "(the distance of the closest order from propagating/evanescent)")
    # Every case stands clear of a threshold by far more than rounding, so the counts are firm.
    assert table["threshold_margin"].min() > 1e-3               # measured 0.0089 (532 nm, 20 deg, azimuth 0)

    # Both azimuths are mirror planes of the array, so s and p do not mix: the cross-polarized
    # zero order is zero in the exact problem. The parallelogram of retained orders does not
    # share those mirrors, which leaves a remainder that must shrink as orders are added. It is
    # largest on a resonance, so the worst entry of the table is solved again with more orders.
    cross = np.abs(table["cross"])
    w, c, p, d = np.unravel_index(cross.argmax(), cross.shape)
    theta, phi = directions(grid["angles"])[c]
    more = grid["orders"][0]+3
    refined = rcwa.run(problem(float(grid["diameters"][d]), theta, phi, POLARIZATIONS[p],
                               wavelength=WAVELENGTHS[w]), numerics=rcwa.Numerics(orders=(more, more)))
    refined_cross = float(np.abs(zero_order(refined, POLARIZATIONS[p])[1])[0])
    print(f"cross-polarized zero order, forbidden by the mirror planes: max |t| = {cross.max():.1e} "
          f"({WAVELENGTHS[w]:.0f} nm, {theta:.0f} deg, d = {grid['diameters'][d]:.0f} nm, "
          f"T = {table['T'][w, c, p, d]:.2f}); {refined_cross:.1e} at {(2*more+1)**2} orders")
    # Measured 3.3e-2 at 121 orders against 1.0e-2 at 289 (450 nm, 0 deg, d = 240 nm); with --full,
    # 1.5e-2 at 289 orders against 2.8e-3 at 529 (450 nm, 0 deg, d = 150 nm).
    assert refined_cross < cross.max()/2


def check_reference(table, grid, numerics):
    """Normal incidence at 532 nm against the numbers the normal-incidence script prints."""
    w = WAVELENGTHS.index(532.0)
    reference = rcwa.Numerics(orders=REFERENCE_ORDERS)
    for diameter, expected in REFERENCE_T.items():
        result = rcwa.run(problem(diameter, 0.0, 0.0, "s", wavelength=532.0), numerics=reference)
        t = float(result.T.sum())
        line = f"0 deg, 532 nm, d = {diameter:.0f} nm at 289 orders: T = {t:.5f} (expected {expected:.5f})"
        # The expected values are printed to five decimals: agreement within half a unit of the last.
        assert abs(t-expected) < 5e-6                            # measured 2.1e-6 and 1.4e-6
        d = np.flatnonzero(grid["diameters"] == diameter)
        if d.size:
            line += f"; this sweep's table differs by {abs(table['T'][w, 0, 0, d[0]]-t):.1e}"
            if grid["orders"] == REFERENCE_ORDERS:
                assert table["T"][w, 0, 0, d[0]] == t
        print(line)
    if grid is FULL_GRID:
        # Same diameters and orders as the normal-incidence script: the same usable entries and span.
        usable = table["T"][w, 0, 0] >= MIN_TRANSMISSION
        phase = np.unwrap(np.angle(table["co"][w, 0, 0][usable]))
        print(f"0 deg, 532 nm, s: left out {grid['diameters'][~usable].tolist()}, "
              f"span {phase[-1]-phase[0]:.3f} rad over {usable.sum()} entries")
        assert grid["diameters"][~usable].tolist() == REFERENCE_LEFT_OUT
        assert abs(phase[-1]-phase[0]-REFERENCE_SPAN) < 5e-4


def check_symmetry(numerics, theta, diameter=150.0):
    """Three statements about one oblique point; each compares complex amplitudes at all wavelengths."""
    refracted = float(np.degrees(np.arcsin(np.sin(np.radians(theta))/N_SUBSTRATE)))
    # Amplitude ratios are of electric fields; reciprocity equates them after this flux factor.
    flux = np.cos(np.radians(theta))/(N_SUBSTRATE*np.cos(np.radians(refracted)))
    worst = dict(reciprocity=0., mirror=0., inversion=0.)
    for polarization in POLARIZATIONS:
        def amplitudes(phi, side="top", angle=theta):
            return zero_order(rcwa.run(problem(diameter, angle, phi, polarization, side=side),
                                       numerics=numerics), polarization)
        for phi in AZIMUTHS:
            forward, _ = amplitudes(phi)
            reverse, _ = amplitudes(phi+180.0)
            # Reciprocity: air to substrate with in-plane wavevector -k is substrate to air with +k.
            from_substrate, _ = amplitudes(phi, side="bottom", angle=refracted)
            worst["reciprocity"] = max(worst["reciprocity"], np.abs(from_substrate*flux-reverse).max())
            # Inversion through the pillar axis maps the array onto itself and k to -k.
            worst["inversion"] = max(worst["inversion"], np.abs(forward-reverse).max())
        # The mirror through the 60 degree line exchanges azimuths 30 and 90 degrees and the
        # reciprocal basis vectors, so it also maps the retained orders onto themselves. A mirror
        # reverses handedness: the co-polarized amplitudes agree, the cross-polarized ones flip sign.
        co_90, cross_90 = amplitudes(90.0)
        co_30, cross_30 = amplitudes(30.0)
        worst["mirror"] = max(worst["mirror"], np.abs(co_90-co_30).max(), np.abs(cross_90+cross_30).max())
    print(f"at {theta:.0f} deg, d = {diameter:.0f} nm: reciprocity {worst['reciprocity']:.1e}, "
          f"mirror (azimuth 30 against 90) {worst['mirror']:.1e}, inversion {worst['inversion']:.1e}")
    assert worst["reciprocity"] < 1e-10                          # measured 3.8e-13; 1.4e-12 with --full
    assert worst["mirror"] < 1e-10                               # measured 2.1e-13; 1.8e-12 with --full
    # Inversion is exact for the array but not for the 1024-point grid the circle's normal field
    # is sampled on; the remainder halves when that grid is doubled (1.5e-7 at 1024, 7.2e-8 at 2048).
    assert worst["inversion"] < 1e-6                             # measured 2.1e-7; 1.5e-7 with --full


def run(directory, grid):
    cases = directions(grid["angles"])
    points = len(cases)*len(POLARIZATIONS)*len(grid["diameters"])
    retained = (2*grid["orders"][0]+1)*(2*grid["orders"][1]+1)
    print(f"{points} records, {len(WAVELENGTHS)} wavelengths each, {retained} orders")

    started = time.perf_counter()
    table, solved = sweep(directory, grid)
    elapsed = time.perf_counter()-started
    print(f"sweep: {elapsed:.1f} s, {solved} of {points} points solved and saved, {points-solved} loaded "
          "(this machine, NumPy backend; not a benchmark)")

    report(table, grid)
    check_table(table, grid)
    numerics = rcwa.Numerics(orders=grid["orders"])
    check_reference(table, grid, numerics)
    check_symmetry(numerics, max(grid["angles"]))

    # The records alone rebuild the table: a second pass finds every point on disk, runs no
    # solver and returns the same numbers bit for bit. This is also how an interrupted sweep resumes.
    started = time.perf_counter()
    again, solved_again = sweep(directory, grid)
    print(f"read back from {points} records in {time.perf_counter()-started:.1f} s, {solved_again} solves")
    assert solved_again == 0
    for name, value in table.items():
        np.testing.assert_array_equal(again[name], value)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--full", action="store_true",
                        help="every 10 nm diameter, five angles, 289 orders (about 30 minutes on four cores)")
    parser.add_argument("--output", type=Path,
                        help="directory for the run records; existing records in it are reused")
    args = parser.parse_args()
    grid = FULL_GRID if args.full else DEFAULT_GRID
    started = time.perf_counter()
    if args.output is None:
        with tempfile.TemporaryDirectory(prefix="wavebit-phase-library-") as directory:
            run(directory, grid)
        print("Temporary records removed. Pass --output to keep them.")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        run(args.output, grid)
        print("Records kept in:", args.output.resolve())
    print(f"total: {time.perf_counter()-started:.1f} s")


if __name__ == "__main__":
    main()
