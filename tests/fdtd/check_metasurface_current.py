"""Qualify M1 p with the original C7 criteria before exporting a current-source row.

Run from a configured checkout with its source and repository root on PYTHONPATH.
Archived hard-source rows remain immutable; only a separately qualified current
row can be exported. Stored RCWA ladders are not fresh external-solver executions.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import hashlib
import json
from pathlib import Path
from studies.cross_solver.metasurface import fdtd_metasurface_rcwa as h
from tests.fdtd.check_planewave_acceptance import write


def references():
    path = Path(h.__file__).parent / "records/fdtd_metasurface_rcwa_reference.jsonl"
    stored = {
        r["key"]: r
        for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())
    }

    def at(trunc):
        out = {}
        for i in range(3):
            row = stored[h.ladder_key("M1", i, trunc)]
            for k, v in row["values"]["p"].items():
                out.setdefault(k, [None] * 3)[i] = complex(*v)
        return out

    levels = [at(t) for t in h.SCENES["M1"]["ladder"][-4:]]
    instruments = [
        r
        for r in stored.values()
        if r.get("kind") == "instrument" and r.get("scene") == "M1"
    ]
    wanted = {f"M1|shift|{n}" for n in (20, 40, 80)}
    selected = [r for r in instruments if r["key"] in wanted]
    if {r["key"] for r in selected} != wanted or not all(
        (r["passed"] for r in selected)
    ):
        raise ValueError(
            "Stored RCWA shift-instrument evidence is incomplete or failed"
        )
    return (
        levels[-1],
        levels[:-1],
        dict(
            path=path.relative_to(Path(h.__file__).parents[3]).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            reference="Stored historical RCWA ladder and shift instruments; no fresh external solve.",
            instruments=selected,
        ),
    )


def judge(runs, mid, long, backends):
    ref, ladder, provenance = references()
    j = h.judge("M1", "p", runs, ref, ladder)
    controls = dict(
        misplaced=h.judge("M1", "p", runs, ref, ladder, misplace=1.0),
        displaced=h.judge("M1", "p", runs, ref, ladder, displace=1.0),
    )
    control_ok = not (
        controls["misplaced"]["second"]
        or controls["misplaced"]["consistent"]
        or controls["displaced"]["consistent"]
    )
    validity = {str(r["spec"]): h.validity("M1", r) for r in runs + [long]}
    mid_valid = h.validity("M1", mid, settle=False)
    gap = h.gap_over_cells("M1", mid, long)
    cap = (h.FLOOR_SHARE * j["A"][-1], h.FLOOR_SHARE * j["F"][-1])
    backend_validity = {
        b: h.validity("M1", r, settle=False) for b, r in backends.items()
    }
    backend_gap = {
        b: h.gap_over_cells("M1", backends["numpy"], backends[b], relative=True)
        for b in ("jax", "torch")
    }
    backend_ok = all(
        (a <= h.BACKEND_GAP and f <= h.BACKEND_GAP for a, f in backend_gap.values())
    ) and (not any(backend_validity.values()))
    layout = next((r for r in h.part_layout() if r["scene"] == "M1"))
    passed = all(
        (
            j["second"],
            j["judgeable"],
            j["consistent"],
            control_ok,
            not any(validity.values()),
            not mid_valid,
            all((a <= b for a, b in zip(gap, cap))),
            backend_ok,
            layout["passed"],
        )
    )
    return dict(
        passed=passed,
        physical=j,
        controls=controls,
        control_ok=control_ok,
        validity=validity,
        mid_validity=mid_valid,
        time_gap=gap,
        time_cap=cap,
        backend_gap=backend_gap,
        backend_validity=backend_validity,
        layout=layout,
        reference=provenance,
        scope="M1 p only. Source qualification uses original C7 formulas plus p-backend agreement; not whole-C7 acceptance.",
    )


def measure(case, records):
    os.environ["JAX_PLATFORMS"] = "cpu"
    h.PROV = h.provenance()
    row = h.fd_row(case)
    path = records / f"m1-n{case['n']}-T{case['periods']}-{case['backend']}.json"
    write(path, row)
    if row["status"] != "ok":
        raise RuntimeError(f"Qualification solve failed; preserved in {path}")
    return row


def collect(records, workers):
    records.mkdir(parents=True, exist_ok=False)
    cases = [
        dict(scene="M1", n=n, pol="p", periods=T, backend=b)
        for n, T, b in (
            (80, 120, "jax"),
            (40, 180, "jax"),
            (20, 120, "jax"),
            (20, 120, "numpy"),
            (20, 120, "torch"),
            (40, 120, "jax"),
        )
    ]
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        max_tasks_per_child=1,
    ) as pool:
        rows = list(pool.map(measure, cases, [records] * len(cases)))
        repeats = [
            dict(r["spec"], periods=180)
            for r in rows
            if r["spec"]["backend"] == "jax"
            and r["spec"]["periods"] == 120
            and (r["spec"]["n"] in (20, 80))
            and (not h.settled(r))
        ]
        list(pool.map(measure, repeats, [records] * len(repeats)))


def grade(records):
    expected = h.provenance()["digest"]

    def read(n, T=120, b="jax"):
        r = json.loads((records / f"m1-n{n}-T{T}-{b}.json").read_text(encoding="utf-8"))
        if r["spec"] != dict(scene="M1", n=n, pol="p", periods=T, backend=b):
            raise ValueError("Qualification record has the wrong case")
        if r["prov"] != expected or r["status"] != "ok":
            raise ValueError(
                "Qualification record has different source/environment provenance or failed"
            )
        return r

    initial = [read(n) for n in (20, 40, 80)]
    rungs = [read(r["spec"]["n"], 180) if not h.settled(r) else r for r in initial]
    backends = {b: read(20, b=b) for b in ("numpy", "jax", "torch")}
    result = judge(rungs, read(40), read(40, 180), backends)
    result["source_provenance"] = expected
    return (result, backends["numpy"])


def export_current(result, row, path):
    if result.get("passed") is not True:
        raise ValueError("M1 p must pass all unchanged physical criteria before export")
    expected_case = dict(scene="M1", n=20, pol="p", periods=120, backend="numpy")
    if row.get("spec") != expected_case or row.get("status") != "ok":
        raise ValueError(
            "Only the verified current NumPy M1 p n20 T120 row can be exported"
        )
    if row.get("prov") != result.get("source_provenance"):
        raise ValueError("Regression row and physical qualification provenance differ")
    archive = Path(h.__file__).parents[3] / "tests/references/fd_metasurface_rows.json"
    old = json.loads(archive.read_text(encoding="utf-8"))
    old = next(
        (r for r in old["runs"] if (r["scene"], r["n"], r["pol"]) == ("M1", 20, "p"))
    )
    values = {k: row["values"][k] for k in old["values"]}
    amp = []
    eff = []
    for k, v in values.items():
        side, m, n, q = k.split("|")
        x = [complex(*a) for a in v]
        reference = [complex(*a) for a in old["reference"][k]]
        if q not in ("R", "T"):
            x = [a * h.phase("M1", 20, int(m), int(n)) for a in x]
        (eff if q in ("R", "T") else amp).append(
            max((abs(a - b) for a, b in zip(x, reference)))
        )
    entry = dict(
        scene="M1",
        n=20,
        pol="p",
        periods=120,
        key=old["key"],
        values=values,
        reference=old["reference"],
        A=max(amp),
        F=max(eff),
    )
    provenance = h.provenance()
    report = {
        k: result[k]
        for k in (
            "passed",
            "physical",
            "controls",
            "control_ok",
            "time_gap",
            "time_cap",
            "backend_gap",
            "source_provenance",
            "scope",
            "validity",
            "mid_validity",
            "backend_validity",
            "layout",
            "record_sha256",
        )
    }
    write(
        path,
        dict(
            schema_version=1,
            source="differentiated-current PlaneWave",
            source_commit=row["commit"],
            versions=provenance["versions"],
            qualifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            criteria_sha256=hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest(),
            historical_archive_sha256=h._sha(archive),
            reference_ladder_sha256=result["reference"]["sha256"],
            qualification=report,
            runs=[entry],
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--regression", type=Path)
    p.add_argument(
        "--run",
        action="store_true",
        help="Measure all required rows in a new directory",
    )
    p.add_argument("--workers", type=int, choices=(1, 2), default=1)
    a = p.parse_args()
    if a.run:
        collect(a.records, a.workers)
    result, row = grade(a.records)
    result["record_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(a.records.glob("m1-*.json"))
    }
    write(a.output, result)
    if a.regression is not None:
        export_current(result, row, a.regression)
    print("M1 p physical qualification:", "PASS" if result["passed"] else "FAIL")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
