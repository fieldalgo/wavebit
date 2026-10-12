"""Re-evaluate saved paired phase controls with the derived Yee radiation map.

No time stepping, fit or rephasing of a measured field occurs here. The input
JSON and its SHA256-pinned NPZ come from fdtd_subpixel_phase_control.py.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
import numpy as np
import wavebit.fd as fd
from wavebit.fd._lower import plan
from wavebit.fd._lower.monitors import box_faces
from wavebit.fd._post import box_flux
from wavebit.fd._runtime import runtime
from wavebit.fd.results import FarFieldResult
from studies.fdtd_subpixel_phase_control import scene
from references.mie import sphere_amplitudes
from fd_helpers import sphere_integral


def evaluate(path):
    path = Path(path)
    header = json.loads(path.read_text(encoding='utf8'))
    raw_path = path.with_suffix('.npz')
    if header['status'] != 'completed_phase_diagnostic_not_C3_pass':
        raise ValueError('LatticeReplayInputRefused: phase control did not finish')
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != header['raw_array_sha256']:
        raise ValueError('LatticeReplayDigestRefused: saved arrays differ from the completed measurement')
    p = plan(scene(header['cpw'], True, header['eps'], header['periods'], header['reflection']))
    Omega = 2*np.sin(np.pi*p.dt)/p.dt
    incident = .5*np.sqrt(1-(Omega*p.dx/2)**2)
    theta = np.linspace(0., 180., 37)
    exact = sphere_amplitudes(np.sqrt(header['eps']), np.pi/2, theta)
    rt, rows = runtime('numpy'), []
    with np.load(raw_path) as raw:
        for old in header['surfaces']:
            name, half = old['name'], old['half']
            faces = box_faces('replay', p.grid, fd.Region(size=(2*half,)*3))
            data = {face: {label: raw[name+'__'+face+'__'+label]-raw['empty__'+name+'__'+face+'__'+label]
                          for label, _, _ in layout.parts()} for face, _, layout in faces}
            net, parts = box_flux(rt, data, faces, p.dx)
            ff = FarFieldResult(name, (1.,), rt, data, faces, p.grid, 1., 1., net, parts, dt=p.dt, propagation='yee')
            forward = complex(ff.fields(0., 0.)[0][0])
            # Pext=.5 Re W, F=i Omega W/(4 pi), Iinc=.5 g for normal
            # incidence. The measured backward illumination remains a
            # separate control, not a fitted correction to this amplitude.
            optical = 4*np.pi*forward.imag/(Omega*2*incident)
            scatter = float(net[0])/incident
            absorption = old['absorption']/incident
            cuts = {}
            for phi in (0., 90.):
                measured = ff.intensity(theta, phi)[0]/incident
                reference = abs(exact[1 if phi == 0 else 0])**2/(2*np.pi)**2
                cuts[str(phi)] = dict(values=measured.tolist(), reference=reference.tolist(),
                    peak_error=float(np.max(abs(measured-reference))/np.max(reference)))
            integral = float(sphere_integral(ff, 24, 32)[0])
            refined = float(sphere_integral(ff, 48, 64)[0])
            rows.append(dict(surface=name, forward=[forward.real, forward.imag],
                forward_reference=[float((1j*exact[1][0]/(2*np.pi)).real), float((1j*exact[1][0]/(2*np.pi)).imag)],
                sigma_scatter=scatter, sigma_absorption=absorption, optical=optical,
                optical_closure=optical/(scatter+absorption)-1,
                integral_closure=integral/float(net[0])-1,
                quadrature_change=refined/integral-1, pattern=cuts,
                historical_continuum_forward=old['forward'],
                historical_two_wave_closure=old['two_wave_closure']))
    return dict(cpw=header['cpw'], eps=header['eps'], periods=header['periods'], reflection=header['reflection'],
        input_json_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        input_array_sha256=header['raw_array_sha256'],
        historical_continuum_optical_closure=header['sphere']['optical_closure'],
        backward_ratio=header['sphere_source']['backward_ratio'], rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('inputs', type=Path, nargs='+')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('DuplicateMeasurementRefused')
    watched = sorted((ROOT/'src/wavebit/fd').rglob('*.py'))+[Path(__file__)]
    def hashes():
        return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    before = hashes()
    rows = [evaluate(p) for p in args.inputs]
    if hashes() != before:
        raise ValueError('MeasurementSourceChangedRefused')
    args.output.write_text(json.dumps(dict(status='completed_lattice_radiation_controls_not_C3_pass',
        source_sha256=before, source_unchanged=True, cases=rows), indent=2)+'\n', encoding='utf8')
    for case in rows:
        print(json.dumps({k: v for k, v in case.items() if k != 'rows'}), flush=True)
        for row in case['rows']:
            print(json.dumps({k: v for k, v in row.items() if k != 'pattern'}), flush=True)


if __name__ == '__main__':
    main()
