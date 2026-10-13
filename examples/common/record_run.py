"""Save a measured RCWA run and inspect it without re-executing the solver.

Use --output artifacts/runs/film-001 to retain a record. The default checks a
round trip in a temporary directory and leaves no generated files in the repo.
"""
import argparse
from pathlib import Path
import tempfile

import numpy as np

from wavebit import Job, Material, Units, rcwa
from wavebit.records import load_run, run_recorded


def record_and_check(path):
    problem = rcwa.Problem(units=Units(length='um'),
        stack=rcwa.Stack(period=(.8, .8), layers=(rcwa.Layer(thickness=.2, background=Material(2.25)),)),
        excitation=rcwa.Excitation(wavelength=(.6, .8), sources=(rcwa.PlaneWave(polarization='s'),)),
        outputs=rcwa.Outputs(fields=True))
    job = Job(solver='rcwa', problem=problem, numerics=rcwa.Numerics())
    observed = {}
    def sample(result):
        observed['T'] = np.asarray(result.T).copy()
        field = result.field(x=(0., .1), y=(0.,), z=(.1,))
        observed['field'] = np.asarray(field.values).copy()
        return {'inside_film': field}
    saved = run_recorded(job, path, sample=sample)
    restored = load_run(saved.path)
    assert restored.job.key == job.key
    np.testing.assert_array_equal(restored.outputs['transmitted']['efficiency'], observed['T'])
    np.testing.assert_array_equal(restored.samples['inside_film']['values'], observed['field'])
    assert restored.samples['inside_film']['units'] == {'E': 'V/m', 'H': 'A/m', 'coordinates': 'um'}
    assert not restored.outputs['transmitted']['efficiency'].flags.writeable
    print('Transmission:', restored.outputs['transmitted']['efficiency'].ravel())
    print('Field units:', dict(restored.samples['inside_film']['units']))
    print('Installed source identity:', restored.manifest['source']['identity'])
    print('Job, arrays and field units preserved exactly; loading did not repeat the calculation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='new destination directory for a retained record')
    args = parser.parse_args()
    if args.output is None:
        with tempfile.TemporaryDirectory(prefix='wavebit-record-example-') as directory:
            record_and_check(Path(directory)/'run')
        print('Temporary smoke record removed. Pass --output to retain a run.')
    else:
        record_and_check(args.output)
        print('Saved:', args.output.resolve())


if __name__ == '__main__':
    main()
