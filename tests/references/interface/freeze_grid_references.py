"""Freeze external outputs from validate_grid_meep.py / validate_grid_fdtdx.py.

Input directory must contain meep-grid-{20,40,80}.json, meep-grid-runtime.json
and fdtdx-grid.json. Deliberately never reads Wavebit-generated outputs. Refuses
an existing destination: changing an oracle needs explicit review.
"""
import argparse
import hashlib
from pathlib import Path
import json

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--kind',choices=['base','bloch'],default='base')
    args=parser.parse_args()
    values={}
    metadata={}
    if args.kind=='bloch':
        raw=(args.input/'fdtdx-bloch.json').read_bytes()
        data=json.loads(raw)
        if data['backend']!='fdtdx':
            raise ValueError('Expected external FDTDX data.')
        values={key:np.array(data[key]) for key in
                ('E','H','E_imag','H_imag','value','gradient','finite_difference')}
        generator=Path(__file__).with_name('validate_grid_fdtdx_bloch.py')
        metadata=dict(version=data['version'],update_sha256=data['update_sha256'],
                      input_sha256=hashlib.sha256(raw).hexdigest(),
                      generator_sha256=hashlib.sha256(generator.read_bytes()).hexdigest())
        values['metadata']=np.array(json.dumps(metadata,sort_keys=True))
        with args.output.open('xb') as target:
            np.savez_compressed(target,**values)
        return
    for label in ('meep-grid-20','meep-grid-40','meep-grid-80','meep-grid-runtime','fdtdx-grid'):
        path=args.input/(label+'.json')
        raw=path.read_bytes()
        data=json.loads(raw)
        expected='fdtdx' if label.startswith('fdtdx') else 'meep'
        if data['backend'] != expected:
            raise ValueError(f'{label}: expected an external {expected} result.')
        metadata[label]=dict(version=data['version'],input_sha256=hashlib.sha256(raw).hexdigest())
        if expected=='meep':
            values[label+'_fields']=np.array(data['field_real'])+1j*np.array(data['field_imag'])
            values[label+'_power']=np.array(data['transmission'])
            values['wavelengths']=np.array(data['wavelengths'])
        else:
            for key in ('E','H','value','gradient','finite_difference'):
                values['fdtdx_'+key]=np.array(data[key])
            metadata[label]['update_sha256']=data['update_sha256']
    metadata['generators_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in ('validate_grid_meep.py','validate_grid_fdtdx.py')}
    values['metadata']=np.array(json.dumps(metadata,sort_keys=True))
    with args.output.open('xb') as target:
        np.savez_compressed(target,**values)


if __name__=='__main__':
    main()
