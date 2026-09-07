"""Freeze only external Meep CPML/material and FDTDX ADE outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    arrays={};metadata={}
    labels=[f'meep-{m}-{r}-band' for m in ('dielectric','lorentz','drude','mixed') for r in (40,80)]
    for label in labels+['fdtdx-ade']:
        raw=(args.input/(label+'.json')).read_bytes();data=json.loads(raw)
        expected='meep' if label.startswith('meep') else 'fdtdx'
        if data['backend']!=expected:
            raise ValueError('Expected external solver output.')
        metadata[label]=dict(version=data['version'],input_sha256=hashlib.sha256(raw).hexdigest())
        if expected=='meep':
            arrays[label+'_fields']=np.array(data['field_real'])+1j*np.array(data['field_imag'])
            arrays[label+'_power']=np.array(data['transmission'])
            arrays[label+'_airy']=np.array(data['airy_real'])+1j*np.array(data['airy_imag'])
        else:
            metadata[label]['update_sha256']=data['update_sha256']
            for key in ('E','H','value','gradient','finite_difference'):
                arrays['fdtdx_'+key]=np.array(data[key])
    metadata['generators_sha256']={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                   for n in ('validate_grid_dispersive.py','validate_grid_fdtdx_ade.py')}
    arrays['metadata']=np.array(json.dumps(metadata,sort_keys=True))
    with args.output.open('xb') as target:
        np.savez_compressed(target,**arrays)


if __name__=='__main__':
    main()
