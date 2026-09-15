"""Exact polynomial identities remove duplicate corner coordinates only.

Multiply every degree-k prefactor by s**(K-k). The resulting homogeneous
polynomials have common degree K and integer Bernstein-hat data. On each
triangle, group identical corner hats even when their global labels differ.
Fraction-free elimination identifies exact relations without numerical ranks,
materials or tolerances. Taylor subtraction changes each relation only by a
piecewise polynomial already represented by the complete nodal Pp space.
"""
from functools import reduce
from math import gcd
import numpy as np


def product(forms):
    out={(0,0,0):1}
    for form in forms:
        new={}
        for powers,value in out.items():
            for axis,coefficient in enumerate(form):
                if coefficient:
                    key=tuple(v+(i==axis) for i,v in enumerate(powers))
                    new[key]=new.get(key,0)+value*int(coefficient)
        out={key:value for key,value in new.items() if value}
    return out


def signature(fields,key,degree):
    corner,neighbors,angular=key
    rows=fields.obj.data['contact_numerators'][fields.obj.data['mapping']]
    out={}
    for index,triangle in enumerate(fields.obj.data['triangles'][:,:3]):
        hats=rows[triangle]
        if not np.any(hats[:,corner]):continue
        forms=[hats[:,n] for n in neighbors]
        if angular:forms += [hats[:,neighbors[0]]-hats[:,neighbors[1]]]*angular
        forms += [np.sum(hats[:,np.arange(10)!=corner],axis=1)]*(degree-len(forms))
        argument=tuple(int(v) for v in hats[:,corner])
        out.update({(index,argument,*monomial):coefficient for monomial,coefficient in product(forms).items()})
    return out


def primitive(vector):
    vector={k:v for k,v in vector.items() if v}
    if not vector:return vector
    factor=reduce(gcd,vector.values())
    if vector[min(vector)]<0:factor=-factor
    return {k:v//factor for k,v in vector.items()}


def remainder(vector,basis):
    vector=primitive(vector)
    while vector:
        pivot=min(vector)
        if pivot not in basis:break
        other=basis[pivot];a,b=vector[pivot],other[pivot]
        vector=primitive({k:b*vector.get(k,0)-a*other.get(k,0) for k in vector.keys()|other.keys()})
    return vector


def selection(fields):
    keys=[(c,(n,),0) for c,n,a in fields.powers]
    keys += [(c,(n,m),0) for c,n,m,a in fields.bubbles]
    keys += [(c,(n,m),j) for c,n,m,a,j in fields.angular]
    keys=list(dict.fromkeys(keys));selected=set();basis={};removed=[]
    for key in keys:
        vector=signature(fields,key,2+fields.angular_order)
        reduced=remainder(vector,basis)
        if reduced:
            basis[min(reduced)]=reduced;selected.add(key)
        else:removed.append(key)
    # Every omitted polynomial is exactly in the retained integer span.
    for key in removed:
        if remainder(signature(fields,key,2+fields.angular_order),basis):
            raise ArithmeticError('discarded corner prefactor is not in the exact retained span')
    return selected,removed


def remove_duplicates(fields):
    """Retain the whole enriched trial space in independent exact coordinates."""
    selected,removed=selection(fields)
    fields.powers=[row for row in fields.powers if (row[0],(row[1],),0) in selected]
    fields.bubbles=[row for row in fields.bubbles if (row[0],row[1:3],0) in selected]
    fields.angular=[row for row in fields.angular if (row[0],row[1:3],row[4]) in selected]
    return tuple(removed)
