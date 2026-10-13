"""Exact angular expansion of every oscillatory corner coordinate.

Normalize the complementary vertex weights by their complete sum. All
homogeneous angular factors then have a common Bernstein representation.
Exact rational coefficients separate the four real radial kernels and the
represented polynomial overlap before the full stored coordinate transform.
No field is selected by magnitude, material or numerical rank.
"""
from fractions import Fraction as F
from math import factorial
from math import prod
import numpy as np
from wavebit.rcwa._engine.conforming.triangle_polynomials import multi_indices
from wavebit.rcwa._engine.conforming.pair_oscillatory_coefficients import coefficients
from wavebit.rcwa._engine.conforming.pair_oscillatory_coefficients import div
from wavebit.rcwa._engine.conforming.pair_oscillatory_coefficients import mul
from wavebit.rcwa._engine.conforming.pair_corner_precision import D

def add(a,b):
    out=a.copy()
    for powers,value in b.items():out[powers]=out.get(powers,F(0))+value
    return {k:v for k,v in out.items() if v}

def scale(a,b):return {k:b*v for k,v in a.items() if b*v}

def multiply(a,b):
    out={}
    for p,x in a.items():
        for q,y in b.items():
            k=tuple(i+j for i,j in zip(p,q));out[k]=out.get(k,F(0))+x*y
    return {k:v for k,v in out.items() if v}

def power(a,n):
    out={(0,0,0):F(1)}
    for _ in range(n):out=multiply(out,a)
    return out

def linear(values):return {tuple(int(i==j) for i in range(3)):value for j,value in enumerate(values) if value}

ONE=linear([F(1)]*3)

def bernstein(polynomial,degree):
    homogeneous={}
    for powers,value in polynomial.items():
        homogeneous=add(homogeneous,multiply({powers:value},power(ONE,degree-sum(powers))))
    return [homogeneous.get(tuple(a),F(0))*prod(factorial(int(v)) for v in a)/factorial(degree) for a in multi_indices(degree)]

def pair(array):
    high=np.array(array,float)
    low=np.array([float(v-F.from_float(float(h))) for v,h in zip(np.asarray(array,dtype=object).ravel(),high.ravel())]).reshape(high.shape)
    return D(high,low)

def prepare(fields,tri):
    data=fields.obj.data;degree=fields.obj.degree;angular_degree=2+fields.angular_order
    numbers=data['contact_numerators'][data['mapping'][tri[:3]]]
    hats=[[F(int(n),int(data['weight_denominator'])) for n in row] for row in numbers]
    local=fields.interpolant[data['pmap'][data['periodic'][tri]]]
    result=[]
    for corner,ids,transform in zip(fields.controls,fields.groups,fields.transforms):
        if not any(row[corner] for row in hats):continue
        s=[1-row[corner] for row in hats]
        angular=[linear([row[n]/sv if sv else F(0) for row,sv in zip(hats,s)]) for n in range(10)]
        raw=[linear([row[n] for row in hats]) for n in range(10)]
        radial=[[[F(0) for _ in ids] for _ in multi_indices(angular_degree)] for _ in range(4)]
        ordinary=[[F(0) for _ in ids] for _ in multi_indices(degree)]
        for column,index in enumerate(ids):
            _,c,ns,cutoff,j,nodes=fields.newton[index]
            assert c==corner
            p=power(ONE,0);q=power(ONE,0)
            for n in ns:p=multiply(p,angular[n]);q=multiply(q,raw[n])
            if j:
                p=multiply(p,power(add(angular[ns[0]],scale(angular[ns[1]],-1)),j))
                q=multiply(q,power(add(raw[ns[0]],scale(raw[ns[1]],-1)),j))
            k=len(ns)+j;r=max(0,cutoff+1)
            weights,leads,taylor,factor,_=coefficients(k,r,len(nodes))
            vector=bernstein(p,angular_degree);polynomial={}
            for node,(weight,lead,row) in enumerate(zip(weights,leads,taylor)):
                w=div(mul((factor,F(0)),weight),lead)
                for a,coefficient in enumerate(vector):
                    if node<2:radial[node][a][column]+=coefficient*w[0]
                    else:
                        radial[2][a][column]+=coefficient*w[0]
                        radial[3][a][column]+=coefficient*w[1]*(-1 if node==2 else 1)
                for n,t in enumerate(row):
                    v=mul(w,t)[0]
                    polynomial=add(polynomial,scale(multiply(q,power(raw[corner],n+1)),v))
            for a,value in enumerate(bernstein(polynomial,degree)):ordinary[a][column]=value
        keep=np.array([all(s[j] or not alpha[j] for j in range(3)) for alpha in multi_indices(angular_degree)])
        mapping=D(transform)
        radial=[(pair(np.asarray(row,dtype=object)[keep])*fields.column_scale[ids])@mapping for row in radial]
        ordinary=((pair(ordinary)+D(local[:,ids]))*fields.column_scale[ids])@mapping
        result.append((corner,ids,pair(s),keep,radial,ordinary))
    return result
