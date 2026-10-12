"""Exact real radial remainders with both polynomial and exponent overlap removed.

For h=1-s and spatial degree k, remove r=max(0,p-k) represented Taylor
terms. Divide each radial remainder by its leading Taylor coefficient,
then take successive divided differences in the fixed real exponent nodes.
Multiplication by s**k makes this a polynomial in u=s**(1/20). Exact
rational division removes (1-u)**(r+len(nodes)-1) before evaluation.
No direction is removed, and no material enters the coordinate change.
"""
from fractions import Fraction
from functools import lru_cache
from math import comb
import numpy as np

@lru_cache(None)
def factored_coefficients(k,r,prefix):
    nodes=tuple(Fraction(round(20*a),20) for a in prefix)
    maximum=max([int(20*a) for a in nodes]+[20*(k+r-1) if r else 0])
    coefficients=[Fraction(0)]*(maximum+1)
    for index,a in enumerate(nodes):
        weight=Fraction(1)
        for j,b in enumerate(nodes):
            if j!=index:weight/=a-b
        normal=Fraction(1)
        for j in range(r):normal*=(k+j-a)/(j+1)
        weight/=normal
        coefficients[int(20*a)]+=weight
        coefficient=Fraction(1)
        for j in range(r):
            for ell in range(j+1):coefficients[20*(k+ell)]-=weight*coefficient*comb(j,ell)*(-1)**ell
            coefficient*=(k+j-a)/(j+1)
    degree=r+len(nodes)-1
    for _ in range(degree):
        quotient=[];running=Fraction(0)
        for coefficient in coefficients[:-1]:
            running+=coefficient;quotient.append(running)
        if running+coefficients[-1]!=0:raise ArithmeticError('nonzero exact normalized factor remainder')
        coefficients=quotient
    return np.asarray(coefficients,float)

def normalized_power_remainder(complement,hat,k,r,nodes):
    s,h=np.asarray(complement),np.asarray(hat)
    u=s**.05
    roots=np.polynomial.polynomial.polyval(u,np.ones(20))
    v=h/roots
    c=factored_coefficients(k,r,tuple(nodes));degree=r+len(nodes)-1
    P=np.polynomial.polynomial.polyval(u,c)
    Pu=np.polynomial.polynomial.polyval(u,np.arange(1,len(c))*c[1:])
    Puu=np.polynomial.polynomial.polyval(u,np.arange(2,len(c))*np.arange(1,len(c)-1)*c[2:])
    A=v**degree*P
    Au=v**degree*Pu
    Auu=v**degree*Puu
    if degree:
        Au-=degree*v**(degree-1)*P
        Auu-=2*degree*v**(degree-1)*Pu
    if degree>1:Auu+=degree*(degree-1)*v**(degree-2)*P
    us=1/(20*u**19);uss=-19/(400*u**39)
    return A,Au*us,Auu*us**2+Au*uss
