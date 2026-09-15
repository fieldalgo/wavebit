"""Exact original-equation controls for balanced Schur response derivatives."""
from fractions import Fraction as F
import numpy as np
import pytest
from wavebit._core.conforming.pair_balanced_response import solve


@pytest.mark.parametrize('gap',('.001','.0000001','.0000000001',
                                '.000000000001','.000000000000001'))
@pytest.mark.parametrize('direction',('1','-.75','0'))
def test_original_state_and_derivative_match_exact_rational_equations(gap,direction):
    s,ds=F(gap),F(direction)
    a,b,c=F(4),F(1),F(2)
    da,db,dc=F('.1'),F('.2'),F('.3')
    f,g=F('.7'),F('.8');df,dg=F('.04'),F('.03')
    determinant=a*c-s*b*b
    u=-(c*f-s*b*g)/determinant
    z=-(-b*f+a*g)/determinant
    first=-df-da*u-(ds*b+s*db)*z
    second=-dg-db*u-dc*z
    du=(c*first-s*b*second)/determinant
    dz=(-b*first+a*second)/determinant
    h=F(3)+f*u+s*g*z
    dh=F('.06')+df*u+f*du+ds*g*z+s*dg*z+s*g*dz
    M=np.array([[float(a),float(s*b)],[float(b),float(c)]])
    DM=np.array([[float(da),float(ds*b+s*db)],[float(db),float(dc)]])
    result=solve(M,DM,np.array([[float(f)],[float(g)]]),
                 np.array([[float(df)],[float(dg)]]),1,float(s),float(ds),
                 (np.array([[3.]]),np.array([[.06]])),np.array([0]))
    for actual,expected in zip(result,([u,z],[du,dz],[h],[dh])):
        np.testing.assert_allclose(actual.ravel(),list(map(float,expected)),atol=3e-15,rtol=0)


@pytest.mark.parametrize('step',(3e-5,1.5e-5))
def test_complex_original_equations_and_fourier_reversal_match_finite_variations(step):
    cc=np.array([[4+.3j,.2-.1j],[.2-.1j,3+.4j]])
    dcc=np.array([[.1+.2j,-.03+.02j],[-.03+.02j,.07-.04j]])
    c=np.array([[.4+.2j,.7-.1j]])
    dc=np.array([[.03-.04j,-.02+.01j]])
    f=np.array([[.7+.2j,.4-.1j,.1+.4j,-.2+.5j],
                [.3+.1j,.2+.6j,.2-.1j,.5+.2j],
                [.4+.3j,.5-.2j,.8+.1j,.3+.7j]])
    df=(np.arange(12).reshape(3,4)-5)*(.013+.017j)
    reverse=np.array([1,0,3,2]);s=.002;ds=-.75
    A=np.eye(4)*3;DA=np.eye(4)*(.06+.02j)
    def original(t):
        gap=s+t*ds;cross=c+t*dc
        matrix=np.block([[cc+t*dcc,gap*cross.T],
                         [cross,np.array([[2+.2j+t*(.3+.07j)]])]])
        force=f+t*df
        u=-np.linalg.solve(matrix,force)
        H=A+t*DA+force[:,reverse].T@(np.r_[np.ones(2),gap][:,None]*u)
        return u,H
    M=np.block([[cc,s*c.T],[c,np.array([[2+.2j]])]])
    DM=np.block([[dcc,ds*c.T+s*dc.T],[dc,np.array([[.3+.07j]])]])
    _,du,_,DH,_=solve(M,DM,f,df,2,s,ds,(A,DA),reverse)
    points=[original(j*step) for j in (-2,-1,1,2)]
    for i,expected in enumerate((du,DH)):
        values=[point[i] for point in points]
        finite=(values[0]-8*values[1]+8*values[2]-values[3])/(12*step)
        np.testing.assert_allclose(expected,finite,atol=3e-10,rtol=0)
