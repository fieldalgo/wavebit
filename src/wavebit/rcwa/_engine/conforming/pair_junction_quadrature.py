"""Moving nodal, corner and continuous-trace fields on clipped corner rays."""
from fractions import Fraction as F
import numpy as np
from wavebit.rcwa._engine.conforming.pair_junction_partition import scalar
from wavebit.rcwa._engine.conforming.pair_junction_partition import point
from wavebit.rcwa._engine.conforming.pair_junction_transport import samples as transport_samples
from wavebit.rcwa._engine.conforming.cartesian_pair_seam_galerkin import _geometry
from wavebit.rcwa._engine.conforming.real_trace_coupling import _contract


def quadrature(obj,base,settings,order,transform,split=True,normalized=False,weighted=False,*,coordinate_records=False):
    if coordinate_records and (not normalized or weighted):
        raise ValueError('coordinate records require unweighted normalized rows')
    info=base[-1];original=info['base_original'];fields=info['corner_fields']
    nold,nc=info['original_nc'],info['nc'];ds=info['ds']
    chart=_geometry(obj,settings)
    length,s,zero,velocity,period,widths,height,center,trace,tg=chart
    delta,delta_slope=trace._taper(tg);ddelta=delta_slope*ds
    common,projection,change=transform
    nx,ny=obj.order;modes=np.array([(m,n) for n in range(-ny,ny+1) for m in range(-nx,nx+1)])
    wave=2*np.pi*modes/period
    h=scalar(height);dd=scalar(delta);ddd=scalar(ddelta);a,b=map(scalar,widths);sf,dsf=scalar(s),scalar(ds)
    for tri,local,p,dp,poly,bb,dbb,measure,linear in transport_samples(obj,base,settings,chart,order,split):
        ids,ci,zi,normal,n1,c,c1,z,z1,d0,d1,thin,ep,dep,*_=local
        determinant=d0+s*d1;ddeterminant=ds*d1
        pf=np.asarray(p,float);dpf=np.asarray(dp,float)
        jac=np.column_stack([pf[1]-pf[0],pf[2]-pf[0]]);djac=np.column_stack([dpf[1]-dpf[0],dpf[2]-dpf[0]])
        if thin:
            vv=velocity[tri[:3]]
            j1=np.column_stack([vv[1]-vv[0],vv[2]-vv[0]])
            inverse=np.array([[j1[1,1],-j1[0,1]],[-j1[1,0],j1[0,0]]])/d1
            dinverse=np.zeros_like(inverse)
        else:
            inverse=np.linalg.inv(jac);dinverse=-inverse@djac@inverse
        average=tuple(sum(point(v[0],p)[i] for v in poly)/len(poly) for i in range(2))
        if -h/2-dd<average[1]<-h/2+dd:side,edge=1.,-h/2
        elif h/2-dd<average[1]<h/2+dd:side,edge=-1.,h/2
        else:side,edge=0.,F(0)
        region=0 if average[0]<-a-sf/2 else 1 if average[0]<-sf/2 else 2 if average[0]<sf/2 else 3 if average[0]<b+sf/2 else 4
        weights,dweights=measure(False)
        weights_bar,dweights_bar=measure(True) if thin or region==2 else (weights,dweights)
        positions=[linear([row[axis] for row in p],[row[axis] for row in dp]) for axis in range(2)]
        points=np.column_stack([row[0] for row in positions])
        speeds=np.column_stack([row[1] for row in positions])
        polynomial,dpolynomial=obj._gradient_basis(bb,dbb)
        def gradient(coeff,dcoeff):
            g=_contract('qv,vai->qai',polynomial,coeff)/determinant
            dg=(_contract('qv,vai->qai',dpolynomial,coeff)+ds*_contract('qv,vai->qai',polynomial,dcoeff))/determinant-g*ddeterminant/determinant
            return g,dg
        closed_dbb=dbb
        if thin:
            hats=fields.hats[tri[:3]]
            pair=next((i,j) for i in range(3) for j in range(i+1,3) if np.array_equal(hats[i],hats[j]))
            other=next(i for i in range(3) if i not in pair)
            closed_dbb=np.zeros_like(dbb)
            closed_dbb[:,other]=dbb[:,other]
            closed_dbb[:,pair[0]]=-dbb[:,other]
            closed_polynomial=obj._gradient_basis(bb,closed_dbb)[1]
            gc=_contract('qv,vai->qai',polynomial,c1)/d1
            dgc=_contract('qv,vai->qai',closed_polynomial,c1)/d1
            sgz=_contract('qv,vai->qai',polynomial,z)/d1
            dsgz=(_contract('qv,vai->qai',dpolynomial,z)+ds*_contract('qv,vai->qai',polynomial,z1))/d1
            gz=sgz/s;dgz=(dsgz-ds*gz)/s
        else:
            gc,dgc=gradient(c,c1);gz,dgz=gradient(z,z1)
            sgz=s*gz;dsgz=ds*gz+s*dgz
        gn,dgn=gradient(normal.swapaxes(-1,-2),n1.swapaxes(-1,-2))
        added,dadded=fields.gradients(tri,bb,inverse,dinverse,closed_dbb)
        ci_all=np.r_[ci,np.arange(nold,nc)]
        gc_all=np.concatenate([gc,added],axis=2);dgc_all=np.concatenate([dgc,dadded],axis=2)
        if side:
            fractions=[F(1,2)+F(int(side))*(xy[1]-edge)/(2*dd) for xy in p]
            directions=[F(int(side))*(dxy[1]-(xy[1]-edge)*ddd/dd)/(2*dd) for xy,dxy in zip(p,dp)]
            fraction,dfraction=linear(fractions,directions)
            complement,_=linear([1-f for f in fractions],[-df for df in directions])
            hat=fraction**2*(1+2*complement);dhat=6*fraction*complement*dfraction
            hy=side*3*fraction*complement/delta
            dhy=side*3*((complement-fraction)*dfraction/delta-fraction*complement*ddelta/delta**2)
        else:
            fraction=None;hat=np.full(len(bb),float(-h/2<average[1]<h/2));dhat=np.zeros(len(bb));hy=dhat;dhy=dhat
        T,dT,Ty,dTy=trace.tapered_basis(points[:,1],speeds[:,1],tg,hat,dhat,hy,dhy,side,fraction,delta)
        if region in (1,3):
            if region==1:
                vals=[-(xy[0]+a+sf/2)/(2*a) for xy in p];dirs=[-(dxy[0]+dsf/2)/(2*a) for dxy in dp]
            else:
                vals=[(b+sf/2-xy[0])/(2*b) for xy in p];dirs=[(dsf/2-dxy[0])/(2*b) for dxy in dp]
            f,df=linear(vals,dirs);fx=-1/(2*widths[region//2])
            barle=np.stack([fx*T,f[:,None]*Ty],axis=1)
            dbarle=np.stack([fx*dT,df[:,None]*Ty+f[:,None]*dTy],axis=1)
        else:
            barle=np.zeros((len(bb),2,change.shape[0]));dbarle=np.zeros_like(barle)
        if region==2:
            f,df=linear([xy[0] for xy in p],[dxy[0] for dxy in dp])
            le=np.stack([T,f[:,None]*Ty],axis=1)
            dle=np.stack([dT,df[:,None]*Ty+f[:,None]*dTy],axis=1)
        else:
            le=s*barle;dle=ds*barle+s*dbarle
        if thin or region==2:
            le=(le-s*gc@common[ci]-sgz@projection[zi])@change
            dle=(dle-(ds*gc+s*dgc)@common[ci]-dsgz@projection[zi])@change
            barle,dbarle=le,dle
        else:
            barle=(barle-gc@common[ci]-gz@projection[zi])@change
            dbarle=(dbarle-dgc@common[ci]-dgz@projection[zi])@change
            le=s*barle;dle=ds*barle+s*dbarle
        phase=np.exp(-1j*(points+center)@wave.T);dphase=-1j*(speeds@wave.T)*phase
        row=(ids,ci_all,zi,weights,dweights,ep,dep,phase,dphase,gc_all,dgc_all,gz,dgz,le,dle,gn,dgn)
        if normalized:
            result=(*row,weights_bar,dweights_bar,sgz,dsgz,barle,dbarle)
            if coordinate_records:
                yield (*result,dict(triangle=tri,barycentric=bb,tangent=dbb,
                                    common_tangent=closed_dbb))
            else:
                yield (*result,thin or region==2) if weighted else result
        else:yield row
