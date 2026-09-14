"""Continuous real gap traces in the complete Cartesian-pair Galerkin tensor.

Explicit NumPy finite family for an internal horizontal, equal-height pair in
an axis-aligned periodic cell. Geometry-only potentials join across the gap
and vanish at the outer body faces. Their cubic tangential tapers shrink as
sqrt(gap), admitting the full retained normal trace at exact contact.
"""
from numbers import Integral
import numpy as np
from numpy.polynomial.legendre import leggauss
from .periodic_seam_trace import PeriodicSeamTrace
from .cartesian_pair_galerkin import CartesianPairGalerkinGapCell
from .cartesian_pair_polynomial_galerkin import CartesianPairPolynomialGalerkinGapCell
from .cartesian_pair_opening import real_array
from .projected_pair_positive import projected_positive_response
from .projected_pair_contact import projected_contact_response
from .real_trace_coupling import coupled_positive, coupled_contact, _contract
from .pair_contact_energy import verify_contact_energy


def _geometry(obj, settings):
    """Normalized local chart and geometry-only tangential Fourier basis.

    The seam remains at x=0 even below a coordinate ulp. Translation enters
    the plane-wave phase only. Exact graph identities place material edges.
    """
    period=np.diag(settings['basis'])
    if np.any(np.asarray(settings['basis']) != np.diag(period)) or min(period)<=0:
        raise ValueError('positive diagonal period required')
    length,s,zero,vel=obj._chart_geometry(settings['gap'], {k:np.asarray(settings[k]) for k in ('basis','widths','height','offset')})
    widths=np.asarray(settings['widths'])/length
    height=settings['height']/length
    center=period/(2*length)+np.asarray(settings['offset'])/length
    center[0]+=(widths[0]-widths[1])/2
    # Build local coordinates from exact control identities. This preserves
    # material-edge equality under unit changes and never shifts a tiny gap.
    numerators=obj.data['contact_numerators'][obj.data['mapping']]
    denominator=obj.data['weight_denominator']
    rows=numerators/denominator
    # Combine equal control coordinates as integers before multiplying by
    # physical lengths. Distinct barycentric representations of one horizontal
    # edge must have identical y values; separately rounded products create a
    # fictitious one-ulp sweep strip with no representable interior point.
    group=lambda ids:numerators[:,ids].sum(axis=1)
    outer=group([0,1,2,3])
    x=((group([1,2])-group([0,3]))*(period[0]/length)
       -(outer+2*group([4,9]))*widths[0]
       +(outer+2*group([6,7]))*widths[1])/(2*denominator)
    y=((group([2,3])-group([0,1]))*(period[1]/length)
       +(group([7,8,9])-group([4,5,6]))*height)/(2*denominator)
    zero=np.column_stack([x,y])
    for indices,x in (([4,9],-widths[0]),([5,8],0.),([6,7],widths[1])):
        zero[np.sum(rows[:,indices],axis=1)==1,0]=x
    for indices,y in (([4,5,6],-height/2),([7,8,9],height/2)):
        zero[np.sum(rows[:,indices],axis=1)==1,1]=y
    trace=PeriodicSeamTrace(obj.order[1], quadrature=max(64,2*obj.order[1]+4))
    tg=trace._geometry(settings['gap'],period[::-1],settings['height'],0.)
    return length,s,zero,vel,period/length,widths,height,center,trace,tg

def _quadrature(obj, base, settings, qorder, split=True, transform=None, normalized=False):
    """Differentiate horizontal triangle sections and continuous trace fields.

    Sweep events include taper endpoints, original vertices and vertical
    profile/edge intersections. Thin annulus sections use their exact gap
    factor and barycentric edge interpolation, avoiding subtraction of two
    rounded cell-scale coordinates to recover a subpixel width. The unsplit
    path supplies only the original mesh energy, as in the periodic family.
    """
    length,s,p0,velocity,period,widths,height,center,trace,tg=_geometry(obj,settings)
    ds=base[-1]['ds']
    delta,delta_slope=trace._taper(tg)
    ddelta=delta_slope*ds
    a,b=widths
    t,w=leggauss(qorder);t=(t+1)/2;w=w/2
    xplanes=[(-a-s/2,-ds/2),(-s/2,-ds/2),(s/2,ds/2),(b+s/2,ds/2)]
    ymoving=[(-height/2-delta,-ddelta,0.),(-height/2,0.,.5),(-height/2+delta,ddelta,1.),(height/2-delta,-ddelta,1.),(height/2,0.,.5),(height/2+delta,ddelta,0.)]
    nx,ny=obj.order
    modes=np.array([(m,n) for n in range(-ny,ny+1) for m in range(-nx,nx+1)])
    wave=2*np.pi*modes/period
    for ie,triangle in enumerate(obj.data['triangles']):
        zero=p0[triangle[:3]];v=velocity[triangle[:3]]
        p=zero+s*v;dp=ds*v
        ids,ci,zi,normal,n1,c,c1,z,z1,d0,d1,thin,mat,dmat,*_=base[-1]['local'][ie]
        det=d0+s*d1;ddet=ds*d1
        lo,hi=min(p[:,1]),max(p[:,1])
        cuts=[(float(y),0.) for y in p[:,1]]
        if split:
            cuts += [(y,dy) for y,dy,_ in ymoving if lo<y<hi]
            for j in range(3):
                k=(j+1)%3
                ex=p[k,0]-p[j,0];dex=dp[k,0]-dp[j,0]
                if ex==0: continue
                for x,dx in xplanes:
                    f=(x-p[j,0])/ex
                    if 0<f<1:
                        df=(dx-dp[j,0]-f*dex)/ex
                        y=p[j,1]+f*(p[k,1]-p[j,1])
                        dy=df*(p[k,1]-p[j,1])
                        if lo<y<hi: cuts.append((y,dy))
        cuts=sorted(set(cuts))
        jmat=np.column_stack([p[1]-p[0],p[2]-p[0]])
        djmat=np.column_stack([dp[1]-dp[0],dp[2]-dp[0]])
        inverse=np.array([[jmat[1,1],-jmat[0,1]],[-jmat[1,0],jmat[0,0]]])/det
        for (bottom,dbottom),(top,dtop) in zip(cuts,cuts[1:]):
            if top<=bottom:continue
            mid=(bottom+top)/2
            y=bottom+(top-bottom)*t;dy=dbottom+(dtop-dbottom)*t
            edges=[]
            for j in range(3):
                k=(j+1)%3
                ey=p[k,1]-p[j,1]
                if ey==0 or not min(p[j,1],p[k,1])<mid<max(p[j,1],p[k,1]):continue
                f=(y-p[j,1])/ey;df=dy/ey
                x=p[j,0]+f*(p[k,0]-p[j,0])
                dx=dp[j,0]+df*(p[k,0]-p[j,0])+f*(dp[k,0]-dp[j,0])
                xm=p[j,0]+(mid-p[j,1])/ey*(p[k,0]-p[j,0])
                eb=np.zeros((qorder,3));deb=eb.copy()
                eb[:,j]=1-f;eb[:,k]=f;deb[:,j]=-df;deb[:,k]=df
                vx=v[j,0]+f*(v[k,0]-v[j,0])
                dvx=df*(v[k,0]-v[j,0])
                vm=v[j,0]+(mid-p[j,1])/ey*(v[k,0]-v[j,0])
                leading=zero[j,0]+f*(zero[k,0]-zero[j,0])
                dleading=df*(zero[k,0]-zero[j,0])
                edges.append((vm if thin else xm,x,dx,eb,deb,vx,dvx,leading,dleading))
            if len(edges)!=2:raise ValueError('triangle sweep must have exactly two active edges')
            edges.sort(key=lambda item:item[0])
            if thin:
                leading,dleading=edges[0][7:9]
                lv,rv=edges[0][5],edges[1][5]
                dlv,drv=edges[0][6],edges[1][6]
                span=s*(rv-lv);dspan=ds*(rv-lv)+s*(drv-dlv)
                xl=leading+s*lv;dxl=dleading+ds*lv+s*dlv
                slabs=[(np.zeros(qorder),np.zeros(qorder))]
                if split:
                    for x,dx in xplanes:
                        # Inclusion is decided at the strip midpoint; sweep
                        # events keep the same clipping pattern inside it.
                        lead_mid=np.interp(mid,y,leading)
                        lmid=lead_mid+s*edges[0][0];rmid=lead_mid+s*edges[1][0]
                        if lmid<x<rmid:
                            alpha=((x-leading)-s*lv)/span
                            dalpha=(dx-dleading-ds*lv-s*dlv-alpha*dspan)/span
                            slabs.append((alpha,dalpha))
                slabs += [(np.ones(qorder),np.zeros(qorder))]
            else:
                slabs=[(edges[0][1],edges[0][2])]
                if split:
                    slabs += [(np.full(qorder,x),np.full(qorder,dx)) for x,dx in xplanes if edges[0][0]<x<edges[1][0]]
                slabs += [(edges[1][1],edges[1][2])]
            # The known taper knots retain exact fractions at tiny gaps.
            if -height/2-delta<mid<-height/2+delta:side,edge=1.,-height/2
            elif height/2-delta<mid<height/2+delta:side,edge=-1.,height/2
            else:side,edge=0.,0.
            if side:
                def endpoint(y,dy):
                    for knot,speed,fraction in ymoving:
                        if y==knot:return fraction,0.
                    return .5+side*(y-edge)/(2*delta),side*(dy-(y-edge)*ddelta/delta)/(2*delta)
                f0,df0=endpoint(bottom,dbottom);f1,df1=endpoint(top,dtop)
                fraction=f0+(f1-f0)*t;dfraction=df0+(df1-df0)*t
                hat=fraction**2*(3-2*fraction);dhat=6*fraction*(1-fraction)*dfraction
                hy=side*3*fraction*(1-fraction)/delta
                dhy=side*3*((1-2*fraction)*dfraction/delta-fraction*(1-fraction)*ddelta/delta**2)
            else:
                hat=np.full(qorder,float(-height/2<mid<height/2));dhat=np.zeros(qorder);hy=dhat;dhy=dhat
            B,By,Byy=trace._basis(y,tg)
            T=hat[:,None]*B;dT=dhat[:,None]*B+hat[:,None]*By*dy[:,None]
            Ty=hy[:,None]*B+hat[:,None]*By
            dTy=dhy[:,None]*B+hy[:,None]*By*dy[:,None]+dhat[:,None]*By+hat[:,None]*Byy*dy[:,None]
            for (left,dleft),(right,dright) in zip(slabs,slabs[1:]):
                coordinate=left[:,None]+(right-left)[:,None]*t
                dcoordinate=dleft[:,None]+(dright-dleft)[:,None]*t
                yy=np.broadcast_to(y[:,None],coordinate.shape);dyy=np.broadcast_to(dy[:,None],coordinate.shape)
                if thin:
                    xx=xl[:,None]+span[:,None]*coordinate
                    dxx=dxl[:,None]+dspan[:,None]*coordinate+span[:,None]*dcoordinate
                    width=span*(right-left);dwidth=dspan*(right-left)+span*(dright-dleft)
                    eb0,eb1=edges[0][3],edges[1][3]
                    deb0,deb1=edges[0][4],edges[1][4]
                    bb=(eb0[:,None,:]+coordinate[:,:,None]*(eb1-eb0)[:,None,:]).reshape(-1,3)
                    dbb=(deb0[:,None,:]+dcoordinate[:,:,None]*(eb1-eb0)[:,None,:]+coordinate[:,:,None]*(deb1-deb0)[:,None,:]).reshape(-1,3)
                else:
                    xx,dxx=coordinate,dcoordinate
                    width=right-left;dwidth=dright-dleft
                    point=np.column_stack([xx.ravel(),yy.ravel()]);dpoint=np.column_stack([dxx.ravel(),dyy.ravel()])
                    b12=(point-p[0])@inverse.T
                    db12=(dpoint-dp[0]-b12@djmat.T)@inverse.T
                    bb=np.column_stack([1-b12.sum(axis=1),b12]);dbb=np.column_stack([-db12.sum(axis=1),db12])
                weights=(top-bottom)*width[:,None]*w[:,None]*w
                dw=((dtop-dbottom)*width+(top-bottom)*dwidth)[:,None]*w[:,None]*w
                if normalized:
                    if thin:
                        width_bar=(rv-lv)*(right-left)
                        dwidth_bar=(drv-dlv)*(right-left)+(rv-lv)*(dright-dleft)
                        weights_bar=(top-bottom)*width_bar[:,None]*w[:,None]*w
                        dweights_bar=((dtop-dbottom)*width_bar+(top-bottom)*dwidth_bar)[:,None]*w[:,None]*w
                    else:
                        weights_bar=weights/s
                        dweights_bar=(dw-ds*weights_bar)/s
                point=np.column_stack([xx.ravel(),yy.ravel()]);dpoint=np.column_stack([dxx.ravel(),dyy.ravel()])
                polynomial,dpolynomial=obj._gradient_basis(bb,dbb)
                def grad(n,dn,den,dden):
                    g=_contract('qv,vai->qai',polynomial,n)/den
                    dg=(_contract('qv,vai->qai',dpolynomial,n)+ds*_contract('qv,vai->qai',polynomial,dn))/den-g*dden/den
                    return g,dg
                gc,dgc=grad(c1,np.zeros_like(c1),d1,0.) if thin else grad(c,c1,det,ddet)
                gz,dgz=grad(z,z1,det,ddet)
                sgz,dsgz=grad(z,z1,d1,0.) if thin else (s*gz,ds*gz+s*dgz)
                gn,dgn=grad(normal.swapaxes(-1,-2),n1.swapaxes(-1,-2),det,ddet)
                shape=xx.shape
                def expand(v):return np.broadcast_to(v[:,None,:],(*shape,v.shape[-1])).reshape(-1,v.shape[-1])
                tr,dtr,trace_y,dtrace_y=map(expand,(T,dT,Ty,dTy))
                x,dx=xx.ravel(),dxx.ravel()
                f=np.zeros_like(x);fx=f.copy();df=f.copy();dfx=f.copy()
                region=np.where(x<-a-s/2,0,np.where(x<-s/2,1,np.where(x<s/2,2,np.where(x<b+s/2,3,4))))
                mask=region==1
                f[mask]=-s*(x[mask]+a+s/2)/(2*a);fx[mask]=-s/(2*a)
                df[mask]=-ds*(x[mask]+a+s)/(2*a)-s*dx[mask]/(2*a);dfx[mask]=-ds/(2*a)
                mask=region==2
                f[mask]=x[mask];fx[mask]=1.;df[mask]=dx[mask]
                mask=region==3
                f[mask]=s*(b+s/2-x[mask])/(2*b);fx[mask]=-s/(2*b)
                df[mask]=ds*(b+s-x[mask])/(2*b)-s*dx[mask]/(2*b);dfx[mask]=-ds/(2*b)
                le=np.stack([fx[:,None]*tr,f[:,None]*trace_y],axis=1)
                dle=np.stack([dfx[:,None]*tr+fx[:,None]*dtr,df[:,None]*trace_y+f[:,None]*dtrace_y],axis=1)
                body_factor=not thin and not np.any(region==2)
                if body_factor:
                    # In a regular body slab L=s*F. Differentiate F directly;
                    # subtracting dL/s-L*ds/s**2 would lose the finite result.
                    bf=np.zeros_like(x);bfx=bf.copy();dbf=bf.copy()
                    mask=region==1
                    bf[mask]=-(x[mask]+a+s/2)/(2*a);bfx[mask]=-1/(2*a)
                    dbf[mask]=-(dx[mask]+ds/2)/(2*a)
                    mask=region==3
                    bf[mask]=(b+s/2-x[mask])/(2*b);bfx[mask]=-1/(2*b)
                    dbf[mask]=(ds/2-dx[mask])/(2*b)
                    barle=np.stack([bfx[:,None]*tr,bf[:,None]*trace_y],axis=1)
                    dbarle=np.stack([bfx[:,None]*dtr,dbf[:,None]*trace_y+bf[:,None]*dtrace_y],axis=1)
                if transform is not None:
                    common,projection,change=transform
                    # The physical s*Z derivative is finite on a thin cell.
                    # Evaluate it before dividing by s to avoid cancellation.
                    le=(le-s*gc@common[ci]-sgz@projection[zi])@change
                    dle=(dle-(ds*gc+s*dgc)@common[ci]-dsgz@projection[zi])@change
                    if body_factor:
                        barle=(barle-gc@common[ci]-gz@projection[zi])@change
                        dbarle=(dbarle-dgc@common[ci]-dgz@projection[zi])@change
                if body_factor:
                    le=s*barle
                    dle=ds*barle+s*dbarle
                phase=np.exp(-1j*(point+center)@wave.T);dphase=-1j*(dpoint@wave.T)*phase
                row=(ids,ci,zi,weights.ravel(),dw.ravel(),mat,dmat,phase,dphase,gc,dgc,gz,dgz,le,dle,gn,dgn)
                if normalized:
                    if body_factor:
                        weights_bar,dweights_bar=weights,dw
                    else:
                        barle,dbarle=le,dle
                    yield (*row,weights_bar.ravel(),dweights_bar.ravel(),sgz,dsgz,barle,dbarle)
                else:
                    yield row

def _free_fourier_energy(order,settings):
    """Independent exact pair convolution for the pure Fourier energy check."""
    nx,ny=order;modes=np.array([(m,n) for n in range(-ny,ny+1) for m in range(-nx,nx+1)])
    difference=modes[:,None,:]-modes[None,:,:]
    period=np.diag(settings['basis']);widths=np.asarray(settings['widths']);height=settings['height'];gap=settings['gap']
    center=period/2+np.asarray(settings['offset'])
    shifts=[-(widths[1]+gap)/2,(widths[0]+gap)/2]
    dirs=[-settings['gap_tangent']/2,settings['gap_tangent']/2]
    bg=settings['background'];db=settings['background_tangent']
    e=bg*np.eye(len(modes),dtype=complex);de=db*np.eye(len(modes),dtype=complex)
    for width,ep,dep,shift,dshift in zip(widths,settings['epsilon'],settings['epsilon_tangent'],shifts,dirs):
        indicator=width*height/np.prod(period)*np.sinc(difference[:,:,0]*width/period[0])*np.sinc(difference[:,:,1]*height/period[1])*np.exp(-2j*np.pi*(difference@((center+[shift,0.])/period)))
        change=-2j*np.pi*difference[:,:,0]*dshift/period[0]*indicator
        e+=(ep-bg)*indicator;de+=(dep-db)*indicator+(ep-bg)*change
    return np.kron(np.eye(2),e),np.kron(np.eye(2),de)

def _contact_blocks(obj,settings,quadrature=32):
    """Exact contact coefficients, including the two material-side edge measures.

    The limiting profile F=f/g has a unit jump at the seam. Normal moments
    include that background gap measure. Tangential moments include half
    the inside and outside materials at each shrinking cubic taper edge.
    No small positive gap is sampled to construct these coefficients.
    """
    base=obj._assemble(**settings)
    k,dk,q,dq,r,dr,qt,dqt,source,dsource,e,de,info=base
    length,s,zero,vel,period,widths,height,center,trace,tg=_geometry(obj,settings)
    n,ny=obj.order;N=len(e);NE=2*ny+1;nc,nz=info['nc'],info['nz']
    C=np.zeros((nc,NE),complex);Z=np.zeros((nz,NE),complex)
    t,w=leggauss(quadrature);t=(t+1)/2;w=w/2
    bary=np.stack([np.broadcast_to(1-t[:,None],(quadrature,quadrature)),t[:,None]*(1-t),t[:,None]*t],axis=-1).reshape(-1,3)
    polynomial=obj._gradient_basis(bary,np.zeros_like(bary))[0]
    weight=(t[:,None]*w[:,None]*w).ravel()
    rows=(obj.data['contact_numerators']/obj.data['weight_denominator'])[obj.data['mapping']]
    boundary=[]
    for inds,edge in (([4,5,6],-height/2),([7,8,9],height/2)):
        mask=np.sum(rows[:,inds],axis=1)==1
        zero[mask,1]=edge
        boundary.append(mask)
    a,b=widths
    def F(x):return np.where(x<0,-(x+a)/(2*a),(b-x)/(2*b))
    for ie,tri in enumerate(obj.data['triangles']):
        ids,ci,zi,normal,n1,c,c1,z,z1,d0,d1,thin,mat,*_=info['local'][ie]
        p=zero[tri[:3]]
        point=bary@p
        if thin:
            if np.all(p[:,0]==0) and min(p[:,1])>=-height/2 and max(p[:,1])<=height/2:
                B=trace._basis(point[:,1],tg)[0]
                G=_contract('qv,vai->qai',polynomial,z)
                Z[zi]+=mat*_contract('q,qi,qj->ij',weight,G[:,0],B)
            continue
        if obj.data['tags'][ie]:
            B,By,_=trace._basis(point[:,1],tg)
            fx=-1/(2*widths[obj.data['tags'][ie]-1])
            L=np.stack([fx*B,F(point[:,0])[:,None]*By],axis=1)
            G=_contract('qv,vai->qai',polynomial,c)/d0
            C[ci]+=mat*d0*_contract('q,qai,qaj->ij',weight,G,L)
        for mask,edge,sign in zip(boundary,(-height/2,height/2),(1.,-1.)):
            local=np.flatnonzero(mask[tri[:3]])
            if len(local)!=2:continue
            ia,ib=local
            if p[ia,0]>p[ib,0]:ia,ib=ib,ia
            left,right=p[ia,0],p[ib,0]
            if left<-a-1e-15 or right>b+1e-15:raise ValueError('unexpected material boundary extent')
            if right<=left:continue
            bb=np.zeros((quadrature,3));bb[:,ia]=1-t;bb[:,ib]=t
            x=left+(right-left)*t
            B=trace._basis(np.array([edge]),tg)[0][0]
            boundary_polynomial=obj._gradient_basis(bb,np.zeros_like(bb))[0]
            G=_contract('qv,vai->qai',boundary_polynomial,c)/d0
            C[ci]+=.5*sign*mat*_contract('q,qi,j->ij',w*(right-left)*F(x),G[:,1],B)
    modes=np.array([(m,j) for j in range(-ny,ny+1) for m in range(-n,n+1)])
    wave=2*np.pi*modes/period
    y=height*(t-.5);B,By,_=trace._basis(y,tg)
    phase=np.exp(-1j*y[:,None]*wave[:,1])
    Y=phase.T@((height*w)[:,None]*B)
    Yp=phase.T@((height*w)[:,None]*By)
    edge=trace._basis(np.array([-height/2,height/2]),tg)[0]
    jump=np.exp(1j*height/2*wave[:,1,None])*edge[0]-np.exp(-1j*height/2*wave[:,1,None])*edge[1]
    XF=np.zeros(N,complex);XEF=XF.copy();XEx=XF.copy()
    epsilon=settings['epsilon'];background=settings['background']
    for left,right,ep in ((-a,0.,epsilon[0]),(0.,b,epsilon[1])):
        x=left+(right-left)*t
        phase=np.exp(-1j*x[:,None]*wave[:,0])
        moment=phase.T@(w*(right-left)*F(x))
        XF+=moment;XEF+=ep*moment
        XEx+=-ep/(2*(right-left))*(phase.T@(w*(right-left)))
    translation=np.exp(-1j*wave@center)[:,None]
    Qbar=np.vstack([1j*wave[:,0,None]*XF[:,None]*Y,1j*wave[:,1,None]*XF[:,None]*Y])*np.vstack([translation,translation])
    Rbar=np.vstack([(background+XEx)[:,None]*Y,XEF[:,None]*Yp+.5*(XEF+background*XF)[:,None]*jump])*np.vstack([translation,translation])
    EE=background*B.T@((height*w)[:,None]*B)
    return base,C,Z,Qbar,Rbar,EE


def _leading_trace_products(obj, settings, base, projection, change, quadrature, *, factor=False):
    """Integrate the changed real fields before forming their small energy.

    The contact normal field is the original gap trace minus a real nodal
    difference field. Subtracting two already integrated, nearly equal
    energies would destroy the precision supplied by this basis change.
    """
    nc,nz=base[-1]['nc'],base[-1]['nz']
    count=change.shape[1]
    cross=np.zeros((nz,count),complex)
    energy=np.zeros((count,count),complex)
    upper=np.zeros((0,count))
    _,_,points,_,_,_,height,_,trace,tg=_geometry(obj,settings)
    t,w=leggauss(quadrature);t=(t+1)/2;w=w/2
    bary=np.stack([np.broadcast_to(1-t[:,None],(quadrature,quadrature)),
                   t[:,None]*(1-t),t[:,None]*t],axis=-1).reshape(-1,3)
    weight=(t[:,None]*w[:,None]*w).ravel()
    polynomial=obj._gradient_basis(bary,np.zeros_like(bary))[0]
    for triangle,local in zip(obj.data['triangles'],base[-1]['local']):
        _,_,zi,_,_,_,_,z,_,_,d1,thin,material,*_=local
        if not thin:
            continue
        p=points[triangle[:3]]
        old=np.zeros((len(bary),2,count))
        if np.all(p[:,0]==0) and p[:,1].min()>=-height/2 and p[:,1].max()<=height/2:
            old[:,0]=trace._basis((bary@p)[:,1],tg)[0]
        gradient=_contract('qv,vai->qai',polynomial,z)/d1
        field=(old-gradient@projection[zi])@change
        if factor:
            rows=(np.sqrt(d1*weight)[:,None,None]*field).reshape(-1,count)
            upper=np.linalg.qr(np.vstack([upper,rows]),mode='r')
        cross[zi]+=material*d1*_contract('q,qai,qaj->ij',weight,gradient,field)
        energy+=material*d1*_contract('q,qai,qaj->ij',weight,field,field)
    return (cross,energy,upper) if factor else (cross,energy)


def _trace_change(obj, settings, quadrature):
    """A real, material-independent change of the complete gap field space."""
    geometric=type(obj)(obj.plan,obj.order,degree=obj.degree,coercivity_angle=0.)
    homogeneous=dict(settings,gap=0.,epsilon=(1.,1.),background=1.,gap_tangent=0.,
                     epsilon_tangent=(0.,0.),background_tangent=0.)
    base,common_cross,cross,_,_,energy=_contact_blocks(geometric,homogeneous,quadrature)
    nc=base[-1]['nc']
    projection=np.linalg.solve(base[0][nc:,nc:].real,cross.real)
    # A trace's body continuation can also nearly coincide with a common
    # nodal potential. Remove s*C times its real geometric projection before
    # normalizing the contact trace; this leaves the closed tensor unchanged.
    common=np.linalg.solve(base[0][:nc,:nc].real,
                           common_cross.real-base[0][nc:,:nc].real.T@projection)
    _,gram,factor=_leading_trace_products(geometric,homogeneous,base,projection,np.eye(len(energy)),quadrature,factor=True)
    if (not np.isfinite(gram).all() or np.linalg.norm(gram.imag)>1e-12
            or np.linalg.norm(gram-gram.T)>1e-10*max(1.,np.linalg.norm(gram))):
        raise ValueError('real gap trace metric failed its finite/symmetry guard')
    try:
        # QR operates on the residual fields before squaring their condition
        # in a Gram matrix. Every real trace column is retained.
        change=np.linalg.solve(factor,np.eye(len(factor)))
    except np.linalg.LinAlgError as error:
        raise ValueError('real gap traces lost independence at the available precision') from error
    common.flags.writeable=False;projection.flags.writeable=False;change.flags.writeable=False
    return common,projection,change


def _changed_contact(obj, settings, blocks, transform, quadrature):
    """Integrate every changed contact source in its physical real basis."""
    base=blocks[0]
    common,projection,change=transform
    nc,nz,size=(base[-1][k] for k in ('nc','nz','size'))
    count=change.shape[1]
    c=np.zeros((nc,count),complex)
    q=np.zeros((2*size,count),complex);r=np.zeros_like(q)
    _,_,zero,_,period,widths,height,center,trace,tg=_geometry(obj,settings)
    t,w=leggauss(quadrature);t=(t+1)/2;w=w/2
    bary=np.stack([np.broadcast_to(1-t[:,None],(quadrature,quadrature)),
                   t[:,None]*(1-t),t[:,None]*t],axis=-1).reshape(-1,3)
    weight=(t[:,None]*w[:,None]*w).ravel()
    polynomial=obj._gradient_basis(bary,np.zeros_like(bary))[0]
    nx,ny=obj.order
    modes=np.array([(m,n) for n in range(-ny,ny+1) for m in range(-nx,nx+1)])
    wave=2*np.pi*modes/period
    rows=obj.data['contact_numerators'][obj.data['mapping']]
    denominator=obj.data['weight_denominator']
    boundary=[rows[:,inds].sum(axis=1)==denominator for inds in ([4,5,6],[7,8,9])]
    a,b=widths
    def profile(x):
        return np.where(x<0,-(x+a)/(2*a),(b-x)/(2*b))
    for triangle,tag,local in zip(obj.data['triangles'],obj.data['tags'],base[-1]['local']):
        _,ci,zi,_,_,cc,c1,zz,_,d0,d1,thin,material,*_=local
        p=zero[triangle[:3]];point=bary@p
        denominator=d1 if thin else d0
        gc=_contract('qv,vai->qai',polynomial,c1 if thin else cc)/denominator
        gz=_contract('qv,vai->qai',polynomial,zz)/denominator
        old=np.zeros((len(bary),2,count))
        if thin:
            if np.all(p[:,0]==0) and p[:,1].min()>=-height/2 and p[:,1].max()<=height/2:
                old[:,0]=trace._basis(point[:,1],tg)[0]
        elif tag:
            functions,derivatives,_=trace._basis(point[:,1],tg)
            old[:,0]=-functions/(2*widths[tag-1])
            old[:,1]=profile(point[:,0])[:,None]*derivatives
        field=old-gz@projection[zi]
        if not thin:
            field-=gc@common[ci]
        field=field@change
        phase=np.exp(-1j*(point+center)@wave.T)
        moment=denominator*_contract('q,qm,qai->ami',weight,phase,field).reshape(2*size,count)
        q+=moment;r+=material*moment
        c[ci]+=material*denominator*_contract('q,qai,qaj->ij',weight,gc,field)
        if thin:
            continue
        # The shrinking tangential taper contributes half of each adjacent
        # material side. The nodal subtraction has no such delta measure.
        for mask,edge,sign in zip(boundary,(-height/2,height/2),(1.,-1.)):
            local_ids=np.flatnonzero(mask[triangle[:3]])
            if len(local_ids)!=2:
                continue
            ia,ib=local_ids
            if p[ia,0]>p[ib,0]:ia,ib=ib,ia
            left,right=p[ia,0],p[ib,0]
            if right<=left:
                continue
            bb=np.zeros((quadrature,3));bb[:,ia]=1-t;bb[:,ib]=t
            x=left+(right-left)*t
            functions=trace._basis(np.array([edge]),tg)[0][0]@change
            edge_weight=.5*sign*w*(right-left)*profile(x)
            phase=np.exp(-1j*(np.column_stack([x,np.full_like(x,edge)])+center)@wave.T)
            moment=phase.T@(edge_weight[:,None]*functions)
            q[size:]+=moment;r[size:]+=material*moment
            poly=obj._gradient_basis(bb,np.zeros_like(bb))[0]
            gradient=_contract('qv,vai->qai',poly,cc)/d0
            c[ci]+=material*_contract('q,qi,j->ij',edge_weight,gradient[:,1],functions)
    z,energy=_leading_trace_products(obj,settings,base,projection,change,quadrature)
    return base,c,z,q,r,energy


class CartesianPairSeamGalerkinGapCell:
    """Polynomial mesh and continuous real gap potentials in one weak solve.

    The explicit opening plan stays fixed. A positive diagonal basis, widths,
    height and translation stay fixed in a JVP; gap and both complex materials
    and background may vary together. Gap zero has right-derivative meaning.
    This finite NumPy family does not change ordinary NVF dispatch or certify
    optical convergence. No third-party solver or material-dependent trial
    frame is used.
    """

    def __init__(self, plan, order=1, *, degree=2, quadrature=32, coercivity_angle=0.):
        if (isinstance(degree,(bool,np.bool_)) or not isinstance(degree,Integral) or degree not in (2,3,4)):
            raise ValueError('coupled pair polynomial degree must be 2, 3 or 4')
        family=CartesianPairGalerkinGapCell if degree==2 else CartesianPairPolynomialGalerkinGapCell
        self._base=family(plan,order,degree=degree,coercivity_angle=coercivity_angle)
        self.order = self._base.order
        if (isinstance(quadrature, (bool, np.bool_)) or not isinstance(quadrature, Integral)
                or quadrature < max(16, 2*max(self.order)+4)):
            raise ValueError('quadrature must be an integer at least max(16, 2*max(order)+4)')
        self.quadrature = int(quadrature)
        self._trace_geometry=None
        self._trace_transform=None

    def closure(self, *, gap, basis, widths, height, epsilon, background=1.,
                offset=(0., 0.), gap_tangent=1., epsilon_tangent=(0., 0.),
                background_tangent=0.):
        gap = real_array(gap, (), 'gap').item()
        settings = dict(gap=gap, basis=basis, widths=widths, height=height,
                        epsilon=epsilon, background=background, offset=offset,
                        gap_tangent=gap_tangent, epsilon_tangent=epsilon_tangent,
                        background_tangent=background_tangent)
        working_quadrature=max(self.quadrature,4*max(self.order)+16) if self._base.degree>2 else self.quadrature
        transform=None
        if self._base.degree>2:
            key=tuple(tuple(np.asarray(settings[k]).ravel()) for k in ('basis','widths','height','offset'))+(working_quadrature,)
            if key!=self._trace_geometry:
                self._trace_transform=_trace_change(self._base,settings,working_quadrature)
                self._trace_geometry=key
            transform=self._trace_transform
        if gap == 0:
            blocks = _contact_blocks(self._base, settings, working_quadrature)
            verifier=None
            response_solver=None
            if transform is not None:
                blocks=_changed_contact(self._base,settings,blocks,transform,working_quadrature)
                chart=_geometry(self._base,settings)
                free=_free_fourier_energy(self.order,settings)
                physical_quadrature=max(32,working_quadrature)
                response_solver=lambda:projected_contact_response(self._base,blocks,chart,
                    transform,free,physical_quadrature)
                verifier=lambda x,dx,u:verify_contact_energy(self._base,settings,blocks[0],chart,
                    transform,x,dx,u,physical_quadrature)
            value, tangent, metadata = coupled_contact(self._base, *blocks, energy_verifier=verifier,
                                                        response_solver=response_solver)
            if transform is not None:
                metadata['physical_quadrature']=physical_quadrature
        else:
            base = self._base._assemble(**settings)
            quadrature=lambda split=True: _quadrature(self._base, base, settings, working_quadrature, split, transform)
            free=_free_fourier_energy(self.order, settings)
            normalized=None if transform is None else lambda: _quadrature(
                self._base,base,settings,working_quadrature,True,transform,True)
            response_solver=None if transform is None else lambda q,dq,blocks:projected_positive_response(
                base[-1],normalized,free,q,dq,blocks)
            value, tangent, metadata = coupled_positive(self._base, base, quadrature,
                free, self.order[1], response_solver=response_solver, normalized_quadrature=normalized)
        residuals = [v for k, v in metadata.items() if k.endswith('residual') and v is not None]
        if (not np.isfinite(np.asarray(value)).all() or not np.isfinite(np.asarray(tangent)).all()
                or not all(np.isfinite(v) for v in residuals) or max(residuals) > 1e-10
                or max(metadata['reciprocity'], metadata['derivative_reciprocity']) > 1e-9
                or metadata['minimum_rotated_storage'] <= 0 or metadata['minimum_dissipation'] < -1e-9):
            raise ValueError('coupled pair Galerkin response failed original/energy/physical guards')
        metadata.update(closure='real_coupled_pair_seam_galerkin', order=self.order,
                        refinement=self._base.plan.refinement, shift=self._base.plan.shift,
                        quadrature=self.quadrature, integration_quadrature=working_quadrature, degree=self._base.degree, gap=gap,
                        trace_basis='nodal_orthogonal' if transform is not None else 'restricted_fourier')
        return value, tangent, metadata
