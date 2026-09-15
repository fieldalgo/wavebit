"""Complete homogeneous radial fields in a real invertible coordinate basis."""
from math import factorial
import numpy as np
from .pair_radial_powers import normalized_power_remainder


def prepare(self):
    rows=[(c,(n,),self.obj.degree-2,0,a) for c,n,a in self.powers]
    rows += [(c,(n,m),self.obj.degree-3,0,a) for c,n,m,a in self.bubbles]
    rows += [(c,(n,m),self.obj.degree-3-j,j,a) for c,n,m,a,j in self.angular]
    groups={};self.newton=[]
    for index,(c,ns,cutoff,j,a) in enumerate(rows):
        nodes=groups.setdefault((c,ns,cutoff,j),[]);nodes.append(a)
        self.newton.append((index,c,ns,cutoff,j,tuple(nodes)))

def sample(self,tri,bary,inverse,dinverse,bary_tangent=None):
    value=np.zeros((len(bary),self.count));gradient=np.zeros((len(bary),2,self.count));change=np.zeros_like(gradient)
    hats=self.hats[tri[:3]];hv=bary@hats
    dhv=np.zeros_like(hv) if bary_tangent is None else bary_tangent@hats
    reference=np.array([[-1.,-1.],[1.,0.],[0.,1.]])
    gh=(reference@inverse).T@hats;dgh=(reference@dinverse).T@hats
    support=tuple(np.flatnonzero(np.any(hats!=0,axis=0)))
    definitions=self._support_cache.get(support)
    if definitions is None:
        active=set(support)
        definitions=[row for row in self.newton
                     if row[1] in active and all(n in active for n in row[2])]
        self._support_cache[support]=definitions
    cache={};products={}
    for index,c,ns,cutoff,j,nodes in definitions:
        s=bary@hats[:,np.arange(10)!=c].sum(axis=1);h=hv[:,c];dh=dhv[:,c];ds=-dh
        if np.any(s<=0):raise ValueError('corner requires a positive complement')
        key=c,ns,j
        if key not in products:
            n=ns[0];a=hv[:,n]/s;da=(dhv[:,n]+a*dh)/s
            p=a;dp=da;gp=np.broadcast_to(gh[:,n],(len(bary),2)).copy();dgp=np.broadcast_to(dgh[:,n],(len(bary),2)).copy()
            if len(ns)==2:
                m=ns[1];b=hv[:,m]/s;db=(dhv[:,m]+b*dh)/s
                p=a*b;dp=da*b+a*db
                gp=b[:,None]*gh[:,n]+a[:,None]*gh[:,m]
                dgp=db[:,None]*gh[:,n]+b[:,None]*dgh[:,n]+da[:,None]*gh[:,m]+a[:,None]*dgh[:,m]
            if j:
                z=a-b;dz=da-db;gz=gh[:,n]-gh[:,m];dgz=dgh[:,n]-dgh[:,m]
                f=z**j;df=j*z**(j-1)*dz;gf=(j*z**(j-1))[:,None]*gz;dgf=(j*z**(j-1))[:,None]*dgz
                if j>1:dgf+=(j*(j-1)*z**(j-2)*dz)[:,None]*gz
                dgp=dgp*f[:,None]+gp*df[:,None]+dp[:,None]*gf+p[:,None]*dgf
                gp=gp*f[:,None]+p[:,None]*gf;dp=dp*f+p*df;p=p*f
            products[key]=p,dp,gp,dgp
        p,dp,gp,dgp=products[key]
        k=len(ns)+j;r=max(0,cutoff+1);key=c,k,r,nodes
        if key not in cache:
            # Normalize each successive leading Taylor coefficient to one.
            # This real scaling retains every radial direction.
            scale=(-1.)**(len(nodes)-1)*factorial(r+len(nodes)-1)/factorial(r)
            cache[key]=tuple(scale*a for a in normalized_power_remainder(s,h,k,r,nodes))
        A,As,Ass=cache[key]
        b=A/s;db=(As/s-A/s**2)*ds
        a=A+h*(k*b-As);da=As*ds+dh*(k*b-As)+h*(k*db-Ass*ds)
        value[:,index]=h*p*A
        gradient[:,:,index]=(p*a)[:,None]*gh[:,c]+(h*b)[:,None]*gp
        change[:,:,index]=((dp*a+p*da)[:,None]*gh[:,c]+(p*a)[:,None]*dgh[:,c]
                           +(dh*b+h*db)[:,None]*gp+(h*b)[:,None]*dgp)
    return value,gradient,change
