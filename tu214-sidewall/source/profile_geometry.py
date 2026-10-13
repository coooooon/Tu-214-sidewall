"""Eight tangent circles with controlled G2 transitions, not an ellipse.

Top and bottom crowns/corners are independent. The curve is evaluated by
polar angle so a source contour with different segmentation can be lofted
without twisting correspondence.
"""
from dataclasses import dataclass
from math import sin,cos,atan2,sqrt,pi,comb
import numpy as np
from scipy.optimize import brentq


@dataclass
class Arc:
    center:np.ndarray
    radius:float
    a0:float
    a1:float
    def data(self,t):
        a=self.a0+(self.a1-self.a0)*t;da=self.a1-self.a0
        n=np.array([cos(a),sin(a)]);tangent=np.array([-sin(a),cos(a)])
        return self.center+self.radius*n,self.radius*da*tangent,-self.radius*da*da*n


@dataclass
class Bezier:
    poles:np.ndarray
    def data(self,t):
        def evaluate(p):
            k=len(p)-1
            return sum(comb(k,i)*(1-t)**(k-i)*t**i*p[i] for i in range(k+1))
        n=len(self.poles)-1
        return evaluate(self.poles),evaluate(n*np.diff(self.poles,axis=0)),evaluate(n*(n-1)*np.diff(self.poles,n=2,axis=0))


def corner(width,height,side,crown,radius):
    a,b=width/2,height/2;top=np.array([0.,b-crown]);right=np.array([a-side,0.])
    rt,rs=crown-radius,side-radius;delta=right-top;distance=np.linalg.norm(delta);direction=delta/distance
    k=(rt*rt-rs*rs+distance*distance)/(2*distance);perp=np.array([-direction[1],direction[0]])
    q2=rt*rt-k*k
    if q2<=0:raise ValueError("Incompatible crown/side/corner radii")
    q=next(q for q in [top+k*direction+s*sqrt(q2)*perp for s in (-1,1)] if 0<q[0]<a and 0<q[1]<b)
    bt=atan2(*(q-top)[::-1]);bs=atan2(*(q-right)[::-1])
    return top,right,q,bt,bs


def contour_arcs(width,height,side,top_radius,bottom_radius,upper_corner,lower_corner):
    top,right,qt,bt,st=corner(width,height,side,top_radius,upper_corner)
    reflected_bottom,_,qb,bb,sb=corner(width,height,side,bottom_radius,lower_corner)
    return [Arc(-reflected_bottom,bottom_radius,pi+bb,2*pi-bb),
        Arc(qb*np.array([1,-1]),lower_corner,2*pi-bb,2*pi-sb),
        Arc(right,side,2*pi-sb,2*pi+st),
        Arc(qt,upper_corner,2*pi+st,2*pi+bt),
        Arc(top,top_radius,2*pi+bt,3*pi-bt),
        Arc(qt*np.array([-1,1]),upper_corner,3*pi-bt,3*pi-st),
        Arc(-right,side,3*pi-st,3*pi+sb),
        Arc(-qb,lower_corner,3*pi+sb,3*pi+bb)]


class RoundedProfile:
    def __init__(self,arcs,blend_half_length=4.,offset=0.,speed_factor=.9):
        arcs=[Arc(a.center.copy(),a.radius+offset,a.a0,a.a1) for a in arcs]
        self.segments=[]
        for i,original in enumerate(arcs):
            trim=blend_half_length;arc=Arc(original.center,original.radius,original.a0+trim/original.radius,original.a1-trim/original.radius)
            self.segments.append(arc)
            following=arcs[(i+1)%len(arcs)]
            angle0=arc.a1;angle1=following.a0+trim/following.radius
            n0=np.array([cos(angle0),sin(angle0)]);n1=np.array([cos(angle1),sin(angle1)])
            t0=np.array([-n0[1],n0[0]]);t1=np.array([-n1[1],n1[0]])
            p0=arc.center+arc.radius*n0;p1=following.center+following.radius*n1;length=2*trim*speed_factor
            d0=length*t0;d1=length*t1;dd0=-length*length/arc.radius*n0;dd1=-length*length/following.radius*n1
            b0=p0;b1=p0+d0/5;b2=dd0/20+2*b1-b0;b5=p1;b4=p1-d1/5;b3=dd1/20+2*b4-b5
            self.segments.append(Bezier(np.array([b0,b1,b2,b3,b4,b5])))
        self.start=atan2(*self.segments[0].data(0)[0][::-1]);self.phases=[]
        previous=self.start
        for segment in self.segments:
            end=atan2(*segment.data(1)[0][::-1]);end=previous+(end-previous)%(2*pi)
            self.phases.append((previous,end));previous=end
        if abs(previous-self.start-2*pi)>1e-9:raise ValueError("Profile is not a complete convex loop")

    def data(self,phi):
        phi=self.start+(phi-self.start)%(2*pi)
        index=next(i for i,(_,end) in enumerate(self.phases) if phi<=end+1e-12)
        segment=self.segments[index];direction=np.array([cos(phi),sin(phi)])
        if isinstance(segment,Arc):
            dot=np.dot(segment.center,direction)
            radial=dot+sqrt(dot*dot+segment.radius**2-np.dot(segment.center,segment.center))
            q=direction*radial;angle=atan2(*(q-segment.center)[::-1])
            angle=segment.a0+(angle-segment.a0)%(2*pi)
            t=(angle-segment.a0)/(segment.a1-segment.a0)
        else:
            def cross(t):
                q=segment.data(t)[0];return q[0]*direction[1]-q[1]*direction[0]
            if abs(cross(0))<1e-11:t=0.
            elif abs(cross(1))<1e-11:t=1.
            else:t=brentq(cross,0,1,xtol=1e-13)
        q,d,dd=segment.data(t);r2=np.dot(q,q)
        cross=q[0]*d[1]-q[1]*d[0];phid=cross/r2
        phidd=(q[0]*dd[1]-q[1]*dd[0])/r2-2*np.dot(q,d)*cross/r2**2
        return q,d/phid,(dd-d*phidd/phid)/phid**2

    def points(self,count=401):return np.array([self.data(p)[0] for p in np.linspace(-pi,pi,count)])


if __name__=="__main__":
    arcs=contour_arcs(266,368,1800,720,540,100,108)
    shape=RoundedProfile(arcs)
    curvature=[]
    for phi in np.linspace(-pi,pi,2001):
        q,d,dd=shape.data(phi);curvature.append((d[0]*dd[1]-d[1]*dd[0])/np.linalg.norm(d)**3)
    print("B contour extents:",np.ptp(shape.points(),axis=0),"minimum curvature",min(curvature))
