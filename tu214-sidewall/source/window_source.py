"""Read existing V5 contours and construct fitted B-rep surfaces; never rebuild the panel."""
from dataclasses import dataclass
import os,math
os.environ.setdefault("XDG_CACHE_HOME","/tmp/tu214-cache")
import numpy as np
from scipy.interpolate import make_interp_spline
import cadquery as cq
from OCP.BRepAdaptor import BRepAdaptor_Curve,BRepAdaptor_Curve2d,BRepAdaptor_Surface
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.Geom import Geom_BSplineSurface
from OCP.TColgp import TColgp_Array2OfPnt
from OCP.TColStd import TColStd_Array1OfReal,TColStd_Array1OfInteger
from OCP.gp import gp_Pnt,gp_Vec,gp_Pnt2d,gp_Vec2d

@dataclass
class Frame:
    cx:float
    cy:float
    cz:float
    radius:float
    theta:float

    @property
    def n(self):return np.array([0.,math.cos(self.theta),math.sin(self.theta)])
    @property
    def up(self):return np.array([0.,-math.sin(self.theta),math.cos(self.theta)])
    @property
    def origin(self):return np.array([self.cx,self.cy+self.radius*self.n[1],self.cz+self.radius*self.n[2]])
    def point(self,x,y,depth):return self.origin+np.array([x,0.,0.])+y*self.up+depth*self.n
    def plane(self,x=0.,y=0.,depth=0.,outward=False):
        return cq.Plane(origin=cq.Vector(*self.point(x,y,depth)),xDir=(1,0,0),normal=cq.Vector(*(self.n if outward else -self.n)))
    def xy(self,p):return np.array([p[0]-self.cx,np.dot(p-self.origin,self.up)])

def array(values,kind=TColStd_Array1OfReal):
    a=kind(1,len(values))
    for i,v in enumerate(values,1):a.SetValue(i,v)
    return a

def frames_from_panel(panel):
    candidates=[]
    for face in panel.Faces():
        if face.geomType()!="CYLINDER" or len(face.Wires())!=2:continue
        cyl=face._geomAdaptor().Cylinder();b=face.BoundingBox()
        if not (1710<cyl.Radius()<1800 and 300<b.xlen<450):continue
        cx=round(face.Center().x/10)*10
        if abs(cx)<150:continue
        axis=cyl.Axis();c=axis.Location()
        inner=next(w for w in face.Wires() if not w.isSame(face.outerWire()))
        pts=np.array([v.toTuple() for e in inner.Edges() for v in e.positions(np.linspace(0,1,61))])
        theta=np.arctan2(pts[:,2]-c.Z(),pts[:,1]-c.Y())
        candidates.append(Frame(cx,c.Y(),c.Z(),cyl.Radius(),(theta.min()+theta.max())/2))
    result=[]
    for cx in sorted(set(f.cx for f in candidates)):
        result.append(min((f for f in candidates if f.cx==cx),key=lambda f:f.radius))
    if len(result)!=2:raise RuntimeError("Cannot identify the two V5 window reference cylinders")
    return result

def oriented_curve(edge,frame,panel):
    c=BRepAdaptor_Curve(edge.wrapped);first,last=c.FirstParameter(),c.LastParameter()
    p=gp_Pnt();v=gp_Vec();c.D1((first+last)/2,p,v)
    xy=frame.xy(np.array([p.X(),p.Y(),p.Z()]))
    t=np.array([v.X(),np.dot([v.X(),v.Y(),v.Z()],frame.up)])
    reverse=xy[0]*t[1]-xy[1]*t[0]<0
    adjacent=[f for f in panel.Faces() if any(e.isSame(edge) for e in f.Edges())]
    walls=[f for f in adjacent if f.geomType()=="BSPLINE" and f._geomAdaptor().VDegree()>1]
    if len(walls)!=1:raise RuntimeError("Cannot identify the source window wall")
    return (edge,reverse,walls[0])

def curve_data(item,u):
    edge,reverse,_=item;c=BRepAdaptor_Curve(edge.wrapped)
    a,b=c.FirstParameter(),c.LastParameter();dt=(a-b) if reverse else (b-a)
    t=b+dt*u if reverse else a+dt*u
    p=gp_Pnt();d=gp_Vec();dd=gp_Vec();c.D2(t,p,d,dd)
    return np.array([p.X(),p.Y(),p.Z()]),np.array([d.X(),d.Y(),d.Z()])*dt,np.array([dd.X(),dd.Y(),dd.Z()])*dt*dt

def extract_throat(panel,frame):
    edges=[];target=frame.radius+18.
    for edge in panel.Edges():
        if abs(edge.Center().x-frame.cx)>180:continue
        pts=np.array([v.toTuple() for v in edge.positions(np.linspace(0,1,25))])
        radii=np.hypot(pts[:,1]-frame.cy,pts[:,2]-frame.cz)
        if radii.min()<target-.1 or radii.max()>target+.3:continue
        p=np.array(edge.positionAt(.5).toTuple());t=np.array(edge.tangentAt(.5).toTuple())
        xy=frame.xy(p);tt=np.array([t[0],np.dot(t,frame.up)])
        # Remove the 3 mm spokes joining the two concentric end contours.
        dot=abs(np.dot(xy,tt)/(np.linalg.norm(xy)*np.linalg.norm(tt)))
        if dot<.75:edges.append(edge)
    def key(v):return tuple(round(x,3) for x in v.toTuple())
    vertices=[{key(e.startPoint()),key(e.endPoint())} for e in edges]
    remaining=set(range(len(edges)));components=[]
    while remaining:
        queue=[remaining.pop()];part=[];nodes=set()
        while queue:
            i=queue.pop();part.append(i);nodes|=vertices[i]
            nearby=[j for j in remaining if nodes&vertices[j]]
            for j in nearby:remaining.remove(j);queue.append(j)
        wire=cq.Wire.assembleEdges([edges[j] for j in part])
        if not wire.IsClosed():raise RuntimeError("An extracted window end contour is open")
        components.append((wire,[edges[j] for j in part]))
    if len(components)!=2:raise RuntimeError("Expected exactly two end contours per window")
    components.sort(key=lambda c:c[0].BoundingBox().xlen)
    inner,outer=components
    def phase(edge):return math.atan2(*frame.xy(np.array(edge.Center().toTuple()))[::-1])
    ordered_inner=sorted(inner[1],key=phase);available=list(outer[1]);ordered_outer=[]
    for edge in ordered_inner:
        selected=min(available,key=lambda e:abs(math.atan2(math.sin(phase(e)-phase(edge)),math.cos(phase(e)-phase(edge)))))
        ordered_outer.append(selected);available.remove(selected)
    if len(ordered_inner)!=len(ordered_outer):raise RuntimeError("Unmatched front contour pieces")
    curves=[[oriented_curve(e,frame,panel) for e in seq] for seq in (ordered_inner,ordered_outer)]
    for loop in curves:
        for a,b in zip(loop,loop[1:]+loop[:1]):
            if np.linalg.norm(curve_data(a,1)[0]-curve_data(b,0)[0])>1e-4:raise RuntimeError("Contour ordering error")
    return curves,(inner[0],outer[0])

def source_normal_data(item,u):
    """Surface-normal derivatives on the actual source edge, via its p-curve."""
    edge,reverse,face=item;c=BRepAdaptor_Curve(edge.wrapped)
    a,b=c.FirstParameter(),c.LastParameter();dt=(a-b) if reverse else (b-a)
    t=b+dt*u if reverse else a+dt*u
    pc=BRepAdaptor_Curve2d(edge.wrapped,face.wrapped)
    uv=gp_Pnt2d();d=gp_Vec2d();dd=gp_Vec2d();pc.D2(t,uv,d,dd)
    ud,vd=d.X()*dt,d.Y()*dt;udd,vdd=dd.X()*dt*dt,dd.Y()*dt*dt
    vectors=[gp_Vec() for _ in range(9)];p=gp_Pnt()
    BRepAdaptor_Surface(face.wrapped).D3(uv.X(),uv.Y(),p,*vectors)
    su,sv,suu,svv,suv,suuu,svvv,suuv,suvv=[np.array([v.X(),v.Y(),v.Z()]) for v in vectors]
    sud=suu*ud+suv*vd;svd=suv*ud+svv*vd
    sudd=suuu*ud*ud+2*suuv*ud*vd+suvv*vd*vd+suu*udd+suv*vdd
    svdd=suuv*ud*ud+2*suvv*ud*vd+svvv*vd*vd+suv*udd+svv*vdd
    return np.cross(su,sv),np.cross(sud,sv)+np.cross(su,svd),np.cross(sudd,sv)+2*np.cross(sud,svd)+np.cross(su,svdd)

def make_surface(values,first,second):
    samples=np.linspace(0,1,len(values));spl=make_interp_spline(samples,values,k=5,bc_type=(
        [(1,first[0]),(2,second[0])],[(1,first[1]),(2,second[1])]))
    knots,mults=np.unique(spl.t,return_counts=True);count,rows=spl.c.shape[:2]
    poles=TColgp_Array2OfPnt(1,count,1,rows)
    for i in range(count):
        for j in range(rows):poles.SetValue(i+1,j+1,gp_Pnt(*spl.c[i,j]))
    surface=Geom_BSplineSurface(poles,array(knots.tolist()),array([0.,1.]),array(mults.tolist(),TColStd_Array1OfInteger),
        array([rows,rows],TColStd_Array1OfInteger),5,rows-1,False,False)
    return cq.Face(BRepBuilderAPI_MakeFace(surface,1e-7).Face())
