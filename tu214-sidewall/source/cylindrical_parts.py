"""Native cylindrical thin-wall B-reps and concentric rotation about X."""
from math import pi,cos,sin
import numpy as np
import cadquery as cq
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge,BRepBuilderAPI_MakeFace,BRepBuilderAPI_Sewing
from OCP.BRepLib import BRepLib
from OCP.Geom import Geom_CylindricalSurface
from OCP.Geom2d import Geom2d_BezierCurve
from OCP.TColgp import TColgp_Array1OfPnt2d
from OCP.TColStd import TColStd_Array1OfReal
from OCP.gp import gp_Ax3,gp_Pnt,gp_Dir,gp_Pnt2d
from OCP.TopoDS import TopoDS
from profile_geometry import Arc,Bezier


def rounded_rectangle(width,height,top_radius,bottom_radius=None,center_s=0.):
    rt=top_radius;rb=bottom_radius if bottom_radius is not None else rt
    x=width/2;bottom=center_s-height/2;top=center_s+height/2
    def line(a,b):return Bezier(np.array([a,b],dtype=float))
    return [line([-x+rb,bottom],[x-rb,bottom]),Arc(np.array([x-rb,bottom+rb]),rb,3*pi/2,2*pi),
        line([x,bottom+rb],[x,top-rt]),Arc(np.array([x-rt,top-rt]),rt,0,pi/2),
        line([x-rt,top],[-x+rt,top]),Arc(np.array([-x+rt,top-rt]),rt,pi/2,pi),
        line([-x,top-rt],[-x,bottom+rb]),Arc(np.array([-x+rb,bottom+rb]),rb,pi,3*pi/2)]


def cylinder(frame,radius):
    return Geom_CylindricalSurface(gp_Ax3(gp_Pnt(0,frame.cy,frame.cz),gp_Dir(1,0,0),gp_Dir(0,1,0)),radius)


def edge_on_cylinder(segment,frame,radius,development_radius):
    weights=None
    if isinstance(segment,Arc):
        mid=(segment.a0+segment.a1)/2;half=(segment.a1-segment.a0)/2;weight=cos(half)
        pts=[segment.center+segment.radius*np.array([cos(segment.a0),sin(segment.a0)]),
            segment.center+segment.radius/weight*np.array([cos(mid),sin(mid)]),
            segment.center+segment.radius*np.array([cos(segment.a1),sin(segment.a1)])]
        weights=[1.,weight,1.]
    else:pts=segment.poles
    poles=TColgp_Array1OfPnt2d(1,len(pts))
    for i,(x,s) in enumerate(pts,1):poles.SetValue(i,gp_Pnt2d(frame.theta+s/development_radius,frame.cx+x))
    if weights:
        w=TColStd_Array1OfReal(1,len(weights))
        for i,value in enumerate(weights,1):w.SetValue(i,value)
        curve=Geom2d_BezierCurve(poles,w)
    else:curve=Geom2d_BezierCurve(poles)
    edge=BRepBuilderAPI_MakeEdge(curve,cylinder(frame,radius)).Edge()
    if not BRepLib.BuildCurves3d_s(edge,1e-7):raise RuntimeError("Cannot construct native cylindrical edge")
    return cq.Edge(edge)


def cylindrical_slab(segments,frame,radius,thickness,development_radius=None):
    reference=development_radius or radius
    boundaries=[[edge_on_cylinder(seg,frame,r,reference) for seg in segments] for r in (radius,radius+thickness)]
    faces=[]
    for r,edges in zip((radius,radius+thickness),boundaries):
        wire=cq.Wire.assembleEdges(edges)
        faces.append(cq.Face(BRepBuilderAPI_MakeFace(cylinder(frame,r),wire.wrapped,True).Face()))
    for a,b in zip(*boundaries):faces.append(cq.Face.makeRuledSurface(a,b))
    sew=BRepBuilderAPI_Sewing(1e-5,True,True,True,False)
    for face in faces:sew.Add(face.wrapped)
    sew.Perform();shell=cq.Shape.cast(sew.SewedShape())
    if sew.NbFreeEdges() or len(shell.Shells())!=1:raise RuntimeError("Cylindrical slab is not watertight")
    solid=cq.Solid.makeSolid(shell.Shells()[0]).fix()
    if solid.Volume()<0:solid=cq.Solid(TopoDS.Solid_s(solid.wrapped.Reversed()))
    if not solid.isValid() or len(solid.Solids())!=1:raise RuntimeError("Invalid cylindrical slab")
    return solid


def move_on_cylinder(shape,frame,travel_mm,radius):
    return shape.rotate((0,frame.cy,frame.cz),(1,frame.cy,frame.cz),travel_mm/radius*180/pi)
