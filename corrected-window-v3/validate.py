"""Check delivered BReps, controlled sections and continuous motion envelopes."""
from pathlib import Path
import json,math,hashlib
from dataclasses import replace
import numpy as np
import cadquery as cq
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.gp import gp_Pnt,gp_Vec
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Common
from build import P,ROOT,CAD,SOURCE,frames_from_panel,extract_throat,curve_data,rows_data,exact_bounds,mesh_bounds,pure_solid
from cylindrical_parts import cylindrical_slab as make_slab,move_on_cylinder
from OCP.gp import gp_Pnt
from profile_geometry import Bezier
from preserve_step_panel import records,references,TOKEN

def cylindrical_slab(*args):
    return pure_solid(make_slab(*args))

def load(name):
    shape=cq.importers.importStep(str(CAD/f"{name}.step")).val()
    return pure_solid(shape) if len(shape.Solids())==1 else shape

def digest_geometry(items,root):
    cache={}
    def one(i):
        if i not in cache:
            text=TOKEN.sub(lambda m:"@"+one(int(m.group(1))) if m.group(1) else m.group(0),items[i])
            cache[i]=hashlib.sha256(text.encode()).hexdigest()
        return cache[i]
    return one(root)

def panel_representation(items):
    product=next(i for i,r in items.items() if r.startswith("PRODUCT('Wall_Panel'"))
    formation=next(i for i,r in items.items() if r.startswith("PRODUCT_DEFINITION_FORMATION(") and product in references(r))
    definition=next(i for i,r in items.items() if r.startswith("PRODUCT_DEFINITION(") and formation in references(r))
    pshape=next(i for i,r in items.items() if r.startswith("PRODUCT_DEFINITION_SHAPE(") and definition in references(r))
    sd=next(r for r in items.values() if r.startswith("SHAPE_DEFINITION_REPRESENTATION(") and pshape in references(r))
    return next(i for i in references(sd) if items[i].startswith("ADVANCED_BREP_SHAPE_REPRESENTATION("))

def annular_sector(frame,width,low,high,inner,outer,reference):
    # Independent exact 3D arcs and a straight X extrusion. This avoids
    # periodic surface UV trimming in the verification volume.
    a0,a1=frame.theta+low/reference,frame.theta+high/reference
    am=(a0+a1)/2
    def point(radius,angle):
        return cq.Vector(frame.cx-width/2,frame.cy+radius*math.cos(angle),
                         frame.cz+radius*math.sin(angle))
    p1,p2,p3,p4=point(inner,a0),point(outer,a0),point(outer,a1),point(inner,a1)
    edges=[cq.Edge.makeLine(p1,p2),
           cq.Edge.makeThreePointArc(p2,point(outer,am),p3),
           cq.Edge.makeLine(p3,p4),
           cq.Edge.makeThreePointArc(p4,point(inner,am),p1)]
    wire=cq.Wire.assembleEdges(edges)
    return pure_solid(cq.Solid.extrudeLinear(wire,[],cq.Vector(width,0,0)))

def handle_sweep(frame,center,travel):
    # The lens bounds the whole handle: |z| <= H/2*(1-(2*x/W)^2).
    # Ten X intervals give certified angular bounds without the excess
    # upper corners of one large rectangle.
    r=P["shade_radius_mm"];width=P["handle_width_mm"]
    depth=P["handle_depth_mm"];height=P["handle_height_mm"]
    bounds=np.linspace(-width/2-.1,width/2+.1,11)
    sectors=[]
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        lo-=.03;hi+=.03
        near=0 if lo<=0<=hi else min(abs(lo),abs(hi))
        h=height/2*max(0,1-(2*near/width)**2)
        arc=r*math.atan2(h,r-depth-.2)+.15
        shifted=replace(frame,cx=frame.cx+(lo+hi)/2)
        sectors.append(annular_sector(shifted,hi-lo,center-arc,center+arc+travel,r-depth-.2,r+.2,r))
    return pure_solid(sectors[0].fuse(*sectors[1:]).fix())

def rectangle(width,low,high):
    x=width/2; pts=np.array([[-x,low],[x,low],[x,high],[-x,high]])
    return [Bezier(np.array([pts[i],pts[(i+1)%4]])) for i in range(4)]

def vol_intersection(a,b):
    op=BRepAlgoAPI_Common(a.wrapped,b.wrapped)
    op.SetNonDestructive(True)
    op.SetFuzzyValue(.00005)
    op.Build()
    print("Common operation",op.IsDone(),flush=True)
    assert op.IsDone(),"Failed Boolean common operation"
    intersection=cq.Shape.cast(op.Shape())
    assert intersection.isValid()
    return sum(max(0,float(s.Volume())) for s in intersection.Solids())

def surface(face,u,v):
    p=gp_Pnt();du=gp_Vec();dv=gp_Vec()
    BRepAdaptor_Surface(face.wrapped).D1(u,v,p,du,dv)
    point=np.array([p.X(),p.Y(),p.Z()])
    a=np.array([du.X(),du.Y(),du.Z()]);b=np.array([dv.X(),dv.Y(),dv.Z()])
    n=np.cross(a,b);n/=np.linalg.norm(n)
    return point,n

def cup_check(panel,frame,i):
    cup=load(f"cup_{i}");curves,_=extract_throat(panel,frame)
    walls=[f for f in cup.Faces() if f.geomType()=="BSPLINE" and f._geomAdaptor().VDegree()==2]
    remaining=walls[:];loops=[];front_errors=[];thickness=[];area=[]
    for outer in (False,True):
        matched=[]
        for item in curves[0]:
            expected=rows_data(item,.5,frame,outer)[0][0]
            f=min(remaining,key=lambda f:np.linalg.norm(surface(f,.5,0)[0]-expected))
            remaining.remove(f);matched.append(f)
            for u in np.linspace(0,1,13):
                front_errors.append(np.linalg.norm(surface(f,u,0)[0]-rows_data(item,u,frame,outer)[0][0]))
                for v in np.linspace(0,1,9):
                    # Quadratic depth rows guarantee one C bend without an S bend.
                    r=rows_data(item,u,frame,outer)[0]
                    d2=2*(r[2]-2*r[1]+r[0])
                    firsts=[2*((1-t)*(r[1]-r[0])+t*(r[2]-r[1])) for t in (0,.5,1)]
                    assert all(np.dot(d,frame.n)>0 for d in firsts)
                    assert all(np.dot(np.cross(firsts[0],d2),np.cross(d,d2))>=-0.000001 for d in firsts)
                    area.append(float(np.linalg.norm(np.cross(firsts[0],d2))))
            for u in np.linspace(.02,.98,17):
                r=rows_data(item,u,frame,False)[0]
                a,b=r[0],r[2]
                assert abs(b[0]-frame.cx)<abs(a[0]-frame.cx)+.01
                assert abs(np.dot(b-frame.origin,frame.up))<abs(np.dot(a-frame.origin,frame.up))+.01
        loops.append(matched)
    assert not remaining
    for a,b in zip(*loops):
        for u in np.linspace(.001,.999,21):
            for v in np.linspace(0,1,9):
                q,n=surface(a,u,v);other=surface(b,u,v)[0]
                thickness.append(abs(float(np.dot(other-q,n))))
    assert max(front_errors)<.002,(i,"front mismatch",max(front_errors))
    assert min(thickness)>1.5,(i,"thin wall",min(thickness))
    return {"window":i,"front_profile_error_mm":max(front_errors),
            "normal_wall_thickness_mm":[min(thickness),max(thickness)],
            "quadratic_depth_generators_no_inflection":True,
            "shrinks_in_width_and_height":True,"wall_faces":len(walls)}

def main():
    (ROOT/"validation.json").unlink(missing_ok=True)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==P["source_sha256"]
    panel=cq.importers.importStep(str(SOURCE)).val()
    frames=frames_from_panel(panel)
    original=records(SOURCE.read_text())
    source_rep=next(i for i,r in original.items() if r.lstrip().startswith("ADVANCED_BREP_SHAPE_REPRESENTATION("))
    reference=digest_geometry(original,source_rep)
    report={"source_sha256":P["source_sha256"],"panel_geometry_hash":reference,
            "assemblies":[],"cups":[],"motion":[]}
    for state in ("open","half","closed"):
        path=CAD/f"assembly_{state}.step"
        shape=load(f"assembly_{state}")
        assert shape.isValid() and len(shape.Solids())==7
        items=records(path.read_text())
        assert digest_geometry(items,panel_representation(items))==reference
        report["assemblies"].append({"file":path.name,"valid_solids":7,"panel_geometry_identical":True})
    canonical=load("handle")
    bb=exact_bounds(canonical)
    dims=[bb.xlen,bb.ylen,bb.zlen]
    assert np.allclose(dims,[100,40,25],atol=.015),dims
    report["handle_dimensions_mm"]=dims
    report["handle_volume_mm3"]=canonical.Volume()
    # Surface symmetries required by the three projections.
    for plane in ("YZ","XY"):
        difference=canonical.cut(canonical.mirror(plane)).Volume()
        assert difference<2.0,(plane,difference)
    report["handle_x_and_z_symmetry"]=True
    for i,frame in enumerate(frames,1):
        print("Validate cup",i,flush=True)
        report["cups"].append(cup_check(panel,frame,i))
        cup=load(f"cup_{i}");shade=load(f"shade_{i}_closed");handle=load(f"handle_{i}_closed")
        cylinder_faces=[f for f in shade.Faces() if f.geomType()=="CYLINDER" and f.Area()>10000]
        radii=sorted(f._geomAdaptor().Cylinder().Radius() for f in cylinder_faces)
        assert np.allclose(radii,[P["shade_radius_mm"],P["shade_radius_mm"]+P["shade_thickness_mm"]])
        for f in cylinder_faces:
            a=f._geomAdaptor().Cylinder().Axis()
            assert abs(a.Location().Y()-frame.cy)<1e-8 and abs(a.Location().Z()-frame.cz)<1e-8
        r=P["shade_radius_mm"];travel=P["shade_open_travel_mm"]
        shade_envelope=annular_sector(frame,P["shade_width_mm"],P["shade_closed_lower_edge_mm"],
            P["shade_closed_lower_edge_mm"]+P["shade_height_mm"]+travel,
            r,r+P["shade_thickness_mm"],r)
        center=P["shade_closed_lower_edge_mm"]+P["handle_center_from_shade_lower_edge_mm"]
        handle_closed_envelope=handle_sweep(frame,center,0)
        op=BRepAlgoAPI_Cut(handle.wrapped,handle_closed_envelope.wrapped)
        op.SetNonDestructive(True)
        op.SetFuzzyValue(.00005)
        op.Build()
        print("Containment cut operation",op.IsDone(),flush=True)
        assert op.IsDone(),"Failed Boolean containment operation"
        difference=cq.Shape.cast(op.Shape())
        outside=sum(max(0,float(s.Volume())) for s in difference.Solids())
        a=frame.theta+center/r
        witness=cq.Vector(frame.cx+7,frame.cy+(r-10)*math.cos(a)-3*math.sin(a),frame.cz+(r-10)*math.sin(a)+3*math.cos(a))
        print("Envelope witness is inside",handle_closed_envelope.isInside(witness),"body inside witness",handle.isInside(witness),flush=True)
        print("Handle envelope residual",outside,"handle volume",handle.Volume(),"envelope volume",handle_closed_envelope.Volume(),flush=True)
        assert outside<.002,("handle envelope containment",outside)
        handle_envelope=handle_sweep(frame,center,travel)
        print("Check continuous swept envelopes",i,flush=True)
        checks={}
        for name,a,b in [("panel/cup",panel,cup),("panel/shade_sweep",panel,shade_envelope),
                          ("cup/shade_sweep",cup,shade_envelope),("panel/handle_sweep",panel,handle_envelope),
                          ("cup/handle_sweep",cup,handle_envelope),("shade/handle",shade,handle)]:
            volume=vol_intersection(a,b)
            checks[name]=volume
            print(name,volume,flush=True)
            assert volume<.002,(i,name,volume)
        distances={}
        for name,a,b in [("panel/cup",panel,cup),("panel/shade_sweep",panel,shade_envelope),("cup/shade_sweep",cup,shade_envelope)]:
            measure=BRepExtrema_DistShapeShape(a.wrapped,b.wrapped)
            measure.Perform()
            assert measure.IsDone()
            distances[name]=measure.Value()
        cq.exporters.export(shade_envelope,str(CAD/f"check_shade_sweep_{i}.step"),exportType="STEP")
        cq.exporters.export(handle_envelope,str(CAD/f"check_handle_sweep_{i}.step"),exportType="STEP")
        positions=[]
        for t in np.linspace(0,travel,P["collision_positions"]):
            s=move_on_cylinder(shade,frame,float(t),r)
            h=move_on_cylinder(handle,frame,float(t),r)
            # Full sweeps cover all positions; confirm the two requested positions independently.
            if abs(t-travel)<1e-7 or abs(t-travel/2)<1e-7:
                for name,a,b in [("panel/shade",panel,s),("cup/shade",cup,s),
                                 ("panel/handle",panel,h),("cup/handle",cup,h),("shade/handle",s,h)]:
                    volume=vol_intersection(a,b)
                    assert volume<.002,(i,t,name,volume)
                positions.append({"travel_mm":float(t),"intersections_mm3":0})
        report["motion"].append({"window":i,"shade_radii_mm":radii,"concentric":True,
           "full_motion_envelopes_checked":True,"swept_intersections_mm3":checks,"minimum_clearances_mm":distances,
           "open_and_half_confirmed":positions})
    report["all_checks_passed"]=True
    (ROOT/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)

if __name__=="__main__":
    main()
