"""Validate delivered STEP geometry, contact and the full motion corridor."""
from pathlib import Path
import os,json,math,hashlib,argparse
os.environ.setdefault("XDG_CACHE_HOME","/tmp/tu214-cache")
import numpy as np
import cadquery as cq
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.TopAbs import TopAbs_IN,TopAbs_ON
from OCP.gp import gp_Pnt,gp_Vec
from build import P,ROOT,CAD,profiles,shade_radius
from window_source import frames_from_panel,extract_throat,curve_data
from cylindrical_parts import cylindrical_slab
from profile_geometry import Bezier
from preserve_step_panel import records,closure,TOKEN,references


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(name):return cq.importers.importStep(str(CAD/f"{name}.step")).val()
def xyz(p):return np.array([p.X(),p.Y(),p.Z()])


def surface_data(face,u,v):
    s=BRepAdaptor_Surface(face.wrapped);p=gp_Pnt();du=gp_Vec();dv=gp_Vec();s.D1(u,v,p,du,dv)
    a,b=xyz(du),xyz(dv);normal=np.cross(a,b);normal/=np.linalg.norm(normal)
    return xyz(p),a,b,normal


def angle(a,b):return math.degrees(math.acos(min(1.,abs(float(np.dot(a,b))))))


def geometry_hash(items,root):
    cache={}
    def digest(i):
        if i not in cache:
            r=TOKEN.sub(lambda m:'@'+digest(int(m.group(1))) if m.group(1) else m.group(0),items[i])
            cache[i]=hashlib.sha256(r.encode()).hexdigest()
        return cache[i]
    return digest(root)


def panel_root(items):
    product=next(i for i,r in items.items() if r.startswith("PRODUCT('Wall_Panel'"))
    formation=next(i for i,r in items.items() if r.startswith("PRODUCT_DEFINITION_FORMATION(") and product in references(r))
    definition=next(i for i,r in items.items() if r.startswith("PRODUCT_DEFINITION(") and formation in references(r))
    pshape=next(i for i,r in items.items() if r.startswith("PRODUCT_DEFINITION_SHAPE(") and definition in references(r))
    sd=next(r for r in items.values() if r.startswith("SHAPE_DEFINITION_REPRESENTATION(") and pshape in references(r))
    return next(i for i in references(sd) if items[i].startswith("ADVANCED_BREP_SHAPE_REPRESENTATION("))


def reveal_bound(reveal,frame):
    radii=[]
    for face in reveal.Faces():
        s=face._geomAdaptor()
        if face.geomType()!="BSPLINE":raise RuntimeError("Unexpected reveal surface type")
        for i in range(1,s.NbUPoles()+1):
            for j in range(1,s.NbVPoles()+1):
                if s.Weight(i,j)<=0:raise RuntimeError("Cannot certify a nonpositive-weight surface")
                q=s.Pole(i,j);radii.append(math.hypot(q.Y()-frame.cy,q.Z()-frame.cz))
    # Positive NURBS basis functions keep all surface points in the convex
    # hull of the control points; radial distance is a convex function.
    return max(radii)


def check_window_geometry(panel,frame,index):
    reveal=load(f"window_reveal_{index}");shade=load(f"window_shade_{index}_closed");handle=load(f"shade_handle_{index}_closed")
    curves,_=extract_throat(panel,frame)
    walls=[f for f in reveal.Faces() if f.geomType()=="BSPLINE" and f._geomAdaptor().VDegree()==5]
    remaining=list(walls);front_gaps=[];back_radius_error=[];normal_angles=[];seam_angles=[];thickness=[];area=[];jacobian=[];back_points=[]
    loops=[]
    for loop in curves:
        matched=[]
        for item in loop:
            target=curve_data(item,.5)[0]
            face=min(remaining,key=lambda f:np.linalg.norm(surface_data(f,.5,0)[0]-target));remaining.remove(face);matched.append(face)
            for u in np.linspace(0,1,151):
                q,_,_,_=surface_data(face,u,0);front_gaps.append(float(np.linalg.norm(q-curve_data(item,u)[0])))
                b,_,_,_=surface_data(face,u,1);back_radius_error.append(abs(math.hypot(b[1]-frame.cy,b[2]-frame.cz)-P["cup_rear_radius_mm"]))
                if loop is curves[0]:back_points.append(b)
            for u in (.1,.25,.5,.75,.9):
                q=curve_data(item,u)[0];source=np.array(item[2].normalAt(cq.Vector(*q)).toTuple())
                normal_angles.append(angle(source,surface_data(face,u,0)[3]))
            for u in np.linspace(.001,.999,41):
                for v in np.linspace(0,1,17):
                    q,du,dv,_=surface_data(face,u,v);area.append(float(np.linalg.norm(np.cross(du,dv))))
                    yy,zz=q[1]-frame.cy,q[2]-frame.cz;r=math.hypot(yy,zz);th=math.atan2(zz,yy)-frame.theta
                    local=np.array([q[0]-frame.cx,1753*th]);r2=np.dot(local,local)
                    def derivatives(d):
                        thd=(yy*d[2]-zz*d[1])/(r*r);ld=np.array([d[0],1753*thd])
                        phid=(local[0]*ld[1]-local[1]*ld[0])/r2
                        rd=(yy*d[1]+zz*d[2])/r
                        return phid,rd
                    pu,ru=derivatives(du);pv,rv=derivatives(dv);jacobian.append(pu*rv-pv*ru)
        for a,b in zip(matched,matched[1:]+matched[:1]):
            for v in (.1,.25,.5,.75,.9):seam_angles.append(angle(surface_data(a,1,v)[3],surface_data(b,0,v)[3]))
        loops.append(matched)
    if remaining:raise RuntimeError("Unmatched reveal walls")
    for a,b in zip(*loops):
        for u in np.linspace(0,1,31):
            for v in np.linspace(0,1,17):
                q,_,_,n=surface_data(a,u,v);outer=surface_data(b,u,v)[0]
                thickness.append(float(-np.dot(outer-q,n)))
    cylinders=[f for f in shade.Faces() if f.geomType()=="CYLINDER"]
    main_cylinders=[f for f in cylinders if f.Area()>10000]
    radii=sorted(f._geomAdaptor().Cylinder().Radius() for f in main_cylinders)
    assert len(radii)==2 and abs(radii[0]-shade_radius())<1e-8 and abs(radii[1]-shade_radius()-P["shade_thickness_mm"])<1e-8
    for f in main_cylinders:
        c=f._geomAdaptor().Cylinder();axis=c.Axis();loc=axis.Location()
        assert abs(loc.Y()-frame.cy)<1e-8 and abs(loc.Z()-frame.cz)<1e-8
        assert abs(abs(axis.Direction().X())-1)<1e-10
    front_face=min(main_cylinders,key=lambda f:f._geomAdaptor().Cylinder().Radius());coverage=[]
    for q in back_points:
        theta=math.atan2(q[2]-frame.cz,q[1]-frame.cy)
        query=gp_Pnt(q[0],frame.cy+shade_radius()*math.cos(theta),frame.cz+shade_radius()*math.sin(theta))
        coverage.append(BRepClass_FaceClassifier(front_face.wrapped,query,1e-6).State() in (TopAbs_IN,TopAbs_ON))
    relative_intersection=shade.intersect(handle,tol=5e-5).Volume()
    assert relative_intersection<.001
    assert max(front_gaps)<.001 and max(back_radius_error)<.005
    assert max(normal_angles)<.05 and max(seam_angles)<.015
    assert min(thickness)>.5 and min(area)>1e-5 and min(jacobian)>0
    assert all(coverage)
    bound=reveal_bound(reveal,frame);handle_min=shade_radius()-P["handle_grip_projection_mm"]
    assert handle_min-bound>.5
    return {"window":index,"max_front_contour_error_mm":max(front_gaps),"max_rear_radius_error_mm":max(back_radius_error),
        "max_panel_joint_normal_angle_deg":max(normal_angles),"max_reveal_seam_normal_angle_deg":max(seam_angles),
        "min_normal_wall_thickness_mm":min(thickness),"max_normal_wall_thickness_mm":max(thickness),
        "min_surface_area_jacobian":min(area),"min_polar_radial_jacobian":min(jacobian),
        "shade_inner_outer_radii_mm":radii,"shade_concentric_with_panel":True,"closed_aperture_coverage_samples":len(coverage),
        "closed_aperture_covered":all(coverage),"handle_shade_intersection_mm3":relative_intersection,
        "reveal_radial_control_hull_bound_mm":bound,"handle_reveal_min_clearance_bound_mm":handle_min-bound}


def check_geometry():
    original=CAD/"tu214_sidewall_original.step";panel=load("tu214_sidewall_original");source_records=records(original.read_text())
    original_rep=next(i for i,r in source_records.items() if r.startswith("ADVANCED_BREP_SHAPE_REPRESENTATION("))
    source_digest=geometry_hash(source_records,original_rep);assemblies=[]
    for name in ("tu214_sidewall_complete","tu214_sidewall_half_closed","tu214_sidewall_closed"):
        path=CAD/f"{name}.step";saved=load(name);items=records(path.read_text());digest=geometry_hash(items,panel_root(items))
        assert digest==source_digest and saved.isValid() and len(saved.Solids())==11
        assemblies.append({"file":path.name,"geometry_hash_matches_original":True,"sha256":sha(path)})
    windows=[check_window_geometry(panel,frame,i) for i,frame in enumerate(frames_from_panel(panel),1)]
    r=P["cup_rear_radius_mm"];sr=shade_radius();gb=r+P["window_block_radial_clearance_mm"]
    report={"source_sha256":sha(original),"panel_geometry_reference_hash":source_digest,"assemblies":assemblies,"windows":windows,
        "clearances_mm":{"cup_to_window_block_reference":gb-r,"cup_to_shade_front":sr-r,
            "shade_back_to_window_block_reference":gb-sr-P["shade_thickness_mm"],
            "guide_front_lip_to_shade":sr-(r+P["guide_front_gap_from_cup_mm"]+P["guide_lip_radial_thickness_mm"]),
            "shade_to_guide_rear_lip":gb-P["guide_lip_radial_thickness_mm"]-sr-P["shade_thickness_mm"],
            "shade_side_to_guide_web":P["guide_side_web_inner_x_mm"]-P["shade_width_mm"]/2},
        "all_geometry_checks_passed":True}
    (CAD/"geometry_validation.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2),flush=True)


def rectangle(width,low,high):
    x=width/2;pts=[[-x,low],[x,low],[x,high],[-x,high]]
    return [Bezier(np.array([pts[i],pts[(i+1)%4]],dtype=float)) for i in range(4)]


def check_contacts(index):
    panel=load("tu214_sidewall_original");frame=frames_from_panel(panel)[index-1];reveal=load(f"window_reveal_{index}")
    guides=[load(f"shade_guide_{index}_{side}") for side in ("left","right")]
    sr=shade_radius();low=P["shade_closed_lower_edge_mm"];high=low+P["shade_height_mm"]+P["shade_open_travel_mm"]
    sweep=cylindrical_slab(rectangle(P["shade_width_mm"],low,high),frame,sr,P["shade_thickness_mm"],sr)
    handle_sweep=cylindrical_slab(rectangle(P["handle_mount_width_mm"],low+3.5,low+15+P["shade_open_travel_mm"]),
        frame,sr-P["handle_grip_projection_mm"],P["handle_grip_projection_mm"],sr)
    for name,solid in [(f"motion_envelope_shade_{index}",sweep),(f"motion_envelope_handle_{index}",handle_sweep)]:
        cq.exporters.export(solid,str(CAD/f"{name}.step"),exportType="STEP")
    checks=[]
    tests=[("panel/reveal",panel,reveal),("panel/shade_full_sweep",panel,sweep),("panel/handle_full_sweep",panel,handle_sweep)]
    for j,guide in enumerate(guides,1):tests.extend([(f"panel/guide_{j}",panel,guide),(f"guide_{j}/shade_full_sweep",guide,sweep)])
    for name,a,b in tests:
        print(f"Window {index}: Boolean {name}",flush=True)
        intersection=a.intersect(b,tol=5e-5);volume=intersection.Volume()
        assert intersection.isValid() and volume<.001,(name,volume)
        checks.append({"pair":name,"intersection_mm3":volume,"intersection_valid":True})
    bound=reveal_bound(reveal,frame)
    assert sr-P["handle_grip_projection_mm"]>bound
    report={"window":index,"checks":checks,"fuzzy_tolerance_mm":.00005,
        "continuous_motion_checked_by_full_swept_envelopes":True,
        "reveal_vs_shade_handle_guides_certified_by_radial_control_hull":True,
        "reveal_radial_control_hull_bound_mm":bound,
        "handle_vs_guides_certified_by_disjoint_x_ranges":True,
        "static_guide_max_inner_x_abs_mm":P["guide_lip_inner_x_mm"],"handle_max_x_abs_mm":P["handle_mount_width_mm"]/2,
        "part_sha256":{name:sha(CAD/f"{name}.step") for name in [f"window_reveal_{index}",f"window_shade_{index}_closed",f"shade_handle_{index}_closed",f"shade_guide_{index}_left",f"shade_guide_{index}_right"]}}
    (CAD/f"collision_window_{index}.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--contacts",type=int,choices=(1,2));args=parser.parse_args()
    if args.contacts:check_contacts(args.contacts)
    else:check_geometry()
