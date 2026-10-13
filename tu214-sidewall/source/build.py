"""Reconstruct a formed window reveal and a concentric sliding shade.

The original panel is imported and kept as a separate, unchanged B-rep.
Its original STEP records are retained verbatim apart from entity numbers.
"""
from pathlib import Path
import os,json,math,hashlib
os.environ.setdefault("XDG_CACHE_HOME","/tmp/tu214-cache")
import numpy as np
import cadquery as cq
from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
from OCP.TopoDS import TopoDS
from window_source import frames_from_panel,extract_throat,curve_data,source_normal_data,make_surface
from profile_geometry import RoundedProfile,contour_arcs,Arc,Bezier
from cylindrical_parts import cylindrical_slab,rounded_rectangle,move_on_cylinder
from preserve_step_panel import preserve_panel

ROOT=Path(__file__).resolve().parents[1]
P=json.loads((ROOT/"source/parameters.json").read_text())
CAD=ROOT/"cad"


def profiles():
    arcs=contour_arcs(P["rear_opening_width_mm"],P["rear_opening_height_mm"],P["rear_side_radius_mm"],
        P["rear_top_crown_radius_mm"],P["rear_bottom_crown_radius_mm"],
        P["rear_upper_corner_radius_mm"],P["rear_lower_corner_radius_mm"])
    return [RoundedProfile(arcs,P["rear_g2_blend_half_length_mm"],offset,P["rear_g2_blend_speed_factor"])
        for offset in (0.,P["cup_rear_wall_width_mm"])]


def rear_data(q,d,dd,frame,profile):
    y,z=q[1]-frame.cy,q[2]-frame.cz;dy,dz=d[1],d[2];ddy,ddz=dd[1],dd[2]
    rr=y*y+z*z;cross=y*dz-z*dy
    angle=math.atan2(z,y)-frame.theta
    ad=cross/rr;add=(y*ddz-z*ddy)/rr-2*(y*dy+z*dz)*cross/rr**2
    a=np.array([q[0]-frame.cx,1753.*angle]);da=np.array([d[0],1753.*ad]);dda=np.array([dd[0],1753.*add])
    r2=np.dot(a,a);c=a[0]*da[1]-a[1]*da[0]
    phi=math.atan2(a[1],a[0]);pd=c/r2;pdd=(a[0]*dda[1]-a[1]*dda[0])/r2-2*np.dot(a,da)*c/r2**2
    b,db,ddb=profile.data(phi);dbu=db*pd;ddbu=ddb*pd*pd+db*pdd
    radius=P["cup_rear_radius_mm"];theta=frame.theta+b[1]/radius;ct,st=math.cos(theta),math.sin(theta)
    point=np.array([frame.cx+b[0],frame.cy+radius*ct,frame.cz+radius*st])
    first=np.array([dbu[0],-st*dbu[1],ct*dbu[1]])
    second=np.array([ddbu[0],-st*ddbu[1]-ct*dbu[1]**2/radius,ct*ddbu[1]-st*dbu[1]**2/radius])
    return point,first,second


def cup_rows(q,normal,rear,frame):
    def unit(v):return v/np.sqrt(np.dot(v,v))
    radial=unit(np.array([0.,q[1]-frame.cy,q[2]-frame.cz]));normal=unit(normal)
    start_direction=unit(radial-np.dot(radial,normal)*normal)
    rear_radial=unit(np.array([0.,rear[1]-frame.cy,rear[2]-frame.cz]))
    length=np.dot(rear-q,radial)
    d0=P["cup_front_tangent_factor"]*length*start_direction
    d1=P["cup_rear_tangent_factor"]*length*rear_radial
    return np.array([q,q+d0/5,q+2*d0/5,rear-2*d1/5,rear-d1/5,rear])


def rows_data(item,u,frame,profile):
    q,dq,ddq=curve_data(item,u);n,dn,ddn=source_normal_data(item,u);r,dr,ddr=rear_data(q,dq,ddq,frame,profile)
    h=1e-20;eps=1e-5
    rows=cup_rows(q,n,r,frame)
    first=np.imag(cup_rows(q+1j*h*dq,n+1j*h*dn,r+1j*h*dr,frame))/h
    def shifted(sign):
        return np.imag(cup_rows(q+sign*eps*dq+.5*eps*eps*ddq+1j*h*(dq+sign*eps*ddq),
            n+sign*eps*dn+.5*eps*eps*ddn+1j*h*(dn+sign*eps*ddn),
            r+sign*eps*dr+.5*eps*eps*ddr+1j*h*(dr+sign*eps*ddr),frame))/h
    return rows,first,(shifted(1)-shifted(-1))/(2*eps)


def build_reveal(curves,frame):
    prof=profiles();samples=np.linspace(0,1,P["cup_u_samples"]);faces=[]
    for inner,outer in zip(*curves):
        data=[];endpoints=[]
        for item,profile in zip((inner,outer),prof):
            values=np.array([rows_data(item,u,frame,profile)[0] for u in samples])
            start=rows_data(item,0,frame,profile);end=rows_data(item,1,frame,profile)
            faces.append(make_surface(values,[start[1],end[1]],[start[2],end[2]]))
            data.append(values);endpoints.append((start,end))
        for side in (0,5):
            values=np.stack([r[:,side,:] for r in data],axis=1)
            first=[np.stack([ends[k][1][side] for ends in endpoints]) for k in (0,1)]
            second=[np.stack([ends[k][2][side] for ends in endpoints]) for k in (0,1)]
            faces.append(make_surface(values,first,second))
    sew=BRepBuilderAPI_Sewing(P["sewing_tolerance_mm"],True,True,True,False)
    for face in faces:sew.Add(face.wrapped)
    sew.Perform();shell=cq.Shape.cast(sew.SewedShape())
    if sew.NbFreeEdges() or len(shell.Shells())!=1:raise RuntimeError(f"Reveal free edges: {sew.NbFreeEdges()}")
    solid=cq.Solid.makeSolid(shell.Shells()[0]).fix()
    if solid.Volume()<0:solid=cq.Solid(TopoDS.Solid_s(solid.wrapped.Reversed()))
    if not solid.isValid() or not solid.Shells()[0].Closed():raise RuntimeError("Invalid reveal solid")
    return solid


def shade_radius():return P["cup_rear_radius_mm"]+P["shade_radius_offset_from_cup_mm"]


def shifted_x(segments,delta):
    result=[]
    for segment in segments:
        if isinstance(segment,Arc):result.append(Arc(segment.center+np.array([delta,0.]),segment.radius,segment.a0,segment.a1))
        else:result.append(Bezier(segment.poles+np.array([delta,0.])))
    return result


def build_shade(frame):
    center=P["shade_closed_lower_edge_mm"]+P["shade_height_mm"]/2
    outline=rounded_rectangle(P["shade_width_mm"],P["shade_height_mm"],P["shade_upper_corner_radius_mm"],P["shade_lower_corner_radius_mm"],center)
    return cylindrical_slab(outline,frame,shade_radius(),P["shade_thickness_mm"])


def build_handle(frame):
    r=shade_radius();bottom=P["shade_closed_lower_edge_mm"]
    mount=cylindrical_slab(rounded_rectangle(P["handle_mount_width_mm"],P["handle_mount_height_mm"],3.,center_s=bottom+11),
        frame,r-P["handle_mount_thickness_mm"],P["handle_mount_thickness_mm"],r)
    lip=cylindrical_slab(rounded_rectangle(P["handle_grip_width_mm"],P["handle_grip_height_mm"],P["handle_grip_corner_radius_mm"],center_s=bottom+8),
        frame,r-P["handle_grip_projection_mm"],P["handle_grip_projection_mm"]-P["handle_mount_thickness_mm"]+.25,r)
    # Round the forward perimeter into a shallow finger lip. The mounting
    # strip sits on the front cylinder of the shade, without large brackets.
    front=r-P["handle_grip_projection_mm"]
    edges=[]
    for edge in lip.Edges():
        pts=[p.toTuple() for p in edge.positions(np.linspace(0,1,7))]
        radii=[math.hypot(y-frame.cy,z-frame.cz) for x,y,z in pts]
        if max(abs(x-front) for x in radii)<1e-4:edges.append(edge)
    lip=cq.Workplane().newObject([lip]).newObject(edges).fillet(P["handle_edge_fillet_radius_mm"]).val()
    handle=mount.fuse(lip).clean()
    if not handle.isValid() or len(handle.Solids())!=1:raise RuntimeError("Invalid low-profile handle")
    return handle


def build_guides(frame):
    r0=P["cup_rear_radius_mm"]+P["guide_front_gap_from_cup_mm"]
    rb=P["cup_rear_radius_mm"]+P["window_block_radial_clearance_mm"]
    low=P["shade_closed_lower_edge_mm"]-P["guide_end_margin_mm"]
    high=P["shade_closed_lower_edge_mm"]+P["shade_height_mm"]+P["shade_open_travel_mm"]+P["guide_end_margin_mm"]
    height=high-low;center=(high+low)/2;development=shade_radius();result=[]
    for side in (-1.,1.):
        xi,xo=P["guide_side_web_inner_x_mm"],P["guide_side_web_outer_x_mm"]
        web=cylindrical_slab(shifted_x(rounded_rectangle(xo-xi,height,1.3,center_s=center),side*(xi+xo)/2),frame,r0,rb-r0,development)
        lipwidth=xo-P["guide_lip_inner_x_mm"];lipcenter=side*(xo+P["guide_lip_inner_x_mm"])/2
        outline=shifted_x(rounded_rectangle(lipwidth,height,2.,center_s=center),lipcenter)
        front=cylindrical_slab(outline,frame,r0,P["guide_lip_radial_thickness_mm"],development)
        back=cylindrical_slab(outline,frame,rb-P["guide_lip_radial_thickness_mm"],P["guide_lip_radial_thickness_mm"],development)
        guide=web.fuse(front,back).clean()
        if not guide.isValid() or len(guide.Solids())!=1:raise RuntimeError("Invalid cylindrical U-channel guide")
        result.append(guide)
    return result


def export_part(shape,name):
    if not shape.isValid() or len(shape.Solids())!=1:raise RuntimeError(f"Invalid part {name}")
    cq.exporters.export(shape,str(CAD/f"{name}.step"),exportType="STEP")


def main():
    CAD.mkdir(exist_ok=True);source=ROOT/P["original_file"];panel=cq.importers.importStep(str(source)).val()
    frames=frames_from_panel(panel);parts=[]
    for i,frame in enumerate(frames,1):
        print(f"Window {i}: existing contour -> independent rounded B",flush=True)
        curves,_=extract_throat(panel,frame);reveal=build_reveal(curves,frame)
        print(f"Window {i}: cylindrical shade, shallow handle, two guides",flush=True)
        shade=build_shade(frame);handle=build_handle(frame);guides=build_guides(frame)
        for name,part in [(f"window_reveal_{i}",reveal),(f"window_shade_{i}_closed",shade),(f"shade_handle_{i}_closed",handle),
            (f"shade_guide_{i}_left",guides[0]),(f"shade_guide_{i}_right",guides[1])]:export_part(part,name)
        parts.append((frame,reveal,shade,handle,guides))
    report={"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"source_panel_volume_mm3":panel.Volume(),"parameters":P,"positions":[]}
    for position,travel,filename in (("open",P["shade_open_travel_mm"],"tu214_sidewall_complete"),
        ("half",P["shade_open_travel_mm"]/2,"tu214_sidewall_half_closed"),("closed",0.,"tu214_sidewall_closed")):
        print(f"Export {position} assembly",flush=True)
        assembly=cq.Assembly(name=f"Tu214_Window_Node_{position}")
        assembly.add(panel,name="Wall_Panel",color=cq.Color(.86,.845,.79))
        for i,(frame,reveal,shade,handle,guides) in enumerate(parts,1):
            moved_shade=move_on_cylinder(shade,frame,travel,shade_radius());moved_handle=move_on_cylinder(handle,frame,travel,shade_radius())
            assembly.add(reveal,name=f"Window_Reveal_{i}",color=cq.Color(.955,.947,.922))
            assembly.add(moved_shade,name=f"Window_Shade_{i}",color=cq.Color(.76,.755,.705))
            assembly.add(moved_handle,name=f"Shade_Handle_{i}",color=cq.Color(.86,.807,.685))
            for j,guide in enumerate(guides):assembly.add(guide,name=f"Shade_Guide_{i}_{j+1}",color=cq.Color(.59,.61,.62))
            export_part(moved_shade,f"window_shade_{i}_{position}");export_part(moved_handle,f"shade_handle_{i}_{position}")
        path=CAD/f"{filename}.step";assembly.export(str(path),exportType="STEP");preserved=preserve_panel(source,path)
        restored=cq.importers.importStep(str(path)).val();saved=max(restored.Solids(),key=lambda s:s.Volume())
        if not restored.isValid() or len(restored.Solids())!=11:raise RuntimeError("STEP assembly must contain 11 valid solids")
        if abs(saved.Volume()-panel.Volume())>1e-5:raise RuntimeError("Panel geometry changed during STEP export")
        report["positions"].append({"position":position,"travel_mm":travel,"rotation_deg":travel/shade_radius()*180/math.pi,
            "file":path.name,"solids":11,"valid":True,"panel_volume_difference_mm3":saved.Volume()-panel.Volume(),**preserved})
    (CAD/"build_validation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="parameters"},indent=2),flush=True)


if __name__=="__main__":main()
