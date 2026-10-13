"""Tu-214 window revision 3: preserve the panel; rebuild accessories.

Only source contour extraction, cylindrical BRep construction and STEP-record
preservation helpers are reused. No geometry from either previous cup or handle
is used. Coordinates are millimetres.
"""
from pathlib import Path
import sys, os, json, math, hashlib
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/tu214-v3-cache")
import numpy as np
import cadquery as cq
from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeEdge
from OCP.TopoDS import TopoDS

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "tu214-sidewall/source"))
from window_source import frames_from_panel, extract_throat, curve_data, make_surface
from cylindrical_parts import cylindrical_slab, rounded_rectangle, move_on_cylinder
from preserve_step_panel import preserve_panel

P = json.loads((ROOT / "parameters.json").read_text())
CAD = ROOT / "cad"
SOURCE = REPO / P["source_step"]

def xyz_on_cylinder(frame, x, s, radius):
    a = frame.theta + s / radius
    return np.array([frame.cx + x, frame.cy + radius * math.cos(a),
                     frame.cz + radius * math.sin(a)])

def section_rows(q, dq, frame, outer=False):
    n, up = frame.n, frame.up
    yy, zz = q[1] - frame.cy, q[2] - frame.cz
    ang = math.atan2(zz, yy) - frame.theta
    a = np.array([q[0] - frame.cx, 1753.0 * ang])
    da = np.array([dq[0], 1753.0 * (yy*dq[2] - zz*dq[1])/(yy*yy+zz*zz)])
    power = P["rear_superellipse_power"]
    bw, bh = P["rear_opening_width_mm"]/2, P["rear_opening_height_mm"]/2
    scale = ((abs(a[0])/bw)**power+(abs(a[1])/bh)**power)**(-1/power)
    b = a * scale
    if outer:
        na = np.array([da[1], -da[0]])
        na /= np.linalg.norm(na)
        nb = np.sign(b)*np.abs(b)**(power-1)/np.array([bw**power, bh**power])
        nb /= np.linalg.norm(nb)
        a += P["cup_wall_mm"]*na
        b += P["cup_wall_mm"]*nb
        # Retain the source front's depth while offsetting its contour.
        q = q + P["cup_wall_mm"]*(na[0]*np.array([1.,0.,0.])+na[1]*up)
    front = q + (P["cup_front_radius_mm"]-1753.0)*n
    back = xyz_on_cylinder(frame, b[0], b[1], P["cup_rear_radius_mm"])
    delta = back-front
    inplane_delta = delta-n*np.dot(delta,n)
    middle = (front+back)/2+P["cup_curve_bias"]*inplane_delta
    return np.array([front,middle,back])

def rows_data(item, u, frame, outer=False):
    q,dq,ddq = curve_data(item,u)
    h = 0.0001
    def shifted(t):
        return section_rows(q+t*dq+0.5*t*t*ddq, dq+t*ddq, frame, outer)
    m2,m1,z,p1,p2 = [shifted(t) for t in (-2*h,-h,0,h,2*h)]
    first = (m2-8*m1+8*p1-p2)/(12*h)
    second = (-m2+16*m1-30*z+16*p1-p2)/(12*h*h)
    return z,first,second

def solid_from_faces(faces):
    sew = BRepBuilderAPI_Sewing(P["sewing_tolerance_mm"],True,True,True,False)
    for face in faces: sew.Add(face.wrapped)
    sew.Perform()
    shell = cq.Shape.cast(sew.SewedShape())
    if sew.NbFreeEdges() or len(shell.Shells())!=1:
        raise RuntimeError(f"Unclosed BRep: {sew.NbFreeEdges()} free edges")
    solid = cq.Solid.makeSolid(shell.Shells()[0]).fix()
    if solid.Volume()<0:
        solid = cq.Solid(TopoDS.Solid_s(solid.wrapped.Reversed()))
    assert solid.isValid() and len(solid.Solids())==1 and solid.Shells()[0].Closed()
    return solid

def build_cup(panel, frame):
    curves,_ = extract_throat(panel,frame)
    samples=np.linspace(0,1,P["cup_u_samples"])
    faces=[]
    for item in curves[0]:
        data=[]; ends=[]
        for outer in (False,True):
            values=np.array([rows_data(item,u,frame,outer)[0] for u in samples])
            start,end=rows_data(item,0,frame,outer),rows_data(item,1,frame,outer)
            faces.append(make_surface(values,[start[1],end[1]],[start[2],end[2]]))
            data.append(values);ends.append((start,end))
        for row in (0,2):
            values=np.stack([v[:,row,:] for v in data],axis=1)
            first=[np.stack([e[k][1][row] for e in ends]) for k in (0,1)]
            second=[np.stack([e[k][2][row] for e in ends]) for k in (0,1)]
            faces.append(make_surface(values,first,second))
    return solid_from_faces(faces)

def build_shade(frame):
    center=P["shade_closed_lower_edge_mm"]+P["shade_height_mm"]/2
    outline=rounded_rectangle(P["shade_width_mm"],P["shade_height_mm"],45,12,center_s=center)
    return cylindrical_slab(outline,frame,P["shade_radius_mm"],P["shade_thickness_mm"])

def affine(shape, rows):
    return shape.transformGeometry(cq.Matrix(rows))

def build_handle():
    # Front silhouette: two symmetric parabolic arcs, 100 x 25.
    a=cq.Edge.makeBezier([cq.Vector(-50,-50,0),cq.Vector(0,-50,25),cq.Vector(50,-50,0)])
    b=cq.Edge.makeBezier([cq.Vector(50,-50,0),cq.Vector(0,-50,-25),cq.Vector(-50,-50,0)])
    lens=cq.Solid.extrudeLinear(cq.Wire.assembleEdges([a,b]),[],cq.Vector(0,55,0))
    # Top silhouette: one shallow convex back and a pronounced concave grip.
    pts=[(-50,-3),(-45,-1),(-25,-.2),(0,0),(25,-.2),(45,-1),(50,-3),
         (49,-13),(45,-32),(36,-40),(25,-36.5),(0,-30),
         (-25,-36.5),(-36,-40),(-45,-32),(-49,-13)]
    edge=cq.Edge.makeSpline([cq.Vector(x,y,-20) for x,y in pts],periodic=True)
    adaptor=edge._geomAdaptor()
    curve=adaptor.Curve().Curve()
    start,end=adaptor.FirstParameter(),adaptor.LastParameter()
    pieces=[]
    for lo,hi in zip(np.linspace(start,end,5)[:-1],np.linspace(start,end,5)[1:]):
        piece=curve.Copy()
        piece.Segment(float(lo),float(hi))
        pieces.append(cq.Edge(BRepBuilderAPI_MakeEdge(piece).Edge()))
    wire=cq.Wire.assembleEdges(pieces)
    bb=wire.BoundingBox()
    sx,sy=100/bb.xlen,40/bb.ylen
    wire=affine(wire,[[sx,0,0,-sx*(bb.xmin+bb.xmax)/2],
                      [0,sy,0,-sy*bb.ymax],[0,0,1,0],[0,0,0,1]])
    top=cq.Solid.extrudeLinear(wire,[],cq.Vector(0,0,40))
    # Side silhouette is symmetric in height only and tapers toward the grip.
    points=[cq.Vector(-60,y,z) for y,z in [(-45,-5.75),(5,-13.25),(5,13.25),(-45,5.75)]]
    wedge=cq.Solid.extrudeLinear(cq.Wire.makePolygon(points,close=True),[],cq.Vector(120,0,0))
    grip=lens.intersect(top).intersect(wedge).fix()
    assert grip.isValid() and len(grip.Solids())==1
    filleted=False
    for radius in (.6,.3,.12):
        try:
            candidate=cq.Workplane().newObject([grip]).edges().fillet(radius).val()
            if candidate.isValid() and len(candidate.Solids())==1:
                grip=candidate;filleted=True;print("Handle edge fillet",radius,flush=True);break
        except Exception as error:
            print("Handle fillet retry",radius,type(error).__name__,flush=True)
    if not filleted: print("Handle: smooth planform, intersection edges retained",flush=True)
    bb=grip.BoundingBox()
    sx,sy,sz=100/bb.xlen,40/bb.ylen,25/bb.zlen
    grip=affine(grip,[[sx,0,0,-sx*(bb.xmin+bb.xmax)/2],
                     [0,sy,0,-sy*bb.ymax],[0,0,sz,-sz*(bb.zmin+bb.zmax)/2],[0,0,0,1]])
    assert grip.isValid() and len(grip.Solids())==1
    return grip,filleted

def mount_handle(canonical,frame):
    s=P["shade_closed_lower_edge_mm"]+P["handle_center_from_shade_lower_edge_mm"]
    a=frame.theta+s/P["shade_radius_mm"]
    # Use a native rigid rotation. A 0.05-mm seating allowance avoids a
    # tangential Boolean trim across the lens's rounded upper edge.
    r=P["shade_radius_mm"]-.05
    mounted=canonical.rotate((0,0,0),(1,0,0),math.degrees(a)).translate(
        (frame.cx,frame.cy+r*math.cos(a),frame.cz+r*math.sin(a)))
    assert mounted.isValid() and len(mounted.Solids())==1
    # Integral small mounting land at the blade's lower edge. Its back face
    # is exactly R1761 and touches the shade without entering its volume.
    outline=rounded_rectangle(20,2,.4,center_s=P["shade_closed_lower_edge_mm"]+2)
    land=cylindrical_slab(outline,frame,P["shade_radius_mm"]-.3,.3,P["shade_radius_mm"])
    mounted=mounted.fuse(land).fix()
    print("Mounted handle",mounted.isValid(),len(mounted.Solids()),mounted.Volume(),flush=True)
    assert mounted.isValid() and len(mounted.Solids())==1
    return mounted

def export(shape,name):
    assert shape.isValid() and len(shape.Solids())==1,(name,"invalid solid")
    path=CAD/f"{name}.step"
    cq.exporters.export(shape,str(path),exportType="STEP")
    restored=cq.importers.importStep(str(path)).val()
    assert restored.isValid() and len(restored.Solids())==1,(name,"STEP round trip")
    return restored

def main():
    CAD.mkdir(parents=True,exist_ok=True)
    original_bytes=SOURCE.read_bytes()
    assert hashlib.sha256(original_bytes).hexdigest()==P["source_sha256"]
    panel=cq.importers.importStep(str(SOURCE)).val()
    assert panel.isValid() and len(panel.Solids())==1
    frames=frames_from_panel(panel)
    canonical,filleted=build_handle()
    export(canonical,"handle")
    parts=[]
    for i,frame in enumerate(frames,1):
        print("Build cup",i,flush=True)
        cup=export(build_cup(panel,frame),f"cup_{i}")
        print("Build shade and handle",i,flush=True)
        shade=build_shade(frame);handle=mount_handle(canonical,frame)
        parts.append((frame,cup,shade,handle))
    report={"source_sha256":P["source_sha256"],"panel_volume_mm3":panel.Volume(),
            "handle_edges_filleted":filleted,"assemblies":[]}
    for position,travel in [("open",P["shade_open_travel_mm"]),
                            ("half",P["shade_open_travel_mm"]/2),("closed",0)]:
        assembly=cq.Assembly(name=f"Tu214_Corrected_{position}")
        assembly.add(panel,name="Wall_Panel",color=cq.Color(.87,.85,.81))
        for i,(frame,cup,shade,handle) in enumerate(parts,1):
            r=P["shade_radius_mm"]
            moved_shade=move_on_cylinder(shade,frame,travel,r)
            moved_handle=move_on_cylinder(handle,frame,travel,r)
            export(moved_shade,f"shade_{i}_{position}")
            export(moved_handle,f"handle_{i}_{position}")
            assembly.add(cup,name=f"Cup_{i}",color=cq.Color(.97,.96,.93))
            assembly.add(moved_shade,name=f"Shade_{i}",color=cq.Color(.69,.68,.63))
            assembly.add(moved_handle,name=f"Handle_{i}",color=cq.Color(.89,.81,.63))
        path=CAD/f"assembly_{position}.step"
        assembly.export(str(path),exportType="STEP")
        preserved=preserve_panel(SOURCE,path)
        restored=cq.importers.importStep(str(path)).val()
        assert restored.isValid() and len(restored.Solids())==7
        saved=max(restored.Solids(),key=lambda s:s.Volume())
        assert abs(saved.Volume()-panel.Volume())<0.00001
        report["assemblies"].append({"position":position,"valid":True,"solids":7,
            "travel_mm":travel,"panel_volume_difference_mm3":saved.Volume()-panel.Volume(),**preserved})
        print("Exported",position,flush=True)
    assert SOURCE.read_bytes()==original_bytes
    (ROOT/"build_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)

if __name__=="__main__":
    main()
