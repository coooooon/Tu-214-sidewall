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
from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box
from OCP.BRepTools import BRepTools
from OCP.BRepLib import BRepLib
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.TopAbs import TopAbs_OUT
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "tu214-sidewall/source"))
from window_source import frames_from_panel, extract_throat, curve_data, make_surface
from cylindrical_parts import cylindrical_slab, rounded_rectangle, move_on_cylinder
from preserve_step_panel import preserve_panel

P = json.loads((ROOT / "parameters.json").read_text())
CAD = ROOT / "cad"
SOURCE = REPO / P["source_step"]

def pure_solid(shape):
    assert len(shape.Solids())==1
    solid=cq.Solid(TopoDS.Solid_s(shape.Solids()[0].wrapped))
    BRepTools.Clean_s(solid.wrapped)
    infinity=BRepClass3d_SolidClassifier(solid.wrapped)
    infinity.PerformInfinitePoint(.00001)
    if infinity.State()!=TopAbs_OUT:
        assert BRepLib.OrientClosedSolid_s(solid.wrapped)
        infinity=BRepClass3d_SolidClassifier(solid.wrapped)
        infinity.PerformInfinitePoint(.00001)
    assert infinity.State()==TopAbs_OUT,"The material must be bounded"
    assert solid.isValid() and solid.Shells()[0].Closed()
    return solid

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

def exact_bounds(shape):
    box=Bnd_Box()
    BRepBndLib.AddOptimal_s(shape.wrapped,box,False,False)
    xmin,ymin,zmin,xmax,ymax,zmax=box.Get()
    return SimpleNamespace(xmin=xmin,ymin=ymin,zmin=zmin,xmax=xmax,ymax=ymax,zmax=zmax,
                           xlen=xmax-xmin,ylen=ymax-ymin,zlen=zmax-zmin)

def mesh_bounds(shape):
    vertices,_=shape.tessellate(.01,.08)
    pts=np.array([v.toTuple() for v in vertices])
    xmin,ymin,zmin=pts.min(axis=0);xmax,ymax,zmax=pts.max(axis=0)
    return SimpleNamespace(xmin=xmin,ymin=ymin,zmin=zmin,xmax=xmax,ymax=ymax,zmax=zmax,
                           xlen=xmax-xmin,ylen=ymax-ymin,zlen=zmax-zmin)

def affine(shape, rows):
    BRepTools.Clean_s(shape.wrapped)
    return shape.transformGeometry(cq.Matrix(rows))

def build_handle():
    # Directly controlled orthographic silhouettes. Six nonperiodic cubic
    # Bezier edges define a symmetric convex back and a concave grip.
    half=P["handle_width_mm"]/2
    height=P["handle_height_mm"]
    depth=P["handle_depth_mm"]
    def point(x,y,z):
        return cq.Vector(x*half/50,y*depth/40,z*height/25)
    a=cq.Edge.makeBezier([point(-50,-45,0),point(0,-45,25),point(50,-45,0)])
    b=cq.Edge.makeBezier([point(50,-45,0),point(0,-45,-25),point(-50,-45,0)])
    lens=cq.Solid.extrudeLinear(cq.Wire.assembleEdges([a,b]),[],cq.Vector(0,50*depth/40,0))
    controls=[
        [(0,0),(25,0),(50,0),(50,-5)],
        [(50,-5),(50,-25),(46,-40),(36,-40)],
        [(36,-40),(26,-40),(18,-30),(0,-30)],
        [(0,-30),(-18,-30),(-26,-40),(-36,-40)],
        [(-36,-40),(-46,-40),(-50,-25),(-50,-5)],
        [(-50,-5),(-50,0),(-25,0),(0,0)]]
    edges=[cq.Edge.makeBezier([point(x,y,-20) for x,y in pts]) for pts in controls]
    top=cq.Solid.extrudeLinear(cq.Wire.assembleEdges(edges),[],cq.Vector(0,0,40*height/25))
    pts=[point(-55,y,z) for y,z in [(-45,-5.75),(5,-13.25),(5,13.25),(-45,5.75)]]
    wedge=cq.Solid.extrudeLinear(cq.Wire.makePolygon(pts,close=True),[],cq.Vector(110*half/50,0,0))
    grip=pure_solid(lens.intersect(top).intersect(wedge).fix())
    dims=exact_bounds(grip)
    print("Native handle dimensions",dims.xlen,dims.ylen,dims.zlen,flush=True)
    assert np.allclose([dims.xlen,dims.ylen,dims.zlen],[2*half,depth,height],atol=.015)
    return grip,False

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
    land=cylindrical_slab(outline,frame,P["shade_radius_mm"]-1.5,1.5,P["shade_radius_mm"])
    measure=BRepExtrema_DistShapeShape(mounted.wrapped,land.wrapped)
    measure.Perform()
    p1,p2=measure.PointOnShape1(1),measure.PointOnShape2(1)
    print("Mount distance",measure.Value(),"body",p1.X(),p1.Y(),p1.Z(),"land",p2.X(),p2.Y(),p2.Z(),flush=True)
    print("Canonical bounds",vars(exact_bounds(canonical)),"centroid",canonical.Center().toTuple(),flush=True)
    print("Mount overlap",mounted.intersect(land).Volume(),flush=True)
    mounted=mounted.fuse(land).fix()
    print("Mounted handle",mounted.isValid(),len(mounted.Solids()),mounted.Volume(),flush=True)
    assert mounted.isValid() and len(mounted.Solids())==1
    return pure_solid(mounted)

def export(shape,name):
    shape=pure_solid(shape)
    assert shape.isValid() and len(shape.Solids())==1,(name,"invalid solid")
    path=CAD/f"{name}.step"
    cq.exporters.export(shape,str(path),exportType="STEP")
    restored=cq.importers.importStep(str(path)).val()
    assert restored.isValid() and len(restored.Solids())==1,(name,"STEP round trip")
    return pure_solid(restored)

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
        shade=pure_solid(build_shade(frame));handle=pure_solid(mount_handle(canonical,frame))
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
