"""Analytical formed cup: concentric flange, drafted circular walls, R10 blend."""
from pathlib import Path
import importlib.util,sys,os,math,json,hashlib,zipfile,shutil
os.environ.setdefault("MPLCONFIGDIR","/tmp/tu214-v5-mpl")
import numpy as np
import cadquery as cq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from PIL import Image
from OCP.Geom import Geom_ConicalSurface
from OCP.gp import gp_Ax3,gp_Pnt,gp_Dir
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import TopoDS
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent;CAD=ROOT/"cad";OUT=ROOT/"previews"
for p in (CAD,OUT):p.mkdir(exist_ok=True)
BASE=REPO/"corrected-window-v3"
spec=importlib.util.spec_from_file_location("baseline_v3",BASE/"build.py");base=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=base;spec.loader.exec_module(base)
from profile_geometry import contour_arcs
from cylindrical_parts import move_on_cylinder,rounded_rectangle,cylindrical_slab
from preview_mesh import load_mesh,merge,rasterize
from preserve_step_panel import preserve_panel
CFG=json.loads((ROOT/"parameters.json").read_text())
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
RF=CFG["flange_radius_mm"];RB=CFG["floor_radius_mm"];T=CFG["wall_mm"]
ARCS=contour_arcs(247,345,1000,350,350,90,90)
SLOPE=CFG["mouth_offset_mm"]/(RB-RF)
ALPHA=math.atan(SLOPE)
XMIN=-300.;XLEN=600.
YFRONT=RF-80.;YBACK=RB+20.
R=base.P["shade_radius_mm"]

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def protected():
    paths=list(BASE.rglob("*.step"))+list((REPO/"handle-revision-v4/cad").glob("*.step"))+[base.SOURCE,BASE/"build.py",BASE/"parameters.json"]
    return {str(p.relative_to(REPO)):digest(p) for p in paths}

def wire(y,offset=0):
    edges=[]
    for a in ARCS:
        def p(t):
            ang=a.a0+(a.a1-a.a0)*t
            q=a.center+(a.radius+offset)*np.array([math.cos(ang),math.sin(ang)])
            return cq.Vector(q[0],y,q[1])
        edges.append(cq.Edge.makeThreePointArc(p(0),p(.5),p(1)))
    w=cq.Wire.assembleEdges(edges);assert w.IsClosed()
    return w

def prism(offset=0):
    return cq.Solid.extrudeLinear(wire(YFRONT,offset),[],cq.Vector(0,YBACK-YFRONT+30,0))

def radial_band(a,b):
    outer=cq.Solid.makeCylinder(b,XLEN,cq.Vector(XMIN,0,0),cq.Vector(1,0,0))
    inner=cq.Solid.makeCylinder(a,XLEN,cq.Vector(XMIN,0,0),cq.Vector(1,0,0))
    return outer.cut(inner)

def taper(offset=0):
    """Eight native cone patches; same tangent circles, no Loft or fitted spline."""
    start_offset=SLOPE*(RB-YBACK)+offset
    front_offset=SLOPE*(RB-YFRONT)+offset
    faces=[cq.Face.makeFromWires(wire(YFRONT,front_offset)),
           cq.Face(TopoDS.Face_s(cq.Face.makeFromWires(wire(YBACK,start_offset)).wrapped.Reversed()))]
    for a in ARCS:
        ax=gp_Ax3(gp_Pnt(float(a.center[0]),YBACK,float(a.center[1])),gp_Dir(0,-1,0),gp_Dir(math.cos(a.a0),0,math.sin(a.a0)))
        cone=Geom_ConicalSurface(ax,ALPHA,a.radius+start_offset)
        u0=0.;u1=a.a1-a.a0
        f=BRepBuilderAPI_MakeFace(cone,u0,u1,0.,(YBACK-YFRONT)/math.cos(ALPHA),1e-7).Face()
        faces.append(cq.Face(f))
    return base.pure_solid(base.solid_from_faces(faces))

def opening_offset(p):
    # Closest point on the finite circle arcs, including tangency endpoints.
    candidates=[]
    for a in ARCS:
        q=p-a.center;ang=math.atan2(q[1],q[0])
        mid=(a.a0+a.a1)/2
        ang+=2*math.pi*round((mid-ang)/(2*math.pi))
        clipped=np.clip(ang,a.a0,a.a1)
        normal=np.array([math.cos(clipped),math.sin(clipped)])
        foot=a.center+a.radius*normal
        delta=p-foot
        candidates.append((float(np.linalg.norm(delta)),float(np.dot(delta,normal))))
    return min(candidates,key=lambda pair:pair[0])[1]

def lip_edges(shape,radius):
    found=[]
    for e in shape.Edges():
        p=np.array([v.toTuple() for v in e.positions(np.linspace(0,1,9))])
        radii=np.hypot(p[:,1],p[:,2])
        offsets=np.array([opening_offset(q[[0,2]]) for q in p])
        if np.max(np.abs(radii-radius))<.003 and 8<np.median(offsets)<55:
            found.append(e)
    return found

def signature(face):
    c=face.Center();return [face.Area(),c.x,c.y,c.z]
def fillet(shape,radius,edges):
    assert edges,("No fillet edges",radius)
    op=BRepFilletAPI_MakeFillet(shape.wrapped)
    for e in edges:op.Add(radius,e.wrapped)
    op.Build()
    assert op.IsDone(),("Fillet operation failed",radius,len(edges))
    body=base.pure_solid(cq.Shape.cast(op.Shape()).fix())
    generated=[]
    for e in edges:
        for s in op.Generated(e.wrapped):
            if s.ShapeType()==TopAbs_FACE:generated.append(signature(cq.Face(TopoDS.Face_s(s))))
    return body,generated

def build_cup():
    inner=taper(0)
    outer=taper(T/math.cos(ALPHA))
    flange=radial_band(RF,RF+T).intersect(prism(CFG["mouth_offset_mm"]+CFG["flange_width_mm"]+CFG["fillet_radius_mm"]))
    walls=outer.cut(inner).intersect(radial_band(RF,RB+T))
    floor=radial_band(RB,RB+T).intersect(outer)
    raw=base.pure_solid(flange.fuse(walls).fuse(floor).cut(inner).cut(prism(0)).fix())
    inner_edges=lip_edges(raw,RF)
    print("R10 selected edges",len(inner_edges),flush=True)
    rounded,signatures=fillet(raw,10.,inner_edges)
    rear_edges=lip_edges(rounded,RF+T)
    print("Rear R7 selected edges",len(rear_edges),flush=True)
    rounded,back_signatures=fillet(rounded,10.-T,rear_edges)
    assert rounded.isValid() and len(rounded.Solids())==1
    # Enforce exact manufacturing symmetry by mirroring one finished quadrant.
    corner_sigs=[s for s in signatures if s[1]>20 and s[3]>50]
    assert corner_sigs,("No upper-right R10 corner surface",signatures)
    quadrant=base.pure_solid(rounded.intersect(cq.Solid.makeBox(500,400,400,cq.Vector(0,1600,0))).fix())
    upper=base.pure_solid(quadrant.fuse(quadrant.mirror("YZ")).fix())
    rounded=base.pure_solid(upper.fuse(upper.mirror("XY")).fix())
    signatures=[[s[0],sx*s[1],s[2],sz*s[3]] for s in corner_sigs for sx in (-1,1) for sz in (-1,1)]
    lip=lip_edges(rounded,RF)
    lip_offsets=[opening_offset(np.array(p.toTuple())[[0,2]]) for e in lip for p in e.positions(np.linspace(0,1,17))]
    outer_offset=CFG["mouth_offset_mm"]+CFG["flange_width_mm"]+CFG["fillet_radius_mm"]
    actual_flange=[outer_offset-max(lip_offsets),outer_offset-min(lip_offsets)]
    assert 40<min(actual_flange) and max(actual_flange)<60,actual_flange
    return rounded,signatures,{"actual_flange_width_range_mm":actual_flange,"inner_fillet_radius_mm":10.,"outer_parallel_radius_mm":10.-T,
       "inner_edge_count":len(inner_edges),"outer_edge_count":len(rear_edges),
       "skin_construction":"concentric cylindrical flange + eight native tangent cones + drilled analytical opening",
       "wall_generator_slope":SLOPE,"wall_generator_angle_deg":math.degrees(ALPHA)}

def to_world(shape,frame):
    return shape.rotate((0,0,0),(1,0,0),math.degrees(frame.theta)).translate((frame.cx,frame.cy,frame.cz))

def export(shape,name):
    path=CAD/(name+".step");shape=base.pure_solid(shape)
    cq.exporters.export(shape,str(path),exportType="STEP")
    result=base.pure_solid(cq.importers.importStep(str(path)).val())
    assert result.isValid() and len(result.Solids())==1
    return result

def analytic_check():
    data=[];gap=[];tangent=[]
    for a,b in zip(ARCS,ARCS[1:]+ARCS[:1]):
        qa,da,_=a.data(1);qb,db,_=b.data(0)
        gap.append(float(np.linalg.norm(qa-qb)))
        tangent.append(float(math.acos(np.clip(np.dot(da,db)/np.linalg.norm(da)/np.linalg.norm(db),-1,1))))
    w=wire(0)
    cq.exporters.export(w,str(CAD/"opening_contour.step"),exportType="STEP")
    rw=cq.Wire.assembleEdges(cq.importers.importStep(str(CAD/"opening_contour.step")).val().Edges())
    assert rw.IsClosed()
    radii=[]
    for e in rw.Edges():
        assert e.geomType()=="CIRCLE",e.geomType()
        radii.append(float(e._geomAdaptor().Circle().Radius()))
    bb=base.exact_bounds(cq.Compound.makeCompound([rw]))
    assert np.allclose([bb.xlen,bb.zlen],[247,345],atol=1e-6)
    assert np.allclose(sorted(radii),[90]*4+[350]*2+[1000]*2,atol=1e-6)
    assert max(gap)<1e-8 and max(tangent)<1e-7
    pts=np.vstack([np.array([a.data(t)[0] for t in np.linspace(0,1,101)]) for a in ARCS])
    fig,ax=plt.subplots(figsize=(8,10),facecolor="white")
    colors={90:"#e48b28",350:"#2482b5",1000:"#248c62"}
    seen=set()
    for a in ARCS:
        q=np.array([a.data(t)[0] for t in np.linspace(0,1,201)])
        ax.plot(q[:,0],q[:,1],color=colors[a.radius],lw=2,label=f"R{a.radius:g}" if a.radius not in seen else None)
        seen.add(a.radius);ax.scatter(*q[0],s=20,color="black")
        data.append({"center_mm":a.center.tolist(),"radius_mm":a.radius,"angles_rad":[a.a0,a.a1]})
    ax.axhline(0,color="#888",ls="--",lw=.7);ax.axvline(0,color="#888",ls="--",lw=.7)
    ax.annotate("",(-123.5,195),(123.5,195),arrowprops={"arrowstyle":"<->"})
    ax.text(0,202,"247 мм",ha="center")
    ax.annotate("",(150,-172.5),(150,172.5),arrowprops={"arrowstyle":"<->"})
    ax.text(158,0,"345 мм",rotation=90,va="center")
    ax.set_aspect("equal");ax.set_xlabel("X, мм");ax.set_ylabel("Z, мм");ax.grid(alpha=.12);ax.legend()
    ax.set_title("Аналитический контур иллюминатора / восемь дуг / G1")
    fig.tight_layout();fig.savefig(OUT/"opening_analytic_front.png",dpi=160);plt.close(fig)
    return {"dimensions_mm":[bb.xlen,bb.zlen],"radii_mm":sorted(radii),"closed":True,
       "max_join_gap_mm":max(gap),"max_tangent_angle_rad":max(tangent),
       "two_symmetry_axes":True,"arcs":data,"dimension_convention":"orthogonal XZ projection, not cylindrical development"}


def opening_actual_compare(cup):
    curves=[];errors=[]
    for e in cup.Edges():
        p=np.array([v.toTuple() for v in e.positions(np.linspace(0,1,201))])
        if np.ptp(p[:,[0,2]],axis=0).max()<.2:continue
        err=[abs(opening_offset(q[[0,2]])) for q in p]
        if p[:,1].min()>RB-30:
            print("Opening edge",e.geomType(),"XZ min/max",p[:,[0,2]].min(0),p[:,[0,2]].max(0),"error",max(err),"mid",p[100],flush=True)
        if max(err)<.003 and p[:,1].min()>RB-30:
            curves.append(p[:,[0,2]]);errors.extend(err)
    assert len(curves)>=8,("Actual hole edges not found",len(curves))
    pts=np.vstack(curves)
    assert np.allclose(np.ptp(pts,axis=0),[247,345],atol=.003),np.ptp(pts,axis=0)
    fig,ax=plt.subplots(figsize=(8,10),facecolor="white")
    for i,q in enumerate(curves):
        ax.plot(q[:,0],q[:,1],color="#2563eb",lw=3,label="Контур отверстия из STEP чашки" if i==0 else None)
    for i,a in enumerate(ARCS):
        q=np.array([a.data(t)[0] for t in np.linspace(0,1,301)])
        ax.plot(q[:,0],q[:,1],color="#dc2626",lw=1.2,ls="--",label="Аналитический контур по размерам эскиза" if i==0 else None)
    ax.axhline(0,color="#999",ls=":",lw=.7);ax.axvline(0,color="#999",ls=":",lw=.7)
    ax.set_aspect("equal");ax.grid(alpha=.15);ax.legend(fontsize=8)
    ax.set_xlabel("X, мм");ax.set_ylabel("Z, мм")
    ax.set_title("Фронтально: отверстие реальной чашки и размерный эскиз")
    fig.text(.04,.015,"Красный контур восстановлен по247×345 и R1000/R350/R90; это не растровая трассировка.",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(OUT/"opening_sketch_overlay.png",dpi=160);plt.close(fig)
    return {"actual_STEP_projected_edges":len(curves),"max_circle_offset_residual_mm":max(errors),
            "projected_dimensions_mm":np.ptp(pts,axis=0).tolist(),"reference":"analytic reconstruction of supplied dimension sketch"}

def mount_handle(canonical,frame):
    station=base.P["shade_closed_lower_edge_mm"]+3
    a=frame.theta+station/R;r=R-.05
    h=canonical.rotate((0,0,0),(1,0,0),math.degrees(a)).translate((frame.cx,frame.cy+r*math.cos(a),frame.cz+r*math.sin(a)))
    land=cylindrical_slab(rounded_rectangle(20,2,.4,center_s=station+2),frame,R-1.5,1.5,R)
    return base.pure_solid(h.fuse(land).fix())

def plot_handle(handle):
    vertices,tri=handle.tessellate(.025,.06);vv=np.array([p.toTuple() for p in vertices]);tri=np.asarray(tri)
    from shapely.geometry import Polygon
    from shapely import union_all
    fig,axs=plt.subplots(1,3,figsize=(15,5),facecolor="white")
    old=cq.importers.importStep(str(REPO/"handle-revision-v4/cad/handle.step")).val()
    ov,ot=old.tessellate(.025,.06);ov=np.array([p.toTuple() for p in ov]);ot=np.asarray(ot)
    metrics={}
    for ax,axes,label,scale,kind in zip(axs,[[0,2],[1,2],[0,1]],["Спереди / 120×15","Сбоку / 30×15","Сверху / 120×30"],[[1.2,.6],[1.,.6],[1.2,1.]],["front","side","top"]):
        def outline(points,ids):
            p=points[:,axes].copy()
            if axes==[1,2]:p[:,0]*=-1
            if axes==[0,1]:p[:,1]*=-1
            u=union_all([Polygon(p[t]) for t in ids if Polygon(p[t]).area>1e-7],grid_size=.002)
            return max(u.geoms,key=lambda x:x.area) if u.geom_type=="MultiPolygon" else u
        newp=outline(vv,tri);oldp=outline(ov,ot)
        from shapely.affinity import scale as poly_scale
        reference=poly_scale(oldp,xfact=scale[0],yfact=scale[1],origin=(0,0))
        metrics[kind]=float(newp.boundary.hausdorff_distance(reference.boundary))
        q=np.array(newp.exterior.coords);a=np.array(reference.exterior.coords)
        ax.plot(*q.T,color="#2563eb",lw=2,label="Новый STEP")
        ax.plot(*a.T,color="#dc2626",lw=1,ls="--",label="Предыдущая форма × масштаб")
        if kind=="top":ax.invert_yaxis()
        ax.set_title(label);ax.set_aspect("equal");ax.grid(alpha=.18);ax.legend(fontsize=8)
    fig.suptitle("Ручка: сохранение формы при изменении размеров")
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/"handle_three_views.png",dpi=160);plt.close(fig)
    return metrics

def save_img(img,name,title,note):
    fig,ax=plt.subplots(figsize=(10,9),facecolor="white");ax.imshow(img);ax.axis("off");ax.set_title(title)
    fig.text(.025,.015,note,fontsize=9);fig.tight_layout(rect=(0,.04,1,1));fig.savefig(OUT/name,dpi=150);plt.close(fig)

def colored_mesh(path,signatures):
    mesh=load_mesh(path,[245,239,225])
    shape=cq.importers.importStep(str(path)).val()
    BRepMesh_IncrementalMesh(shape.wrapped,.12,False,.08,True)
    shape.tessellate(.2,.12)
    offset=0;colors=mesh[3].copy();ids=[]
    for face in shape.Faces():
        loc=TopLoc_Location();poly=BRep_Tool.Triangulation_s(face.wrapped,loc)
        if poly is None:raise RuntimeError("Face triangulation missing")
        n=poly.NbNodes();sig=np.array(signature(face))
        hit=any(abs(sig[0]-s[0])<.1 and np.linalg.norm(sig[1:]-np.array(s[1:]))<.01 for s in signatures)
        if hit:colors[offset:offset+n]=[255,110,15];ids.append(face)
        offset+=n
    assert offset==len(colors),(offset,len(colors))
    return (*mesh[:3],colors),ids

def section_check(cup,signatures):
    part=cq.Workplane(cq.Plane(origin=(0,0,0),xDir=(0,1,0),normal=(1,0,0))).newObject([cup]).section().val()
    lines=[];circle_radii=[];fitted_radii=[]
    fig,ax=plt.subplots(figsize=(12,7),facecolor="white")
    for e in part.Edges():
        p=np.array([v.toTuple() for v in e.positions(np.linspace(0,1,101))])
        ax.plot(p[:,2],p[:,1]-RF,color="#466d91",lw=1.5)
        if e.geomType()=="CIRCLE":
            rr=e._geomAdaptor().Circle().Radius();circle_radii.append(float(rr))
            if abs(rr-10)<.03:ax.plot(p[:,2],p[:,1]-RF,color="#e67d1c",lw=3,label="R10")
        if e.geomType()!="CIRCLE" and len(p)>5:
            q=p[:,[2,1]];origin=q.mean(0);qq=q-origin
            A=np.column_stack((2*qq[:,0],2*qq[:,1],np.ones(len(q))))
            cx,cy,c=np.linalg.lstsq(A,np.sum(qq*qq,axis=1),rcond=None)[0]
            rad=math.sqrt(max(0,c+cx*cx+cy*cy))
            residual=np.max(np.abs(np.linalg.norm(qq-[cx,cy],axis=1)-rad))
            if residual<.001 and 5<rad<20:
                fitted_radii.append({"radius_mm":rad,"max_circle_residual_mm":float(residual)})
                if abs(rad-10)<.03:ax.plot(p[:,2],p[:,1]-RF,color="#e67d1c",lw=3,label="R10 (сечение)")
        lines.append(p.tolist())
    ax.set_aspect("equal");ax.grid(alpha=.18);ax.set_xlabel("Высота от центра, мм");ax.set_ylabel("Радиальное направление относительно Rфланца, мм")
    ax.set_title("Сечение X=0: цилиндрический фланец, сформованная стенка и R10")
    ax.legend();fig.tight_layout();fig.savefig(OUT/"cup_section.png",dpi=150);plt.close(fig)
    (OUT/"cup_section_curves.json").write_text(json.dumps(lines))
    
    fig,ax=plt.subplots(figsize=(10,6),facecolor="white")
    flange_lengths=[]
    for e in part.Edges():
        p=np.array([v.toTuple() for v in e.positions(np.linspace(0,1,101))])
        ax.plot(p[:,2],p[:,1]-RF,color="#466d91",lw=1.5)
        if e.geomType()=="CIRCLE":
            rr=float(e._geomAdaptor().Circle().Radius())
            if abs(rr-10)<.03:ax.plot(p[:,2],p[:,1]-RF,color="#e67d1c",lw=3)
            if abs(rr-RF)<.003 and p[:,2].min()>150:
                length=float(e.Length());flange_lengths.append(length)
                q=p[len(p)//2]
                ax.annotate(f"Фланец {length:.1f} мм по дуге цилиндра",(q[2],q[1]-RF),
                            (q[2]-35,q[1]-RF+30),arrowprops={"arrowstyle":"->"},fontsize=10)
    assert flange_lengths and all(40<l<60 for l in flange_lengths),flange_lengths
    ax.set_xlim(155,275);ax.set_ylim(-35,70);ax.set_aspect("equal");ax.grid(alpha=.2)
    ax.set_xlabel("Z, мм");ax.set_ylabel("Y − Rфланца, мм")
    ax.set_title("Увеличенное сечение: фланец около50мм и переходR10")
    fig.tight_layout();fig.savefig(OUT/"cup_section_flange_R10.png",dpi=160);plt.close(fig)

    cq.exporters.export(part,str(CAD/"cup_section.step"),exportType="STEP")
    assert (any(abs(r-10)<.03 for r in circle_radii) or any(abs(r["radius_mm"]-10)<.03 for r in fitted_radii)),("R10 not verified in section",circle_radii)
    return {"native_circle_radii_in_section_mm":circle_radii,"fitted_circle_radii_in_section":fitted_radii,"R10_section_verified":True,"upper_flange_arc_lengths_mm":flange_lengths}

def common(a,b):
    v=float(a.intersect(b).Volume());return max(0.,v)

def main():
    original=protected();report={"input_commit":"7ac8f70e035a7864f45363456c2ee388d4d0ca76","analytic_opening":analytic_check()}
    print("Opening verified",report["analytic_opening"],flush=True)
    local,blend_sig,cupmeta=build_cup();local=export(local,"cup_local")
    report["cup"]=cupmeta;report["cup"]["section"]=section_check(local,blend_sig)
    report["cup"]["dimension_sketch_comparison"]=opening_actual_compare(local)
    for plane in ("YZ","XY"):
        error=float(local.cut(local.mirror(plane)).Volume())
        assert error<.02,(plane,error)
    report["cup"]["two_axis_symmetry_volume_error_mm3"]=[float(local.cut(local.mirror(p)).Volume()) for p in ("YZ","XY")]
    panel=cq.importers.importStep(str(base.SOURCE)).val();frames=base.frames_from_panel(panel)
    cups=[];handles=[]
    old=cq.importers.importStep(str(REPO/"handle-revision-v4/cad/handle.step")).val()
    matrix=cq.Matrix([[1.2,0,0,0],[0,1,0,0],[0,0,.6,0],[0,0,0,1]])
    handle=export(old.transformGeometry(matrix),"handle")
    dims=base.exact_bounds(handle);assert np.allclose([dims.xlen,dims.ylen,dims.zlen],[120,30,15],atol=.001)
    inv=handle.transformGeometry(cq.Matrix([[1/1.2,0,0,0],[0,1,0,0],[0,0,1/.6,0],[0,0,0,1]]))
    assert abs(handle.Volume()/old.Volume()-.72)<1e-5
    assert float(old.cut(inv).Volume())<.002
    report["handle"]={"dimensions_mm":[dims.xlen,dims.ylen,dims.zlen],"affine_scale":[1.2,1,.6],
        "previous_geometry_preserved":True,"normalized_projection_errors_mm":plot_handle(handle),"mount_offset_mm":3}
    for i,frame in enumerate(frames,1):
        cups.append(export(to_world(local,frame),f"cup_{i}"))
        handles.append(mount_handle(handle,frame))
    assemblies=[]
    for state,travel in [("open",380),("half",190),("closed",0)]:
        asm=cq.Assembly(name="Tu214_Window_v5_"+state);asm.add(panel,name="Wall_Panel")
        checks=[]
        for i,(frame,cup,h0) in enumerate(zip(frames,cups,handles),1):
            shade=cq.importers.importStep(str(BASE/f"cad/shade_{i}_{state}.step")).val()
            h=export(move_on_cylinder(h0,frame,travel,R),f"handle_{i}_{state}")
            shutil.copy2(BASE/f"cad/shade_{i}_{state}.step",CAD/f"shade_{i}_{state}.step")
            overlaps={"panel_cup":common(panel,cup),"shade_cup":common(shade,cup),"handle_cup":common(h,cup),"panel_handle":common(panel,h),"shade_handle":common(shade,h)}
            assert max(overlaps.values())<.01,(i,state,overlaps)
            checks.append({"window":i,"overlaps_mm3":overlaps})
            asm.add(cup,name=f"Cup_{i}");asm.add(shade,name=f"Shade_{i}");asm.add(h,name=f"Handle_{i}")
        path=CAD/f"assembly_{state}.step";asm.export(str(path),exportType="STEP")
        kept=preserve_panel(base.SOURCE,path);restored=cq.importers.importStep(str(path)).val()
        assert restored.isValid() and len(restored.Solids())==7
        assemblies.append({"state":state,**kept,"checks":checks})
    report["assemblies"]=assemblies
    # Native front cup and assembly images, with R10 marked for visibility checks.
    cm,filletfaces=colored_mesh(CAD/"cup_local.step",blend_sig)
    assert filletfaces,("R10 faces lost after STEP",blend_sig)
    localfront=rasterize(cm,900,1120,[0,-1,0],target=(0,RF,0),physical_size=(450,550))
    save_img(localfront,"cup_front_R10.png","Чашка / фронтально / R10 выделен оранжевым","Проём247×345, R1000/R350/R90; передний фланец цилиндрический.")
    fm=load_mesh(base.SOURCE,[222,218,208]);shade=load_mesh(CAD/"shade_2_open.step",[166,163,147]);hm=load_mesh(CAD/"handle_2_open.step",[230,211,165])
    # World signatures preserve face areas and transform centers.
    frame=frames[1]
    ws=[]
    for sig in blend_sig:
        center=to_world(cq.Vertex.makeVertex(*sig[1:]),frame).Center()
        ws.append([sig[0],center.x,center.y,center.z])
    cm_world,world_faces=colored_mesh(CAD/"cup_2.step",ws)
    assert world_faces
    scene=merge([fm,cm_world,shade,hm])
    front=rasterize(scene,1000,1250,-frame.n,frame.origin,physical_size=(400,500))
    save_img(front,"assembly_front_R10.png","Сборка / ортогонально / видимость R10 в углах","Оранжевый — переходR10. Панель и положение шторки исходные.")
    # Segmentation uses black fillet faces, bright all other geometry.
    def recolor(mesh,c):return (*mesh[:3],np.broadcast_to(np.array(c,float),mesh[0].shape).copy())
    segcolors=np.full(cm_world[3].shape,240.)
    orange=np.all(cm_world[3]==[255,110,15],axis=1);segcolors[orange]=0
    segcup=(*cm_world[:3],segcolors)
    segscene=merge([recolor(fm,[240]*3),segcup,recolor(shade,[240]*3),recolor(hm,[240]*3)])
    im=rasterize(segscene,1000,1250,-frame.n,frame.origin,physical_size=(400,500))
    mask=np.all(im<20,axis=2);yy,xx=np.where(mask)
    counts={}
    for name,sel in [("upper_left",(xx<500)&(yy<625)),("upper_right",(xx>500)&(yy<625)),
                     ("lower_left",(xx<500)&(yy>625)),("lower_right",(xx>500)&(yy>625))]:
        counts[name]=int(sel.sum())
    assert all(v>20 for v in counts.values()),("R10 corners invisible",counts)
    report["cup"]["R10_visible_pixels_assembly_front"]=counts
    natural_scene=merge([fm,recolor(cm_world,[245,239,225]),shade,hm])
    natural=rasterize(natural_scene,1000,1250,-frame.n,frame.origin,physical_size=(400,500))
    save_img(natural,"assembly_front.png","Оконный узел / ортогональный фронтальный вид","Проверка скруглений в сборке, а не только на отдельной чашке.")
    view=.25*np.array([1.,0.,0.])-.97*frame.n+.13*frame.up
    perspective=rasterize(natural_scene,1050,1300,view,frame.origin,physical_size=(400,500),perspective=True,distance=1450)
    save_img(perspective,"assembly_perspective.png","Ту-214 / новая чашка и ручка120×30×15","Исходная панель сохранена; положение самой шторки не менялось.")
    
    # Qualitative comparison with a readable repository photograph.
    photo=np.asarray(Image.open(REPO/"tu214-sidewall/references/18205_marked.png").convert("RGB"))
    fig,axs=plt.subplots(1,2,figsize=(13,7),facecolor="white")
    axs[0].imshow(photo);axs[0].set_xlim(965,1294);axs[0].set_ylim(790,270)
    axs[0].set_title("Фото из репозитория / перспективный ракурс")
    axs[1].imshow(natural);axs[1].set_title("Новый STEP / ортогональная сборка")
    for ax in axs:ax.axis("off")
    fig.text(.02,.025,"Качественное сравнение: переходы чашки и компактная ручка. Фото обрезано справа и не используется как обмер.",fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(OUT/"photo_assembly_compare.png",dpi=160);plt.close(fig)
    report["photograph_comparison"]={"repository_photo":"tu214-sidewall/references/18205_marked.png",
        "current_chat_photo_visually_read":True,"quantitative_photo_measurement":False}
    report["cup"]["fillet_face_signatures_local"]=blend_sig
    assert original==protected(),"Baseline files changed"
    report["source_files_unchanged"]=True;report["source_sha256"]=original;report["all_checks_passed"]=True
    (ROOT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    readme=f"""# Ту-214: аналитическая чашка и ручка120×30×15

Проём:247×345мм в ортогональной фронтальной проекции, восемь нативных окружностей R1000/R350/R90. Центры и углы вычислены по внутренней касательности; сплайновые переходы в контуре не применяются. [Проверка](report.json), [контурSTEP](cad/opening_contour.step), [чертёж](previews/opening_analytic_front.png).

Чашка строится из концентричного цилиндрического фланцаR{RF:g}, восьми нативных конусов с согласованными касательными и цилиндрического днаR{RB:g}, с последующим вырезанием точного проёма. Это аналитическая конструкция формованного углубления; произвольный Loft не применяется. Передняя поверхность плавно сопряжена R10; параллельная задняя сторона R7 сохраняет толщину3мм. Фланец приблизительно50мм по наружному нормальному направлению от устья; внешняя кромка скрывается за исходной панелью. [Сечение](previews/cup_section.png), [фланец и R10 крупно](previews/cup_section_flange_R10.png).\n\n![Отверстие чашки и размерный эскиз](previews/opening_sketch_overlay.png)

![Чашка и углыR10 в сборке](previews/assembly_front_R10.png)

Ручка — непосредственное преобразование ранее принятого STEP с коэффициентами(1.2,1,0.6). Габариты120×30×15мм; изменение геометрической концепции исключено, сохранение формы проверено обратным преобразованием и тремя проекциями. Посадка+3мм относительно нижнего края шторки; сама шторка не перемещалась.

![Ручка](previews/handle_three_views.png)
![Сборка](previews/assembly_perspective.png)

ВершиныR10 действительно видны при ортогональном рендере исходной панели и всех деталей. Числа видимых пикселей каждого угла находятся вreport.json. [Фото и сборка](previews/photo_assembly_compare.png) используют доступный файл tu214-sidewall/references/18205_marked.png. Последняя фотография и размерный эскиз пользователя визуально прочитаны в чате; имена/растровые байты вложений не переданы, поэтому их оригиналы не включены. Геометрия проверена по точным размерам эскиза, симметрии и касательности; визуальное сравнение с фото качественное.

Исходные CAD-файлы и панель не перезаписаны; их SHA-256 совпадает. В сборочныхSTEP сохранены исходные BRep-записи панели. [Открытая сборка](cad/assembly_open.step), [Чашка1](cad/cup_1.step), [Чашка2](cad/cup_2.step), [Ручка](cad/handle.step).

Проверки перед экспортом:валидность BRep, повторный импорт, радиусы8дуг, касательность, симметрия, круговоеR10всечении, видимость всех4углов в сборке и пересечения в open/half/closed. Проверка движения между этими состояниями отдельно не выполнена. Воспроизведение: python -m pip install -r requirements.txt; python build.py.
"""
    (ROOT/"README.md").write_text(readme)
    manifest={str(p.relative_to(ROOT)):{"bytes":p.stat().st_size,"sha256":digest(p)} for folder in (CAD,OUT) for p in folder.iterdir() if p.is_file()}
    (ROOT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    with zipfile.ZipFile(ROOT/"window_cup_handle_STEP.zip","w",zipfile.ZIP_DEFLATED) as z:
        for folder in (CAD,OUT):
            for p in folder.iterdir():z.write(p,str(p.relative_to(ROOT)))
        for name in ("README.md","report.json","parameters.json","manifest.json"):z.write(ROOT/name,name)
    print("COMPLETE",flush=True)

if __name__=="__main__":main()
