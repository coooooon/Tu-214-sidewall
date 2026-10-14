"""Local BRep surgery on v5 cup; rigidly move the existing mounted handles."""
from pathlib import Path
import importlib.util,sys,math,json,hashlib,shutil,zipfile,os
os.environ.setdefault("MPLCONFIGDIR","/tmp/tu214-v6-mpl")
import numpy as np
import cadquery as cq
from scipy.interpolate import make_interp_spline,BSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from OCP.Geom import Geom_BSplineSurface
from OCP.TColgp import TColgp_Array2OfPnt
from OCP.gp import gp_Pnt
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace,BRepBuilderAPI_MakeEdge
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut,BRepAlgoAPI_Common,BRepAlgoAPI_Fuse
from OCP.TopTools import TopTools_ListOfShape
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeOffsetShape
from OCP.BRepOffset import BRepOffset_Skin
from OCP.GeomAbs import GeomAbs_Intersection,GeomAbs_Arc
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopAbs import TopAbs_REVERSED
from OCP.BRepLib import BRepLib_ToolTriangulatedShape
from OCP.BRep import BRep_Tool
from OCP.TopLoc import TopLoc_Location
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent;OLD=REPO/"window-revision-v5";CAD=ROOT/"cad";OUT=ROOT/"previews"
for p in (CAD,OUT):p.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location("v5",OLD/"build.py");v5=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v5;spec.loader.exec_module(v5)
from window_source import array
from cylindrical_parts import move_on_cylinder
from preview_mesh import load_mesh,merge,rasterize
from preserve_step_panel import preserve_panel
base=v5.base;RF=v5.RF;RB=v5.RB;T=v5.T;R=v5.R
D=json.loads((ROOT/"diagnostic.json").read_text());ARCS=v5.ARCS
CFG={"baseline_commit":"4db64ea3848f3ec188f8cd6854606298099a5fad","mount_gap_mm":7.,
     "corner_follow_clearance_mm":5.0,"corner_sector_margin_rad":.04,
     "local_depth_limit_radius_mm":RF+43.,"blend_y_start_mm":RF-10.,"blend_y_end_mm":RF+25.}
(ROOT/"parameters.json").write_text(json.dumps(CFG,indent=2)+"\n")
C=ARCS[3].center;A0=ARCS[3].a0;SPAN=ARCS[3].a1-A0
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def protected():
    return {str(p.relative_to(REPO)):sha(p) for p in list(OLD.rglob("*.step"))+[base.SOURCE]}
def sample_parts(parts):return [np.array(x)[:,[0,2]] for x in parts]
def ray_hit(parts,a):
    n=np.array([math.cos(a),math.sin(a)]);hits=[]
    for p in parts:
        q=p[:-1];d=p[1:]-q;w=q-C
        den=n[0]*d[:,1]-n[1]*d[:,0]
        good=np.abs(den)>1e-10
        ts=np.zeros(len(d));ks=ts.copy()
        ts[good]=(w[good,0]*d[good,1]-w[good,1]*d[good,0])/den[good]
        ks[good]=(w[good,0]*n[1]-w[good,1]*n[0])/den[good]
        sel=good&(ts>30)&(ks>=-1e-9)&(ks<=1+1e-9)
        hits.extend(ts[sel].tolist())
    assert hits,("Ray missed contour",a)
    return min(hits)
panel_parts=sample_parts(D["panel_throat_contours"][0])
old_lip=sample_parts(D[f"cup_lip_{RF}"])
def smooth(t):return np.where(t<=0,0,np.where(t>=1,1,10*t**3-15*t**4+6*t**5))
u=np.linspace(0,1,121)
window=smooth((u-.025)/.15)*smooth((.975-u)/.15)
delta=np.maximum(0.,np.array([ray_hit(panel_parts,A0+SPAN*t)-ray_hit(old_lip,A0+SPAN*t)+CFG["corner_follow_clearance_mm"] for t in u]))*window
SHIFT=make_interp_spline(u,delta,k=5,bc_type=([(1,0.),(2,0.)],[(1,0.),(2,0.)]))
(ROOT/"corner_shift.json").write_text(json.dumps({"u":u.tolist(),"delta_mm":delta.tolist(),"window":window.tolist()},indent=2))
# One-sided C profile: cubic decay to the original wall with C2 continuity.\n# A quadratic continuation lies ahead of the physical flange and closes the tool solid.
Y0=v5.YBACK;Y1=v5.YFRONT;DY=Y1-Y0
ve=(CFG["blend_y_end_mm"]-Y0)/DY;vs=(CFG["blend_y_start_mm"]-Y0)/DY
vknots=np.array([0.]*6+[ve]*3+[vs]*3+[1.]*6)
nv=len(vknots)-6;greville=np.array([vknots[i+1:i+6].mean() for i in range(nv)])
def gfun(v):
    yy=Y0+DY*v;end=CFG["blend_y_end_mm"];front=CFG["blend_y_start_mm"];a=end-front
    t=yy-front
    return np.where(yy>=end,0.,np.where(yy>=front,(end-yy)**3,a**3-3*a*a*t+3*a*t*t))
def depth_scale(ref):
    phi=A0+SPAN*ref;d=max(0.,float(SHIFT(ref)))
    b=v5.SLOPE*math.sin(phi)
    k=C[1]+(90.+v5.SLOPE*RB+d)*math.sin(phi)
    mouth_y=(k*b+math.sqrt((1+b*b)*RF*RF-k*k))/(1+b*b)
    return (CFG["blend_y_end_mm"]-mouth_y)**3
basis=BSpline.design_matrix(greville,vknots,5).toarray()
gcoeff=np.linalg.solve(basis,gfun(greville));yc=Y0+DY*greville
corner_surfaces={};G1_ERRORS=[]
def corner_surface(index):
    a=ARCS[index];rev=index in (1,5);us=np.linspace(0,1,41)
    vals=[]
    for t in us:
        phi=a.a0+(a.a1-a.a0)*t;n=np.array([math.cos(phi),math.sin(phi)])
        ref=1-t if rev else t;sh=float(SHIFT(ref))/depth_scale(ref)
        rad=a.radius+v5.SLOPE*(RB-yc)+sh*gcoeff
        vals.append(np.column_stack((a.center[0]+rad*n[0],yc,a.center[1]+rad*n[1])))
    vals=np.array(vals);sp=a.a1-a.a0
    def derivatives(t,order):
        phi=a.a0+sp*t;n=np.array([math.cos(phi),math.sin(phi)])
        rad=a.radius+v5.SLOPE*(RB-yc)
        if order==1:return np.column_stack((-rad*sp*n[1],np.zeros(nv),rad*sp*n[0]))
        return np.column_stack((-rad*sp*sp*n[0],np.zeros(nv),-rad*sp*sp*n[1]))
    spl=make_interp_spline(us,vals,k=5,bc_type=([(1,derivatives(0,1)),(2,derivatives(0,2))],[(1,derivatives(1,1)),(2,derivatives(1,2))]))
    uk,um=np.unique(spl.t,return_counts=True);vk,vm=np.unique(vknots,return_counts=True)
    poles=TColgp_Array2OfPnt(1,len(spl.c),1,nv)
    for i in range(len(spl.c)):
        for j in range(nv):poles.SetValue(i+1,j+1,gp_Pnt(*spl.c[i,j]))
    from OCP.TColStd import TColStd_Array1OfInteger
    s=Geom_BSplineSurface(poles,array(uk.tolist()),array(vk.tolist()),array(um.tolist(),TColStd_Array1OfInteger),array(vm.tolist(),TColStd_Array1OfInteger),5,5,False,False)
    from OCP.gp import gp_Vec
    for t in (0.,1.):
        ph=a.a0+sp*t
        expected=np.array([math.cos(ph),v5.SLOPE,math.sin(ph)]);expected/=np.linalg.norm(expected)
        for v in np.linspace(0,1,17):
            p=gp_Pnt();du=gp_Vec();dv=gp_Vec();s.D1(t,float(v),p,du,dv)
            normal=np.cross([du.X(),du.Y(),du.Z()],[dv.X(),dv.Y(),dv.Z()]);normal/=np.linalg.norm(normal)
            G1_ERRORS.append(float(np.linalg.norm(normal-expected)))
    assert max(G1_ERRORS)<1e-7,max(G1_ERRORS)
    corner_surfaces[index]=s;return s
def inner_taper():
    faces=[];caps={0:[],1:[]}
    for i,a in enumerate(ARCS):
        if i%2:
            s=corner_surface(i);faces.append(cq.Face(BRepBuilderAPI_MakeFace(s,1e-7).Face()))
            for v in (0,1):caps[v].append(cq.Edge(BRepBuilderAPI_MakeEdge(s.VIso(float(v)),0.,1.).Edge()))
        else:
            # Native v5 cone patch, unchanged on every main contour segment.
            from OCP.Geom import Geom_ConicalSurface
            from OCP.gp import gp_Ax3,gp_Dir
            ax=gp_Ax3(gp_Pnt(float(a.center[0]),Y0,float(a.center[1])),gp_Dir(0,-1,0),gp_Dir(math.cos(a.a0),0,math.sin(a.a0)))
            s=Geom_ConicalSurface(ax,v5.ALPHA,a.radius+v5.SLOPE*(RB-Y0))
            faces.append(cq.Face(BRepBuilderAPI_MakeFace(s,0.,a.a1-a.a0,0.,(Y0-Y1)/math.cos(v5.ALPHA),1e-7).Face()))
            for v,y in ((0,Y0),(1,Y1)):
                def p(t):
                    ph=a.a0+(a.a1-a.a0)*t;r=a.radius+v5.SLOPE*(RB-y)
                    return cq.Vector(a.center[0]+r*math.cos(ph),y,a.center[1]+r*math.sin(ph))
                caps[v].append(cq.Edge.makeThreePointArc(p(0),p(.5),p(1)))
    for v in (0,1):
        w=cq.Wire.assembleEdges(caps[v]);assert w.IsClosed()
        f=cq.Face.makeFromWires(w)
        if v==0:f=cq.Face(v5.TopoDS.Face_s(f.wrapped.Reversed()))
        faces.append(f)
    return base.pure_solid(base.solid_from_faces(faces))
def build_candidate():
    inside=inner_taper();print("Local cavity valid",inside.isValid(),flush=True)
    outside=None
    for intersect,join in ((False,GeomAbs_Arc),(True,GeomAbs_Intersection),(True,GeomAbs_Arc),(False,GeomAbs_Intersection)):
        op=BRepOffsetAPI_MakeOffsetShape()
        op.PerformByJoin(inside.wrapped,T,1e-5,BRepOffset_Skin,intersect,False,join,False)
        print("Offset attempt",intersect,join,op.IsDone(),op.Shape().IsNull() if op.IsDone() else True,flush=True)
        if op.IsDone() and not op.Shape().IsNull():
            outside=base.pure_solid(cq.Shape.cast(op.Shape()).fix());break
    assert outside is not None,"Normal wall offset failed"
    flange=v5.radial_band(RF,RF+T).intersect(v5.prism(85))
    walls=outside.cut(inside).intersect(v5.radial_band(RF,RB))
    floor=v5.radial_band(RB,RB+T).intersect(outside)
    raw=base.pure_solid(flange.cut(inside).fuse(walls).fuse(floor).cut(v5.prism()).fix())
    es=v5.lip_edges(raw,RF);print("R10 edges",len(es),flush=True)
    result,sigs=v5.fillet(raw,10.,es)
    es=v5.lip_edges(result,RF+T);print("R7 edges",len(es),flush=True)
    result,_=v5.fillet(result,7.,es)
    # Four identical corner patches come from one finished quadrant.
    sigs=[s for s in sigs if s[1]>20 and s[3]>50]
    q=base.pure_solid(result.intersect(cq.Solid.makeBox(500,400,400,cq.Vector(0,1600,0))).fix())
    u=base.pure_solid(q.fuse(q.mirror("YZ")).fix())
    result=base.pure_solid(u.fuse(u.mirror("XY")).fix())
    sigs=[[s[0],sx*s[1],s[2],sz*s[3]] for s in sigs for sx in (-1,1) for sz in (-1,1)]
    return result,sigs
def sew_local_regions(retained,patches):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
    lo=(A0%(2*math.pi))-CFG["corner_sector_margin_rad"]
    hi=lo+SPAN+2*CFG["corner_sector_margin_rad"]
    def is_cut_cap(face):
        if face.geomType()=="CYLINDER":
            ad=face._geomAdaptor().Cylinder()
            return abs(ad.Radius()-CFG["local_depth_limit_radius_mm"])<1e-5 and abs(ad.Axis().Direction().X())>.999
        if face.geomType()!="PLANE":return False
        center=face.Center();q=np.array([abs(center.x)-C[0],abs(center.z)-C[1]])
        if q[0]<-.01 or q[1]<-.01:return False
        return min(abs(q[0]*math.sin(t)-q[1]*math.cos(t)) for t in (lo,hi))<1e-4
    faces=[f for body in (retained,patches) for f in body.Faces() if not is_cut_cap(f)]
    print("Splice faces kept",len(faces),"caps excluded",len(retained.Faces())+len(patches.Faces())-len(faces),flush=True)
    for tol in (1e-4,.001):
        sew=BRepBuilderAPI_Sewing(tol,True,True,True,False)
        for f in faces:sew.Add(f.wrapped)
        sew.Perform();shape=cq.Shape.cast(sew.SewedShape())
        print("Local sewing",tol,"free edges",sew.NbFreeEdges(),"shells",len(shape.Shells()),flush=True)
        if sew.NbFreeEdges()==0 and len(shape.Shells())==1:
            result=cq.Solid.makeSolid(shape.Shells()[0]).fix()
            if result.isValid():return base.pure_solid(result)
    raise RuntimeError("Local patch boundary did not sew")
def masks():
    lo=A0-CFG["corner_sector_margin_rad"];hi=A0+SPAN+CFG["corner_sector_margin_rad"]
    def q(r,a):return cq.Vector(C[0]+r*math.cos(a),1600,C[1]+r*math.sin(a))
    w=cq.Wire.makePolygon([q(0,lo),q(600,lo),q(600,hi)],close=True)
    wedge=cq.Solid.extrudeLinear(w,[],cq.Vector(0,400,0)).intersect(v5.radial_band(RF-.001,CFG["local_depth_limit_radius_mm"]))
    return cq.Compound.makeCompound([wedge,wedge.mirror("YZ"),wedge.mirror("XY"),wedge.mirror("YZ").mirror("XY")])
def export(s,name):
    path=CAD/(name+".step");s=base.pure_solid(s)
    if name.startswith("cup"):
        from OCP.ShapeFix import ShapeFix_FixSmallFace
        small=ShapeFix_FixSmallFace();small.Init(s.wrapped);small.SetPrecision(1e-5);small.SetMaxTolerance(.0001);small.Perform()
        clean=cq.Shape.cast(small.FixShape())
        print("Small-face cleanup",name,clean.isValid(),len(s.Faces()),len(clean.Faces()),flush=True)
        if clean.isValid():s=base.pure_solid(clean)
    cq.exporters.export(s,str(path),exportType="STEP")
    restored=cq.importers.importStep(str(path)).val()
    print("STEP import",name,restored.isValid(),len(restored.Solids()),flush=True)
    if not restored.isValid():
        from OCP.BRepCheck import BRepCheck_Analyzer
        check=BRepCheck_Analyzer(restored.wrapped)
        for i,face in enumerate(restored.Faces()):
            if not check.IsValid(face.wrapped):
                print("Invalid imported face",i,face.geomType(),"area",face.Area(),"center",face.Center().toTuple(),list(check.Result(face.wrapped).Status()),flush=True)
                for edge in face.Edges():
                    try:status=list(check.Result(edge.wrapped).StatusOnShape(face.wrapped))
                    except Exception:status=list(check.Result(edge.wrapped).Status())
                    print("Edge status",edge.geomType(),status,flush=True)
        from OCP.ShapeFix import ShapeFix_Shape,ShapeFix_FixSmallFace
        small=ShapeFix_FixSmallFace();small.Init(restored.wrapped);small.SetPrecision(.0001);small.SetMaxTolerance(.001);small.Perform()
        restored=cq.Shape.cast(small.FixShape())
        print("Imported small-face cleanup",name,restored.isValid(),flush=True)
        from OCP.BRepLib import BRepLib
        healer=ShapeFix_Shape(restored.wrapped);healer.SetPrecision(1e-6);healer.SetMinTolerance(1e-7);healer.SetMaxTolerance(.001);healer.Perform()
        restored=cq.Shape.cast(healer.Shape())
        BRepLib.SameParameter_s(restored.wrapped,1e-6,True)
        print("STEP healed",name,restored.isValid(),flush=True)
        cq.exporters.export(restored,str(path),exportType="STEP")
        restored=cq.importers.importStep(str(path)).val()
        print("STEP reimport after healing",name,restored.isValid(),flush=True)
    return base.pure_solid(restored)
def outside_surface_check(old,new):
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    def clear_outside(p):
        x,y,z=p;x,z=abs(x),abs(z);rad=math.hypot(y,z)
        if rad>CFG["local_depth_limit_radius_mm"]+.1:return True
        if abs(rad-CFG["local_depth_limit_radius_mm"])<.1:return False
        if x<C[0]-.1 or z<C[1]-.1:return True
        q=np.array([x-C[0],z-C[1]]);a=math.atan2(q[1],q[0])
        lo=(A0%(2*math.pi))-CFG["corner_sector_margin_rad"];hi=lo+SPAN+2*CFG["corner_sector_margin_rad"]
        if min(abs(np.linalg.norm(q)*math.sin(a-lo)),abs(np.linalg.norm(q)*math.sin(a-hi)))<.1:return False
        return not lo<a<hi
    checks=[]
    for src,dst in ((old,new),(new,old)):
        BRepMesh_IncrementalMesh(src.wrapped,.02,False,.06,True)
        verts,_=src.tessellate(.02,.06)
        points=np.array([v.toTuple() for v in verts]);points=np.array([p for p in points if clear_outside(p)])
        ids=np.unique(np.linspace(0,len(points)-1,min(256,len(points))).astype(int))
        shell=cq.Compound.makeCompound(dst.Shells())
        distances=[]
        for p in points[ids]:
            op=BRepExtrema_DistShapeShape(cq.Vertex.makeVertex(*p).wrapped,shell.wrapped);op.Perform()
            assert op.IsDone()
            distances.append(float(op.Value()))
        assert distances
        checks.append({"points":len(distances),"max_boundary_distance_mm":max(distances),"rms_boundary_distance_mm":float(np.sqrt(np.mean(np.square(distances))))})
    print("Preserved boundary distances",checks,flush=True)
    assert max(v["max_boundary_distance_mm"] for v in checks)<.001,checks
    return checks
def precise_volume(shape):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    props=GProp_GProps()
    err=BRepGProp.VolumeProperties_s(shape.wrapped,props,1e-10,False,False)
    return float(props.Mass())
def boolean(a,b,kind):
    cls={"cut":BRepAlgoAPI_Cut,"common":BRepAlgoAPI_Common,"fuse":BRepAlgoAPI_Fuse}[kind]
    op=cls();args=TopTools_ListOfShape();args.Append(a.wrapped);ops=TopTools_ListOfShape();ops.Append(b.wrapped)
    op.SetArguments(args);op.SetTools(ops);op.SetFuzzyValue(1e-7);op.SetNonDestructive(True);op.SetRunParallel(True);op.Build()
    assert op.IsDone(),("Boolean failed",kind)
    assert not op.Shape().IsNull(),("Null Boolean result",kind)
    return cq.Shape.cast(op.Shape())
def cut_volume(a,b):
    try:return max(0.,float(a.cut(b).Volume()))
    except ValueError as exc:
        if "Null TopoDS" not in str(exc):raise
        shared=float(a.intersect(b).Volume())
        residual=abs(float(a.Volume())-shared)
        assert residual<.001,("Null difference with nonzero residual",residual)
        return residual
def outside_difference(a,b,region):
    try:diff=a.cut(b)
    except ValueError as exc:
        if "Null TopoDS" not in str(exc):raise
        return cut_volume(a,b)
    volume=max(0.,float(diff.Volume()))
    if volume<1e-8:return 0.
    within=max(0.,float(diff.intersect(region).Volume()))
    return max(0.,volume-within)
def contact_data(shape,frame):
    pts=[]
    for f in shape.Faces():
        if f.geomType()!="CYLINDER":continue
        if abs(f._geomAdaptor().Cylinder().Radius()-R)>.001:continue
        for e in f.Edges():pts.extend(v.toTuple() for v in e.positions(np.linspace(0,1,101)))
    assert pts,"No shade/land surface at R1761"
    p=np.array(pts);s=R*(np.arctan2(p[:,2]-frame.cz,p[:,1]-frame.cy)-frame.theta)
    return p,np.column_stack((p[:,0]-frame.cx,s))
def painted(path,sigs=None,flat=False):
    shape=cq.importers.importStep(str(path)).val()
    BRepMesh_IncrementalMesh(shape.wrapped,.12,False,.08,True)
    vs,ts=shape.tessellate(.2,.12);vs=np.array([v.toTuple() for v in vs]);ts=np.array(ts);normals=[];colors=[]
    for f in shape.Faces():
        loc=TopLoc_Location();pol=BRep_Tool.Triangulation_s(f.wrapped,loc);BRepLib_ToolTriangulatedShape.ComputeNormals_s(f.wrapped,pol)
        sign=-1 if f.wrapped.Orientation()==TopAbs_REVERSED else 1
        hit=False
        if sigs:
            sig=v5.signature(f);hit=any(abs(sig[0]-s[0])<.3 and np.linalg.norm(np.array(sig[1:])-s[1:])<.03 for s in sigs)
        black=flat and f.geomType()=="CYLINDER" and abs(f._geomAdaptor().Cylinder().Radius()-RF)<.002
        col=[0]*3 if black else ([255,115,15] if hit else [245,239,225])
        for i in range(1,pol.NbNodes()+1):
            n=pol.Normal(i).Transformed(loc.Transformation());normals.append([sign*n.X(),sign*n.Y(),sign*n.Z()]);colors.append(col)
    normals=np.array(normals);normals/=np.linalg.norm(normals,axis=1)[:,None]
    assert len(vs)==len(colors)
    return vs,ts,normals,np.array(colors,float)
def recolor(mesh,c):return (*mesh[:3],np.broadcast_to(np.array(c,float),mesh[0].shape).copy())
def control_images(report,frames,blend_sig):
    frame=frames[1]
    fm=load_mesh(base.SOURCE,[222,218,208]);sm=load_mesh(CAD/"shade_2_open.step",[166,163,147]);hm=load_mesh(CAD/"handle_2_open.step",[230,211,165])
    ws=[]
    for s in blend_sig:
        c=v5.to_world(cq.Vertex.makeVertex(*s[1:]),frame).Center();ws.append([s[0],c.x,c.y,c.z])
    cm=painted(CAD/"cup_2.step",ws);scene=merge([fm,cm,sm,hm])
    image=rasterize(scene,1000,1250,-frame.n,frame.origin,physical_size=(400,500))
    def save(im,name,title,note=""):
        fig,ax=plt.subplots(figsize=(9,9),facecolor="white");ax.imshow(im);ax.axis("off");ax.set_title(title)
        fig.text(.02,.015,note,fontsize=9);fig.tight_layout(rect=(0,.04,1,1));fig.savefig(OUT/name,dpi=150);plt.close(fig)
    save(image,"assembly_front.png","Локальная правка четырёх углов / ортогональный вид","R10 выделен оранжевым. Основные контуры, глубина чашки и панель сохранены.")
    natural=merge([fm,recolor(cm,[245,239,225]),sm,hm])
    zoom=rasterize(natural,1100,1000,-frame.n,frame.origin+np.array([115.,0,0])+165*frame.up,physical_size=(105,95))
    save(zoom,"corner_closeup.png","Правый верхний угол / ортогонально","Линия перехода следует скруглению исходного проёма; сопряжение R10.")
    before=painted(OLD/"cad/cup_2.step",flat=True);after=painted(CAD/"cup_2.step",flat=True)
    static=[recolor(fm,[240]*3),recolor(sm,[240]*3),recolor(hm,[240]*3)]
    counts={}
    for label,m in (("before",before),("after",after)):
        bright=m[3].copy();bright[np.any(bright!=0,axis=1)]=240
        im=rasterize(merge(static+[(*m[:3],bright)]),1000,1250,-frame.n,frame.origin,physical_size=(400,500))
        yy,xx=np.where(np.all(im<20,axis=2));x=np.abs((xx+.5-500)/2.5);z=np.abs((625-yy-.5)/2.5)
        u=(np.arctan2(z-C[1],x-C[0])-A0%(2*math.pi))/SPAN
        select=(x>40)&(z>90)&(u>.22)&(u<.85)
        counts[label]=int(select.sum())
    report["visible_corner_flat_flange_pixels"]=counts
    assert counts["after"]<counts["before"]*.45,counts
    # Actual contact footprint and lower shade edge, unwrapped at R1761.
    h=cq.importers.importStep(str(CAD/"handle_2_closed.step")).val();s=cq.importers.importStep(str(CAD/"shade_2_closed.step")).val()
    _,pad=contact_data(h,frame);_,blade=contact_data(s,frame);edge=blade[:,1].min();pad[:,1]-=edge
    fig,ax=plt.subplots(figsize=(10,6),facecolor="white")
    ax.axhline(0,color="#445",lw=2,label="Нижняя кромка шторки")
    # Entire handle projected onto developed shade coordinates, from the real STEP.
    verts,tris=h.tessellate(.01,.04);tris=np.asarray(tris);p=np.array([v.toTuple() for v in verts])
    hs=R*(np.arctan2(p[:,2]-frame.cz,p[:,1]-frame.cy)-frame.theta)-edge
    from shapely.geometry import Polygon
    from shapely import union_all
    hp=np.column_stack((p[:,0]-frame.cx,hs))
    poly=union_all([Polygon(hp[t]) for t in tris if Polygon(hp[t]).area>1e-6],grid_size=.001)
    if poly.geom_type=="MultiPolygon":poly=max(poly.geoms,key=lambda v:v.area)
    q=np.array(poly.exterior.coords);ax.fill(q[:,0],q[:,1],color="#e8ddbd",alpha=.7,label="Проекция ручки")
    ax.scatter(pad[:,0],pad[:,1],s=2,color="#e57819",label="Контур площадки из STEP")
    ax.annotate("",(0,0),(0,7),arrowprops={"arrowstyle":"<->","color":"#c53"})
    ax.text(2,3.5,"7 мм",va="center",color="#c53",fontsize=14)
    ax.set_xlim(-65,65);ax.set_ylim(-3,17);ax.set_aspect("equal");ax.grid(alpha=.2);ax.legend(loc="upper right")
    ax.set_xlabel("Поперёк шторки, мм");ax.set_ylabel("По поверхности шторки, мм")
    ax.set_title("Посадка ручки / развёртка цилиндрической поверхности R1761")
    fig.tight_layout();fig.savefig(OUT/"handle_mount_7mm.png",dpi=160);plt.close(fig)
def main():
    protected_before=protected();old=cq.importers.importStep(str(OLD/"cad/cup_local.step")).val()
    candidate,sigs=build_candidate();region=masks()
    # Replace only the material inside the four bounded corner regions.
    old_local=boolean(old,region,"common")
    new_local=boolean(candidate,region,"common")
    retained=boolean(old,region,"cut")
    print("Corner regions clipped",old_local.Volume(),new_local.Volume(),flush=True)
    result=sew_local_regions(retained,new_local)
    # The v5 remainder was already symmetric. Reflect the finished native
    # quadrant so the four corrected corners are identical, including topology.
    quad=base.pure_solid(result.intersect(cq.Solid.makeBox(500,400,400,cq.Vector(0,1600,0))).fix())
    upper=base.pure_solid(quad.fuse(quad.mirror("YZ")).fix())
    result=base.pure_solid(upper.fuse(upper.mirror("XY")).fix())
    result=export(result,"cup_local")
    old_local_volume=float(old_local.Volume())
    new_local_volume=float(new_local.Volume())
    print("Original cup remainder retained; local volume delta",new_local_volume-old_local_volume,flush=True)
    old_bounds=base.exact_bounds(old);new_bounds=base.exact_bounds(result)
    envelope_errors={k:abs(getattr(old_bounds,k)-getattr(new_bounds,k)) for k in ("xmin","xmax","ymin","ymax","zmin","zmax")}
    assert max(envelope_errors.values())<.001,envelope_errors
    surface_check=outside_surface_check(old,result)
    sym=[cut_volume(result,result.mirror(p)) for p in ("YZ","XY")]
    assert max(sym)<.02,sym
    opening=v5.opening_actual_compare(result)
    # This helper writes its comparison into v5 previews; do not save those changes.
    panel=cq.importers.importStep(str(base.SOURCE)).val();frames=base.frames_from_panel(panel)
    report={"baseline_commit":CFG["baseline_commit"],"local_material_net_change_mm3":new_local_volume-old_local_volume,
      "outside_boundary_preservation":surface_check,"external_envelope_errors_mm":envelope_errors,"cup_symmetry_difference_mm3":sym,"corner_G1_normal_residual":max(G1_ERRORS),"corner_fillet_radius_mm":10.,"back_parallel_fillet_radius_mm":7.,"opening":opening,"handle_mount":[],"assembly_checks":[]}
    for i,frame in enumerate(frames,1):
        export(v5.to_world(result,frame),f"cup_{i}")
        oldh=cq.importers.importStep(str(OLD/f"cad/handle_{i}_closed.step")).val()
        shade=cq.importers.importStep(str(OLD/f"cad/shade_{i}_closed.step")).val()
        _,pad=contact_data(oldh,frame);_,blade=contact_data(shade,frame)
        lower=blade[:,1].min();existing=pad[:,1].min()-lower;shift=7.-existing
        mounted=base.pure_solid(move_on_cylinder(oldh,frame,shift,R));_,newpad=contact_data(mounted,frame)
        gap=float(newpad[:,1].min()-lower);assert abs(gap-7)<1e-6
        assert abs((newpad[:,0].min()+newpad[:,0].max())/2)<1e-6
        inverse=move_on_cylinder(mounted,frame,-shift,R);assert abs(inverse.Volume()-oldh.Volume())<.001
        report["handle_mount"].append({"window":i,"existing_gap_mm":float(existing),"rigid_surface_shift_mm":float(shift),"new_gap_mm":gap,"mount_pad_arc_length_mm":float(np.ptp(newpad[:,1])),"horizontal_center_mm":float(newpad[:,0].mean())})
        for state,travel in (("open",380),("half",190),("closed",0)):
            export(move_on_cylinder(mounted,frame,travel,R),f"handle_{i}_{state}")
            shutil.copy2(OLD/f"cad/shade_{i}_{state}.step",CAD/f"shade_{i}_{state}.step")
    shutil.copy2(OLD/"cad/handle.step",CAD/"handle.step")
    shutil.copy2(OLD/"cad/opening_contour.step",CAD/"opening_contour.step")
    for state in ("open","half","closed"):
        asm=cq.Assembly(name="Tu214_Local_v6_"+state);asm.add(panel,name="Wall_Panel")
        checks=[]
        for i in (1,2):
            cup=cq.importers.importStep(str(CAD/f"cup_{i}.step")).val();h=cq.importers.importStep(str(CAD/f"handle_{i}_{state}.step")).val();s=cq.importers.importStep(str(CAD/f"shade_{i}_{state}.step")).val()
            ov={key:v5.common(a,b) for key,a,b in (("panel_cup",panel,cup),("shade_cup",s,cup),("handle_cup",h,cup),("panel_handle",panel,h),("shade_handle",s,h))}
            assert max(ov.values())<.01,(i,state,ov)
            checks.append({"window":i,"overlaps_mm3":ov})
            asm.add(cup,name=f"Cup_{i}");asm.add(s,name=f"Shade_{i}");asm.add(h,name=f"Handle_{i}")
        p=CAD/f"assembly_{state}.step";asm.export(str(p),exportType="STEP");kept=preserve_panel(base.SOURCE,p)
        check=cq.importers.importStep(str(p)).val();assert check.isValid() and len(check.Solids())==7
        report["assembly_checks"].append({"state":state,"checks":checks,**kept})
    control_images(report,frames,sigs)
    # Remove previews written by imported v5 helper, restore baseline checksums.
    (OLD/"previews/unused.png").unlink(missing_ok=True)
    assert protected()==protected_before
    assert sha(CAD/"handle.step")==sha(OLD/"cad/handle.step")
    report["baseline_STEP_files_unchanged"]=True;report["canonical_handle_unchanged"]=True;report["all_checks_passed"]=True
    (ROOT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    (ROOT/"README.md").write_text("""# Ту-214: локальная коррекция четырёх углов и посадки ручки

Из исходной чашки v5 заменены только объёмы в четырёх симметричных угловых областях. Новая линия перехода ориентирована по нативному контуру проёма панели; сопряжение выполнено R10 с параллельной задней поверхностью и толщиной3мм. Основные участки большого контура, внутреннее отверстие247×345 и глубина чашки сохранены. Сохранность граничных поверхностей вне локальных областей проверена в обоих направлениях; исходный остаток чашки использован без перестроения.

Форма ручки120×30×15 не изменена: канонический STEP скопирован побайтно. Уже собранная ручка с площадкой жёстко перемещена вдоль шторки. От нижнего края шторки до ближайшего края площадки —7мм по поверхности R1761. Положение и STEP-файлы шторки сохранены.

![Фронтально](previews/assembly_front.png)
![Угол крупно](previews/corner_closeup.png)
![Отступ7мм](previews/handle_mount_7mm.png)

Численные проверки сохранены в[report.json](report.json). Исходная стеновая панель не изменена; её BRep-записи сохранены в сборках. Проверены открытое, среднее и закрытое положения.
""")
    with zipfile.ZipFile(ROOT/"local_correction_STEP.zip","w",zipfile.ZIP_DEFLATED) as z:
        for p in list(CAD.glob("*"))+list(OUT.glob("*"))+[ROOT/"report.json",ROOT/"README.md",ROOT/"build.py",ROOT/"parameters.json"]:z.write(p,str(p.relative_to(ROOT)))
    print("COMPLETE",report,flush=True)
if __name__=="__main__":main()
