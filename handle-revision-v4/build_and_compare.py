"""Build only the revised handle and review its real STEP projections/placement."""
from pathlib import Path
import importlib.util, sys, os, math, json, hashlib, zipfile
os.environ.setdefault("MPLCONFIGDIR","/tmp/tu214-handle-v4-mpl")
import numpy as np
import cadquery as cq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import Polygon, LineString
from shapely import union_all
from OCP.TColgp import TColgp_Array2OfPnt
from OCP.gp import gp_Pnt
from OCP.Geom import Geom_BezierSurface
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
BASE=REPO/"corrected-window-v3"
CAD=ROOT/"cad"
OUT=ROOT/"previews"
for p in (CAD,OUT):p.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location("baseline_v3",BASE/"build.py")
base=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=base
spec.loader.exec_module(base)
from preview_mesh import load_mesh, merge, rasterize
from cylindrical_parts import move_on_cylinder, rounded_rectangle, cylindrical_slab

CFG=json.loads((ROOT/"parameters.json").read_text())
TRACE=json.loads((REPO/"reviews/handle-100x30x25-before/trace_points.json").read_text())
BLUE="#2563eb"; RED="#dc2626"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
R=float(base.P["shade_radius_mm"])
TRAVEL=float(base.P["shade_open_travel_mm"])
OLD=BASE/"cad"

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def protected_files():
    paths=list(BASE.rglob("*.step"))+[BASE/"parameters.json",BASE/"build.py",base.SOURCE]
    return {str(p.relative_to(REPO)):digest(p) for p in paths}

def build_handle():
    width,depth,height=CFG["dimensions_mm"]
    def point(x,y,z):return cq.Vector(x*width/100,y*depth/30,z*height/25)
    def face(grid,reverse=False):
        poles=TColgp_Array2OfPnt(1,len(grid),1,len(grid[0]))
        for i,row in enumerate(grid,1):
            for j,p in enumerate(row,1):
                q=point(*p);poles.SetValue(i,j,gp_Pnt(q.x,q.y,q.z))
        f=BRepBuilderAPI_MakeFace(Geom_BezierSurface(poles),1e-7).Face()
        return cq.Shape.cast(f.Reversed() if reverse else f)
    xs,ys=[-50,0,50],[0,-15,-30]
    fc,sc=CFG["front_bezier_height_controls_mm"],CFG["side_bezier_height_controls_mm"]
    upper=[[(x,y,fc[i]*sc[j]/12.5) for j,y in enumerate(ys)] for i,x in enumerate(xs)]
    lower=[[(x,y,-z) for x,y,z in row] for row in upper]
    rear=[[(x,0,-fc[i]),(x,0,fc[i])] for i,x in enumerate(xs)]
    front=[[(x,-30,-fc[i]*.4),(x,-30,fc[i]*.4)] for i,x in enumerate(xs)]
    left=[[(-50,y,-5*sc[j]/12.5),(-50,y,5*sc[j]/12.5)] for j,y in enumerate(ys)]
    right=[[(50,y,-5*sc[j]/12.5),(50,y,5*sc[j]/12.5)] for j,y in enumerate(ys)]
    roof=base.solid_from_faces([face(upper,True),face(lower),face(rear,True),
                               face(front),face(left),face(right,True)])
    controls=CFG["plan_bezier_controls_mm"]
    edges=[cq.Edge.makeLine(point(-50,0,-20),point(50,0,-20))]
    edges += [cq.Edge.makeBezier([point(x,y,-20) for x,y in c]) for c in controls]
    plan=cq.Solid.extrudeLinear(cq.Wire.assembleEdges(edges),[],cq.Vector(0,0,40*height/25))
    return base.pure_solid(roof.intersect(plan).fix())

def export_handle(shape,name):
    shape=base.pure_solid(shape)
    path=CAD/(name+".step")
    cq.exporters.export(shape,str(path),exportType="STEP")
    restored=base.pure_solid(cq.importers.importStep(str(path)).val())
    assert restored.isValid() and len(restored.Solids())==1
    return restored

def mount(canonical,frame,offset):
    station=base.P["shade_closed_lower_edge_mm"]+offset
    a=frame.theta+station/R;r=R-.05
    body=canonical.rotate((0,0,0),(1,0,0),math.degrees(a)).translate(
        (frame.cx,frame.cy+r*math.cos(a),frame.cz+r*math.sin(a)))
    outline=rounded_rectangle(20,2,.4,center_s=station+2)
    land=cylindrical_slab(outline,frame,R-1.5,1.5,R)
    return base.pure_solid(body.fuse(land).fix())

def overlap(a,b):
    op=BRepAlgoAPI_Common(a.wrapped,b.wrapped)
    op.SetNonDestructive(True);op.SetFuzzyValue(.00005);op.Build()
    assert op.IsDone()
    result=cq.Shape.cast(op.Shape())
    assert result.isValid()
    return float(sum(max(0,s.Volume()) for s in result.Solids()))

def mesh_outline(shape,axes,neg_second=False):
    vertices,tri=shape.tessellate(.025,.06)
    tri=np.asarray(tri,dtype=int)
    vv=np.array([p.toTuple() for p in vertices])[:,axes]
    if neg_second:vv[:,1]*=-1
    if axes==[1,2]:vv[:,0]*=-1
    polys=[Polygon(vv[ids]) for ids in tri if Polygon(vv[ids]).area>1e-7]
    outline=union_all(polys,grid_size=.002)
    if outline.geom_type=="MultiPolygon":outline=max(outline.geoms,key=lambda p:p.area)
    assert outline.is_valid and outline.area>0
    return np.array(outline.exterior.coords)

def trace(kind):
    t=TRACE[kind];x0,y0,x1,y1=t["bbox_px"];w,h=t["target_dimensions_mm"]
    return np.array([[(x-x0)/(x1-x0)*w-(0 if kind=="side" else w/2),
       (y-y0)/(y1-y0)*h if kind=="top" else h/2-(y-y0)/(y1-y0)*h]
       for x,y in t["points_px"]])

def distances(a,b):
    la=LineString(np.vstack([a,a[0]]));lb=LineString(np.vstack([b,b[0]]))
    vals=[]
    for one,other in ((la,lb),(lb,la)):
        n=max(100,int(math.ceil(one.length/.1)))
        vals.extend(one.interpolate(float(s)).distance(other) for s in np.linspace(0,one.length,n,endpoint=False))
    d=np.array(vals)
    return {"symmetric_max_mm":float(d.max()),"uniform_arc_rms_mm":float(np.sqrt(np.mean(d*d))),
            "p95_mm":float(np.quantile(d,.95)),"sample_spacing_mm":.1}

def compare(canonical):
    old=cq.importers.importStep(str(OLD/"handle.step")).val()
    metrics={}
    configs=[("front",[0,2],False,(1,1),"Спереди / X–Z",(100,25)),
             ("side",[1,2],False,(.75,1),"Сбоку / глубина–Z",(30,25)),
             ("top",[0,1],True,(1,.75),"Сверху / X–глубина",(100,30))]
    fig,axs=plt.subplots(3,2,figsize=(13,11),facecolor="white")
    for row,(kind,axes,neg,scale,title,dims) in enumerate(configs):
        actual=mesh_outline(canonical,axes,neg)
        before=mesh_outline(old,axes,neg)*np.array(scale)
        reference=trace(kind)
        metrics[kind]={"before":distances(before,reference),"after":distances(actual,reference)}
        assert metrics[kind]["after"]["symmetric_max_mm"]<1.5,(kind,metrics[kind])
        for col,contour in enumerate((before,actual)):
            ax=axs[row,col]
            ax.plot(contour[:,0],contour[:,1],color=BLUE,lw=2,label="Контур STEP")
            ax.plot(*np.vstack([reference,reference[0]]).T,color=RED,lw=1.6,label="Принятый контур")
            ax.scatter(reference[:,0],reference[:,1],color=RED,s=9)
            ax.set_title(("До: глубина нормирована 40→30" if col==0 else "После: реальные размеры 100×30×25")+" / "+title,fontsize=10)
            ax.set_aspect("equal");ax.grid(alpha=.18);ax.legend(fontsize=8)
            if kind=="top":ax.invert_yaxis()
            ax.set_xlabel("Глубина, мм" if kind=="side" else "Ширина X, мм")
            ax.set_ylabel("Глубина, мм" if kind=="top" else "Высота Z, мм")
        single,ax=plt.subplots(figsize=(9,4),facecolor="white")
        ax.plot(actual[:,0],actual[:,1],color=BLUE,lw=2,label="Исправленный STEP")
        ax.plot(*np.vstack([reference,reference[0]]).T,color=RED,lw=1.5,label="Принятый контур эскиза")
        ax.scatter(reference[:,0],reference[:,1],color=RED,s=10)
        ax.set_aspect("equal");ax.grid(alpha=.18);ax.legend(fontsize=8);ax.set_title(title+f" / {dims[0]}×{dims[1]} мм")
        if kind=="top":ax.invert_yaxis()
        single.tight_layout();single.savefig(OUT/f"overlay_{kind}_after.png",dpi=160);plt.close(single)
        np.savetxt(OUT/f"actual_{kind}_outline.csv",actual,delimiter=",",header="horizontal_mm,vertical_mm",comments="")
    fig.suptitle("Три ортогональные проекции экспортированного STEP: до и после",fontsize=16)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/"three_views_before_after.png",dpi=150);plt.close(fig)
    for kind,direction in [("front",(0,-1,0)),("side",(1,0,0)),("top",(0,0,1))]:
        cq.exporters.export(canonical,str(OUT/f"handle_{kind}_after.svg"),exportType="SVG",
            opt={"projectionDir":direction,"showHidden":False,"width":800,"height":400,"strokeWidth":.5})
    return metrics

def save_render(img,name,title,note):
    fig,ax=plt.subplots(figsize=(10,7),facecolor="white")
    ax.imshow(img);ax.axis("off");ax.set_title(title)
    fig.text(.03,.025,note,fontsize=9)
    fig.tight_layout(rect=(0,.045,1,1));fig.savefig(OUT/name,dpi=150);plt.close(fig)

def recolor(mesh,color):
    return mesh[0],mesh[1],mesh[2],np.broadcast_to(np.array(color,float),mesh[0].shape).copy()

def visible_mask(static_mesh,handle_mesh,frame):
    scene=merge([recolor(static_mesh,[240,240,240]),recolor(handle_mesh,[0,0,0])])
    img=rasterize(scene,880,400,-frame.n,frame.origin+185*frame.up,physical_size=(220,100))
    mask=np.all(img<20,axis=2);yy,xx=np.where(mask)
    if not len(xx):return {"visible":False}
    w=int(xx.max()-xx.min()+1);h=int(yy.max()-yy.min()+1)
    return {"visible":True,"bbox_xyxy_px":[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())],
       "width_px":w,"height_px":h,"height_width_ratio":h/w,"area_px":int(mask.sum()),
       "measured_after_scene_occlusion":True}

def project(points,params,origin):
    yaw,pitch,roll,logf,logdistance,tx,ty=params
    camera=np.array([math.sin(yaw)*math.cos(pitch),-math.cos(yaw)*math.cos(pitch),math.sin(pitch)])
    right=np.cross([0,0,1],camera);right/=np.linalg.norm(right);up=np.cross(camera,right)
    r=right*math.cos(roll)+up*math.sin(roll);u=up*math.cos(roll)-right*math.sin(roll)
    rel=points-origin;depth=math.exp(logdistance)-rel@camera
    assert depth.min()>0
    return np.column_stack((math.exp(logf)*(rel@r)/depth+tx,-math.exp(logf)*(rel@u)/depth+ty))

def photo_poly(mesh,params,origin):
    uv=project(mesh[0],params,origin)
    polys=[Polygon(uv[ids]) for ids in mesh[1] if Polygon(uv[ids]).area>1e-6]
    return union_all(polys,grid_size=.001),uv

def compact_metrics(mesh,frame,params,offset):
    verts=mesh[0];uv=project(verts,params,frame.origin)
    radial=np.hypot(verts[:,1]-frame.cy,verts[:,2]-frame.cz)
    delta=np.arctan2(verts[:,2]-frame.cz,verts[:,1]-frame.cy)-frame.theta
    s=R*np.arctan2(np.sin(delta),np.cos(delta));lo,hi=uv.min(0),uv.max(0)
    return {"fixed_camera_bbox_xyxy_px":[*lo.tolist(),*hi.tolist()],
      "fixed_camera_silhouette_bbox_wh_px":(hi-lo).tolist(),"bbox_before_occlusion":True,
      "max_inboard_projection_from_shade_mm":float(R-radial.min()),
      "radial_range_mm":[float(radial.min()),float(radial.max())],
      "tangential_range_mm":[float(s.min()),float(s.max())],
      "handle_mount_station_open_mm":base.P["shade_open_lower_edge_mm"]+offset,
      "shade_lower_edge_open_mm":base.P["shade_open_lower_edge_mm"],
      "lateral_center_mm":float((verts[:,0].min()+verts[:,0].max())/2-frame.cx),"camera_refitted":False}

def render_placement(static_mesh,old_mesh,new_mesh,frame,offset):
    old_scene=merge([static_mesh,old_mesh]);new_scene=merge([static_mesh,new_mesh])
    target=frame.origin
    img=rasterize(new_scene,900,1120,-frame.n,target,physical_size=(380,500))
    save_render(img,"window_open_after.png","Поднятая шторка / исправленная ручка","Панель, чашка и шторка взяты из исходных STEP; заменена только ручка.")
    view=.22*np.array([1.,0.,0.])-.97*frame.n+.12*frame.up
    img=rasterize(new_scene,900,1120,view,target,physical_size=(380,500),perspective=True,distance=1450)
    save_render(img,"window_oblique_after.png","Выступ ручки / косой ракурс","Реальная STEP-геометрия; размеры ручки 100×30×25 мм.")
    fig,axs=plt.subplots(1,2,figsize=(13,5),facecolor="white")
    for ax,scene,title in [(axs[0],old_scene,"До / 100×40×25"),(axs[1],new_scene,"После / 100×30×25")]:
        img=rasterize(scene,1000,450,-frame.n,frame.origin+185*frame.up,physical_size=(220,100))
        ax.imshow(img);ax.axis("off");ax.set_title(title)
    fig.suptitle("Верх оконного проёма / одинаковые камера и масштаб")
    fig.text(.02,.02,f"Смещение только ручки относительно нижнего края шторки: {offset:g} мм.",fontsize=10)
    fig.tight_layout(rect=(0,.07,1,.93));fig.savefig(OUT/"upper_window_before_after.png",dpi=150);plt.close(fig)
    params=json.loads((REPO/"tu214-sidewall/cad/reference_comparison.json").read_text())["camera_params_yaw_pitch_roll_logf_logdistance_tx_ty"]
    photo=np.asarray(Image.open(REPO/"tu214-sidewall/references/18205_marked.png").convert("RGB"))
    old_poly,_=photo_poly(old_mesh,params,frame.origin);new_poly,_=photo_poly(new_mesh,params,frame.origin)
    fig,axs=plt.subplots(1,3,figsize=(15,5),facecolor="white")
    for ax in axs:ax.imshow(photo);ax.set_xlim(975,1290);ax.set_ylim(475,315);ax.set_aspect("equal");ax.axis("off")
    axs[0].set_title("Фотография / верх окна")
    for ax,poly,color,title in [(axs[1],old_poly,RED,"Прежний STEP"),(axs[2],new_poly,BLUE,"Исправленный STEP")]:
        parts=list(poly.geoms) if poly.geom_type=="MultiPolygon" else [poly]
        for g in parts:ax.plot(*np.array(g.exterior.coords).T,color=color,lw=1.6)
        ax.set_title(title)
    fig.suptitle("Проекция STEP на фото / прежняя фиксированная камера")
    fig.text(.02,.015,"Без подгонки ручки; полный силуэт до окклюзии. Фото размыто и размечено.",fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.93));fig.savefig(OUT/"photo_upper_compare.png",dpi=160);plt.close(fig)
    metrics={"before":compact_metrics(old_mesh,frame,params,0),"after":compact_metrics(new_mesh,frame,params,offset),
             "before_visible":visible_mask(static_mesh,old_mesh,frame),"after_visible":visible_mask(static_mesh,new_mesh,frame),
             "photo_comparison_is_qualitative":True}
    metrics["projection_reduction_fraction"]=1-metrics["after"]["max_inboard_projection_from_shade_mm"]/metrics["before"]["max_inboard_projection_from_shade_mm"]
    return metrics

def validate_positions(panel,frames,closed_handles):
    checks=[];worst=0.
    for i,(frame,h0) in enumerate(zip(frames,closed_handles),1):
        cup=cq.importers.importStep(str(OLD/f"cup_{i}.step")).val()
        shade=cq.importers.importStep(str(OLD/f"shade_{i}_closed.step")).val()
        for t in np.linspace(0,TRAVEL,CFG["collision_positions"]):
            h=move_on_cylinder(h0,frame,float(t),R);s=move_on_cylinder(shade,frame,float(t),R)
            volumes={name:overlap(a,h) for name,a in [("panel",panel),("cup",cup),("shade",s)]}
            worst=max(worst,*volumes.values())
            checks.append({"window":i,"travel_mm":float(t),"intersections_mm3":volumes})
            assert max(volumes.values())<.002,(i,float(t),volumes)
        print("Collision positions passed, window",i,flush=True)
    return {"positions_per_window":CFG["collision_positions"],"continuous_sweep_proof":False,
      "max_intersection_mm3":worst,"samples":checks}

def write_readme(report):
    rows=[]
    for kind,label in [("front","Спереди"),("side","Сбоку"),("top","Сверху")]:
        m=report["projection_comparison"][kind]
        rows.append(f"| {label} | {m['before']['symmetric_max_mm']:.3f} | {m['after']['symmetric_max_mm']:.3f} | {m['after']['uniform_arc_rms_mm']:.3f} |")
    p=report["placement"];offset=report["selected_mount_offset_mm"]
    text=f"""# Ручка шторки — исправление по принятым контурам

Пользователь принял красные контуры диагностического коммита51c8a43. Исправлена только ручка; новые файлы находятся в этой папке. Исходные панели, чашки, шторки и сборочные STEP не изменены (SHA-256 проверены).

Габариты экспортированного и повторно импортированного STEP: **100×30×25 мм**. Передняя кромка сверху выпуклая, задняя почти прямая; фронтальные торцы имеют конечную высоту10мм. Верхняя/нижняя кожи — единые би-квадратичные Bézier-поверхности, без внутреннего ребра от пересечения двух профилей. Сохранены симметрии по X и Z.

[Новая ручка](cad/handle.step); [открытое положение окна1](cad/handle_1_open.step); [окно2](cad/handle_2_open.step). Закрытые и половинные положения также сохранены. [ZIP](handle_step_and_review.zip).

## Повторное сравнение реального STEP

Силуэты получены ортогональным проецированием треугольников **повторно импортированного STEP**, их объединением и извлечением внешней границы. Новая модель не масштабируется. Для старой модели в сравнении глубина нормирована40→30мм, чтобы оценивать форму при одинаковом габарите. Красный — ранее принятые пользователем ручные точки; это не точная метрическая оцифровка рисунка.

![До/после](previews/three_views_before_after.png)

| Проекция | Максимальное расстояние до, мм | После, мм | RMS после, мм |
|---|---:|---:|---:|
{chr(10).join(rows)}

Метрика симметрична в обоих направлениях, равномерная выборка по длине≤0.1мм. [Спереди](previews/overlay_front_after.png), [сбоку](previews/overlay_side_after.png), [сверху](previews/overlay_top_after.png). Чистые HLR SVG и CSV-силуэты сохранены в previews. Погрешность тесселяции .025мм; ограничение сравнения — ручное чтение исходного эскиза.

## Посадка на поднятой шторке

Смещение только ручки относительно нижнего края шторки: **{offset:g}мм**. Нижний край самой открытой шторки остаётся197мм. Скрытая контактная площадка20×2×1.5мм сохранена и перемещается вместе с ручкой; она не становится отдельным видимым отростком.

![Верх окна](previews/upper_window_before_after.png)

[Полный оконный вид](previews/window_open_after.png), [косой ракурс](previews/window_oblique_after.png), [сравнение с фото](previews/photo_upper_compare.png). Камера фотографии взята из существующей регистрации по контурам окон и не подгонялась по ручке. Фотографическое наложение показывает полный силуэт до окклюзии и является качественным сравнением, а не обмером. Реальная видимая часть после окклюзии измерена отдельно маской CAD-сцены.

Выступ к салону уменьшен с{p['before']['max_inboard_projection_from_shade_mm']:.2f} до{p['after']['max_inboard_projection_from_shade_mm']:.2f}мм. Отношение видимой высоты к ширине: {p['before_visible'].get('height_width_ratio',0):.3f}→{p['after_visible'].get('height_width_ratio',0):.3f}. Варианты положения и причины выбора сохранены в report.json; ориентир тонкой полоски на фотографии приблизителен.

## Проверки

BRep валиден, одно замкнутое тело; STEP повторно импортирован. Габариты и симметрии проверены отдельно от визуального сравнения. Отсутствие пересечений с исходными панелью, чашкой и шторкой проверено в{CFG['collision_positions']}положениях каждого из двух окон; это дискретная проверка, не доказательство непрерывного sweep. Исходные CAD-файлы проверены по SHA-256. [Полный отчёт](report.json).

Воспроизведение: python -m pip install -r requirements.txt; python build_and_compare.py. Расчёт выполнялся в GitHub Actions с CadQuery/OpenCascade.
"""
    (ROOT/"README.md").write_text(text,encoding="utf8")

def main():
    protected=protected_files()
    report={"baseline_commit":"51c8a43332dbcd9329ece51c5b6a0f7a14443157","accepted_trace":str(REPO/"reviews/handle-100x30x25-before/trace_points.json"),
            "canonical_dimensions_requested_mm":CFG["dimensions_mm"],"pipeline_stage":"building"}
    canonical=export_handle(build_handle(),"handle")
    bb=base.exact_bounds(canonical);dims=[bb.xlen,bb.ylen,bb.zlen]
    assert np.allclose(dims,CFG["dimensions_mm"],atol=.001),dims
    symmetry={plane:float(canonical.cut(canonical.mirror(plane)).Volume()) for plane in ("YZ","XY")}
    assert max(symmetry.values())<.002,symmetry
    report.update({"canonical_dimensions_mm":dims,"canonical_volume_mm3":canonical.Volume(),"symmetry_difference_mm3":symmetry,
                   "projection_comparison":compare(canonical)})
    print("Handle built and compared",report["projection_comparison"],flush=True)
    panel=cq.importers.importStep(str(base.SOURCE)).val();frames=base.frames_from_panel(panel)
    static=merge([load_mesh(base.SOURCE,[225,220,209]),load_mesh(OLD/"cup_2.step",[247,246,239]),
                  load_mesh(OLD/"shade_2_open.step",[181,178,163])])
    old_mesh=load_mesh(OLD/"handle_2_open.step",[233,214,169])
    candidates=[]
    cup2=cq.importers.importStep(str(OLD/"cup_2.step")).val()
    shade2=cq.importers.importStep(str(OLD/"shade_2_open.step")).val()
    for offset in CFG["mount_offset_candidates_mm"]:
        try:
            mounted=mount(canonical,frames[1],offset)
            opened=move_on_cylinder(mounted,frames[1],TRAVEL,R)
            volumes={name:overlap(a,opened) for name,a in [("panel",panel),("cup",cup2),("shade",shade2)]}
            candidate={"offset_mm":offset,"open_intersections_mm3":volumes}
            if max(volumes.values())<.002:
                tmp=CAD/"candidate_placement.step"
                cq.exporters.export(opened,str(tmp),exportType="STEP")
                mesh=load_mesh(tmp,[233,214,169])
                visible=visible_mask(static,mesh,frames[1]);candidate["visible_mask"]=visible
                candidate["score"]=abs(visible.get("height_width_ratio",1)-CFG["visible_height_width_goal"]) if visible.get("visible") else 10
            else:candidate["score"]=10
        except Exception as e:candidate={"offset_mm":offset,"score":10,"error":str(e)}
        candidates.append(candidate);print("Placement candidate",candidate,flush=True)
    valid=sorted([c for c in candidates if c["score"]<10],key=lambda c:(c["score"],c["offset_mm"]))
    assert valid,"No collision-free visible placement"
    selected=None;closed_handles=None;motion=None
    for candidate in valid:
        offset=candidate["offset_mm"]
        handles=[mount(canonical,frame,offset) for frame in frames]
        try:
            motion=validate_positions(panel,frames,handles)
            selected=candidate;closed_handles=handles;break
        except AssertionError as e:
            candidate["motion_rejected"]=str(e);print("Reject placement",offset,e,flush=True)
    assert selected is not None,"All placements failed movement checks"
    offset=selected["offset_mm"]
    for i,(frame,closed) in enumerate(zip(frames,closed_handles),1):
        for state,t in [("closed",0),("half",TRAVEL/2),("open",TRAVEL)]:
            export_handle(move_on_cylinder(closed,frame,t,R),f"handle_{i}_{state}")
    (CAD/"candidate_placement.step").unlink(missing_ok=True)
    new_mesh=load_mesh(CAD/"handle_2_open.step",[233,214,169])
    report.update({"selected_mount_offset_mm":offset,"placement_candidates":candidates,"motion":motion,
                  "placement":render_placement(static,old_mesh,new_mesh,frames[1],offset)})
    handle_mesh=load_mesh(CAD/"handle.step",[233,214,169])
    img=rasterize(handle_mesh,1100,650,[.35,-.8,.48],target=(0,-15,0),physical_size=(125,70),perspective=True,distance=350)
    save_render(img,"handle_perspective_after.png","Исправленная ручка / реальный STEP","Гладкие поверхности, торцы конечной высоты; глубина30мм.")
    assert protected_files()==protected,"Baseline geometry or source changed"
    report.update({"baseline_files_unchanged":True,"baseline_sha256":protected,"all_checks_passed":True,"pipeline_stage":"complete"})
    (ROOT/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    write_readme(report)
    manifest={str(p.relative_to(ROOT)):{"bytes":p.stat().st_size,"sha256":digest(p)} for folder in (CAD,OUT) for p in sorted(folder.iterdir()) if p.is_file()}
    (ROOT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    with zipfile.ZipFile(ROOT/"handle_step_and_review.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for folder in (CAD,OUT):
            for p in sorted(folder.iterdir()):archive.write(p,str(p.relative_to(ROOT)))
        for name in ("README.md","parameters.json","report.json","manifest.json"):archive.write(ROOT/name,name)
    print("COMPLETE",dims,"placement offset",offset,flush=True)

if __name__=="__main__":main()
