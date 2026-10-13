"""Render evidence from exported STEP files, never from illustrative model images."""
import os, math, json, hashlib
os.environ.setdefault("MPLCONFIGDIR","/tmp/tu214-v3-matplotlib")
import numpy as np
import cadquery as cq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely import union_all
from build import P,ROOT,CAD,SOURCE,frames_from_panel,extract_throat,curve_data,rows_data
from preview_mesh import load_mesh,merge,rasterize

OUT=ROOT/"previews"
COLORS={"Panel":[225,220,209],"Cup":[247,246,239],"Shade":[181,178,163],"Handle":[233,214,169]}

def save_image(img,name,title,subtitle,size=(10,10)):
    fig=plt.figure(figsize=size,facecolor="#f6f7f9")
    fig.text(.04,.95,title,fontsize=17,weight="bold",color="#243447")
    fig.text(.04,.91,subtitle,fontsize=10,color="#5e6b77")
    ax=fig.add_axes([.025,.07,.95,.80]);ax.imshow(img);ax.axis("off")
    fig.text(.04,.025,"Проекция экспортированной BRep-геометрии STEP; миллиметры.",fontsize=9,color="#5e6b77")
    fig.savefig(OUT/name,dpi=140);plt.close(fig)

def local(frame,pts):
    d=pts-frame.origin
    return np.column_stack((pts[:,0]-frame.cx,d@frame.up,d@frame.n))

def section_curves(shape,plane,frame,axes):
    part=cq.Workplane(plane).newObject([shape]).section().val()
    curves=[]
    for edge in part.Edges():
        pts=np.array([p.toTuple() for p in edge.positions(np.linspace(0,1,61))])
        curves.append(local(frame,pts)[:,axes])
    assert curves,"Empty required section"
    return curves

def sections(panel,frame):
    parts=[("Панель",panel,"#626e7d"),("Чашка",cq.importers.importStep(str(CAD/"cup_2.step")).val(),"#4383b6"),
           ("Шторка",cq.importers.importStep(str(CAD/"shade_2_half.step")).val(),"#278f6b"),
           ("Ручка",cq.importers.importStep(str(CAD/"handle_2_half.step")).val(),"#c18c35")]
    planes=[("vertical_section.png","Вертикальное сечение A–A",
             cq.Plane(origin=(frame.cx,0,0),xDir=tuple(frame.n),normal=(1,0,0)),
             [2,1],(-65,130),(-260,680),("Глубина от R1735, мм","Высота от центра окна, мм")),
            ("horizontal_section.png","Горизонтальное сечение B–B / +40 мм",
             cq.Plane(origin=tuple(frame.point(0,40,0)),xDir=(1,0,0),normal=tuple(frame.up)),
             [0,2],(-210,210),(-55,115),("Ширина от центра окна, мм","Глубина от R1735, мм"))]
    for name,title,plane,axes,xlim,ylim,labels in planes:
        fig=plt.figure(figsize=(12,10) if "vertical" in name else (10,6),facecolor="white")
        if "vertical" in name:
            full=fig.add_axes([.06,.12,.34,.78])
            zoom=fig.add_axes([.52,.18,.42,.68])
            allaxes=[full,zoom]
            limits=[((-220,140),(-570,710)),((-35,120),(-240,250))]
        else:
            full=fig.add_axes([.09,.18,.88,.68]);allaxes=[full];limits=[(xlim,ylim)]
        for ax,(xl,yl) in zip(allaxes,limits):
            for label,shape,color in parts:
                cs=section_curves(shape,plane,frame,axes) if label!="Ручка" or "vertical" in name else []
                for k,c in enumerate(cs):ax.plot(c[:,0],c[:,1],color=color,lw=1.7,label=label if k==0 else None)
            ax.set_xlim(*xl);ax.set_ylim(*yl);ax.set_aspect("equal")
            ax.set_xlabel(labels[0]);ax.set_ylabel(labels[1]);ax.grid(alpha=.16)
        allaxes[-1].legend(loc="best")
        if len(allaxes)==2:
            full.set_title("Вся сборка / исходная панель целиком",fontsize=10)
            zoom.set_title("Оконная зона / увеличено",fontsize=10)
        fig.suptitle(title+"\nЧастично опущенная шторка между панелью и чашкой",fontsize=14)
        fig.text(.08,.035,"Сечения получены булевым пересечением экспортированных BRep-деталей с плоскостями.",fontsize=9)
        fig.savefig(OUT/name,dpi=155);plt.close(fig)

def cup_front(panel,frame,cup_mesh):
    front=-frame.n
    img=rasterize(cup_mesh,950,1200,front,frame.origin,physical_size=(350,470))
    save_image(img,"cup_front.png","Чашка / ортогональный вид спереди",
               "Наружный контур исходного проёма; внутренний — 246 × 344 мм в развёртке.")
    curves,_=extract_throat(panel,frame)
    fig,ax=plt.subplots(figsize=(8,10),facecolor="white")
    for item in curves[0]:
        a=np.array([rows_data(item,u,frame,False)[0][0] for u in np.linspace(0,1,101)])
        b=np.array([rows_data(item,u,frame,False)[0][2] for u in np.linspace(0,1,101)])
        aa=local(frame,a);bb=local(frame,b)
        ax.plot(aa[:,0],aa[:,1],color="#394958",lw=2)
        ax.plot(bb[:,0],bb[:,1],color="#257eac",lw=2)
    ax.plot([],[],color="#394958",lw=2,label="A: контур исходного проёма")
    ax.plot([],[],color="#257eac",lw=2,label="B: меньший, более округлый")
    ax.legend();ax.set_aspect("equal");ax.grid(alpha=.15)
    ax.set_xlabel("X, мм");ax.set_ylabel("Высота, мм")
    ax.set_title("Контроль сужения по ширине и высоте\nОртогональная проекция управляющих границ")
    fig.tight_layout();fig.savefig(OUT/"cup_contours.png",dpi=155);plt.close(fig)

def outline(mesh,axes):
    verts,triangles,_,_=mesh
    vv=verts[:,axes]
    polys=[]
    for ids in triangles:
        p=Polygon(vv[ids])
        if p.area>0.000001:polys.append(p)
    shape=union_all(polys,grid_size=.002)
    if shape.geom_type=="MultiPolygon":shape=max(shape.geoms,key=lambda p:p.area)
    return np.array(shape.exterior.coords)

def handle_views():
    mesh=load_mesh(CAD/"handle.step",COLORS["Handle"])
    fig,axs=plt.subplots(2,2,figsize=(13,9),facecolor="white")
    front=outline(mesh,[0,2]);side=outline(mesh,[1,2]);top=outline(mesh,[0,1])
    traces=[]
    x=np.array([-50,-42,-25,0,25,42,50],float)
    z=np.array([0,5,10.5,12.5,10.5,5,0],float)
    traces.append(np.column_stack((np.r_[x,x[::-1]],np.r_[z,-z[::-1]])))
    traces.append(np.array([[0,-12.5],[-40,-6],[-40,6],[0,12.5],[0,-12.5]],float))
    traces.append(np.array([[-50,-3],[-35,-.5],[0,0],[35,-.5],[50,-3],
                            [47,-25],[36,-40],[25,-36],[0,-30],[-25,-36],[-36,-40],[-47,-25],[-50,-3]],float))
    for ax,c,trace,title,labels in zip(axs.flat[:3],[front,side,top],traces,
           ["Спереди / 100 × 25","Сбоку / глубина 40","Сверху / выпуклый и вогнутый контуры"],
           [("X, мм","Z, мм"),("Y, мм","Z, мм"),("X, мм","Y, мм")]):
        ax.fill(c[:,0],c[:,1],color="#e6d7b7",alpha=.5)
        ax.plot(c[:,0],c[:,1],color="#6d5131",lw=1.7,label="Контур STEP")
        ax.plot(trace[:,0],trace[:,1],"--",color="#3c9786",lw=1,label="Ручной перенос эскиза 6")
        ax.set_aspect("equal");ax.grid(alpha=.14);ax.set_title(title,fontsize=13)
        ax.set_xlabel(labels[0]);ax.set_ylabel(labels[1]);ax.legend(fontsize=8)
    axs[1,1].axis("off")
    axs[1,1].text(0,.9,"Проверка по трём проекциям",fontsize=16,weight="bold",va="top")
    axs[1,1].text(0,.72,"Спереди: две оси симметрии, линза.\nСбоку: симметрия по высоте, клин.\nСверху: симметрия по ширине;\nслабовыпуклая спинка и вогнутый захват.\n\nГабариты: 100 × 40 × 25 мм.\n\nПунктир — нормированный ручной перенос\nвидимых линий предоставленного эскиза,\nа не скан и не точный обмер.",fontsize=11,linespacing=1.6,va="top")
    fig.suptitle("Ручка шторки / ортогональные проекции и сравнение с эскизом",fontsize=17,weight="bold")
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/"handle_three_views.png",dpi=150);plt.close(fig)
    for name,direction in [("handle_front",(0,-1,0)),("handle_side",(1,0,0)),("handle_top",(0,0,1))]:
        shape=cq.importers.importStep(str(CAD/"handle.step")).val()
        cq.exporters.export(shape,str(OUT/f"{name}.svg"),exportType="SVG",
            opt={"projectionDir":direction,"showHidden":False,"width":800,"height":400,"strokeWidth":.6})
    img=rasterize(mesh,1100,700,[.35,-.8,.48],physical_size=(125,70),target=(0,-20,0),perspective=True,distance=350)
    save_image(img,"handle_perspective.png","Ручка / форма захвата","Линза, клиновидное сечение и вогнутый продольный контур.",(11,7))

def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
    panel=cq.importers.importStep(str(SOURCE)).val();frame=frames_from_panel(panel)[1]
    panel_mesh=load_mesh(SOURCE,COLORS["Panel"])
    meshes=[panel_mesh];window=[panel_mesh];half=[panel_mesh]
    for i in (1,2):
        cup=load_mesh(CAD/f"cup_{i}.step",COLORS["Cup"]);meshes.append(cup)
        for kind in ("shade","handle"):
            mesh=load_mesh(CAD/f"{kind}_{i}_open.step",COLORS[kind.title()])
            meshes.append(mesh)
            if i==2:window.append(mesh)
        if i==2:
            window.append(cup)
            half.append(cup)
            half.extend(load_mesh(CAD/f"{kind}_2_half.step",COLORS[kind.title()]) for kind in ("shade","handle"))
            cup_front(panel,frame,cup)
    allmesh=merge(meshes);wm=merge(window);hm=merge(half)
    print("Render assembly perspective",flush=True)
    img=rasterize(allmesh,1500,1500,[.4,-.9,.18],perspective=True,distance=2500)
    save_image(img,"assembly_perspective.png","Ту-214 / оконная сборка","Исходная панель и шесть новых отдельных деталей. Шторки подняты.")
    print("Render window views",flush=True)
    img=rasterize(wm,1050,1300,-frame.n,frame.origin,physical_size=(380,500))
    save_image(img,"window_open.png","Открытое положение","Пластина скрыта над окном за панелью; ручка доступна в верхней части проёма.")
    img=rasterize(wm,1050,1300,[.28,-.94,.16],frame.origin,(380,500),True,1450)
    save_image(img,"window_perspective.png","Оконный узел / перспектива","Внутренние боковые стенки чашки плавно сужаются к оконному блоку.")
    img=rasterize(hm,1050,1300,[.2,-.96,.12],frame.origin,(380,500),True,1450)
    save_image(img,"shade_half_closed.png","Частично опущенная шторка","190 мм хода по дуге; жёсткая цилиндрическая пластина сохраняет радиус.")
    print("Render actual BRep sections",flush=True)
    sections(panel,frame);handle_views()
    checksums={}
    for p in sorted((ROOT/"cad").glob("*.step")):
        checksums[str(p.relative_to(ROOT))]={"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
    (ROOT/"manifest.json").write_text(json.dumps(checksums,indent=2)+"\n")
    print("All required views saved",flush=True)

if __name__=="__main__":main()
