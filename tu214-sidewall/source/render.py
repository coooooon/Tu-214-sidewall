"""Five required views, rendered and sectioned from the delivered STEP parts."""
from pathlib import Path
import os,math,json
os.environ.setdefault("XDG_CACHE_HOME","/tmp/tu214-cache")
os.environ.setdefault("MPLCONFIGDIR","/tmp/tu214-matplotlib")
import numpy as np
import cadquery as cq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from preview_mesh import load_mesh,merge,rasterize
from window_source import frames_from_panel
from build import P,ROOT,CAD,shade_radius

OUT=ROOT/"previews"


def save_view(image,name,title,subtitle):
    fig=plt.figure(figsize=(10,10),facecolor="#f6f7f9")
    fig.text(.05,.957,title,fontsize=19,weight="bold",color="#243447")
    fig.text(.05,.917,subtitle,fontsize=10.5,color="#647281")
    ax=fig.add_axes([.025,.085,.95,.80]);ax.imshow(image);ax.axis("off")
    fig.text(.05,.033,"Геометрия изображения получена из экспортированных STEP-деталей.",fontsize=9,color="#647281")
    fig.savefig(OUT/name,dpi=145,facecolor=fig.get_facecolor());plt.close(fig)


def section_faces(path,x):
    shape=cq.importers.importStep(str(path)).val()
    plane=cq.Plane(origin=(x,0,0),xDir=(0,1,0),normal=(1,0,0))
    section=cq.Workplane(plane).newObject([shape]).section().val()
    triangles=[];curves=[]
    for face in section.Faces():
        vertices,indices=face.tessellate(.015,.06);points=np.array([p.toTuple() for p in vertices])
        triangles.extend(points[list(ids),1:3] for ids in indices)
    for edge in section.Edges():curves.append(np.array([p.toTuple()[1:3] for p in edge.positions(np.linspace(0,1,81))]))
    return triangles,curves


def section_view(frame):
    data=[]
    for file,color,label in [("tu214_sidewall_original","#718294","Панель v5"),
        ("window_reveal_2","#b1bac4","Чашка"),("window_shade_2_open","#438f88","Шторка"),
        ("shade_handle_2_open","#d5aa6d","Ручка")]:
        print("Section",file,flush=True);triangles,curves=section_faces(CAD/f"{file}.step",frame.cx);data.append((triangles,curves,color,label))
    fig=plt.figure(figsize=(13,10),facecolor="white");fig.text(.045,.955,"ВЕРТИКАЛЬНОЕ СЕЧЕНИЕ / X = 300 мм",fontsize=20,weight="bold",color="#243447")
    full=fig.add_axes([.045,.11,.42,.76]);detail=fig.add_axes([.55,.42,.41,.43])
    upper=frame.theta+(P["rear_opening_height_mm"]/2+P["cup_rear_wall_width_mm"])/P["cup_rear_radius_mm"]
    point=np.array([frame.cy+P["cup_rear_radius_mm"]*math.cos(upper),frame.cz+P["cup_rear_radius_mm"]*math.sin(upper)])
    for ax in (full,detail):
        for triangles,curves,color,label in data:
            if triangles:ax.add_collection(PolyCollection(triangles,facecolor=color,edgecolor="none",alpha=.85))
            for j,curve in enumerate(curves):ax.plot(curve[:,0],curve[:,1],color=color,lw=1.25,label=label if j==0 else None)
        th=np.linspace(frame.theta-.23,frame.theta+.4,501);reference=P["cup_rear_radius_mm"]+P["window_block_radial_clearance_mm"]
        ax.plot(frame.cy+reference*np.cos(th),frame.cz+reference*np.sin(th),color="#ae7662",ls="--",lw=1.,label="Примыкание оконного блока, принятое")
        ax.set_aspect("equal");ax.set_xlabel("Y, мм");ax.set_ylabel("Z, мм");ax.grid(alpha=.14)
    full.set_xlim(-200,frame.cy+P["cup_rear_radius_mm"]+P["window_block_radial_clearance_mm"]+35);full.set_ylim(220,1215);full.legend(loc="lower left",fontsize=8)
    detail.set_xlim(point[0]-44,point[0]+40);detail.set_ylim(point[1]-30,point[1]+65)
    detail.set_title("Рабочий зазор за чашкой",loc="left",fontsize=13,color="#243447")
    a=point;b=point+P["window_block_radial_clearance_mm"]*np.array([math.cos(upper),math.sin(upper)])
    detail.annotate("",xy=a,xytext=b,arrowprops={"arrowstyle":"<->","color":"#b26746","lw":1.3})
    detail.text((a[0]+b[0])/2+2,(a[1]+b[1])/2+6,"15 мм",color="#b26746",fontsize=12)
    fig.text(.55,.33,f"Чашка: задняя кромка R{P['cup_rear_radius_mm']:g}",fontsize=12,color="#243447")
    fig.text(.55,.29,f"Шторка: R{shade_radius():g} → R{shade_radius()+P['shade_thickness_mm']:g}",fontsize=12,color="#438f88")
    fig.text(.55,.25,"Зазоры до шторки: 7,0 мм спереди / 6,2 мм сзади",fontsize=11,color="#647281")
    fig.text(.55,.21,"Движение — поворот относительно общей оси X",fontsize=11,color="#647281")
    fig.text(.045,.05,"Штриховая линия задаёт принятое положение примыкания оконного блока; заводская геометрия блока не предоставлена.",fontsize=9,color="#647281")
    fig.savefig(OUT/"vertical_section.png",dpi=155);plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True);plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
    panel=cq.importers.importStep(str(CAD/"tu214_sidewall_original.step")).val();frame=frames_from_panel(panel)[1]
    meshes=[load_mesh(CAD/"tu214_sidewall_original.step",[234,227,214])]
    for i in (1,2):
        for name,color in [(f"window_reveal_{i}",[248,246,237]),(f"window_shade_{i}_open",[210,208,194]),
            (f"shade_handle_{i}_open",[237,222,188]),(f"shade_guide_{i}_left",[147,154,162]),(f"shade_guide_{i}_right",[147,154,162])]:
            meshes.append(load_mesh(CAD/f"{name}.step",color))
    allmesh=merge(meshes);window=merge([meshes[0]]+meshes[6:11])
    target=frame.origin;front=[0,-math.cos(frame.theta),-math.sin(frame.theta)]
    print("Render orthographic front",flush=True)
    image=rasterize(window,1200,1450,front,target,physical_size=(430,570))
    save_view(image,"front.png","ИЛЛЮМИНАТОР / ОРТОГОНАЛЬНЫЙ ВИД",f"Контуры A и B различаются; внутренние углы R{P['rear_upper_corner_radius_mm']:g} / R{P['rear_lower_corner_radius_mm']:g} в развёртке")
    print("Render true perspective",flush=True)
    image=rasterize(allmesh,1550,1550,[.4,-.89,.18],perspective=True,distance=2200)
    save_view(image,"perspective.png","ПАНЕЛЬ v5 / ВИД ИЗ САЛОНА","Открытое положение. Чашки, шторки, ручки и скрытые направляющие — отдельные тела.")
    print("Render handle detail",flush=True)
    theta=frame.theta+(P["shade_closed_lower_edge_mm"]+8+P["shade_open_travel_mm"])/shade_radius()
    r=shade_radius()-P["handle_grip_projection_mm"]
    focus=[frame.cx,frame.cy+r*math.cos(theta),frame.cz+r*math.sin(theta)+8]
    image=rasterize(window,1450,900,[.12,-.985,.08],focus,(215,125),True,1100)
    save_view(image,"handle_detail.png","РУЧКА / ОТКРЫТОЕ ПОЛОЖЕНИЕ",f"Низкая планка шириной {P['handle_grip_width_mm']:g} мм; выступ {P['handle_grip_projection_mm']:g} мм; плавные закругления")
    half=merge([meshes[0],meshes[6],load_mesh(CAD/"window_shade_2_half.step",[210,208,194]),
        load_mesh(CAD/"shade_handle_2_half.step",[237,222,188]),meshes[9],meshes[10]])
    print("Render half closed",flush=True)
    image=rasterize(half,1200,1450,[.25,-.96,.1],target,(430,570),True,1400)
    save_view(image,"shade_half_closed.png","ШТОРКА / ЧАСТИЧНО ЗАКРЫТО",f"Полхода: {P['shade_open_travel_mm']/2:g} мм по дуге; цилиндрическая кривизна сохранена")
    section_view(frame)
    print("Five required STEP views saved",flush=True)


if __name__=="__main__":main()
