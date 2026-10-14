from pathlib import Path
import importlib.util,sys,math,json,os
os.environ.setdefault("MPLCONFIGDIR","/tmp/tu214-v6-mpl")
import numpy as np
import cadquery as cq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parent
spec=importlib.util.spec_from_file_location("v5",REPO/"window-revision-v5/build.py");v5=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v5;spec.loader.exec_module(v5)
from window_source import extract_throat,curve_data
OUT=ROOT/"diagnostic";OUT.mkdir(exist_ok=True)
panel=cq.importers.importStep(str(v5.base.SOURCE)).val()
frames=v5.base.frames_from_panel(panel);frame=frames[1]
def local(q):
    q=np.array(q)-np.array([frame.cx,frame.cy,frame.cz])
    c,s=math.cos(frame.theta),math.sin(frame.theta)
    return np.array([q[0],c*q[1]+s*q[2],-s*q[1]+c*q[2]])
loops,_=extract_throat(panel,frame)
data={"frame":vars(frame),"panel_throat_contours":[],"panel_cylinder_loops":[]}
fig,ax=plt.subplots(figsize=(10,10))
for i,loop in enumerate(loops):
    parts=[]
    for item in loop:
        pts=np.array([local(curve_data(item,t)[0]) for t in np.linspace(0,1,201)])
        parts.append(pts.tolist());ax.plot(pts[:,0],pts[:,2],lw=2,label=f"Panel throat {i}" if len(parts)==1 else None)
    data["panel_throat_contours"].append(parts)
for face in panel.Faces():
    if face.geomType()!="CYLINDER":continue
    bb=face.BoundingBox()
    if abs(face.Center().x-frame.cx)>100 or len(face.Wires())!=2:continue
    radius=float(face._geomAdaptor().Cylinder().Radius())
    if not 1700<radius<1800:continue
    inner=next(w for w in face.Wires() if not w.isSame(face.outerWire()))
    pts=np.array([local(v.toTuple()) for e in inner.Edges() for v in e.positions(np.linspace(0,1,101))])
    data["panel_cylinder_loops"].append({"radius":radius,"points":pts.tolist()})
    ax.plot(pts[:,0],pts[:,2],ls="--",label=f"Panel R{radius:g}")
cup=cq.importers.importStep(str(REPO/"window-revision-v5/cad/cup_local.step")).val()
for rad in (v5.RF,v5.RF+v5.T):
    edges=v5.lip_edges(cup,rad);parts=[]
    for e in edges:
        pts=np.array([p.toTuple() for p in e.positions(np.linspace(0,1,201))]);parts.append(pts.tolist())
        ax.plot(pts[:,0],pts[:,2],color="#d56a22",lw=1.5,label=f"Cup lip R{rad:g}" if len(parts)==1 else None)
    data[f"cup_lip_{rad}"]=parts
for a in v5.ARCS:
    pts=np.array([a.data(t)[0] for t in np.linspace(0,1,201)])
    ax.plot(pts[:,0],pts[:,1],color="#777",lw=1)
ax.set_aspect("equal");ax.grid(alpha=.2);ax.legend();ax.set_xlabel("X mm");ax.set_ylabel("Z mm")
fig.tight_layout();fig.savefig(OUT/"panel_cup_contours.png",dpi=150)
fig,ax=plt.subplots(figsize=(9,8))
for loop in data["panel_throat_contours"]:
    for pts in loop:
        p=np.array(pts);ax.plot(p[:,0],p[:,2],lw=2)
for key in (f"cup_lip_{v5.RF}",f"cup_lip_{v5.RF+v5.T}"):
    for pts in data[key]:
        p=np.array(pts);ax.plot(p[:,0],p[:,2],color="#d56a22",lw=2)
ax.set_xlim(20,170);ax.set_ylim(80,220);ax.set_aspect("equal");ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/"corner_contours.png",dpi=150)
(ROOT/"diagnostic.json").write_text(json.dumps(data,indent=2))
print("FRAMES",[vars(f) for f in frames],flush=True)
print("LOOPS",[(r["radius"],np.ptp(np.array(r["points"])[:,[0,2]],axis=0).tolist()) for r in data["panel_cylinder_loops"]],flush=True)
print("DIAGNOSTIC_COMPLETE",flush=True)
