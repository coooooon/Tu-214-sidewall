"""Compare actual STEP contours with the marked photo in a fitted view.

Only the visible outer lip is used to fit the camera. The inner contour is
then projected without changing its dimensions. Landmarks are approximate
pixel readings from the supplied, cropped and partially obscured image.
"""
from pathlib import Path
import os,json,math
os.environ.setdefault("XDG_CACHE_HOME","/tmp/tu214-cache")
os.environ.setdefault("MPLCONFIGDIR","/tmp/tu214-matplotlib")
import numpy as np
import cadquery as cq
from OCP.BRepAdaptor import BRepAdaptor_Surface
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from window_source import frames_from_panel,extract_throat,curve_data
from profile_geometry import RoundedProfile,contour_arcs

ROOT=Path(__file__).resolve().parents[1]
OUTER=np.array([[1084,343],[1068,350],[1051,374],[1035,416],[1021,481],[1008,546],
    [997,607],[995,650],[1007,688],[1040,715],[1092,737],[1150,751],[1200,754],[1240,747],[1256,736]],dtype=float)
INNER=np.array([[1096,364],[1078,389],[1061,435],[1048,496],[1047,559],[1049,603],
    [1064,636],[1092,657],[1132,675],[1180,686],[1221,687],[1250,677]],dtype=float)
MIDDLE=np.array([[592,293],[595,266],[601,241],[613,221],[629,212],[647,214],[677,222],[715,233],[753,245],[773,253]],dtype=float)
FAR=np.array([[296,122],[312,121],[339,127],[372,137],[399,150],[416,169],[424,194],[426,223]],dtype=float)


def project(points,params,origin):
    yaw,pitch,roll,logf,logdistance,tx,ty=params
    camera=np.array([math.sin(yaw)*math.cos(pitch),-math.cos(yaw)*math.cos(pitch),math.sin(pitch)])
    right=np.cross([0,0,1],camera);right/=np.linalg.norm(right);up=np.cross(camera,right)
    r=right*math.cos(roll)+up*math.sin(roll);u=up*math.cos(roll)-right*math.sin(roll)
    relative=points-origin;depth=math.exp(logdistance)-relative@camera
    return np.column_stack((math.exp(logf)*(relative@r)/depth+tx,-math.exp(logf)*(relative@u)/depth+ty))


def closest_vectors(points,curve):
    start=curve;end=np.roll(curve,-1,axis=0);delta=end-start
    relative=points[:,None,:]-start[None,:,:]
    fraction=np.clip(np.sum(relative*delta[None,:,:],axis=2)/np.maximum(np.sum(delta*delta,axis=1),1e-20)[None,:],0,1)
    projected=start[None,:,:]+fraction[:,:,None]*delta[None,:,:]
    difference=points[:,None,:]-projected
    index=np.argmin(np.sum(difference*difference,axis=2),axis=1)
    return difference[np.arange(len(points)),index]


def main():
    photo=ROOT/"references/18205_marked.png"
    if not photo.exists():raise RuntimeError("Marked reference image is needed for this optional comparison")
    panel=cq.importers.importStep(str(ROOT/"cad/tu214_sidewall_original.step")).val();frame=frames_from_panel(panel)[1]
    face=next(f for f in panel.Faces() if f.geomType()=="CYLINDER" and len(f.Wires())==2 and
        abs(f._geomAdaptor().Cylinder().Radius()-frame.radius)<1e-8 and abs(f.Center().x-frame.cx)<100)
    wire=next(w for w in face.Wires() if not w.isSame(face.outerWire()))
    outer=np.vstack([np.array([p.toTuple() for p in e.positions(np.linspace(0,1,101,endpoint=False))]) for e in wire.Edges()])
    # Sort convex projected points by polar phase in the developed window plane.
    xy=np.column_stack((outer[:,0]-frame.cx,(outer-frame.origin)@frame.up));order=np.argsort(np.arctan2(xy[:,1],xy[:,0]));outer=outer[order]
    reveal=cq.importers.importStep(str(ROOT/"cad/window_reveal_2.step")).val();curves,_=extract_throat(panel,frame)
    walls=[f for f in reveal.Faces() if f.geomType()=="BSPLINE" and f._geomAdaptor().VDegree()==5];inner=[]
    def point(f,u,v):
        p=BRepAdaptor_Surface(f.wrapped).Value(u,v);return np.array([p.X(),p.Y(),p.Z()])
    for item in curves[0]:
        target=curve_data(item,.5)[0];selected=min(walls,key=lambda f:np.linalg.norm(point(f,.5,0)-target));walls.remove(selected)
        inner.extend(point(selected,u,1) for u in np.linspace(0,1,81,endpoint=False))
    inner=np.array(inner)
    world_middle=outer-np.array([600.,0.,0.]);world_far=outer-np.array([1200.,0.,0.])
    def residual(params):return np.concatenate([closest_vectors(points,project(world,params,frame.origin)).ravel()
        for points,world in ((OUTER,outer),(MIDDLE,world_middle),(FAR,world_far))])
    best=None
    for yaw in (.15,.45,.75):
        initial=[yaw,.3,.2,math.log(1400),math.log(1300),1170,550]
        fit=least_squares(residual,initial,bounds=([0.,-.2,-.4,math.log(500),math.log(600),1000,400],
            [1.25,.8,.6,math.log(5000),math.log(6000),1450,750]),max_nfev=250,loss="soft_l1",f_scale=3.)
        if best is None or np.linalg.norm(fit.fun)<np.linalg.norm(best.fun):best=fit
    op=project(outer,best.x,frame.origin);ip=project(inner,best.x,frame.origin)
    oe=np.linalg.norm(closest_vectors(OUTER,op),axis=1);ie=np.linalg.norm(closest_vectors(INNER,ip),axis=1)
    report={"purpose":"Approximate view fit and proportional comparison, not metrology",
        "camera_fitted_using":"Three repeated outer front contours at 600 mm pitch; inner contour was not used to fit camera",
        "outer_landmark_rms_pixels":float(np.sqrt(np.mean(oe*oe))),"inner_landmark_rms_pixels":float(np.sqrt(np.mean(ie*ie))),
        "outer_landmarks_pixels":OUTER.tolist(),"inner_landmarks_pixels":INNER.tolist(),
        "middle_outer_landmarks_pixels":MIDDLE.tolist(),"far_outer_landmarks_pixels":FAR.tolist(),
        "camera_params_yaw_pitch_roll_logf_logdistance_tx_ty":best.x.tolist(),"fitted_camera_distance_mm":math.exp(best.x[4]),
        "all_outer_landmark_rms_pixels":float(np.sqrt(np.mean(best.fun*best.fun))),
        "limitations":"Image is cropped, blurred and partly marked; right-hand contour is absent. Camera/dimensions are not uniquely recoverable."}
    (ROOT/"cad/reference_comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    im=np.array(Image.open(photo));fig,axes=plt.subplots(1,2,figsize=(10,10),facecolor="white")
    for ax in axes:
        ax.imshow(im);ax.set_xlim(975,1270);ax.set_ylim(775,315);ax.set_aspect("equal");ax.axis("off")
    axes[0].set_title("Предоставленная фотография",fontsize=12)
    axes[1].plot(np.r_[op[:,0],op[0,0]],np.r_[op[:,1],op[0,1]],lw=1.4,color="#25bed4",label="A: исходный проём v5")
    axes[1].plot(np.r_[ip[:,0],ip[0,0]],np.r_[ip[:,1],ip[0,1]],lw=1.4,color="#f6a04d",label="B: новая чашка, из STEP")
    axes[1].set_title("Контуры CAD в подобранном ракурсе",fontsize=12);axes[1].legend(loc="lower left",fontsize=8)
    fig.suptitle("Проверка пропорций по размеченному снимку",fontsize=16)
    fig.text(.08,.05,"Ракурс подобран по контуру A. Фотография не задаёт однозначных размеров и глубины.",fontsize=10,color="#56616a")
    fig.tight_layout(rect=(0,.07,1,.95));fig.savefig(ROOT/"previews/reference_comparison.png",dpi=150)
    print(json.dumps({k:v for k,v in report.items() if 'landmarks_pixels' not in k},indent=2),flush=True)
    # Diagnostic design fit: camera stays fixed; vary only rear dimensions
    # and depth within plausible bounds. This does not alter model files.
    def rear_world(values):
        width,height,radius=values
        profile=RoundedProfile(contour_arcs(width,height,1800,720,540,100,108))
        b=profile.points(401);theta=frame.theta+b[:,1]/radius
        return np.column_stack((frame.cx+b[:,0],frame.cy+radius*np.cos(theta),frame.cz+radius*np.sin(theta)))
    def rear_residual(values):return closest_vectors(INNER,project(rear_world(values),best.x,frame.origin)).ravel()
    shape_fit=least_squares(rear_residual,[286,360,1820],bounds=([240,330,1780],[296,400,1890]),max_nfev=120,loss="soft_l1",f_scale=3.)
    design={"estimated_rear_width_height_radius_mm":shape_fit.x.tolist(),
        "inner_trace_rms_pixels":float(np.sqrt(np.mean(np.sum(shape_fit.fun.reshape(-1,2)**2,axis=1)))),"camera_fixed_from_three_outer_contours":True,
        "rear_width_upper_bound_reason":"4 mm below the supplied 300 mm throat; preserve passage through the existing mouth"}
    (ROOT/"cad/reference_shape_estimate.json").write_text(json.dumps(design,indent=2)+"\n")
    print(json.dumps(design,indent=2),flush=True)


if __name__=="__main__":main()
