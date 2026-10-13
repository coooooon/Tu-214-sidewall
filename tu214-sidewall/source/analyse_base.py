"""Measure the supplied panel and show why the shade-radius choice matters."""
from pathlib import Path
import os,json,math,hashlib
os.environ.setdefault("MPLCONFIGDIR","/tmp/tu214-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME","/tmp/tu214-cache")
import numpy as np
import cadquery as cq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from window_source import frames_from_panel,extract_throat,curve_data
from profile_geometry import RoundedProfile,contour_arcs
from build import P,shade_radius

ROOT=Path(__file__).resolve().parents[1]


def developed(points,frame):
    return np.column_stack((points[:,0]-frame.cx,1753*(np.arctan2(points[:,2]-frame.cz,points[:,1]-frame.cy)-frame.theta)))


def main():
    path=ROOT/"cad/tu214_sidewall_original.step";panel=cq.importers.importStep(str(path)).val()
    frame=frames_from_panel(panel)[0];curves,wires=extract_throat(panel,frame)
    source=np.vstack([np.array([curve_data(item,u)[0] for u in np.linspace(0,1,41,endpoint=False)]) for item in curves[0]])
    a=developed(source,frame)
    b=RoundedProfile(contour_arcs(P["rear_opening_width_mm"],P["rear_opening_height_mm"],P["rear_side_radius_mm"],
        P["rear_top_crown_radius_mm"],P["rear_bottom_crown_radius_mm"],P["rear_upper_corner_radius_mm"],P["rear_lower_corner_radius_mm"]));bp=b.points(801)
    curvature=[]
    for segment in b.segments:
        for t in np.linspace(0,1,151):
            q,d,dd=segment.data(t);curvature.append(float((d[0]*dd[1]-d[1]*dd[0])/np.linalg.norm(d)**3))
    phi=np.arctan2(a[:,1],a[:,0]);order=np.argsort(phi);phi=phi[order];rad=np.linalg.norm(a,axis=1)[order]
    x=np.r_[phi,phi[0]+2*math.pi];y=np.r_[rad,rad[0]]
    arad=make_interp_spline(x,y,k=3,bc_type="periodic")
    phases=np.linspace(phi[0],phi[0]+2*math.pi,1001)
    ra=arad(phases);rb=np.array([np.linalg.norm(b.data(p)[0]) for p in phases])
    scale=float(np.dot(ra,rb)/np.dot(ra,ra));error=rb-scale*ra
    report={"source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"source_valid":panel.isValid(),
        "source_solid_count":len(panel.Solids()),"source_volume_mm3":panel.Volume(),
        "window_axis_y_mm":frame.cy,"window_axis_z_mm":frame.cz,"original_main_panel_radius_mm":1705,
        "window_floor_radius_mm":frame.radius,"existing_throat_radius_mm":1753,
        "front_contour_width_height_developed_mm":np.ptp(a,axis=0).tolist(),
        "proposed_independent_B_width_height_mm":np.ptp(bp,axis=0).tolist(),
        "B_upper_corner_radius_mm":P["rear_upper_corner_radius_mm"],"B_lower_corner_radius_mm":P["rear_lower_corner_radius_mm"],
        "B_side_radius_mm":P["rear_side_radius_mm"],"B_top_crown_radius_mm":P["rear_top_crown_radius_mm"],"B_bottom_crown_radius_mm":P["rear_bottom_crown_radius_mm"],
        "B_G2_transition_length_mm":8,"B_min_signed_curvature_per_mm":min(curvature),
        "B_vs_scaled_A_best_scale":scale,"B_vs_scaled_A_max_difference_mm":float(np.max(np.abs(error))),
        "radius_requirement_conflict":"R < 1705 is inboard of the panel; a concentric shade behind the reveal must be outside the existing throat R1753."}
    (ROOT/"cad/base_analysis.json").write_text(json.dumps(report,indent=2)+"\n")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11})
    fig,axes=plt.subplots(1,2,figsize=(13,7),facecolor="white")
    ax=axes[0];ax.plot(np.r_[a[:,0],a[0,0]],np.r_[a[:,1],a[0,1]],color="#304a65",label="A: реальный торец проёма v5")
    ax.plot(bp[:,0],bp[:,1],color="#cf8543",label="B: самостоятельный внутренний контур")
    ax.set_aspect("equal");ax.set_xlabel("X от центра окна, мм");ax.set_ylabel("Развёрнутая координата по дуге, мм")
    ax.set_title("Различные контуры A и B");ax.grid(alpha=.15);ax.legend(loc="lower center",fontsize=9)
    ax=axes[1]
    angles=np.linspace(frame.theta-.2,frame.theta+.38,501)
    for radius,color,label in [(1705,"#304a65","Основная панель R1705"),(1735,"#6e8195","Дно выемки R1735"),(1695,"#b95e58","R1695: со стороны салона"),(shade_radius(),"#458c71",f"Шторка R{shade_radius():g} за чашкой")]:
        ax.plot(frame.cy+radius*np.cos(angles),frame.cz+radius*np.sin(angles),color=color,lw=1.5,label=label)
    ax.scatter([frame.cy+1753*math.cos(frame.theta)],[frame.cz+1753*math.sin(frame.theta)],color="#cf8543",s=28,label="Торец проёма R1753")
    ax.set_aspect("equal");ax.set_xlim(-210,200);ax.set_ylim(100,1210);ax.set_xlabel("Y, мм");ax.set_ylabel("Z, мм")
    ax.set_title("Вертикальная плоскость через окно");ax.grid(alpha=.15);ax.legend(loc="upper right",fontsize=8)
    fig.suptitle("Исходная v5 и принятая компоновка оконного узла",fontsize=15)
    fig.tight_layout();fig.savefig(ROOT/"previews/base_analysis.png",dpi=150)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
