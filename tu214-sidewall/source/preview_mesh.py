"""Software CAD preview with analytic normals and a true perspective camera."""
import math
import numpy as np
import cadquery as cq
from OCP.BRep import BRep_Tool
from OCP.BRepLib import BRepLib_ToolTriangulatedShape
from OCP.TopLoc import TopLoc_Location
from OCP.TopAbs import TopAbs_REVERSED
from OCP.BRepMesh import BRepMesh_IncrementalMesh


def load_mesh(path,color):
    shape=cq.importers.importStep(str(path)).val()
    BRepMesh_IncrementalMesh(shape.wrapped,.12,False,.08,True)
    vertices,triangles=shape.tessellate(.2,.12)
    verts=np.array([v.toTuple() for v in vertices]);indices=np.array(triangles);normals=[]
    for face in shape.Faces():
        location=TopLoc_Location();poly=BRep_Tool.Triangulation_s(face.wrapped,location)
        BRepLib_ToolTriangulatedShape.ComputeNormals_s(face.wrapped,poly)
        sign=-1 if face.wrapped.Orientation()==TopAbs_REVERSED else 1
        for i in range(1,poly.NbNodes()+1):
            n=poly.Normal(i).Transformed(location.Transformation());normals.append([sign*n.X(),sign*n.Y(),sign*n.Z()])
    normals=np.array(normals);normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-12)
    if len(normals)!=len(verts):raise RuntimeError("Wrong mesh normal count")
    colors=np.broadcast_to(np.array(color,dtype=float),verts.shape).copy()
    print(path.name,len(indices),"triangles",flush=True)
    return verts,indices,normals,colors


def merge(meshes):
    offsets=np.cumsum([0]+[len(m[0]) for m in meshes])
    return np.vstack([m[0] for m in meshes]),np.vstack([m[1]+off for m,off in zip(meshes,offsets)]),np.vstack([m[2] for m in meshes]),np.vstack([m[3] for m in meshes])


def rasterize(mesh,width,height,camera,target=None,physical_size=None,perspective=False,distance=2100.):
    verts,indices,normals,basecolors=mesh
    camera=np.array(camera,dtype=float);camera/=np.linalg.norm(camera)
    right=np.cross([0,0,1],camera);right/=np.linalg.norm(right);up=np.cross(camera,right)
    if target is None:target=(verts.min(axis=0)+verts.max(axis=0))/2
    centered=verts-np.array(target)
    projected=np.column_stack((centered@right,centered@up,centered@camera))
    if perspective:
        dist=distance-projected[:,2]
        if dist.min()<=0:raise RuntimeError("Geometry is behind the perspective camera")
        projected[:,:2]/=dist[:,None];projected[:,2]=1/dist
    if physical_size:
        scale=min(width/physical_size[0],height/physical_size[1])*(distance if perspective else 1.)
        mid=np.zeros(2)
    else:
        extent=np.ptp(projected[:,:2],axis=0);scale=.91*min(width/extent[0],height/extent[1])
        mid=(projected[:,:2].min(axis=0)+projected[:,:2].max(axis=0))/2
    projected[:,0]=width/2+(projected[:,0]-mid[0])*scale
    projected[:,1]=height/2-(projected[:,1]-mid[1])*scale
    light=np.array([-.4,-.8,.55]);light/=np.linalg.norm(light);half=light+camera;half/=np.linalg.norm(half)
    intensity=.52+.43*np.maximum(normals@light,0)+.10*np.maximum(normals@half,0)**32
    colors=basecolors*intensity[:,None]
    rgb=np.broadcast_to(np.array([246,247,249],dtype=np.uint8),(height,width,3)).copy();depth=np.full((height,width),-np.inf)
    for ids in indices:
        p=projected[ids]
        x0=max(0,int(math.floor(p[:,0].min())));x1=min(width-1,int(math.ceil(p[:,0].max())))
        y0=max(0,int(math.floor(p[:,1].min())));y1=min(height-1,int(math.ceil(p[:,1].max())))
        if x1<x0 or y1<y0:continue
        xx,yy=np.meshgrid(np.arange(x0,x1+1)+.5,np.arange(y0,y1+1)+.5)
        den=(p[1,1]-p[2,1])*(p[0,0]-p[2,0])+(p[2,0]-p[1,0])*(p[0,1]-p[2,1])
        if abs(den)<1e-10:continue
        a=((p[1,1]-p[2,1])*(xx-p[2,0])+(p[2,0]-p[1,0])*(yy-p[2,1]))/den
        b=((p[2,1]-p[0,1])*(xx-p[2,0])+(p[0,0]-p[2,0])*(yy-p[2,1]))/den;c=1-a-b
        zz=a*p[0,2]+b*p[1,2]+c*p[2,2];targetdepth=depth[y0:y1+1,x0:x1+1]
        mask=(a>=-1e-7)&(b>=-1e-7)&(c>=-1e-7)&(zz>targetdepth)
        if perspective:
            value=(a[:,:,None]*colors[ids[0]]*p[0,2]+b[:,:,None]*colors[ids[1]]*p[1,2]+c[:,:,None]*colors[ids[2]]*p[2,2])/zz[:,:,None]
        else:value=a[:,:,None]*colors[ids[0]]+b[:,:,None]*colors[ids[1]]+c[:,:,None]*colors[ids[2]]
        value=np.clip(value,0,255).astype(np.uint8)
        targetdepth[mask]=zz[mask];rgb[y0:y1+1,x0:x1+1][mask]=value[mask]
    return rgb
