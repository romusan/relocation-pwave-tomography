"""Closed triangulated shells and static geographic QA for NURBS fragments."""
from pathlib import Path
import json,argparse
import numpy as np
from collections import Counter
from regional.geometry import Projection
from regional.inputs import sha256

def solid(mid,normal,faces,h):
    n=len(mid);faces=faces.copy();cross=np.cross(mid[faces[:,1]]-mid[faces[:,0]],mid[faces[:,2]]-mid[faces[:,0]])
    wrong=np.einsum('ij,ij->i',cross,normal[faces].mean(axis=1))<0
    faces[wrong]=faces[wrong][:,[0,2,1]]
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);count=Counter(map(tuple,np.sort(edges,axis=1)))
    border=[(a,b) for a,b in edges if count[tuple(sorted((a,b)))]==1]
    sides=np.array([(a,b,n+b) for a,b in border]+[(a,n+b,n+a) for a,b in border],int)
    sf=np.vstack([faces[:,[0,2,1]],faces+n,sides]);vertices=np.vstack([mid-h/2*normal,mid+h/2*normal])
    counts=Counter(map(tuple,np.sort(np.concatenate([sf[:,[0,1]],sf[:,[1,2]],sf[:,[2,0]]]),axis=1)))
    if not all(v==2 for v in counts.values()):raise ValueError('Nonmanifold shell')
    return vertices,sf

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    patches=json.loads((a.model/'superficies_nurbs.json').read_text(encoding='utf-8'));domain=json.loads((a.run/'domain.json').read_text());pr=Projection(domain)
    report=[]
    for p in patches:
        middle=np.asarray(p['surface']);normals=np.asarray(p['normals']);faces=np.asarray(p['faces'])
        for h in [30,50,70]:
            v,f=solid(middle,normals,faces,h)
            np.savez_compressed(a.model/f'{p["id"]}_solido_{h}.npz',vertices=v,triangles=f)
            report.append(dict(fragment=p['id'],thickness_km=h,vertices=len(v),faces=len(f),all_edges_two_faces=True,limitation='Closed topology; no guarantee against self-intersection for a large imposed thickness'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
    from agregar_mapa_superficie_norte import boundaries,lines
    from shapely.geometry import box
    from shapely.ops import unary_union
    fig=plt.figure(figsize=(13,9));ax=fig.add_subplot(projection='3d');ax.set_position([.03,.10,.80,.77]);colors=plt.cm.tab10(np.arange(len(patches)))
    for p,color in zip(patches,colors):
        mid=np.asarray(p['surface']);norm=np.asarray(p['normals']);faces=np.asarray(p['faces'])
        for off,alpha in [(0,.65),(-15,.18),(15,.18)]:
            v=mid+off*norm;lat,lon=pr.latlon(v[:,0],v[:,1]);points=np.c_[lon,lat,v[:,2]]
            ax.add_collection3d(Poly3DCollection(points[faces],facecolor=color,edgecolor='none',alpha=alpha))
        la,lo=pr.latlon(mid[:,0],mid[:,1]);ax.text(float(np.mean(lo)),float(np.mean(la)),float(np.mean(mid[:,2])),p['id'].replace('fragmento_','F'),fontsize=9)
    _,_,features=boundaries();region=box(-79,2,-71,13);poly=[v[1].intersection(region) for v in features.values()]
    for geo,color,width in [(unary_union([p.boundary for p in poly]),'#999999',.3),(unary_union(poly).boundary,'#25353f',.9)]:
        segments=[np.c_[p[:,0],p[:,1],np.zeros(len(p))] for p in lines(geo)];ax.add_collection3d(Line3DCollection(segments,colors=color,linewidths=width))
    ax.set(xlabel='Longitud WGS84',ylabel='Latitud WGS84',zlabel='Profundidad km',xlim=(-79,-71),ylim=(2,13),zlim=(180,0));ax.set_box_aspect([.726,1,.148]);ax.view_init(elev=25,azim=-65)
    fig.suptitle('Cuatro fragmentos NURBS provisionales y espesor normal de 30 km',fontsize=15);fig.text(.04,.04,'Caras superior e inferior translúcidas. Identidad tectónica sin asignar. Huecos preservados.',fontsize=10)
    fig.savefig(a.model/'vista_nurbs_30km.png',dpi=180,bbox_inches='tight');plt.close(fig)
    method=json.loads((a.model/'metodologia.json').read_text(encoding='utf-8'))
    method['closed_shells']=report;method['input_sha256'][str(Path(__file__).resolve())]=sha256(__file__)
    (a.model/'metodologia.json').write_text(json.dumps(method,indent=2),encoding='utf-8')
    print('PASS: 12 closed shells, every edge has exactly two faces')
if __name__=='__main__':main()
