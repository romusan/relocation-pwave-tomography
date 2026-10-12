from pathlib import Path
import sys,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'codigos'))
from nazca_campana_20261010 import projection
from agregar_mapa_superficie_norte import boundaries,lines
OUT=ROOT/'articulo_boletin_geologia_uis_adjunto'
ev=pd.read_csv(OUT/'suplementos_20261011/clasificacion_kmeans_sismos.csv')
pr=projection(json.loads((ROOT/'resultados/tomografia_1d_adjunto_consistente_2026-10-10/domain.json').read_text()))
_,_,features=boundaries();fig,ax=plt.subplots(figsize=(7,7),layout='constrained')
for name,geom in features.values():
    for pts in lines(geom.boundary):ax.plot(pts[:,0],pts[:,1],c='#999',lw=.5)
for label,color in [(1,'#7926b3'),(2,'#ed831b')]:
    e=ev[ev.kmeans_km_k2==label];ax.scatter(e.lon,e.lat,s=10,color=color,alpha=.6,label=f'Grupo {label}')
lon=np.linspace(-79,-71,250);lat=6+(lon+72.5)*3.2/(-3.8);ax.plot(lon,lat,'--',color='#666',lw=1.5,label='Diagonal propuesta')
centers=np.array([ev[ev.kmeans_km_k2==i][['x_km','y_km','depth_km']].mean().to_numpy() for i in [1,2]])
lo,la=np.meshgrid(lon,np.linspace(.8,13,300));x,y=pr.xy(la,lo);z=np.full_like(x,80)
a=((x-centers[0,0])**2+(y-centers[0,1])**2+(z-centers[0,2])**2)
b=((x-centers[1,0])**2+(y-centers[1,1])**2+(z-centers[1,2])**2)
ax.contour(lo,la,a-b,levels=[0],colors='black',linewidths=1.5)
ax.plot([],[],color='black',label='Bisector k-means a 80 km')
ax.set(xlim=(-79,-71),ylim=(.8,13),xlabel='Longitud (°)',ylabel='Latitud (° N)');ax.set_aspect(1/np.cos(np.deg2rad(7)));ax.legend(fontsize=10,loc='lower right');ax.grid(alpha=.15)
fig.savefig(OUT/'figuras/Figura_7_clasificacion_kmeans.png',dpi=300)
