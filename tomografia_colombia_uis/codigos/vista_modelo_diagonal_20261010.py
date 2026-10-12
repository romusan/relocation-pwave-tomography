"""QA of the 1D background plus two candidate slab initial velocity bands."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'codigos'))
from nazca_campana_20261010 import projection
from regional.inputs import sha256
OUT=ROOT/'resultados/modelo_1d_placas_diagonal_2026-10-10'
def main():
    with np.load(OUT/'vp_inicial_1d_placas_30km.npz') as m:
        vp=m['vp'];bg=m['vp_background_1d'];x,y,z=m['xg'],m['yg'],m['zg'];owner=m['slab_owner'];domain=json.loads(str(m['domain_json']))
        delta=100*(vp/bg-1)
    pr=projection(domain);xx,yy=np.meshgrid(x,y);la,lo=pr.latlon(xx.ravel(),yy.ravel());la=la.reshape(xx.shape);lo=lo.reshape(xx.shape)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12,12),layout='constrained')
    from agregar_mapa_superficie_norte import boundaries,lines
    from shapely.geometry import box
    from shapely.ops import unary_union
    path,meta,features=boundaries();polys=[p[1].intersection(box(-79,.8,-71,13)) for p in features.values()]
    borders=[unary_union(polys).boundary,unary_union([p.boundary for p in polys])]
    for ax,depth in zip(axs.ravel(),[40,80,120,160]):
        k=int(np.argmin(abs(z-depth)));im=ax.pcolormesh(lo,la,delta[k],cmap='RdBu_r',vmin=-3,vmax=3,shading='nearest',rasterized=True)
        for value,col in [(1,'#7926b3'),(2,'#ed831b')]:
            mask=(owner[k]==value).astype(float)
            if mask.max()>0:ax.contour(lo,la,mask,levels=[.5],colors=[col],linewidths=.65)
        for i,border in enumerate(borders):
            for p in lines(border):ax.plot(p[:,0],p[:,1],color='black' if i==0 else 'gray',linewidth=.55 if i==0 else .2)
        ax.set(xlim=(-79,-71),ylim=(.8,13),xlabel='Longitud WGS84',ylabel='Latitud WGS84',title=f'Profundidad {z[k]:.0f} km');ax.set_aspect('equal')
    fig.colorbar(im,ax=axs.ravel().tolist(),label='Vp inicial 3D respecto al fondo 1D (%)',shrink=.7)
    fig.suptitle('Modelo inicial: 1D + bandas Nazca y Caribe de 30 km\nContornos morado/naranja: huellas del escenario de placas',fontsize=14)
    fig.savefig(OUT/'modelo_inicial_cortes.png',dpi=150);plt.close(fig)
    (OUT/'vista_hashes.json').write_text(json.dumps({str(p):sha256(p) for p in [Path(__file__),OUT/'vp_inicial_1d_placas_30km.npz',path,meta]},indent=2),encoding='utf-8')
    print('Initial model slices exported',flush=True)
if __name__=='__main__':main()
