"""Explicit measured initial/final velocity comparison, without repeating inversion."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'resultados/tomografia_1d_adjunto_consistente_2026-10-10'
OUT=ROOT/'articulo_boletin_geologia_uis_adjunto'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
with np.load(RUN/'vp_final.npz') as f:
    ini=RegularGridInterpolator((f['zg'],f['yg'],f['xg']),f['vp_initial'])
    fin=RegularGridInterpolator((f['zg'],f['yg'],f['xg']),f['vp'])
    active=f['active'].astype(bool);cz,cy,cx=np.meshgrid(f['inversion_zg'],f['inversion_yg'],f['inversion_xg'],indexing='ij')
    q=np.c_[cz[active],cy[active],cx[active]];v0=ini(q);v1=fin(q)
    depth_direct=f['zg'].copy();profile_direct=f['vp_initial'][:,0,0].copy()
rows=[]
for z in np.unique(q[:,0]):
    mask=q[:,0]==z;vv=v1[mask];ref=v0[mask]
    np.testing.assert_allclose(ref,ref[0],atol=1e-12)
    rows.append(dict(depth_km=float(z),vp_initial_km_s=float(ref[0]),vp_final_min_km_s=float(vv.min()),
        vp_final_p05_km_s=float(np.quantile(vv,.05)),vp_final_median_km_s=float(np.median(vv)),
        vp_final_p95_km_s=float(np.quantile(vv,.95)),vp_final_max_km_s=float(vv.max()),active_cells=int(mask.sum())))
assert sum(r['active_cells'] for r in rows)==10011
profile=pd.DataFrame(rows);profile.to_csv(OUT/'fuentes/comparacion_modelos_vp_profundidad.csv',index=False)
point_table=pd.DataFrame(dict(depth_km=q[:,0],y_km=q[:,1],x_km=q[:,2],vp_initial_km_s=v0,vp_final_km_s=v1))
point_table.to_csv(OUT/'fuentes/modelos_vp_celdas_activas.csv',index=False)
plt.rcParams.update({'font.size':14,'font.family':'DejaVu Sans','axes.labelsize':14,'axes.titlesize':14})
fig,axs=plt.subplots(1,2,figsize=(10,7),layout='constrained',sharey=True)
blue='#2166ac';orange='#b35806';z=profile.depth_km.to_numpy()
axs[0].plot(profile_direct,depth_direct,c=blue,lw=2.3,label='Vp inicial 1D')
nodes=pd.read_csv(ROOT/'datos/perfil_vp_vmm_transferencia.csv')
axs[0].scatter(nodes.vp_km_s,nodes.depth_km,c=blue,s=30,zorder=3)
axs[0].set_title('a  Modelo inicial 1D')
axs[1].fill_betweenx(z,profile.vp_final_min_km_s,profile.vp_final_max_km_s,color=orange,alpha=.12,label='Mínimo–máximo final')
axs[1].fill_betweenx(z,profile.vp_final_p05_km_s,profile.vp_final_p95_km_s,color=orange,alpha=.32,label='Percentiles 5–95 finales')
axs[1].plot(profile.vp_final_median_km_s,z,c=orange,lw=2.3,label='Mediana final por nivel')
axs[1].plot(profile_direct,depth_direct,c=blue,lw=1.6,ls='--',label='Referencia inicial 1D')
axs[1].set_title('b  Modelo final 3D por niveles')
for ax in axs:
    ax.set(xlabel='Vp (km/s)',xlim=(4.4,9),ylim=(180,0),yticks=[0,20,40,60,80,100,120,140,160,180])
    ax.grid(alpha=.18);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.1f}'.replace('.',',')))
    ax.legend(loc='lower left',fontsize=10,frameon=False)
axs[0].set_ylabel('Profundidad (km)');axs[1].tick_params(labelleft=True)
image=OUT/'figuras/Figura_4_modelos_velocidad.png';assert not image.exists();fig.savefig(image,dpi=300);plt.close(fig)
audit={'description':'Initial 1D profile and depth-wise statistics of final 3D Vp at active inversion nodes; no 1D final model is inferred',
    'rows':rows,'active_cells':len(q),'inputs':{str(p):sha(p) for p in [Path(__file__),RUN/'vp_final.npz',ROOT/'datos/perfil_vp_vmm_transferencia.csv']},
    'outputs':{str(p):sha(p) for p in [image,OUT/'fuentes/comparacion_modelos_vp_profundidad.csv',OUT/'fuentes/modelos_vp_celdas_activas.csv']}}
(OUT/'fuentes/auditoria_modelos_velocidad.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(rows,indent=2),flush=True)
