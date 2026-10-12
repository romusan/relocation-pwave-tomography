from pathlib import Path
import numpy as np,pandas as pd
from scipy.interpolate import RegularGridInterpolator
from modelo_3d_continuo import grd
R=Path(__file__).resolve().parents[1]
e=pd.read_csv(R/'resultados/migracion_nazca_cauca_valle_eje_1d_2026-10-10/relocated_train_events.csv')
x,y,z=grd(R/'datos/slab2_2026-10-10/sam_slab2_dep_02.23.18.grd');z=-z
f=RegularGridInterpolator((y,x),z,bounds_error=False,fill_value=np.nan)
e['slab']=f(e[['lat','lon']]);e['match']=(e.depth_km-e.slab-15).abs().le(35)
d=e[e.status.eq('relocated')&e.depth_km.ge(40)&e.gap_deg.le(180)&e.lon.le(-74)&e.lat.lt(8)]
print(d.groupby(np.floor(d.lat)).agg(n=('event_id','size'),compatible=('match','sum'),slab=('slab','count'),lon_min=('lon','min'),lon_max=('lon','max'),z_med=('depth_km','median')).to_string())
xx,yy=np.meshgrid(x,y);valid=np.isfinite(z)&(xx>=-79)&(xx<=-74)&(yy>=.8)&(yy<=8)&(z<=180)
print('Slab nodes',valid.sum(),'latitude',yy[valid].min(),yy[valid].max(),'longitude',xx[valid].min(),xx[valid].max())
