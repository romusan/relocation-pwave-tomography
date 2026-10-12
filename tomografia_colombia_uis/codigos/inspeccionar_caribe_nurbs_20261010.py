from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1]
e=pd.read_csv(R/'resultados/migracion_nazca_cauca_valle_eje_1d_2026-10-10/relocated_train_events.csv')
line=lambda lon:6+(lon+72.5)*(9.2-6)/(-76.3+72.5)
e=e[e.status.eq('relocated')&e.depth_km.ge(40)&e.gap_deg.le(180)]
c=e[e.lat>=line(e.lon)]
print('Caribe candidate',len(c),c[['lon','lat','depth_km']].describe().to_string())
print(c.groupby(np.floor(c.lat)).agg(n=('event_id','size'),median_z=('depth_km','median')).to_string())
