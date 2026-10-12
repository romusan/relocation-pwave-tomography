from pathlib import Path
import numpy as np,pandas as pd,json
from campana_factorizada_cotas_20261011g import GRID,nodes_for
root=Path(__file__).resolve().parents[1]/'resultados/campana_factorizada_cotas_20261011g'
p=pd.read_csv(root/'synthetic_03pct/synthetic_observations.csv',float_precision='round_trip')
p=p[p.split.eq('train')];events=p.drop_duplicates('event_id');nodes=nodes_for(p)
known=events[['true_x_km','true_y_km','true_depth_km']].to_numpy()
cat=events[['source_x_km','source_y_km','ev_prof_km']].to_numpy()
snapped=np.array([GRID.xyz(np.array(nodes[e])) for e in events.event_id])
rmse=lambda xyz:float(np.sqrt(np.mean(np.sum((xyz-known)**2,axis=1))))
r=dict(training_events=len(events),continuous_catalog_rmse_km=rmse(cat),initial_search_nodes_rmse_km=rmse(snapped),grid_spacing_km=GRID.spacing,definition='Euclidean 3D RMS before inversion; same frozen perturbations for both amplitudes')
(root/'hypocentre_initial_baseline.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(r)
