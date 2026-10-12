"""Shared campaign geography with fixed original projection and wider south."""
from pathlib import Path
import json
import numpy as np
from shapely.geometry import shape,Point
from shapely import make_valid
from regional.geometry import Projection
ROOT=Path(__file__).resolve().parents[1]
DEPTS={'19':'Cauca','76':'Valle_del_Cauca','17':'Eje_Cafetero','63':'Eje_Cafetero','66':'Eje_Cafetero'}
def projection(domain):
    # Preserve old physical coordinates while extending bounds southward.
    origin=domain.get('projection_reference_bounds',domain)
    return Projection(origin)
def features():
    data=json.loads((ROOT/'datos/cartografia/departamentos_dane_2025.geojson').read_text(encoding='utf-8'))
    return {str(f['properties']['dpto_ccdgo']).zfill(2):make_valid(shape(f['geometry'])) for f in data['features']}
def departments(lat,lon):
    fs=features();out=[]
    for la,lo in zip(lat,lon):
        p=Point(float(lo),float(la));out.append(next((DEPTS[k] for k,g in fs.items() if k in DEPTS and g.covers(p)),'fuera'))
    return np.array(out)
