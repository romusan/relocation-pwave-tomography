"""Strict loading of a physical-grid initial Vp model; no silent reprojection."""
import json
import numpy as np

def load_initial_model(path,grid,domain):
    with np.load(path,allow_pickle=False) as m:
        for name,expected in [('xg',grid.x),('yg',grid.y),('zg',grid.z)]:
            if not np.array_equal(m[name],expected):raise ValueError('Initial model grid mismatch: '+name)
        if json.loads(str(m['domain_json']))!=domain:raise ValueError('Initial model domain mismatch')
        vp=m['vp'].copy()
    if vp.shape!=grid.shape or not np.isfinite(vp).all() or np.any((vp<4)|(vp>8.8)):
        raise ValueError('Invalid initial Vp volume')
    return vp
